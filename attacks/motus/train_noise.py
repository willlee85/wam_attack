"""White-box universal input noise training for Motus on RoboTwin."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ATTACKS_MOTUS_DIR = Path(__file__).resolve().parent
FASTWAM_ROOT = ATTACKS_MOTUS_DIR.parents[1]
ROBOTWIN_ROOT = FASTWAM_ROOT / "Motus" / "RoboTwin"
ROBOTWIN_POLICY = ROBOTWIN_ROOT / "policy"

for path in (FASTWAM_ROOT, ATTACKS_MOTUS_DIR, ROBOTWIN_ROOT, ROBOTWIN_POLICY, ROBOTWIN_POLICY / "Motus"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from attacks.fastwam.common.noise import LearnableImageNoise, apply_noise_to_tensor_image
from env_utils import sample_task_observations
from whitebox import (
    build_initial_latents,
    compute_action_deviation_loss,
    encode_first_frame_latents,
    infer_actions_whitebox,
)
from Motus.deploy_policy import MotusPolicy
from utils.image_utils import resize_with_padding


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_paths_config() -> dict:
    cfg = _load_yaml(ATTACKS_MOTUS_DIR / "config.yaml")
    return _load_yaml(Path(cfg["paths_config"]))


def _prepare_model_inputs(
    policy: MotusPolicy,
    sample: dict,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, list, list]:
    composite = sample["composite_rgb"]
    resized = resize_with_padding(composite, (
        policy.config_dict["common"]["video_height"],
        policy.config_dict["common"]["video_width"],
    ))
    if resized.dtype == np.uint8:
        resized = resized.astype(np.float32) / 255.0
    frame = torch.from_numpy(resized).permute(2, 0, 1).to(device=device)

    state = torch.from_numpy(np.asarray(sample["state"], dtype=np.float32)).unsqueeze(0).to(device=device)
    scene_prefix = (
        "The whole scene is in a realistic, industrial art style with three views: "
        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
        "The aloha robot is currently performing the following task: "
    )
    instruction = f"{scene_prefix}{sample['instruction']}"
    t5_out = policy.t5_encoder([instruction], device)
    if isinstance(t5_out, torch.Tensor):
        t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
    else:
        t5_list = t5_out

    pil_image = policy._tensor_to_pil_image(frame.cpu())
    vlm_inputs = policy._preprocess_vlm_messages(instruction, pil_image)
    return frame.to(device=device), state, t5_list, [vlm_inputs]


def _build_policy(device: str, paths: dict) -> MotusPolicy:
    policy_dir = ROBOTWIN_POLICY / "Motus"
    return MotusPolicy(
        checkpoint_path=paths["checkpoint_path"],
        wan_path=paths["wan_path"],
        vlm_path=paths["vlm_path"],
        config_path=str(policy_dir / "utils" / "robotwin.yml"),
        device=device,
    )


def train_noise(cfg: dict) -> Path:
    device = str(cfg.get("device", "cuda"))
    paths = _load_paths_config()
    train_cfg = cfg["train"]
    task_name = str(train_cfg["task_name"])
    task_config = str(train_cfg.get("task_config", "demo_randomized"))

    output_dir = Path(train_cfg.get("output_dir", ATTACKS_MOTUS_DIR / "outputs" / task_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = _build_policy(device, paths)
    model = policy.model
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    video_h = int(policy.config_dict["common"]["video_height"])
    video_w = int(policy.config_dict["common"]["video_width"])
    learnable_noise = LearnableImageNoise(
        channels=3,
        height=video_h,
        width=video_w,
        epsilon=float(train_cfg.get("epsilon", 8.0 / 255.0)),
        init_std=float(train_cfg.get("init_std", 1.0 / 255.0)),
        device=device,
    )

    optimizer = torch.optim.Adam([learnable_noise.noise], lr=float(train_cfg.get("lr", 0.01)))
    num_inference_steps = int(train_cfg.get("num_inference_steps", 10))
    num_states = int(train_cfg.get("num_states", 8))
    num_iters = int(train_cfg.get("num_iters", 200))
    seed_start = int(train_cfg.get("seed_start", 4300000))
    loss_type = str(train_cfg.get("loss_type", "mse"))

    print(f"Sampling {num_states} RoboTwin states for {task_name} ...")
    samples_cache = output_dir / "sampled_states.pt"
    if samples_cache.exists():
        print(f"Loading cached states from {samples_cache}")
        samples = torch.load(samples_cache, map_location="cpu")
    else:
        samples = sample_task_observations(
            task_name=task_name,
            task_config=task_config,
            num_states=num_states,
            seed_start=seed_start,
        )
        torch.save(samples, samples_cache)
        print(f"Cached states to {samples_cache}")
    prepared = []
    for sample in samples:
        frame, state, t5_list, vlm_inputs = _prepare_model_inputs(policy, sample, device=device)
        with torch.no_grad():
            cond_latent = encode_first_frame_latents(model, frame.unsqueeze(0))
            video_init, action_init = build_initial_latents(
                model,
                condition_frame_latent=cond_latent,
                seed=int(sample["seed"]),
            )
        prepared.append(
            {
                "frame": frame,
                "state": state,
                "t5_list": t5_list,
                "vlm_inputs": vlm_inputs,
                "video_init": video_init,
                "action_init": action_init,
                "seed": int(sample["seed"]),
            }
        )

    metrics_path = output_dir / "train_metrics.jsonl"
    best_objective = float("-inf")
    best_noise_path = output_dir / "best_noise.pt"

    for iter_idx in range(num_iters):
        sample = prepared[iter_idx % len(prepared)]
        clean_frame = sample["frame"]
        adv_frame = apply_noise_to_tensor_image(clean_frame, learnable_noise())

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            clean_action = infer_actions_whitebox(
                model,
                first_frame=clean_frame.unsqueeze(0),
                state=sample["state"],
                language_embeddings=sample["t5_list"],
                vlm_inputs=sample["vlm_inputs"],
                num_inference_steps=num_inference_steps,
                video_latent_init=sample["video_init"],
                action_latent_init=sample["action_init"],
            )
        adv_action = infer_actions_whitebox(
            model,
            first_frame=adv_frame.unsqueeze(0),
            state=sample["state"],
            language_embeddings=sample["t5_list"],
            vlm_inputs=sample["vlm_inputs"],
            num_inference_steps=num_inference_steps,
            video_latent_init=sample["video_init"],
            action_latent_init=sample["action_init"],
        )
        loss, objective = compute_action_deviation_loss(
            adv_action,
            clean_action,
            loss_type=loss_type,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([learnable_noise.noise], max_norm=float(train_cfg.get("grad_clip_norm", 1.0)))
        optimizer.step()
        learnable_noise.clamp_()

        noise_tensor = learnable_noise().detach()
        metric = {
            "iter": iter_idx,
            "seed": sample["seed"],
            "objective": objective,
            "loss": float(loss.detach().item()),
            "noise_linf": float(noise_tensor.abs().max().item()),
            "noise_l2": float(noise_tensor.pow(2).mean().item()),
        }
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric) + "\n")
        if objective > best_objective:
            best_objective = objective
            learnable_noise.save(best_noise_path)
        if iter_idx % max(1, int(train_cfg.get("log_every", 10))) == 0:
            print(
                f"[{iter_idx}/{num_iters}] objective={objective:.6f} "
                f"noise_linf={metric['noise_linf']:.6f} seed={sample['seed']}"
            )

    final_noise_path = output_dir / "final_noise.pt"
    learnable_noise.save(final_noise_path)
    print(f"Training done. best_noise={best_noise_path}")
    return best_noise_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Motus white-box input noise")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ATTACKS_MOTUS_DIR / "train_config.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    os.chdir(ROBOTWIN_ROOT)
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    if cfg.get("train", {}).get("output_dir") is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.setdefault("train", {})["output_dir"] = str(
            ATTACKS_MOTUS_DIR / "outputs" / cfg["train"]["task_name"] / stamp
        )
    train_noise(cfg)


if __name__ == "__main__":
    main()
