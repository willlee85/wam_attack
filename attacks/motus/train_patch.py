"""White-box 2D patch training for Motus on RoboTwin."""

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
from PIL import Image

ATTACKS_MOTUS_DIR = Path(__file__).resolve().parent
FASTWAM_ROOT = ATTACKS_MOTUS_DIR.parents[1]
ROBOTWIN_ROOT = FASTWAM_ROOT / "Motus" / "RoboTwin"
ROBOTWIN_POLICY = ROBOTWIN_ROOT / "policy"

for path in (FASTWAM_ROOT, ATTACKS_MOTUS_DIR, ROBOTWIN_ROOT, ROBOTWIN_POLICY, ROBOTWIN_POLICY / "Motus"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from attacks.fastwam.common.learnable_patch import LearnablePatch
from attacks.fastwam.common.patch_apply import apply_patch_to_tensor_image
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


def _read_last_iter(metrics_path: Path) -> int | None:
    if not metrics_path.exists():
        return None
    last_iter = None
    with open(metrics_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            last_iter = int(json.loads(line)["iter"])
    return last_iter


def _save_patch_checkpoint(
    learnable_patch: LearnablePatch,
    output_path: Path,
    *,
    upscale: int = 8,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch_tensor = learnable_patch().detach().cpu()
    learnable_patch.save(output_path)

    rgb = patch_tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    img = Image.fromarray((rgb * 255.0).astype(np.uint8))
    png_path = output_path.with_name(output_path.stem + "_rgb.png")
    img.save(png_path)
    if upscale > 1:
        up_path = output_path.with_name(output_path.stem + f"_rgb_x{upscale}.png")
        img.resize((img.width * upscale, img.height * upscale), Image.NEAREST).save(up_path)
    else:
        up_path = png_path
    return png_path, up_path


def _build_policy(device: str, paths: dict) -> MotusPolicy:
    policy_dir = ROBOTWIN_POLICY / "Motus"
    return MotusPolicy(
        checkpoint_path=paths["checkpoint_path"],
        wan_path=paths["wan_path"],
        vlm_path=paths["vlm_path"],
        config_path=str(policy_dir / "utils" / "robotwin.yml"),
        device=device,
    )


def train_patch(cfg: dict) -> Path:
    device = str(cfg.get("device", "cuda"))
    paths = _load_paths_config()
    train_cfg = cfg["train"]
    patch_cfg = train_cfg.get("patch", {})
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
    rel_x = float(patch_cfg.get("rel_x", 0.7))
    rel_y = float(patch_cfg.get("rel_y", 0.1))
    rel_width = float(patch_cfg.get("rel_width", 0.16))
    rel_height = float(patch_cfg.get("rel_height", 0.16))
    alpha = float(patch_cfg.get("alpha", 1.0))

    patch_h = max(1, int(round(video_h * rel_height)))
    patch_w = max(1, int(round(video_w * rel_width)))
    learnable_patch = LearnablePatch(
        height=patch_h,
        width=patch_w,
        init_mode=str(patch_cfg.get("init_mode", "random")),
        device=device,
    )

    resume_from = train_cfg.get("resume_from")
    if resume_from:
        resume_path = Path(str(resume_from)).expanduser().resolve()
        if resume_path.exists():
            resume_tensor = torch.load(resume_path, map_location=device, weights_only=False)
            with torch.no_grad():
                learnable_patch.patch.copy_(
                    resume_tensor.to(device=learnable_patch.patch.device, dtype=learnable_patch.patch.dtype)
                )
            print(f"Resumed patch weights from {resume_path}")
        else:
            print(f"Warning: resume_from not found, training from init: {resume_path}")

    optimizer = torch.optim.Adam([learnable_patch.patch], lr=float(train_cfg.get("lr", 0.01)))
    num_inference_steps = int(train_cfg.get("num_inference_steps", 10))
    num_states = int(train_cfg.get("num_states", 8))
    num_iters = int(train_cfg.get("num_iters", 200))
    save_every = int(train_cfg.get("save_every", 1000))
    seed_start = int(train_cfg.get("seed_start", 4300000))
    loss_type = str(train_cfg.get("loss_type", "mse"))

    print(f"Patch size: {patch_w}x{patch_h}, geometry rel=({rel_x},{rel_y},{rel_width},{rel_height})")
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
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_objective = float("-inf")
    best_patch_path = output_dir / "best_patch.pt"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                best_objective = max(best_objective, float(json.loads(line)["objective"]))

    start_iter = int(train_cfg.get("start_iter", 0))
    if start_iter <= 0 and resume_from and metrics_path.exists():
        last_iter = _read_last_iter(metrics_path)
        if last_iter is not None:
            start_iter = last_iter + 1
    if start_iter >= num_iters:
        print(f"Already completed {start_iter} iters (target={num_iters}), nothing to do.")
        return best_patch_path
    if start_iter > 0:
        print(f"Continuing training from iter {start_iter} to {num_iters - 1}")

    for iter_idx in range(start_iter, num_iters):
        sample = prepared[iter_idx % len(prepared)]
        clean_frame = sample["frame"]
        adv_frame, geometry = apply_patch_to_tensor_image(
            clean_frame,
            learnable_patch(),
            rel_x=rel_x,
            rel_y=rel_y,
            rel_width=rel_width,
            rel_height=rel_height,
            alpha=alpha,
            random_translate=False,
        )

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
            ).detach()
        torch.cuda.empty_cache()
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
        torch.nn.utils.clip_grad_norm_(
            [learnable_patch.patch],
            max_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        )
        optimizer.step()
        learnable_patch.clamp_()

        patch_tensor = learnable_patch().detach()
        metric = {
            "iter": iter_idx,
            "seed": sample["seed"],
            "objective": objective,
            "loss": float(loss.detach().item()),
            "patch_mean": float(patch_tensor.mean().item()),
            "patch_x": int(geometry["x"]),
            "patch_y": int(geometry["y"]),
            "patch_w": int(geometry["patch_width"]),
            "patch_h": int(geometry["patch_height"]),
        }
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric) + "\n")
        if objective > best_objective:
            best_objective = objective
            learnable_patch.save(best_patch_path)

        checkpoint_iter = iter_idx + 1
        if save_every > 0 and checkpoint_iter % save_every == 0:
            ckpt_path = checkpoints_dir / f"patch_iter{checkpoint_iter:05d}.pt"
            png_path, up_path = _save_patch_checkpoint(learnable_patch, ckpt_path)
            print(
                f"[checkpoint] iter={checkpoint_iter} saved {ckpt_path.name}, "
                f"viz={png_path.name}, {up_path.name}, best_objective={best_objective:.6f}"
            )

        if iter_idx % max(1, int(train_cfg.get("log_every", 10))) == 0:
            print(
                f"[{iter_idx}/{num_iters}] objective={objective:.6f} "
                f"patch_mean={metric['patch_mean']:.4f} seed={sample['seed']}"
            )

    final_patch_path = output_dir / "final_patch.pt"
    _save_patch_checkpoint(learnable_patch, final_patch_path)
    print(f"Training done. best_patch={best_patch_path}, best_objective={best_objective:.6f}")
    return best_patch_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Motus white-box 2D patch")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ATTACKS_MOTUS_DIR / "train_patch_config.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    os.chdir(ROBOTWIN_ROOT)
    args = parse_args()
    cfg = _load_yaml(Path(args.config))
    if cfg.get("train", {}).get("output_dir") is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.setdefault("train", {})["output_dir"] = str(
            ATTACKS_MOTUS_DIR / "outputs" / cfg["train"]["task_name"] / f"patch_{stamp}"
        )
    train_patch(cfg)


if __name__ == "__main__":
    main()
