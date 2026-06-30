"""Apply FastWAM optimized input noise to Motus RoboTwin observations."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

FASTWAM_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = FASTWAM_ROOT / "Motus" / "RoboTwin"
ROBOTWIN_POLICY = ROBOTWIN_ROOT / "policy"

for path in (FASTWAM_ROOT, ROBOTWIN_ROOT, ROBOTWIN_POLICY, ROBOTWIN_POLICY / "Motus"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from attacks.fastwam.common.noise import LoadedNoiseAttack, apply_noise_to_tensor_image
from Motus.deploy_policy import MotusPolicy, encode_obs, eval, logger, reset_model


def _load_attack_config(config_path: Path | None = None) -> dict:
    config_path = config_path or Path(__file__).resolve().parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _extract_composite_rgb(observation: dict) -> np.ndarray:
    """Build the same 3-camera composite image used by MotusPolicy."""
    if "observation" in observation:
        obs_data = observation["observation"]
        if (
            "head_camera" in obs_data
            and "left_camera" in obs_data
            and "right_camera" in obs_data
        ):
            head_img = obs_data["head_camera"]["rgb"]
            left_img = obs_data["left_camera"]["rgb"]
            right_img = obs_data["right_camera"]["rgb"]
            left_img_resized = cv2.resize(left_img, (160, 120))
            right_img_resized = cv2.resize(right_img, (160, 120))
            bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
            return np.concatenate([head_img, bottom_row], axis=0)
        raise ValueError("Missing camera data in observation")

    if "head_camera" in observation:
        return observation["head_camera"]
    if "image" in observation:
        return observation["image"]
    raise ValueError("No visual observation found")


class MotusAttackPolicy(MotusPolicy):
    """Motus policy wrapper that applies input noise before or after model resize."""

    def __init__(
        self,
        *args,
        attack_enabled: bool = False,
        attack_noise_path: str | None = None,
        apply_after_resize: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_attack = None
        self.apply_after_resize = bool(apply_after_resize)
        if attack_enabled:
            if not attack_noise_path:
                raise ValueError("attack_noise_path is required when attack_enabled=true")
            self.image_attack = LoadedNoiseAttack(enabled=True, noise_path=attack_noise_path)
            logger.info(
                "Input noise attack enabled (apply_after_resize=%s): %s",
                self.apply_after_resize,
                self.image_attack.describe(),
            )

    def update_obs(self, observation):
        if self.image_attack is None or not self.image_attack.is_enabled():
            return super().update_obs(observation)

        if self.apply_after_resize:
            super().update_obs(observation)
            frame = self.obs_cache[-1].squeeze(0)
            noise = self.image_attack.noise_tensor.to(device=frame.device, dtype=frame.dtype)
            if noise.shape[-2:] != frame.shape[-2:]:
                noise = __import__("torch").nn.functional.interpolate(
                    noise.unsqueeze(0),
                    size=frame.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            attacked = apply_noise_to_tensor_image(frame, noise)
            self.obs_cache[-1] = attacked.unsqueeze(0).to(self.device)
            return

        attacked_image, _ = self.image_attack.apply(_extract_composite_rgb(observation))
        patched_observation = {
            "joint_action": observation["joint_action"],
            "image": attacked_image,
        }
        return super().update_obs(patched_observation)


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_model(usr_args):
    attack_enabled = _parse_bool(usr_args.get("attack_enabled", False))
    attack_noise_path = usr_args.get("attack_noise_path")

    attack_cfg = _load_attack_config().get("attack", {})
    if attack_noise_path in (None, "", "null"):
        attack_enabled = _parse_bool(attack_cfg.get("enabled", attack_enabled))
        attack_noise_path = attack_cfg.get("noise_path")
    apply_after_resize = _parse_bool(
        usr_args.get("attack_apply_after_resize", attack_cfg.get("apply_after_resize", False))
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    wan_path = usr_args.get("wan_path")
    vlm_path = usr_args.get("vlm_path")
    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")

    policy_dir = ROBOTWIN_POLICY / "Motus"
    config_path = policy_dir / "utils" / "robotwin.yml"
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    return MotusAttackPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get("log_dir"),
        task_name=usr_args.get("task_name"),
        attack_enabled=attack_enabled,
        attack_noise_path=attack_noise_path,
        apply_after_resize=apply_after_resize,
    )


__all__ = ["MotusAttackPolicy", "encode_obs", "eval", "get_model", "reset_model"]
