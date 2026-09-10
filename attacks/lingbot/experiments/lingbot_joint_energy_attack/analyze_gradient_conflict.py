from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from wan_va.configs import VA_CONFIGS
from wan_va.wan_va_server import VA_Server

from .gradient_conflict import compute_term_gradients


def _split_real_row(row: np.ndarray) -> dict[str, np.ndarray]:
    """Split the Real Observation row from eval visualization back into 3 cameras."""

    if row.ndim != 3 or row.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB row, got {row.shape}")
    height, width, _ = row.shape
    third = width // 3
    chunks = [
        row[:, 0:third],
        row[:, third : 2 * third],
        row[:, 2 * third :],
    ]
    return {
        "observation.images.cam_high": chunks[0],
        "observation.images.cam_left_wrist": chunks[1],
        "observation.images.cam_right_wrist": chunks[2],
    }


def observation_from_eval_video(video_path: Path, *, title_bar_height: int = 40) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    real_row = frame[title_bar_height : title_bar_height + frame.shape[0] // 2]
    return {"obs": _split_real_row(real_row)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure cosine conflict between joint attack surrogate gradients"
    )
    parser.add_argument("--config-name", default="robotwin")
    parser.add_argument(
        "--video-path",
        required=True,
        help="Episode visualization mp4; first frame supplies the 3-camera observation",
    )
    parser.add_argument("--prompt", default="Lift the long thin bottle head-up from the table.")
    parser.add_argument("--output-json", default="outputs/gradient_conflict/report.json")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--denoise-steps", default="0")
    parser.add_argument("--weight-vae", type=float, default=1.0)
    parser.add_argument("--weight-video", type=float, default=1.0)
    parser.add_argument("--weight-wac", type=float, default=1.0)
    args = parser.parse_args()

    config = deepcopy(VA_CONFIGS[args.config_name])
    config.local_rank = 0
    config.rank = 0
    config.world_size = 1
    server = VA_Server(config)
    server._reset(args.prompt)

    observation = observation_from_eval_video(Path(args.video_path))
    observation["prompt"] = args.prompt
    attack_config = {
        "layers": args.layers,
        "denoise_steps": args.denoise_steps,
        "weight_vae": args.weight_vae,
        "weight_video": args.weight_video,
        "weight_wac": args.weight_wac,
        "num_steps": 0,
        "seed": 42,
    }

    report = {
        "video_path": str(Path(args.video_path).resolve()),
        "prompt": args.prompt,
        "at_delta_zero": compute_term_gradients(
            server, observation, attack_config, delta=None
        ),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
