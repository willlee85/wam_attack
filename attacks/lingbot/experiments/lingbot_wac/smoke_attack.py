from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch.distributed as dist
from PIL import Image

from wan_va.configs import VA_CONFIGS
from wan_va.distributed.util import init_distributed
from wan_va.wan_va_server import VA_Server

from .attack import attack_observation


def main() -> None:
    parser = argparse.ArgumentParser(description="One-observation LingBot WAC smoke test")
    parser.add_argument("--config-name", default="libero")
    parser.add_argument("--output-dir", default="outputs/lingbot_wac_smoke")
    parser.add_argument("--prompt", default="put the black bowl in the bottom drawer")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--layers", default="29")
    parser.add_argument("--denoise-steps", default="0")
    parser.add_argument("--surrogate-video-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Initialize the same single-rank FSDP path used by launch_server.sh.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.distributed:
        init_distributed(world_size, local_rank, rank)

    config = deepcopy(VA_CONFIGS[args.config_name])
    config.local_rank = local_rank
    config.rank = rank
    config.world_size = world_size
    config.save_root = str(output_dir / "server_artifacts")
    server = VA_Server(config)
    server._reset(args.prompt)

    image_root = Path(__file__).resolve().parents[2] / "example" / "libero"
    frame = {
        key: np.asarray(Image.open(image_root / f"{key}.png").convert("RGB"))
        for key in config.obs_cam_keys
    }
    observation = {"obs": frame, "prompt": args.prompt}
    result = attack_observation(
        server,
        observation,
        {
            "num_steps": args.steps,
            "layers": args.layers,
            "denoise_steps": args.denoise_steps,
            "surrogate_video_steps": args.surrogate_video_steps,
            "seed": args.seed,
        },
    )
    np.savez_compressed(output_dir / "delta.npz", **result.delta_by_key)
    (output_dir / "metrics.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(result.metrics, indent=2))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
