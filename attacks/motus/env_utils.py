"""Sample RoboTwin observations for Motus white-box noise training."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from motus_attack import _extract_composite_rgb

FASTWAM_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = FASTWAM_ROOT / "Motus" / "RoboTwin"


def _ensure_robotwin_paths() -> None:
    for path in (ROBOTWIN_ROOT, ROBOTWIN_ROOT / "policy", ROBOTWIN_ROOT / "script"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _get_embodiment_config(robot_file: str) -> dict[str, Any]:
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as handle:
        return yaml.load(handle.read(), Loader=yaml.FullLoader)


def _load_task_args(task_name: str, task_config: str) -> dict[str, Any]:
    from envs import CONFIGS_PATH

    config_path = ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as handle:
        args = yaml.load(handle.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["render_freq"] = 0
    args["eval_mode"] = True

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as handle:
        embodiment_types = yaml.load(handle.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_name: str) -> str:
        robot_file = embodiment_types[embodiment_name]["file_path"]
        if robot_file is None:
            raise ValueError(f"No embodiment file for {embodiment_name}")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as handle:
        camera_config = yaml.load(handle.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])
    return args


def sample_task_observations(
    *,
    task_name: str,
    task_config: str,
    num_states: int,
    seed_start: int,
) -> list[dict[str, Any]]:
    _ensure_robotwin_paths()
    os.chdir(ROBOTWIN_ROOT)
    from envs.utils.create_actor import UnStableError
    from test_render import Sapien_TEST

    Sapien_TEST()

    args = _load_task_args(task_name, task_config)
    envs_module = importlib.import_module(f"envs.{task_name}")
    task_env = getattr(envs_module, task_name)()

    samples: list[dict[str, Any]] = []
    now_seed = int(seed_start)
    print(f"Sampling {num_states} states for {task_name} from seed {now_seed} ...", flush=True)
    while len(samples) < int(num_states):
        try:
            task_env.setup_demo(now_ep_num=len(samples), seed=now_seed, is_test=True, **args)
            observation = task_env.get_obs()
            instruction = task_env.get_instruction()
            composite = _extract_composite_rgb(observation)
            state = observation["joint_action"]["vector"]
            samples.append(
                {
                    "seed": now_seed,
                    "instruction": instruction,
                    "observation": observation,
                    "composite_rgb": composite,
                    "state": state,
                }
            )
            print(f"  collected state {len(samples)}/{num_states} seed={now_seed}", flush=True)
            task_env.close_env(clear_cache=False)
        except UnStableError:
            task_env.close_env(clear_cache=False)
        except Exception as exc:
            print(f"  skip seed={now_seed}: {exc}", flush=True)
            task_env.close_env(clear_cache=True)
        now_seed += 1
    return samples
