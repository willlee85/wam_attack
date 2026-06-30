# wam_attack

Motus + RoboTwin + white-box input attack (noise / patch) for robotic VLA policies.

## Repository layout

```
wam_attack/
├── Motus/                 # Motus VLA model & training code
│   └── RoboTwin/          # RoboTwin simulation & evaluation
├── attacks/
│   ├── motus/             # Motus-specific attack train/eval scripts
│   └── fastwam/common/    # Shared noise/patch utilities
```

## Setup

1. Clone this repo and create conda env (see `Motus/requirements.txt`, RoboTwin docs).

2. Download pretrained models into `Motus/pretrained_models/`:
   - `Motus_robotwin2`
   - `Wan2.2-TI2V-5B`
   - `Qwen3-VL-2B-Instruct`

3. Download RoboTwin assets (see `Motus/RoboTwin/assets/_download.py`).

4. Edit `Motus/RoboTwin/policy/Motus/paths_config.yml` with your local paths.

## Attack workflow (Motus / RoboTwin)

```bash
conda activate robotwin
cd Motus/RoboTwin

# White-box noise training
GPU_ID=7 bash ../../attacks/motus/train.sh

# White-box patch training
GPU_ID=7 bash ../../attacks/motus/train_patch.sh

# Attack evaluation
GPU_ID=7 TASK_NAME=click_alarmclock bash ../../attacks/motus/eval.sh

# Clean baseline
bash policy/Motus/eval.sh
```

## Notes

- Large weights, datasets, eval videos, and trained attack artifacts are **not** included in git.
- Patch/noise training optimizes universal perturbations against Motus action deviation (see `attacks/motus/whitebox.py`).
