#!/usr/bin/env bash
set -euo pipefail

# Example RoboTwin joint-energy attack eval (shared delta, 10 episodes).
# Requires: lingbot-va server running on PORT with --no-fsdp.

LINGBOT_ROOT="${LINGBOT_ROOT:-/path/to/lingbot-va}"
PORT="${PORT:-29056}"
GPU_ID="${GPU_ID:-1}"
TASK_NAME="${TASK_NAME:-adjust_bottle}"
TEST_NUM="${TEST_NUM:-10}"
ST_SEED="${ST_SEED:-10000}"
SAVE_ROOT="${SAVE_ROOT:-outputs/robotwin_joint_energy_shared/attack}"
CLEAN_METRICS="${CLEAN_METRICS:-outputs/robotwin_clean/stseed-10000/metrics/adjust_bottle/res.json}"

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

cd "${LINGBOT_ROOT}"
python -m evaluation.robotwin.eval_polict_client_openpi \
  --config policy/ACT/deploy_policy.yml --overrides \
  --task_name "${TASK_NAME}" \
  --task_config demo_clean \
  --train_config_name 0 --model_name 0 --ckpt_setting 0 --seed 0 \
  --policy_name ACT \
  --save_root "${SAVE_ROOT}" \
  --video_guidance_scale 5 --action_guidance_scale 1 \
  --test_num "${TEST_NUM}" --port "${PORT}" \
  --st_seed_override "${ST_SEED}" \
  --inference_seed 42 --instruction_seed 42 \
  --paired_reference_metrics "${CLEAN_METRICS}" \
  --joint_energy_attack True \
  --joint_energy_layers all \
  --joint_energy_denoise_steps 0 \
  --joint_energy_weight_vae 1.0 \
  --joint_energy_weight_video 1.0 \
  --joint_energy_weight_wac 1.0
