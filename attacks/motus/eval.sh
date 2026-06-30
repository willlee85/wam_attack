#!/bin/bash
# Motus RoboTwin input-noise attack evaluation (standalone entry script)

set -euo pipefail

TASK_NAME="${TASK_NAME:-click_alarmclock}"
GPU_ID="${GPU_ID:-6}"
TEST_NUM="${TEST_NUM:-20}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ATTACK_CONFIG="${SCRIPT_DIR}/config.yaml"

if [ ! -f "$ATTACK_CONFIG" ]; then
    echo "Error: attack config not found: $ATTACK_CONFIG"
    exit 1
fi

PATHS_CONFIG=$(grep "^paths_config:" "$ATTACK_CONFIG" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
NOISE_PATH=$(grep "^  noise_path:" "$ATTACK_CONFIG" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
ATTACK_ENABLED=$(grep "^  enabled:" "$ATTACK_CONFIG" | sed 's/#.*//' | sed 's/.*: *//' | xargs)

if [ -z "$PATHS_CONFIG" ] || [ ! -f "$PATHS_CONFIG" ]; then
    echo "Error: paths_config not found: $PATHS_CONFIG"
    exit 1
fi

ROBOTWIN_ROOT=$(grep "^robotwin_root:" "$PATHS_CONFIG" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CONDA_ENV=$(grep "^conda_env:" "$PATHS_CONFIG" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
CHECKPOINT_PATH=$(grep "^checkpoint_path:" "$PATHS_CONFIG" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
WAN_PATH=$(grep "^wan_path:" "$PATHS_CONFIG" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
VLM_PATH=$(grep "^vlm_path:" "$PATHS_CONFIG" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
TASK_CONFIG=$(grep "^task_config:" "$PATHS_CONFIG" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)
SEED=$(grep "^seed:" "$PATHS_CONFIG" | sed 's/#.*//' | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)

TASK_CONFIG=${TASK_CONFIG:-"demo_randomized"}
SEED=${SEED:-"42"}
ATTACK_ENABLED=${ATTACK_ENABLED:-"true"}
NOISE_PATH="${NOISE_PATH:-${FASTWAM_ROOT}/evaluate_results/video_noise_attack_train/universal_libero/best_noise.pt}"

if [ ! -f "$NOISE_PATH" ]; then
    echo "Error: noise file not found: $NOISE_PATH"
    exit 1
fi

echo "Starting Motus attack evaluation at $(date)"
echo "Attack enabled: ${ATTACK_ENABLED}"
echo "Noise path: ${NOISE_PATH}"

cd "$ROBOTWIN_ROOT"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

export PYTHONPATH="${FASTWAM_ROOT}:${SCRIPT_DIR}:${ROBOTWIN_ROOT}:${ROBOTWIN_ROOT}/policy:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TRITON_CACHE_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

LOG_DIR="${SCRIPT_DIR}/logs_attack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${TASK_NAME}.log"

echo ""
echo "================================================================"
echo "Motus Attack Evaluation"
echo "================================================================"
echo "Task Name:         $TASK_NAME"
echo "GPU:               $GPU_ID"
echo "Test Episodes:     $TEST_NUM"
echo "RoboTwin Root:     $ROBOTWIN_ROOT"
echo "Checkpoint:        $CHECKPOINT_PATH"
echo "Noise Path:        $NOISE_PATH"
echo "Log File:          $LOG_FILE"
echo "================================================================"
echo ""

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
    --config "${SCRIPT_DIR}/deploy_policy.yml" \
    --overrides \
    --task_name "${TASK_NAME}" \
    --task_config "${TASK_CONFIG}" \
    --ckpt_setting "${CHECKPOINT_PATH}" \
    --seed "${SEED}" \
    --policy_name "motus_attack" \
    --test_num "${TEST_NUM}" \
    --log_dir "${LOG_DIR}" \
    --wan_path "${WAN_PATH}" \
    --vlm_path "${VLM_PATH}" \
    --attack_enabled "${ATTACK_ENABLED}" \
    --attack_noise_path "${NOISE_PATH}" \
    2>&1 | tee "$LOG_FILE"

exit_code=${PIPESTATUS[0]}

echo ""
if [ $exit_code -eq 0 ]; then
    echo "Attack eval completed: $TASK_NAME"
else
    echo "Attack eval failed: $TASK_NAME (exit $exit_code)"
    echo "Log file: $LOG_FILE"
fi

exit $exit_code
