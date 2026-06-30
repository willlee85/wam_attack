#!/bin/bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/train_config.yaml"

eval "$(conda shell.bash hook)"
conda activate /DATA_EDS2/AIGC/2312/xuhr2312/.conda/envs/robotwin

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${FASTWAM_ROOT}:${SCRIPT_DIR}:${FASTWAM_ROOT}/Motus/RoboTwin:${FASTWAM_ROOT}/Motus/RoboTwin/policy:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TRITON_CACHE_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${FASTWAM_ROOT}/Motus/RoboTwin"
python "${SCRIPT_DIR}/train_noise.py" --config "${CONFIG}"
