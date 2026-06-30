#!/bin/bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/train_patch_config.yaml"
OUTPUT_DIR="$(grep '^  output_dir:' "${CONFIG}" | sed 's/.*: *"\?\([^"]*\)"\?.*/\1/' | tr -d '"' | xargs)"
NOISE_STATES="${SCRIPT_DIR}/outputs/click_alarmclock_whitebox/sampled_states.pt"

eval "$(conda shell.bash hook)"
conda activate /DATA_EDS2/AIGC/2312/xuhr2312/.conda/envs/robotwin

mkdir -p "${OUTPUT_DIR}" "${TRITON_CACHE_DIR:-/DATA_EDS2/AIGC/2312/xuhr2312/tmp/triton_cache}"
if [ ! -f "${OUTPUT_DIR}/sampled_states.pt" ] && [ -f "${NOISE_STATES}" ]; then
    cp "${NOISE_STATES}" "${OUTPUT_DIR}/sampled_states.pt"
    echo "Reused sampled states from ${NOISE_STATES}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${FASTWAM_ROOT}:${SCRIPT_DIR}:${FASTWAM_ROOT}/Motus/RoboTwin:${FASTWAM_ROOT}/Motus/RoboTwin/policy:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TRITON_CACHE_DIR=/DATA_EDS2/AIGC/2312/xuhr2312/tmp/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

cd "${FASTWAM_ROOT}/Motus/RoboTwin"
python "${SCRIPT_DIR}/train_patch.py" --config "${CONFIG}"
