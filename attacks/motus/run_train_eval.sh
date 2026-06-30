#!/bin/bash
set -euo pipefail

GPU_ID="${GPU_ID:-7}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/click_alarmclock_whitebox"
BEST_NOISE="${OUTPUT_DIR}/best_noise.pt"

echo "=== Step 1: White-box noise training ==="
GPU_ID="${GPU_ID}" bash "${SCRIPT_DIR}/train.sh"

if [ ! -f "${BEST_NOISE}" ]; then
  echo "Error: trained noise not found at ${BEST_NOISE}"
  exit 1
fi

echo "=== Step 2: Update attack config ==="
python - <<PY
from pathlib import Path
import yaml
cfg_path = Path("${SCRIPT_DIR}") / "config.yaml"
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
cfg.setdefault("attack", {})
cfg["attack"]["enabled"] = True
cfg["attack"]["apply_after_resize"] = True
cfg["attack"]["noise_path"] = "${BEST_NOISE}"
cfg_path.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
print("Updated", cfg_path)
PY

echo "=== Step 3: Attack evaluation ==="
GPU_ID="${GPU_ID}" NOISE_PATH="${BEST_NOISE}" bash "${SCRIPT_DIR}/eval.sh"
