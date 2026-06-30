#!/bin/bash
# Batch Motus attack evaluation for multiple tasks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-7}"
TEST_NUM="${TEST_NUM:-20}"
TASKS_FILE="${TASKS_FILE:-${SCRIPT_DIR}/tasks_attack_batch.txt}"

if [ ! -f "$TASKS_FILE" ]; then
    echo "Error: tasks file not found: $TASKS_FILE"
    exit 1
fi

mapfile -t TASKS < <(grep -v '^\s*$' "$TASKS_FILE" | grep -v '^\s*#')
if [ "${#TASKS[@]}" -eq 0 ]; then
    echo "Error: no tasks in $TASKS_FILE"
    exit 1
fi

BATCH_LOG_DIR="${SCRIPT_DIR}/logs_attack_batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BATCH_LOG_DIR"
SUMMARY_FILE="${BATCH_LOG_DIR}/summary.txt"

echo "Batch attack eval: ${#TASKS[@]} tasks, TEST_NUM=${TEST_NUM}, GPU=${GPU_ID}"
echo "Tasks: ${TASKS[*]}"
echo "Summary: ${SUMMARY_FILE}"
echo ""

failed=0
for task in "${TASKS[@]}"; do
    echo "========== ${task} =========="
    if GPU_ID="${GPU_ID}" TEST_NUM="${TEST_NUM}" TASK_NAME="${task}" \
        bash "${SCRIPT_DIR}/eval.sh" 2>&1 | tee "${BATCH_LOG_DIR}/${task}.log"; then
        result=$(grep -oP 'Success rate: \K[0-9]+/[0-9]+ => [0-9.]+%' "${BATCH_LOG_DIR}/${task}.log" | tail -1 || true)
        echo "${task}: OK ${result}" | tee -a "$SUMMARY_FILE"
    else
        echo "${task}: FAILED" | tee -a "$SUMMARY_FILE"
        failed=$((failed + 1))
    fi
    echo ""
done

echo "Batch done. Failed: ${failed}/${#TASKS[@]}"
echo "Summary: ${SUMMARY_FILE}"
exit "$failed"
