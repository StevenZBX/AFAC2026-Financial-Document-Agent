#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/limeixuan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
DATA_ROOT="${DATA_ROOT:-/Users/limeixuan/Desktop/public_dataset_upload}"

"${PYTHON_BIN}" scripts/process_hybrid.py \
  --raw-dir "${DATA_ROOT}/raw" \
  --questions "${DATA_ROOT}/questions/group_a/financial_reports_questions.json" \
  --output-dir processed_data_hybrid \
  --limit-docs 1
