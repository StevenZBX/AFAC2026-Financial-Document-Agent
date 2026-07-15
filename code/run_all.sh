#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$SCRIPT_DIR/process_all_markdown.py" \
  --source-root /Users/limeixuan/Desktop/mineru_otuput \
  --output-root "$SCRIPT_DIR/../output"
