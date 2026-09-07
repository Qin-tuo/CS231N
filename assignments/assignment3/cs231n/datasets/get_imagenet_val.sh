#!/usr/bin/env bash
set -eu
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../../../.." && pwd)"
PYTHON="${CS231N_PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then PYTHON=python3; fi
"$PYTHON" "$REPO_ROOT/scripts/prepare_data.py" --assignment assignment3 --dataset imagenet_val "$@"
