#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)"
if [ -n "${CS231N_PYTHON:-}" ]; then
    SUBMISSION_PYTHON="$CS231N_PYTHON"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    SUBMISSION_PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v python >/dev/null 2>&1; then
    SUBMISSION_PYTHON="$(command -v python)"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    SUBMISSION_PYTHON="$REPO_ROOT/.venv/bin/python"
else
    SUBMISSION_PYTHON=python3
fi
exec "$SUBMISSION_PYTHON" "$REPO_ROOT/scripts/collect_submission.py" --assignment assignment1 "$@"
