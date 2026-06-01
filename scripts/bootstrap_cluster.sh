#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${BOOTSTRAP_PYTHON_BIN:-python3}"
VENV_DIR="${BOOTSTRAP_VENV_DIR:-$REPO_ROOT/.venv}"
PIP_BIN="$VENV_DIR/bin/pip"
BUILD_FRONTEND="${BOOTSTRAP_BUILD_FRONTEND:-1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "missing_python: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$PIP_BIN" install --upgrade pip
"$PIP_BIN" install -e "$REPO_ROOT"

if [[ "$BUILD_FRONTEND" == "1" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "warning=npm not found, skipping frontend build"
  else
    (
      cd -- "$REPO_ROOT/web_console/frontend"
      npm install
      npm run build
    )
  fi
fi

cat <<EOF
bootstrap_complete=true
repo_root=$REPO_ROOT
venv_python=$VENV_DIR/bin/python
next_steps:
  1. cp .env.example .env.local
  2. edit .env.local for LLM and VASP settings
  3. source .venv/bin/activate
  4. python mobality.py --root-path /absolute/path/to/material --dry-run --fresh --json
EOF
