#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PY="$HOME/.venvs/rtk_egfr/bin/python"
STREAMLIT="$HOME/.venvs/rtk_egfr/bin/streamlit"

if [[ ! -x "$PY" ]]; then
  echo "missing venv at $PY" >&2
  echo "create it with: /opt/homebrew/bin/python3 -m venv $HOME/.venvs/rtk_egfr" >&2
  exit 1
fi

"$PY" -m pip install -r requirements.txt >/dev/null
exec "$STREAMLIT" run streamlit_app.py
