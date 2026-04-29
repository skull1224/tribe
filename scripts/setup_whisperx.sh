#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3.12}"
WHISPERX_VENV_DIR="${WHISPERX_VENV_DIR:-.whisperx-venv}"

"$PYTHON_BIN" -m venv "$WHISPERX_VENV_DIR"
"$WHISPERX_VENV_DIR/bin/pip" install --upgrade pip
"$WHISPERX_VENV_DIR/bin/pip" install whisperx==3.8.4
