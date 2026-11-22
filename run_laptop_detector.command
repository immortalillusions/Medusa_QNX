#!/bin/zsh
# Medusa laptop detector (macOS). Open with: open -a Terminal run_laptop_detector.command
# Uses .venv/bin/python if present (pip install mediapipe opencv-python), else python3.
cd "$(dirname "$0")"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
"$PY" -u laptop_detector.py --pi-host "${MEDUSA_PI:-172.20.10.2}" "$@"
