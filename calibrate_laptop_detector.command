#!/bin/zsh
# Bench calibration: one face triggers, Docs switching off, gaze numbers logged to captures/gaze.log.
cd "$(dirname "$0")"
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
mkdir -p captures
echo "CALIBRATION MODE: one face triggers, Docs switching off, gaze logged. Ctrl-C to stop."
"$PY" -u laptop_detector.py --pi-host "${MEDUSA_PI:-172.20.10.2}" --min-faces 1 --no-browser --debug-gaze 2>&1 | tee captures/gaze.log
