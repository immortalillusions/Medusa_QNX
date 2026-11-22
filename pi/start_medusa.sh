#!/bin/sh
# Medusa on the Pi: header UART, then the high-priority control loop (owns the Arduino link, 50 Hz),
# then the AI process at normal priority. Fallback without the control loop: ./start_medusa_simple.sh
cd "$(dirname "$0")"
./start_uart.sh || exit 1
for p in $(pidin ar | grep -E "python3 -u (qnx_ai_service|medusa_control|medusa_sysmon)" | grep -v grep | awk "{print \$1}"); do slay -f "$p"; done
sleep 1
nohup on -p 10 python3 -u medusa_sysmon.py --out /tmp/medusa_system.json > /tmp/medusa_sysmon.log 2>&1 < /dev/null &
nohup on -p 21 python3 -u medusa_control.py --port /dev/ser1 --servo-min 15 --servo-max 165 --servo-home 90 --tilt-min 0 --tilt-max 0 --tilt-home 0 > /tmp/control.log 2>&1 < /dev/null &
sleep 2
nohup python3 -u qnx_ai_service.py --qnx-camera-unit 5 --control-port 8768 --trigger-udp-port 8767 --trigger-from 172.20.10.4 \
  --trigger-timeout 0.3 --servo-min 15 --servo-max 165 --servo-home 90 --camera-on-turret --servo-reversed --tilt-min 0 --tilt-max 0 --tilt-home 0 > /tmp/service.log 2>&1 < /dev/null &
echo "Medusa started: control log /tmp/control.log, AI log /tmp/service.log"
