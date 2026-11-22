#!/bin/zsh
# Deploy any git ref of this repo to the Pi, or roll back to a known-good one.
#   tools/deploy_pi.sh                # deploy the current checkout (HEAD)
#   tools/deploy_pi.sh demo-safe      # roll back to the last version verified live on the turret
#   tools/deploy_pi.sh --dry-run REF  # show what would be copied
# It backs up the Pi's folder first, copies the Python, models and Pi scripts from REF,
# restarts Medusa, and prints the firmware command that matches REF.
set -e
PI=${MEDUSA_PI:-172.20.10.2}
DEST=Medusa_QNX-bench
DRY=0
[ "$1" = "--dry-run" ] && { DRY=1; shift; }
REF=${1:-HEAD}
cd "$(dirname "$0")/.."
SHA=$(git rev-parse --short "$REF")
TMP=$(mktemp -d)
git archive "$REF" | tar -x -C "$TMP"
cd "$TMP"
FILES=( *.py medusa/*.py pi/start_medusa.sh pi/start_uart.sh )
[ -d qnx/cascades ] && FILES+=( qnx/cascades/*.xml )
for m in qnx/*.tflite qnx/*.onnx; do [ -e "$m" ] && FILES+=( "$m" ); done
echo "ref $REF = $SHA, ${#FILES[@]} files -> $PI:$DEST"
if [ $DRY = 1 ]; then printf '  %s\n' "${FILES[@]}"; rm -rf "$TMP"; exit 0; fi
STAMP=$(date +%Y%m%d-%H%M%S)
ssh -o BatchMode=yes qnxuser@$PI "cp -R $DEST $DEST.bak-$STAMP && mkdir -p $DEST/medusa $DEST/qnx/cascades" </dev/null
echo "Pi backup: ~/$DEST.bak-$STAMP"
tar -cf - "${FILES[@]}" | ssh -o BatchMode=yes qnxuser@$PI "cd $DEST && tar -xf - && mv pi/start_medusa.sh start_medusa.sh && mv pi/start_uart.sh start_uart.sh && chmod +x start_medusa.sh start_uart.sh && rmdir pi 2>/dev/null; echo $SHA > DEPLOYED_REF"
ssh -o BatchMode=yes qnxuser@$PI "cd $DEST && ./start_medusa.sh" </dev/null
rm -rf "$TMP"
echo
echo "Deployed $SHA. Firmware for this ref (UNO Q #1 on this Mac's USB-C):"
echo "  git checkout $SHA -- arduino && arduino-cli compile --upload -p \$(arduino-cli board list | awk '/UNO Q/{print \$1; exit}') --fqbn arduino:zephyr:unoq arduino/medusa_controller && git checkout HEAD -- arduino"
echo "Then press the turret's D2 button once to arm."
