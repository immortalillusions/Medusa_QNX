#!/bin/zsh
# Relearn the laptop -> turret pan map after the laptop or turret is moved (e.g. at the venue).
#   tools/relearn_map.sh         # back up + clear the map, switch the laptop to one-person mode, screen switching off
#   tools/relearn_map.sh --done  # back to two-person mode with screen switching on
# A map learned in another spot can point the turret beside a peeper it can see; clearing it
# lets the turret lock onto the only face it sees and learn the new layout as it tracks.
set -e
PI=${MEDUSA_PI:-172.20.10.2}
LAPTOP=http://localhost:8081/settings
if [ "$1" = "--done" ]; then
  curl -s -m 3 -X POST -H "Content-Type: application/json" -d '{"peeper_mode":"two","screen_switch":true}' $LAPTOP; echo
  echo "Back to two-person mode, screen switching on."
  exit 0
fi
ssh -o BatchMode=yes qnxuser@$PI 'cd Medusa_QNX-bench && python3 -m medusa.handoff clear' </dev/null
curl -s -m 3 -X POST -H "Content-Type: application/json" -d '{"peeper_mode":"one","screen_switch":false}' $LAPTOP; echo
cat <<'EOT'
Map cleared. Now, with the turret armed (press D2) and ONLY ONE person in view:
  1. Stand ~2 m away, left of the laptop, look at the screen, hold still ~3 s.
  2. Same in the middle, then on the right.
  3. Check the dashboard's turret panel: "Learned map" should be 3+ points.
Then run: tools/relearn_map.sh --done
EOT
