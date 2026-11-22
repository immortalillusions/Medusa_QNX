# Demo day runbook

Everything to bring, the setup order, the checks, and what to do when something goes wrong.
The Mac copy to run from is `~/medusa-demo` (a full clone with its Python and dashboard
packages already installed).

## 1. Bring

**Compute and network**
- [ ] MacBook + its charger (keep it plugged in; set it never to sleep on power)
- [ ] **Eric's iPhone** for the hotspot — the Pi only knows the network `Eric's Iphone`
- [ ] Raspberry Pi 5 with the QNX microSD card and Active Cooler
- [ ] Pi wall charger (USB-C, the Apple charger) — never power the Pi from the Mac: it browns out

**Turret**
- [ ] Turret: 3D-printed base and yoke, 2 × SG90 servos, counterweight, C270 webcam on the pan stage
- [ ] i-Zoom Rota8 puck light with 2 × CR2032 (spares too), mounted, left on
- [ ] UNO Q #1 (the controller, flashed) + USB-C cable to the Mac
- [ ] UNO Q #2 (servo 5 V supply only) + its USB-C cable and a power source
- [ ] Breadboard with the kill plug (tab), the 470 µF capacitor, the 3 × 10 kΩ kill-sense divider, the D2 arm button
- [ ] Jumpers: UNO Q D0 → Pi pin 8, UNO Q D1 → Pi pin 10, GND → Pi pin 6
- [ ] Spares: jumper wires, 2 spare SG90s, tape, zip ties

**Demo props**
- [ ] A second person (or the dummy head) as the peeper; a mask or scarf for the face-covering beat

## 2. Set up (order matters)

1. Turn on **Eric's hotspot** and connect the Mac to it (Mac should get 172.20.10.4).
2. Plug the **C270 into the Pi**, then the Pi's **wall charger**. About a minute later the Pi joins the
   hotspot; Medusa starts by itself ~30 s after that. (The AI waits for the camera if it's late.)
3. Wire the UNO Q UART to the Pi (D0→8, D1→10, GND→6), plug **UNO Q #1 into the Mac**, power **UNO Q #2**,
   put the **kill plug in**.
4. If the UNO Q needs the latest firmware (fault latch), from `~/medusa-demo`:
   `arduino-cli compile --upload -p $(arduino-cli board list | awk '/UNO Q/{print $1; exit}') --fqbn arduino:zephyr:unoq arduino/medusa_controller`
5. Start the laptop side: `open -a Terminal ~/medusa-demo/run_laptop_detector.command`
   (it must be launched through Terminal: that's where macOS grants the camera).
6. Start the dashboard: `cd ~/medusa-demo/dashboard && npx next dev -p 3000`, open http://localhost:3000.
7. Wait until the dashboard shows Control at 50 Hz and the Arduino chip live, step out of the turret's
   view, then press the turret's **D2 button** once (the firmware boots latched: a human press is the
   safety reset; arming with nobody in view avoids an overshoot while the servos soft-start).
   The red LED lights = armed. After that the dashboard's Arm/Switch off works.
8. **Relearn the pan map for this table layout** (2 minutes; the map learned at home is wrong once the
   laptop or turret moves): `cd ~/medusa-demo && tools/relearn_map.sh`, follow its three steps with one
   person, then `tools/relearn_map.sh --done`.

## 3. Checks before judges arrive (dashboard)

- Top bar shows ARMED (amber) after D2; Laptop, Turret cam, AI, Control, Arduino chips all live.
- Control loop tile: 50.0 Hz. Turret AI ~13 fps. Laptop ~30 fps (lower in dim light).
- Peeper mode **Two people**, Screen switching **On**.
- Walk in as a peeper: Docs comes to the front, the turret turns and centres you.

## 4. The 3-minute demo beats

1. Peeper looks at the screen → Docs flips, turret locks on. Side-eye with the head turned still counts.
2. Two peepers: the newest one takes the turret; a quick glance doesn't steal it.
3. Walk out of the turret camera's view → it keeps aiming from the laptop's view (badge "Following from laptop").
4. CPU stress button → AI fps collapses to ~1, control loop stays at 50 Hz with zero missed deadlines.
   If someone is in the turret's view, it parks home between the slow AI frames: that is the safety
   design (stale AI targets are refused), say so.
5. Safety: pull the kill plug → servos go limp at once; the dashboard can't re-arm until D2 is pressed.

## 5. If something goes wrong

| Symptom | Fix |
| --- | --- |
| Dashboard shows the Pi offline | Is Eric's hotspot on and the Pi on its wall charger? Wait ~90 s after power-up. |
| Turret cam says "waiting for the turret camera" | Re-seat the C270 in the Pi; it is picked up by itself within a few seconds. |
| Arduino chip red / link down | Check D0→8, D1→10, GND→6 and that UNO Q #1 is powered. |
| Arm button greyed "press the turret's button" | Press D2 (safety reset after a fault or a restart). |
| Arm greyed "kill gate open" | Push the kill plug in fully. |
| Laptop doesn't switch to Docs | Turret must be armed; Screen switching On; launched via Terminal. |
| Turret aims beside a peeper who left its view | The map learns as it tracks; let the turret centre on them a few times. |
| Turret sees the peeper but aims beside them | The map is from another layout: `tools/relearn_map.sh` (step 8). |
| Laptop → Pi chip shows no data | Check the Mac is on Eric's hotspot; restart the laptop detector. |
| Turret cam stuck on "waiting for the turret camera" with the C270 plugged in | On the Pi: `grep "camera unit" /tmp/service.log`; if no unit is "using it", set `--qnx-camera-unit 5` in `~/Medusa_QNX-bench/start_medusa.sh` and run it. |
| Anything worse on the Pi | Roll back: `cd ~/medusa-demo && tools/deploy_pi.sh demo-safe` (last version verified live). **Then you must flash its firmware** with the command it prints (the rolled-back Pi can't read the newer firmware's status line) and press D2. |

To redeploy the latest after a rollback: `tools/deploy_pi.sh main` (from an up-to-date `~/medusa-demo`).
Every deploy backs up the Pi folder to `~/Medusa_QNX-bench.bak-<time>` first.
