"""Laptop-triggered aiming: the laptop says WHETHER, the turret camera says WHERE.

The MacBook detector sends ``medusa-peeper`` UDP heartbeats (~15 Hz). The Pi
keeps only the latest valid one and treats the laptop as active while that
message says ``active: true`` and is younger than the timeout. The aim point
comes from the Pi's own turret-mounted camera, using face boxes only: people
look at the laptop, not at the turret, so screen-facing is not required.
"""

from __future__ import annotations

import json
import math
import socket
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from .tflite_detector import FaceBox

PEEPER_MESSAGE_TYPE = "medusa-peeper"
NO_FACE_REASON = "laptop sees a peeper; turret camera sees no face"
CONFIRMING_REASON = "laptop sees a peeper; confirming the turret camera's face"
HELD_REASON = "laptop sees a peeper; holding the last face through a dropout"
_MAX_PACKETS_PER_POLL = 64  # Bound the drain so a flood cannot starve the camera loop.


@dataclass(frozen=True)
class PeeperMessage:
    active: bool
    target_x: float | None = None
    target_y: float | None = None
    seq: int | None = None


def _optional_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def parse_peeper_message(payload: bytes) -> PeeperMessage | None:
    """Return the message, or None for anything that is not a valid medusa-peeper packet."""
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(message, dict) or message.get("type") != PEEPER_MESSAGE_TYPE:
        return None
    active = message.get("active")
    if not isinstance(active, bool):
        return None
    seq = message.get("seq")
    return PeeperMessage(
        active=active,
        target_x=_optional_number(message.get("target_x")),
        target_y=_optional_number(message.get("target_y")),
        seq=seq if isinstance(seq, int) and not isinstance(seq, bool) else None,
    )


class LaptopTrigger:
    """Non-blocking listener holding the latest valid laptop message and its arrival time."""

    def __init__(self, sock, *, timeout: float = 0.5, allowed_sender: str | None = None) -> None:
        self.sock = sock
        self.timeout = timeout
        self.allowed_sender = allowed_sender
        self.latest: PeeperMessage | None = None
        self.received_at: float | None = None

    @classmethod
    def bind(cls, port: int, host: str = "0.0.0.0", **kwargs) -> "LaptopTrigger":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setblocking(False)
            sock.bind((host, port))
        except OSError:
            sock.close()
            raise
        return cls(sock, **kwargs)

    def handle(self, payload: bytes, sender) -> bool:
        """Store a packet if it is valid and from the allowed sender; return whether it was kept."""
        if self.allowed_sender is not None and sender[0] != self.allowed_sender:
            return False
        message = parse_peeper_message(payload)
        if message is None:
            return False
        self.latest, self.received_at = message, time.monotonic()
        return True

    def poll(self) -> None:
        """Drain whatever is waiting on the socket without blocking."""
        for _ in range(_MAX_PACKETS_PER_POLL):
            try:
                payload, sender = self.sock.recvfrom(4096)
            except (BlockingIOError, InterruptedError):
                return
            except OSError:
                return  # e.g. ICMP-reported errors; the next poll tries again.
            self.handle(payload, sender)

    def age(self) -> float | None:
        if self.received_at is None:
            return None
        return max(0.0, time.monotonic() - self.received_at)

    def status(self) -> tuple[bool, str]:
        """Return (laptop_active, reason when inactive)."""
        age = self.age()
        if age is None:
            return False, "waiting for laptop trigger"
        if age > self.timeout:
            return False, "laptop trigger stale"
        if not self.latest.active:
            return False, "laptop sees no peeper"
        return True, ""

    def close(self) -> None:
        self.sock.close()


def pick_aim_target(
    boxes: Sequence[FaceBox], width: int, height: int,
) -> tuple[FaceBox, float, float] | None:
    """Pick the smallest face (the probable farthest onlooker) and normalize its center.

    With one face that face is the target. With several, the smallest box
    overall wins (higher score breaks ties): the same small-means-far
    heuristic as select_snoopers and SnooperSelector. Coordinates are 0..1,
    x to the right and y down.
    """
    if not boxes:
        return None
    box = min(boxes, key=lambda b: ((b.right - b.left) * (b.bottom - b.top), -b.score))
    x = min(1.0, max(0.0, (box.left + box.right) / (2 * width)))
    y = min(1.0, max(0.0, (box.top + box.bottom) / (2 * height)))
    return box, x, y


class FaceConfirm:
    """Debounce the aim target between pick_aim_target and decide_aim.

    A face becomes the target only after ``confirm_frames`` consecutive frames
    near the same spot (centre within ``max_move`` of the frame width), so a
    one-frame false positive never aims the turret. Once confirmed, the target
    survives up to ``dropout_frames`` frames without a face (``held`` is then
    True) before it is dropped; the pan tracker's own coast covers longer gaps.
    """

    def __init__(self, confirm_frames: int = 2, dropout_frames: int = 2, max_move: float = 0.15) -> None:
        self.confirm_frames, self.dropout_frames, self.max_move = confirm_frames, dropout_frames, max_move
        self.reset()

    def reset(self) -> None:
        self.candidate: tuple[FaceBox, float, float] | None = None
        self.streak = 0
        self.target: tuple[FaceBox, float, float] | None = None
        self.missed = 0

    @property
    def held(self) -> bool:
        return self.target is not None and self.missed > 0

    def _near(self, a, b, width: int, height: int) -> bool:
        # Normalized x and y, measured in frame widths.
        return math.hypot(a[1] - b[1], (a[2] - b[2]) * height / width) <= self.max_move

    def update(self, picked: tuple[FaceBox, float, float] | None, width: int, height: int,
               ) -> tuple[FaceBox, float, float] | None:
        """Feed this frame's pick (or None); return the confirmed or held target, or None."""
        if picked is not None:
            if self.target is not None and self._near(picked, self.target, width, height):
                self.candidate, self.streak = picked, max(self.streak, self.confirm_frames)
                self.target, self.missed = picked, 0
                return picked
            near = self.candidate is not None and self._near(picked, self.candidate, width, height)
            self.candidate, self.streak = picked, self.streak + 1 if near else 1
            if self.streak >= self.confirm_frames:
                self.target, self.missed = picked, 0
                return picked
        else:
            self.candidate, self.streak = None, 0
        if self.target is not None and self.missed < self.dropout_frames:
            self.missed += 1
            return self.target
        self.target, self.missed = None, 0
        return None


class FrameRate:
    """Frames per second over a rolling window (the AI loop's rate for status.json)."""

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self.times: deque[float] = deque()

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.times.append(now)
        while now - self.times[0] > self.window_s:
            self.times.popleft()

    def fps(self) -> float | None:
        if len(self.times) < 2:
            return None
        span = self.times[-1] - self.times[0]
        return round((len(self.times) - 1) / span, 2) if span > 0 else None


@dataclass(frozen=True)
class AimDecision:
    active: bool
    target_x: float | None = None
    target_y: float | None = None
    reason: str = ""
    target_box: FaceBox | None = None
    held: bool = False  # True: the last confirmed face, kept through a short detection dropout.


def decide_aim(
    laptop_active: bool, laptop_reason: str, boxes: Sequence[FaceBox] | None,
    width: int, height: int, *, confirm: FaceConfirm | None = None,
) -> AimDecision:
    """Aim only while the laptop is active and the turret camera has a face to aim at.

    With ``confirm`` the face must pass FaceConfirm; it is fed every frame, laptop
    active or not, so a face already in view is aimed at as soon as the laptop fires.
    """
    picked = pick_aim_target(boxes, width, height) if boxes else None
    if confirm is not None:
        picked = confirm.update(picked, width, height)
    if boxes is None:
        return AimDecision(False, reason="AI inference unavailable")
    if not laptop_active:
        return AimDecision(False, reason=laptop_reason)
    if picked is None:
        confirming = confirm is not None and confirm.streak > 0
        return AimDecision(False, reason=CONFIRMING_REASON if confirming else NO_FACE_REASON)
    box, x, y = picked
    if confirm is not None and confirm.held:
        return AimDecision(True, x, y, HELD_REASON, box, held=True)
    reason = ("laptop sees a peeper; aiming at the only face" if len(boxes) == 1
              else "laptop sees a peeper; aiming at the smallest face")
    return AimDecision(True, x, y, reason, box)


def draw_aim_boxes(frame, boxes: Sequence[FaceBox], target: FaceBox | None) -> None:
    """Draw the aim target in red with an AIM label and every other face in green."""
    import cv2

    height, width = frame.shape[:2]
    for box in boxes:
        is_target = box is target
        color = (0, 0, 255) if is_target else (0, 255, 0)
        left, top = int(box.left), int(box.top)
        cv2.rectangle(frame, (left, top), (int(box.right) - 1, int(box.bottom) - 1), color,
                      3 if is_target else 2)
        if is_target:
            cv2.putText(frame, "AIM", (min(left, max(0, width - 50)), min(height - 1, max(20, top - 6))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


# The firmware's motion profile, mirrored here so the Pi knows where the turret
# really points (it eases toward each command instead of jumping). These MUST
# match MAX_SPEED, MAX_ACCEL and MOTION_STEP_MS in
# arduino/medusa_controller/medusa_controller.ino; tests/test_trigger.py reads
# the .ino and fails if they drift apart.
FIRMWARE_MAX_SPEED = 300.0   # degrees per second
FIRMWARE_MAX_ACCEL = 2500.0  # degrees per second squared
FIRMWARE_STEP_S = 0.005      # one stepAxis() call every 5 ms
_FIRMWARE_HISTORY = 200      # 1 s of positions, enough to look back past the camera latency
_FIRMWARE_MAX_GAP_S = 2.0    # a longer gap ends at the target whatever the move was


class FirmwareAxisModel:
    """Python twin of stepAxis() in the firmware: where the eased servo actually is."""

    def __init__(
        self, position: float, minimum: float, maximum: float, *,
        max_speed: float = FIRMWARE_MAX_SPEED, max_accel: float = FIRMWARE_MAX_ACCEL,
        step_s: float = FIRMWARE_STEP_S,
    ) -> None:
        self.position = self.target = float(position)
        self.velocity = 0.0
        self.minimum, self.maximum = float(minimum), float(maximum)
        self.max_speed, self.max_accel, self.step_s = max_speed, max_accel, step_s
        self.time: float | None = None  # time of the last firmware step
        self.history: deque[tuple[float, float]] = deque(maxlen=_FIRMWARE_HISTORY)

    def step(self) -> None:
        error = self.target - self.position
        wanted = min(self.max_speed, math.sqrt(2 * self.max_accel * abs(error)))
        if error < 0:
            wanted = -wanted
        limit = self.max_accel * self.step_s
        self.velocity += max(-limit, min(limit, wanted - self.velocity))
        self.position = max(self.minimum, min(self.maximum, self.position + self.velocity * self.step_s))
        if abs(error) < 0.05 and abs(self.velocity) < 2:
            self.position, self.velocity = self.target, 0.0

    def advance(self, now: float) -> None:
        """Run every firmware step due up to ``now`` toward the current target."""
        if self.time is None or now - self.time > _FIRMWARE_MAX_GAP_S:
            if self.time is not None:
                self.position, self.velocity = self.target, 0.0
            self.time = now
            self.history.clear()
            self.history.append((now, self.position))
            return
        while self.time + self.step_s <= now:
            self.time += self.step_s
            self.step()
            self.history.append((self.time, self.position))

    def position_at(self, when: float) -> float:
        """Estimated position at an earlier time, interpolated between firmware steps."""
        later = None
        for sample in reversed(self.history):
            if sample[0] <= when:
                if later is None or later[0] == sample[0]:
                    return sample[1]
                share = (when - sample[0]) / (later[0] - sample[0])
                return sample[1] + share * (later[1] - sample[1])
            later = sample
        return later[1] if later is not None else self.position


class PanTracker:
    """Pan tracking for a camera riding on the turret's pan stage, in world bearings.

    Each frame: (1) step the firmware model to now, so the Pi knows where the
    turret actually pointed when the frame was captured (``latency_s`` ago);
    (2) face bearing = that pan angle + sign * (x - 0.5) * hfov, which does not
    move when the camera moves, so the turret no longer chases its own motion;
    (3) an alpha-beta filter tracks bearing and velocity, switching to faster
    gains while the residual is past ``maneuver_deg`` (the face started or
    stopped), and treating a jump past ``jump_deg`` as a new target;
    (4) command = filtered bearing + velocity * ``lead_s``, where the lead counts
    from frame capture, so it covers the camera latency as well as servo lag;
    (5) hysteresis: a still command moves only once it would change by more than
    ``start_deg``, then follows until a frame changes it by at most ``stop_deg``
    with the face slower than ``still_speed``, so a still face gives a still turret.
    On loss the command coasts on (velocity fading to zero over ``coast_s``), holds still until ``hold_s`` after the loss (default: no extra
    hold), then glides home at ``return_speed`` deg/s. A face seen again resumes
    tracking at once.

    Acquire (``laptop_active=True`` passed to update while the turret camera sees
    no face, after any coast): with a handoff angle from the laptop's target_x
    (medusa/handoff.py) the turret points there ("handoff"); if no face shows up
    within ``handoff_timeout`` s, or without a map, it sweeps the pan range back
    and forth at ``search_speed`` deg/s ("searching"), starting toward the side
    the laptop reports. ``search_speed`` 0 turns acquire off. The light is not
    the tracker's business: the caller lights it only for a turret-camera face.

    Real angle: observe_actual() feeds the pan angle the firmware reports. While
    that report is fresh (``actual_timeout``), the turret's position at frame
    capture is the real angle carried forward by the model's motion since the
    report, instead of the model alone. A report that the turret is disarmed
    resets to home every frame, so nothing winds up while the firmware ignores us.
    """

    def __init__(
        self, minimum: float, maximum: float, home: float, *, reversed: bool = False,
        hfov_deg: float = 50.0, latency_s: float = 0.07, lead_s: float = 0.15,
        coast_s: float = 0.3, hold_s: float | None = None, return_speed: float = 60.0,
        alpha: float = 0.5, beta: float = 0.15, jump_deg: float = 25.0,
        maneuver_deg: float = 2.0, maneuver_alpha: float = 0.8, maneuver_beta: float = 0.5,
        start_deg: float = 1.0, stop_deg: float = 0.4, still_speed: float = 3.0,
        search_speed: float = 60.0, handoff_timeout: float = 0.6, actual_timeout: float = 0.3,
        mode: str = "predictive", gain: float = 0.3, max_step: float = 4.0,
        start_error: float = 0.04, stop_error: float = 0.02, error_alpha: float = 0.5,
    ) -> None:
        if mode not in ("predictive", "proportional"):
            raise ValueError("pan mode must be 'predictive' or 'proportional'")
        self.mode, self.gain, self.max_step = mode, gain, max_step
        self.start_error, self.stop_error, self.error_alpha = start_error, stop_error, error_alpha
        self.minimum, self.maximum, self.home = float(minimum), float(maximum), float(home)
        self.sign = -1.0 if reversed else 1.0  # Reversed: a face right of centre needs a lower yaw.
        self.hfov_deg, self.latency_s, self.lead_s = hfov_deg, latency_s, lead_s
        self.coast_s, self.return_speed = coast_s, return_speed
        self.hold_s = coast_s if hold_s is None else max(hold_s, coast_s)
        self.alpha, self.beta, self.jump_deg = alpha, beta, jump_deg
        self.maneuver_deg, self.maneuver_alpha, self.maneuver_beta = maneuver_deg, maneuver_alpha, maneuver_beta
        self.start_deg, self.stop_deg, self.still_speed = start_deg, stop_deg, still_speed
        self.search_speed, self.handoff_timeout, self.actual_timeout = search_speed, handoff_timeout, actual_timeout
        self.servo = FirmwareAxisModel(self.home, self.minimum, self.maximum)
        self._last_now: float | None = None
        self.actual: tuple[float, float, bool] | None = None  # (time, real pan, armed)
        self.reset()

    def reset(self) -> None:
        """Command home now (the firmware eases there); forget the target."""
        self.angle, self.state = self.home, "home"
        self.bearing: float | None = None
        self.velocity = 0.0
        self._filter_time = 0.0
        self.moving = False
        self._error_ema: float | None = None
        self.lost_at: float | None = None
        self._lost_velocity = 0.0
        self._acquire_at: float | None = None
        self._search_dir = 1.0
        self.servo.target = float(round(self.home))

    def _clamp(self, angle: float) -> float:
        return max(self.minimum, min(self.maximum, angle))

    def observe_actual(self, pan: float | None, at: float | None, armed: bool | None = True) -> None:
        """Record the firmware's reported pan angle (degrees) and when it was read."""
        self.actual = None if pan is None or at is None else (at, float(pan), armed is not False)

    def _actual_fresh(self, now: float) -> bool:
        return self.actual is not None and now - self.actual[0] <= self.actual_timeout

    def turret_position(self, captured: float, now: float) -> float:
        """Where the turret pointed at ``captured``: the real angle when fresh, else the model."""
        model = self.servo.position_at(captured)
        if not self._actual_fresh(now):
            return model
        at, real, _armed = self.actual
        return real + (model - self.servo.position_at(at))

    def update(
        self, target_x: float | None, now: float | None = None, *,
        laptop_active: bool = False, handoff_angle: float | None = None, laptop_x: float | None = None,
    ) -> float | None:
        """Advance one frame; return the image error (x - 0.5) or None without a target.

        ``laptop_active`` (the laptop sees a peeper), ``handoff_angle`` (the map's pan
        for the laptop's target_x) and ``laptop_x`` drive acquire when ``target_x`` is None.
        """
        now = time.monotonic() if now is None else now
        self.servo.advance(now)
        dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now
        captured = now - self.latency_s
        if self._actual_fresh(now) and not self.actual[2]:
            self.reset()  # Disarmed: the firmware ignores commands, so do not wind up.
            return None
        if target_x is None:
            if laptop_active and self.search_speed > 0 and not (
                    self.state in ("tracking", "coasting") and self._still_coasting(now)):
                self._acquire(now, dt, handoff_angle, laptop_x)
            else:
                self._coast_or_return(now, dt, captured)
            error = None
        else:
            error = target_x - 0.5  # Positive: face right of image centre.
            if self.mode == "proportional":
                self._centre(error)
            else:
                self._measure(self.turret_position(captured, now) + self.sign * error * self.hfov_deg, captured)
                self._follow()
            self.state, self.lost_at = "tracking", None
        self.servo.target = float(round(self._clamp(self.angle)))  # What send_angles() sends.
        return error

    def _measure(self, bearing: float, captured: float) -> None:
        if self.state in ("tracking", "coasting") and self.bearing is not None:
            dt = captured - self._filter_time
            predicted = self.bearing + self.velocity * max(0.0, dt)
            residual = bearing - predicted
            if abs(residual) <= self.jump_deg:
                # A residual past pixel noise means the face started, stopped or turned:
                # take the faster gains so the velocity catches up within a frame or two.
                fast = abs(residual) > self.maneuver_deg
                alpha = self.maneuver_alpha if fast else self.alpha
                beta = self.maneuver_beta if fast else self.beta
                self.bearing = predicted + alpha * residual
                if dt > 1e-3:
                    limit = self.servo.max_speed
                    self.velocity = max(-limit, min(limit, self.velocity + beta * residual / dt))
                self._filter_time = max(self._filter_time, captured)
                return
        # First sight, re-acquired after giving up, or an absurd jump: a new target.
        self.bearing, self.velocity, self._filter_time = bearing, 0.0, captured
        self.moving = True

    def _centre(self, error: float) -> None:
        """Proportional mode: nudge the turret a fraction of the way toward centring the face.

        It needs no camera-latency model, so a wrong latency guess cannot make it hunt:
        each frame moves gain x the remaining error (capped), with a smoothed error and
        a start/stop dead zone so a still face gives a still turret.
        """
        self._error_ema = error if self._error_ema is None else (
            self._error_ema + self.error_alpha * (error - self._error_ema))
        e = self._error_ema
        if self.moving and abs(e) < self.stop_error:
            self.moving = False
        elif not self.moving and abs(e) > self.start_error:
            self.moving = True
        if self.moving:
            step = self.gain * e * self.hfov_deg
            step = max(-self.max_step, min(self.max_step, step))
            self.angle = self._clamp(self.angle + self.sign * step)
        self.bearing, self.velocity = self.angle + self.sign * e * self.hfov_deg, 0.0

    def _follow(self) -> None:
        # The filter sits at the frame's capture time, so lead_s also covers the camera latency.
        wanted = self._clamp(self.bearing + self.velocity * self.lead_s)
        if self.moving:
            if abs(wanted - self.angle) <= self.stop_deg and abs(self.velocity) <= self.still_speed:
                self.moving = False
            self.angle = wanted
        elif abs(wanted - self.angle) > self.start_deg:
            self.moving, self.angle = True, wanted

    def _still_coasting(self, now: float) -> bool:
        """True while a just-lost face should still be coasted on (also on the frame it is lost)."""
        return self.state == "tracking" or now - self.lost_at < self.hold_s

    def _search_direction(self, handoff_angle: float | None, laptop_x: float | None) -> float:
        if handoff_angle is not None and abs(handoff_angle - self.home) > 0.5:
            return math.copysign(1.0, handoff_angle - self.home)
        if laptop_x is not None and abs(laptop_x - 0.5) > 0.02:
            # No map: assume the laptop camera faces the same way as the turret camera.
            return math.copysign(1.0, self.sign * (laptop_x - 0.5))
        return 1.0 if self.maximum - self.angle >= self.angle - self.minimum else -1.0

    def _acquire(self, now: float, dt: float, handoff_angle: float | None, laptop_x: float | None) -> None:
        if self.state not in ("handoff", "searching"):
            self.bearing, self.velocity, self.moving, self.lost_at = None, 0.0, False, None
            self._acquire_at, dt = now, 0.0
            self._search_dir = self._search_direction(handoff_angle, laptop_x)
            self.state = "handoff" if handoff_angle is not None else "searching"
        if self.state == "handoff":
            if handoff_angle is not None and now - self._acquire_at < self.handoff_timeout:
                self.angle = self._clamp(handoff_angle)  # Follows the laptop if the peeper moves.
                return
            self.state, self._search_dir = "searching", self._search_direction(handoff_angle, laptop_x)
        angle = self.angle + self._search_dir * self.search_speed * dt
        if angle >= self.maximum:
            angle, self._search_dir = max(self.minimum, 2 * self.maximum - angle), -1.0
        elif angle <= self.minimum:
            angle, self._search_dir = min(self.maximum, 2 * self.minimum - angle), 1.0
        self.angle = angle

    def _coast_or_return(self, now: float, dt: float, captured: float) -> None:
        if self.state == "home":
            return
        if self.state in ("handoff", "searching"):
            self.state = "returning"  # The laptop no longer sees a peeper: glide home.
        if self.state == "tracking":
            self.state, self.lost_at, self._lost_velocity = "coasting", now, self.velocity
        if self.state == "coasting":
            lost_for = now - self.lost_at
            if lost_for < self.hold_s:
                # Bearing and command both carry on at a velocity fading to zero over
                # coast_s, so the turret slows to a stop instead of reversing its lead.
                fade = max(0.0, 1.0 - lost_for / self.coast_s) if self.coast_s > 0 else 0.0
                velocity = self._lost_velocity * fade
                mean = (self.velocity + velocity) / 2
                self.bearing += mean * max(0.0, captured - self._filter_time)
                self.angle = self._clamp(self.angle + mean * dt)
                self.velocity, self._filter_time = velocity, captured
                return
            self.state, self.bearing, self.velocity, self.moving = "returning", None, 0.0, False
        step = self.return_speed * dt
        self.angle = self.home if abs(self.home - self.angle) <= step else (
            self.angle + math.copysign(step, self.home - self.angle))
        if self.angle == self.home:
            self.state = "home"
