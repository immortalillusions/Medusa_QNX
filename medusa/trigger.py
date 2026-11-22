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
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .tflite_detector import FaceBox

PEEPER_MESSAGE_TYPE = "medusa-peeper"
NO_FACE_REASON = "laptop sees a peeper; turret camera sees no face"
CONFIRMING_REASON = "laptop sees a peeper; confirming the turret camera's face"
HELD_REASON = "laptop sees a peeper; holding the last face through a dropout"
FALLBACK_REASON = "laptop-guided aim (turret camera lost the face)"
NO_MATCH_REASON = "laptop sees a peeper; no turret face matches the laptop's peeper"
MATCH_REASONS = {
    "map": "laptop sees a peeper; aiming at the face where the map expects the peeper",
    "single": "laptop sees a peeper; aiming at the only face",
    "order": "laptop sees a peeper; aiming at the face on the peeper's side of the user",
    "smallest": "laptop sees a peeper; aiming at the smallest face",
}
_MAX_PACKETS_PER_POLL = 64  # Bound the drain so a flood cannot starve the camera loop.


@dataclass(frozen=True)
class PeeperMessage:
    active: bool
    target_x: float | None = None
    target_y: float | None = None
    seq: int | None = None
    user_x: float | None = None  # Optional: the user's x in the laptop image (PeeperMatcher's "order").
    peeper_id: int | None = None  # Optional: the laptop's peeper identity; a change drops the face lock.


def _optional_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _optional_int(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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
        user_x=_optional_number(message.get("user_x")),
        peeper_id=_optional_int(message.get("peeper_id")),
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


def _face(box: FaceBox, width: int, height: int) -> tuple[FaceBox, float, float]:
    return (box, min(1.0, max(0.0, (box.left + box.right) / (2 * width))),
            min(1.0, max(0.0, (box.top + box.bottom) / (2 * height))))


class PeeperMatcher:
    """Pick the turret-camera face that is the laptop's peeper, not the user.

    The laptop sees both people and names the peeper (target_x, and optionally the
    user's x as ``user_x``); the turret camera may see either or both. While the
    laptop is active, in order:

    * "map": with a usable laptop_x -> pan map, the caller passes ``expected_x``,
      where the peeper should appear in the turret image (PanTracker.expected_image_x
      of map(laptop target_x)). The face whose centre x is nearest it wins if within
      ``gate`` (frame widths); with none inside the gate there is no match ("none"),
      so the tracker falls back to aiming at the map instead of at the user.
    * "single": no map, one face: that face.
    * "order": no map, 2+ faces, the laptop sent ``user_x``: the peeper's side of the
      user in the laptop image picks the rightmost or leftmost turret face. This
      assumes both cameras look at the people from the same side, unmirrored.
    * "smallest": otherwise the smallest face (pick_aim_target, the old rule).

    Lock: once a face is picked while the laptop is active, later frames take the
    face nearest the last pick within ``lock_radius`` (x, frame widths), so the turret
    does not jump to the user standing close by. The lock survives ``lock_dropout``
    frames without such a face (returning None, so the user is not picked meanwhile),
    and is dropped at once if the locked face leaves the gate around ``expected_x``;
    matching then runs again. The laptop going inactive drops the lock. ``match`` is
    how the current face was chosen (kept while locked), "none" without one. A new
    ``peeper_id`` from the laptop (it switched to a more recent peeper) drops the lock at
    once, so the turret re-matches instead of staying on the old face.
    """

    def __init__(self, gate: float = 0.25, lock_radius: float = 0.15, lock_dropout: int = 2) -> None:
        self.gate, self.lock_radius, self.lock_dropout = gate, lock_radius, lock_dropout
        self.reset()

    def reset(self) -> None:
        self.lock_x: float | None = None
        self.missed = 0
        self.match = "none"
        self.expected_x: float | None = None
        self.peeper_id: int | None = None

    def pick(
        self, boxes: Sequence[FaceBox] | None, width: int, height: int, *, laptop_active: bool = True,
        expected_x: float | None = None, laptop_x: float | None = None, user_x: float | None = None,
        peeper_id: int | None = None,
    ) -> tuple[FaceBox, float, float] | None:
        """This frame's face (box, x, y normalized 0..1), or None."""
        self.expected_x = expected_x if laptop_active else None
        if not laptop_active:
            self.lock_x, self.missed = None, 0
        elif peeper_id is not None:
            if self.peeper_id is not None and peeper_id != self.peeper_id:
                self.lock_x, self.missed = None, 0  # The laptop switched peepers: re-match.
            self.peeper_id = peeper_id
        faces = [_face(box, width, height) for box in boxes or ()]
        if self.lock_x is not None:
            near = [f for f in faces if abs(f[1] - self.lock_x) <= self.lock_radius]
            if near:
                face = min(near, key=lambda f: abs(f[1] - self.lock_x))
                if self.expected_x is None or abs(face[1] - self.expected_x) <= self.gate:
                    self.lock_x, self.missed = face[1], 0
                    return face
                self.lock_x = None  # The locked face is not where the map puts the peeper: re-match.
            else:
                self.missed += 1
                if self.missed <= self.lock_dropout:
                    return None
                self.lock_x = None
        self.missed = 0
        face, self.match = self._match(faces, laptop_x, user_x)
        if face is not None and laptop_active:
            self.lock_x = face[1]
        return face

    def _match(self, faces, laptop_x, user_x):
        if not faces:
            return None, "none"
        if self.expected_x is not None:
            face = min(faces, key=lambda f: abs(f[1] - self.expected_x))
            return (face, "map") if abs(face[1] - self.expected_x) <= self.gate else (None, "none")
        if len(faces) == 1:
            return faces[0], "single"
        if laptop_x is not None and user_x is not None and laptop_x != user_x:
            pick = max if laptop_x > user_x else min  # Peeper right of the user: the rightmost face.
            return pick(faces, key=lambda f: f[1]), "order"
        area = lambda f: ((f[0].right - f[0].left) * (f[0].bottom - f[0].top), -f[0].score)
        return min(faces, key=area), "smallest"  # The same rule as pick_aim_target.


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
    width: int, height: int, *, confirm: FaceConfirm | None = None, matcher: PeeperMatcher | None = None,
    expected_x: float | None = None, laptop_x: float | None = None, user_x: float | None = None,
    peeper_id: int | None = None,
) -> AimDecision:
    """Aim only while the laptop is active and the turret camera has a face to aim at.

    With ``matcher`` the face is the PeeperMatcher's pick (``expected_x``, ``laptop_x``
    and ``user_x`` are its inputs); without it, pick_aim_target's smallest face.
    With ``confirm`` the face must pass FaceConfirm; it is fed every frame, laptop
    active or not, so a face already in view is aimed at as soon as the laptop fires.
    """
    if matcher is not None:
        picked = matcher.pick(boxes, width, height, laptop_active=laptop_active, expected_x=expected_x,
                              laptop_x=laptop_x, user_x=user_x, peeper_id=peeper_id)
    else:
        picked = pick_aim_target(boxes, width, height) if boxes else None
    if confirm is not None:
        picked = confirm.update(picked, width, height)
    if boxes is None:
        return AimDecision(False, reason="AI inference unavailable")
    if not laptop_active:
        return AimDecision(False, reason=laptop_reason)
    if picked is None:
        confirming = confirm is not None and confirm.streak > 0
        unmatched = matcher is not None and bool(boxes)
        return AimDecision(False, reason=CONFIRMING_REASON if confirming else (
            NO_MATCH_REASON if unmatched else NO_FACE_REASON))
    box, x, y = picked
    if confirm is not None and confirm.held:
        return AimDecision(True, x, y, HELD_REASON, box, held=True)
    if matcher is not None:
        reason = MATCH_REASONS.get(matcher.match, MATCH_REASONS["smallest"])
    else:
        reason = MATCH_REASONS["single" if len(boxes) == 1 else "smallest"]
    return AimDecision(True, x, y, reason, box)


def draw_aim_boxes(frame, boxes: Sequence[FaceBox], target: FaceBox | None) -> None:
    """Aim target in amber with an AIM tag, other faces in blue-grey, a small centre crosshair."""
    import cv2

    from .overlay import AMBER, BLUE_GREY, crosshair, frame_scale, label, thickness

    scale = frame_scale(frame)
    crosshair(cv2, frame, scale)
    for box in boxes:
        is_target = box is target
        color = AMBER if is_target else BLUE_GREY
        left, top = int(box.left), int(box.top)
        cv2.rectangle(frame, (left, top), (int(box.right) - 1, int(box.bottom) - 1), color,
                      thickness(scale, 3.0 if is_target else 1.5), cv2.LINE_AA)
        if is_target:
            label(cv2, frame, "AIM", left, top - thickness(scale), color, scale)


# The firmware's motion profile, mirrored here so the Pi knows where the turret
# really points (it eases toward each command instead of jumping). These MUST
# match MAX_SPEED, MAX_ACCEL and MOTION_STEP_MS in
# arduino/medusa_controller/medusa_controller.ino; tests/test_trigger.py reads
# the .ino and fails if they drift apart.
FIRMWARE_MAX_SPEED = 180.0   # degrees per second
FIRMWARE_MAX_ACCEL = 900.0   # degrees per second squared
FIRMWARE_STEP_S = 0.005      # one stepAxis() call every 5 ms
_PROPORTIONAL_REF_FPS = 15.0  # proportional gain is the share of the error corrected per 1/15 s
_PROPORTIONAL_MAX_DT = 0.2     # a longer frame gap moves no more than a 0.2 s one
_FIRMWARE_HISTORY = 200      # 1 s of positions, enough to look back past the camera latency
_FIRMWARE_MAX_GAP_S = 2.0    # a longer gap ends at the target whatever the move was
_FALLBACK_MAX_VX = 1.5       # laptop image widths per second: a faster estimate is noise
_FALLBACK_SWITCH_X = 0.15    # a laptop x jump past this in one sample is a new peeper, not motion
_FALLBACK_EXTRAPOLATE_X = 0.05  # laptop x this far outside the learned points: the map is a guess


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
    hold), then ("holding") stays still ``lost_hold_s`` more, then glides home at
    ``return_speed`` deg/s ("returning"); fallback, handoff and search that lose the laptop
    hold the same way. ``lost_hold_s=0, return_speed=60`` is the old behaviour. A face
    seen again resumes tracking at once.

    Acquire (``laptop_active=True`` passed to update while the turret camera sees
    no face, after any coast): with a handoff angle from the laptop's target_x
    (medusa/handoff.py) the turret points there ("handoff"); if no face shows up
    within ``handoff_timeout`` s, or without a map, it sweeps the pan range back
    and forth at ``search_speed`` deg/s ("searching"), starting toward the side
    the laptop reports. ``search_speed`` 0 turns acquire off. The light is not
    the tracker's business: the caller lights it for a turret-camera face, and in fallback.

    Fallback (``fallback_map`` passed to update, a usable laptop_x -> pan map, e.g.
    the learned HandoffLearner one): instead of handoff and search, the turret
    follows the laptop ("fallback"). The laptop x is smoothed (EMA, time constant
    ``fallback_tau``, predicted forward by its velocity so a moving x is not lagged) with a
    smoothed velocity from successive samples, and the aim is map(x + vx * ``fallback_lead_s``)
    plus an offset: the angle minus map(x) on entry, decaying over ``fallback_blend_s``, so
    the handoff does not jump. The command follows the aim as a critically damped
    second-order system (natural frequency ``fallback_omega``, velocity feed-forward, so a
    steady ramp is followed without lag) capped at ``max_rate`` and ``fallback_max_accel``;
    a step (the laptop switching peepers) glides there without overshoot. After
    ``fallback_search_after_s`` in fallback without a turret face, or at once while the
    laptop x is past the learned points (``fallback_domain``; the map is extrapolating), it
    sweeps +-``fallback_search_deg`` around the moving map estimate at
    ``fallback_search_speed``, widening to twice that after one full sweep, so a wrong map
    cannot hold the turret where the camera never sees the face. A turret-camera face takes
    over again ("tracking"), the command ramping from the fallback one to the tracking one
    over ``reacquire_blend_s``.

    Real angle: observe_actual() feeds the pan angle the firmware reports. While
    that report is fresh (``actual_timeout``), the turret's position at frame
    capture is the real angle carried forward by the model's motion since the
    report, instead of the model alone. A report that the turret is disarmed, or
    that its kill gate is open, resets to home every frame, so nothing winds up
    (and fallback does not run) while the firmware ignores us; its servos are
    detached then, so the reset itself moves nothing.
    """

    def __init__(
        self, minimum: float, maximum: float, home: float, *, reversed: bool = False,
        hfov_deg: float = 50.0, latency_s: float = 0.07, lead_s: float = 0.15,
        coast_s: float = 0.3, hold_s: float | None = None, return_speed: float = 30.0,
        alpha: float = 0.5, beta: float = 0.15, jump_deg: float = 25.0,
        maneuver_deg: float = 2.0, maneuver_alpha: float = 0.8, maneuver_beta: float = 0.5,
        start_deg: float = 1.0, stop_deg: float = 0.4, still_speed: float = 3.0,
        search_speed: float = 60.0, handoff_timeout: float = 0.6, actual_timeout: float = 0.3,
        mode: str = "predictive", gain: float = 0.12, max_rate: float = 120.0,
        dead_zone: float = 0.012, full_gain_error: float = 0.05, error_tau: float = 0.12,
        fallback_tau: float = 0.25, fallback_blend_s: float = 1.0, fallback_lead_s: float = 0.15,
        fallback_omega: float = 6.0, fallback_max_accel: float = 600.0, reacquire_blend_s: float = 0.3,
        lost_hold_s: float = 1.0, fallback_search_after_s: float = 1.5, fallback_search_deg: float = 20.0,
        fallback_search_speed: float = 30.0,
    ) -> None:
        if mode not in ("predictive", "proportional"):
            raise ValueError("pan mode must be 'predictive' or 'proportional'")
        if not 0 <= dead_zone < full_gain_error:
            raise ValueError("pan dead zone must be at least 0 and below the full-gain error")
        self.mode, self.gain, self.max_rate = mode, gain, max_rate
        self.dead_zone, self.full_gain_error, self.error_tau = dead_zone, full_gain_error, error_tau
        self.fallback_tau, self.fallback_blend_s, self.fallback_lead_s = fallback_tau, fallback_blend_s, fallback_lead_s
        self.fallback_omega, self.fallback_max_accel = fallback_omega, fallback_max_accel
        self.reacquire_blend_s, self.lost_hold_s = reacquire_blend_s, lost_hold_s
        self.fallback_search_after_s, self.fallback_search_deg = fallback_search_after_s, fallback_search_deg
        self.fallback_search_speed = fallback_search_speed
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
        self.actual: tuple[float, float, bool] | None = None  # (time, real pan, live: armed, gate not open)
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
        self._laptop_ema: float | None = None
        self._laptop_vx = 0.0
        self._laptop_prev: tuple[float, float] | None = None  # (time, laptop x) of the last fresh sample
        self._fb_offset = self._fb_stage = self._fb_rate = 0.0  # entry offset, damped-follow state
        self._fb_since = 0.0  # fallback entry (or last peeper switch): the local-search clock
        self._sweep, self._sweep_dir, self._sweep_amp, self._sweep_run = 0.0, 0.0, 0.0, 0.0
        self._blend: tuple[float, float] | None = None  # (start, fallback command) while reacquiring
        self._hold_at: float | None = None
        self.servo.target = float(round(self.home))

    def _clamp(self, angle: float) -> float:
        return max(self.minimum, min(self.maximum, angle))

    def observe_actual(self, pan: float | None, at: float | None, armed: bool | None = True,
                       gate_closed: bool | None = None) -> None:
        """Record the firmware's reported pan angle (degrees) and when it was read.

        The turret is live unless the report says disarmed or the kill gate open.
        """
        live = armed is not False and gate_closed is not False
        self.actual = None if pan is None or at is None else (at, float(pan), live)

    def _actual_fresh(self, now: float) -> bool:
        return self.actual is not None and now - self.actual[0] <= self.actual_timeout

    @property
    def last_update(self) -> float | None:
        """The ``now`` of the last update()."""
        return self._last_now

    def real_pan(self) -> float | None:
        """The firmware's reported pan if it was fresh and live at the last update, else None."""
        if self._last_now is None or not self._actual_fresh(self._last_now) or not self.actual[2]:
            return None
        return self.actual[1]

    def current_pan(self) -> float:
        """Where the turret points now: the fresh real angle, else the firmware model's position."""
        real = self.real_pan()
        return real if real is not None else self.servo.position

    def expected_image_x(self, bearing: float) -> float:
        """Turret-image x (unclamped) of a face at world ``bearing``, the inverse of the
        bearing formula above: 0.5 + sign * (bearing - current pan) / hfov."""
        return 0.5 + self.sign * (bearing - self.current_pan()) / self.hfov_deg

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
        fallback_map: Callable[[float], float | None] | None = None,
        fallback_domain: tuple[float, float] | None = None,
    ) -> float | None:
        """Advance one frame; return the image error (x - 0.5) or None without a target.

        ``laptop_active`` (the laptop sees a peeper), ``handoff_angle`` (the map's pan
        for the laptop's target_x) and ``laptop_x`` drive acquire when ``target_x`` is None;
        ``fallback_map`` (laptop_x -> pan, None while unusable) turns acquire into fallback;
        ``fallback_domain`` (lowest, highest learned laptop x) lets it search at once while the
        map extrapolates.
        """
        now = time.monotonic() if now is None else now
        self.servo.advance(now)
        dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
        self._last_now = now
        captured = now - self.latency_s
        if self._actual_fresh(now) and not self.actual[2]:
            self.reset()  # Disarmed or gate open: the firmware ignores commands, so do not wind up.
            return None
        if target_x is None:
            if self._blend is not None:  # Lost mid-blend: carry on from the angle actually sent.
                self.angle, self._blend = self._blended(now), None
            coasting = self.state in ("tracking", "coasting") and self._still_coasting(now)
            fallback_x = self._fallback_x(fallback_map, laptop_x) if laptop_active and not coasting else None
            if fallback_x is not None:
                self._fallback(now, dt, fallback_map, fallback_x, laptop_x is not None, fallback_domain)
            elif laptop_active and self.search_speed > 0 and not coasting:
                self._acquire(now, dt, handoff_angle, laptop_x)
            else:
                self._coast_or_return(now, dt, captured)
            error = None
        else:
            error = target_x - 0.5  # Positive: face right of image centre.
            if self.state == "fallback" and self.reacquire_blend_s > 0:
                self._blend = (now, self._clamp(self.angle))  # Ramp from here to the tracking command.
            if self.mode == "proportional":
                self._centre(error, dt)
            else:
                self._measure(self.turret_position(captured, now) + self.sign * error * self.hfov_deg, captured)
                self._follow()
            self.state, self.lost_at = "tracking", None
            if self._blend is not None and now - self._blend[0] >= self.reacquire_blend_s:
                self._blend = None
        if error is None and self.state not in ("tracking", "coasting"):
            self._error_ema = None  # A face found again after a loss starts a fresh average.
        self.servo.target = self.command  # What send_angles() sends.
        return error

    @property
    def command(self) -> float:
        """The angle to send: float in proportional mode (N lines carry tenths), whole degrees in predictive."""
        angle = self._clamp(self.angle if self._blend is None else self._blended(self._last_now))
        return angle if self.mode == "proportional" else float(round(angle))

    @property
    def local_search(self) -> bool:
        """True while fallback sweeps around the map estimate (no turret face for a while)."""
        return self.state == "fallback" and self._sweep_dir != 0

    def _blended(self, now: float) -> float:
        """The command ramping (smoothstep) from the fallback one to the tracking one."""
        start, origin = self._blend
        w = max(0.0, min(1.0, (now - start) / self.reacquire_blend_s))
        return origin + w * w * (3.0 - 2.0 * w) * (self.angle - origin)

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

    def _centre(self, error: float, dt: float) -> None:
        """Proportional mode: nudge the turret a fraction of the way toward centring the face.

        It needs no camera-latency model, so a wrong latency guess cannot make it hunt.
        Everything is per second, not per frame, so the motion does not change with the
        frame rate: the image error is smoothed with time constant ``error_tau``; the
        command moves ``gain`` of the remaining error per 1/15 s (compounded over dt),
        capped at ``max_rate`` deg/s; and a soft dead zone fades the gain from zero at
        ``dead_zone`` to full at ``full_gain_error`` (smoothstep), so a still face gives
        a still turret without the jolt of a hard start/stop switch.
        """
        dt = min(dt, _PROPORTIONAL_MAX_DT)
        if self._error_ema is None:
            self._error_ema = error
        elif dt > 0:
            share = 1.0 - math.exp(-dt / self.error_tau) if self.error_tau > 0 else 1.0
            self._error_ema += share * (error - self._error_ema)
        e = self._error_ema
        ramp = (abs(e) - self.dead_zone) / (self.full_gain_error - self.dead_zone)
        ramp = max(0.0, min(1.0, ramp))
        fraction = 1.0 - (1.0 - self.gain) ** (dt * _PROPORTIONAL_REF_FPS)
        step = fraction * ramp * ramp * (3.0 - 2.0 * ramp) * e * self.hfov_deg
        limit = self.max_rate * dt
        self.angle = self._clamp(self.angle + self.sign * max(-limit, min(limit, step)))
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

    def _fallback_x(self, fallback_map, laptop_x: float | None) -> float | None:
        """The laptop x to follow, or None when fallback cannot run (no usable map or x)."""
        if fallback_map is None:
            return None
        if laptop_x is None and self.state == "fallback":
            laptop_x = self._laptop_ema  # A heartbeat without target_x: keep the last aim.
        if laptop_x is None or fallback_map(laptop_x) is None:
            return None
        return laptop_x

    def _fallback(self, now: float, dt: float, fallback_map, laptop_x: float, fresh: bool,
                  domain: tuple[float, float] | None) -> None:
        dt = min(dt, _PROPORTIONAL_MAX_DT)
        share = (1.0 - math.exp(-dt / self.fallback_tau) if self.fallback_tau > 0 else 1.0) if dt > 0 else 0.0
        if self.state != "fallback" or self._laptop_ema is None:
            self.state, self._laptop_ema = "fallback", laptop_x
            self.bearing, self.velocity, self.moving, self.lost_at = None, 0.0, False, None
            self._laptop_vx, self._laptop_prev = 0.0, (now, laptop_x)
            self._fb_offset = self.angle - fallback_map(laptop_x)  # No jump: start from where we point.
            self._fb_stage, self._fb_rate, dt = self.angle, 0.0, 0.0
            self._start_local_search(now)
        elif not fresh:
            self._laptop_vx = 0.0  # A heartbeat without target_x: hold the last aim.
        else:
            prev_t, prev_x = self._laptop_prev
            self._laptop_prev = (now, laptop_x)
            if abs(laptop_x - prev_x) > _FALLBACK_SWITCH_X:
                # The laptop picked another peeper: aim there now (the entry offset belonged to the
                # old one); the damped follow glides over.
                self._laptop_ema, self._laptop_vx, self._fb_offset = laptop_x, 0.0, 0.0
                self._start_local_search(now)
            else:
                if now - prev_t > 1e-3:
                    raw = (laptop_x - prev_x) / (now - prev_t)
                    raw = max(-_FALLBACK_MAX_VX, min(_FALLBACK_MAX_VX, raw))
                    self._laptop_vx += share * (raw - self._laptop_vx)
                predicted = self._laptop_ema + self._laptop_vx * dt  # Predicted forward: no ramp lag.
                self._laptop_ema = predicted + share * (laptop_x - predicted)
        if dt > 0:
            self._fb_offset *= math.exp(-dt / self.fallback_blend_s) if self.fallback_blend_s > 0 else 0.0
        look = max(0.0, min(1.0, self._laptop_ema + self._laptop_vx * self.fallback_lead_s))
        mapped = fallback_map(look)
        if mapped is None:
            mapped = fallback_map(self._laptop_ema)
            if mapped is None:
                return
        ahead, behind = fallback_map(min(1.0, look + 0.01)), fallback_map(max(0.0, look - 0.01))
        slope = (ahead - behind) / (min(1.0, look + 0.01) - max(0.0, look - 0.01)) if (
            ahead is not None and behind is not None) else 0.0
        sweep, sweep_rate = self._local_search(now, dt, mapped, domain)
        self._damped_follow(dt, self._clamp(mapped + self._fb_offset + sweep), slope * self._laptop_vx + sweep_rate)

    def _damped_follow(self, dt: float, aim: float, aim_rate: float) -> None:
        """Critically damped (two equal first-order stages, exact for any dt) with feed-forward."""
        if dt <= 0:
            return
        share = 1.0 - math.exp(-self.fallback_omega * dt) if self.fallback_omega > 0 else 1.0
        predicted = self._fb_stage + aim_rate * dt
        self._fb_stage = predicted + share * (aim - predicted)
        predicted = self.angle + aim_rate * dt
        rate = (predicted + share * (self._fb_stage - predicted) - self.angle) / dt
        accel = self.fallback_max_accel * dt
        rate = max(self._fb_rate - accel, min(self._fb_rate + accel, rate))
        rate = max(-self.max_rate, min(self.max_rate, rate))
        self.angle = self._clamp(self.angle + rate * dt)
        self._fb_rate = rate

    def _start_local_search(self, now: float) -> None:
        self._fb_since = now
        self._sweep, self._sweep_dir, self._sweep_amp, self._sweep_run = 0.0, 0.0, self.fallback_search_deg, 0.0

    def _local_search(self, now: float, dt: float, mapped: float,
                      domain: tuple[float, float] | None) -> tuple[float, float]:
        """(sweep offset, its rate) around the map estimate once fallback has gone a while without a face."""
        if self.fallback_search_deg <= 0 or self.fallback_search_speed <= 0:
            return 0.0, 0.0
        if self._sweep_dir == 0:
            extrapolating = domain is not None and not (
                domain[0] - _FALLBACK_EXTRAPOLATE_X <= self._laptop_ema <= domain[1] + _FALLBACK_EXTRAPOLATE_X)
            if not extrapolating and now - self._fb_since < self.fallback_search_after_s:
                return 0.0, 0.0
            # Toward home first: an extrapolated estimate most often overshoots outward.
            self._sweep_dir = math.copysign(1.0, self.home - mapped) if abs(self.home - mapped) > 0.5 else 1.0
            return 0.0, 0.0
        step = self.fallback_search_speed * dt
        sweep = self._sweep + self._sweep_dir * step
        low, high = max(-self._sweep_amp, self.minimum - mapped), min(self._sweep_amp, self.maximum - mapped)
        if sweep >= high:
            sweep, self._sweep_dir = high, -1.0
        elif sweep <= low:
            sweep, self._sweep_dir = low, 1.0
        self._sweep_run += step
        if self._sweep_amp == self.fallback_search_deg and self._sweep_run >= 4 * self.fallback_search_deg:
            self._sweep_amp = 2 * self.fallback_search_deg  # Nothing after one full sweep: widen once.
        self._sweep = sweep
        return sweep, self._sweep_dir * self.fallback_search_speed

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
        if self.state in ("handoff", "searching", "fallback"):
            self._start_hold(now)  # The laptop no longer sees a peeper: hold, then glide home.
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
            self._start_hold(now)
        if self.state == "holding":
            if now - self._hold_at < self.lost_hold_s:
                return
            self.state = "returning"
        step = self.return_speed * dt
        self.angle = self.home if abs(self.home - self.angle) <= step else (
            self.angle + math.copysign(step, self.home - self.angle))
        if self.angle == self.home:
            self.state = "home"

    def _start_hold(self, now: float) -> None:
        self.state = "holding" if self.lost_hold_s > 0 else "returning"
        self.bearing, self.velocity, self.moving, self._hold_at = None, 0.0, False, now
