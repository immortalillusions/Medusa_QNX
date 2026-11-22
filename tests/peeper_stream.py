"""Synthetic noisy face streams for the peeper tracker tests (pure Python, seeded)."""

import math
import random

from medusa.gaze import GazeConfig, GazeEstimate, looking_at_screen
from medusa.models import FaceObservation
from medusa.peeper_tracker import Detection

ASPECT = 720 / 1280  # frame height / width
CONFIG = GazeConfig()


def gaze(head_yaw=0.0, eye_yaw=0.0, pitch=-10.0, eye_pitch=0.0, closed=False):
    """What estimate_gaze returns: eyes closed -> head only."""
    if closed:
        eye_yaw = eye_pitch = 0.0
    gaze_yaw, gaze_pitch = head_yaw + eye_yaw, pitch + eye_pitch
    return GazeEstimate(head_yaw, pitch, eye_yaw, eye_pitch, gaze_yaw, gaze_pitch,
                        looking_at_screen(gaze_yaw, gaze_pitch, CONFIG),
                        "head" if closed else "eyes")


def det(cx, g=None, size=0.1, cy=0.4, looking=None):
    """A square face ``size`` frame-widths wide."""
    return Detection(cx, cy, size, size / ASPECT,
                     looking=g.looking if looking is None and g is not None else bool(looking),
                     gaze=g)


def observation(d: Detection) -> FaceObservation:
    """The same face as the old SnooperSelector saw it."""
    return FaceObservation("mac-camera", d.center_x, d.center_y, d.width, d.looking)


def noisy_viewer(seconds, fps, *, seed=1, with_user=False, head_yaw=10.0, eye_yaw=4.0,
                 x0=0.45, sway=0.02, noise_deg=3.0, spike_p=0.015):
    """(t, detections) for someone sitting still-ish and looking at the screen.

    Gaze noise is uniform +-noise_deg; every 3-5 s a 4-frame blink (head only) bracketed by
    one-frame eye-pitch spikes; ``spike_p`` per frame of an eye-yaw saturation at +35 deg.
    The face centre sways around x0 (crossing the old selector's 0.45 bucket edge).
    """
    rng = random.Random(seed)
    frames = int(seconds * fps)
    next_blink = rng.uniform(3, 5)
    blink_left = 0
    spike_next = False
    for n in range(frames):
        t = n / fps
        eye, eye_pitch, closed = eye_yaw + rng.uniform(-noise_deg, noise_deg), \
            rng.uniform(-noise_deg, noise_deg), False
        if spike_next:
            eye_pitch, spike_next = -35.0, False
        if t >= next_blink and blink_left == 0:
            blink_left, next_blink = 5, t + rng.uniform(3, 5)
            eye_pitch = -35.0  # lids closing: the iris reads far down for a frame
        elif blink_left:
            blink_left -= 1
            closed = blink_left > 0
            spike_next = blink_left == 0
        if rng.random() < spike_p:
            eye = 35.0
        x = x0 + sway * math.sin(2 * math.pi * t / 7.0) + rng.gauss(0, 0.004)
        faces = []
        if with_user:
            faces.append(det(0.3 + rng.gauss(0, 0.003), gaze(0, rng.uniform(-3, 3)), size=0.3,
                             cy=0.55))
        faces.append(det(x, gaze(head_yaw, eye, -10.0, eye_pitch, closed),
                         size=0.1 + rng.gauss(0, 0.003)))
        yield t, faces


def edges(actives):
    """(activations, flips) in a sequence of booleans."""
    rises = sum(1 for a, b in zip([False] + actives, actives) if b and not a)
    flips = sum(1 for a, b in zip([False] + actives, actives) if a != b)
    return rises, flips
