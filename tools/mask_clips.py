"""Synthetic face masks: paint an opaque cloth mask over the lower face of recorded clips.

The mask is placed from the FaceLandmarker's own landmarks on each (unmasked) frame: it
runs across the middle of the nose (landmark 195, between the bridge and the tip) and the
cheeks below the eyes, then down the jaw line to the chin, grown a little past the jaw.
It is a flat cloth colour with slight per-pixel noise, so the mouth, jaw and nose tip are
gone while the eyes, brows and forehead are untouched.

With --shrink the masked 1 m clip is also written at the 3/5/7/10 m sizes the same way the
unmasked distance copies were made (face scaled to 76/45/33/23 px on a dim blurred canvas,
sensor noise re-added), so masked and unmasked copies line up frame for frame.

  .venv/bin/python tools/mask_clips.py clips/d1_look.mp4 clips/d1_away.mp4 --out masked --shrink
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Lower face oval, cheek (subject's left, 454) round the chin (152) to cheek (234).
LOWER_OVAL = (454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
              148, 176, 149, 150, 136, 172, 58, 132, 93, 234)
# Top edge, from 234's side back across: under the right eye, mid-nose, under the left eye.
TOP_EDGE = (227, 116, 117, 118, 195, 347, 346, 345, 447)
GROW = 1.06          # mask grown about the lower-face centre: cloth sits past the jaw
CLOTH_BGR = (92, 64, 48)  # dark navy cloth
NOISE = 6.0
FACE_1M = {"look": 244, "away": 358}  # median face px in the 1 m clips (as eyes_work/shrink2)
METRES = ((3, 76), (5, 45), (7, 33), (10, 23))


def mask_polygon(points):
    """(N, 2) int32 polygon in pixels from 478 pixel (x, y) landmarks."""
    import numpy as np

    upper = [points[i] for i in TOP_EDGE]
    lower = [points[i] for i in LOWER_OVAL]
    poly = np.array(lower + upper, dtype=np.float64)  # 454 -> chin -> 234 -> nose -> 447
    # Grown about the top edge's mean point, so it spreads down and out, not up over the eyes.
    anchor = np.mean(np.array(upper, dtype=np.float64), axis=0)
    poly = anchor + (poly - anchor) * GROW
    return np.round(poly).astype(np.int32)


def paint_mask(frame, points, rng, colour=CLOTH_BGR, noise: float = NOISE):
    """Paint the mask in place onto a BGR frame; returns the boolean mask that was painted."""
    import cv2
    import numpy as np

    stencil = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(stencil, [mask_polygon(points)], 255)
    region = stencil > 0
    count = int(region.sum())
    if count:
        cloth = np.array(colour, np.float64) + rng.normal(0, noise, (count, 1))
        frame[region] = np.clip(cloth, 0, 255).astype(np.uint8)
    return region


def shrink(frame, k: float, rng):
    """The frame scaled by 1/k, a little off centre on a dim blurred 1920x1080 canvas."""
    import cv2
    import numpy as np

    bg = cv2.GaussianBlur(cv2.resize(frame, (192, 108)), (0, 0), 3)
    canvas = (cv2.resize(bg, (1920, 1080)) * 0.6).astype(np.uint8)
    sw, sh = round(1920 / k), round(1080 / k)
    small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
    small = np.clip(small + rng.normal(0, 3, small.shape), 0, 255).astype(np.uint8)
    x0, y0 = (1920 - sw) // 2 + 300, (1080 - sh) // 2 - 60
    canvas[y0:y0 + sh, x0:x0 + sw] = small
    return canvas


def mask_clip(src: Path, out: Path, shrink_to: bool = False) -> dict:
    """Write <out>/<stem>_masked.mp4 (and m<d>_<kind>_masked.mp4 copies). Returns counts."""
    import cv2
    import mediapipe as mp
    import numpy as np

    from laptop_detector import MODELS

    vision = mp.tasks.vision
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    kind = "look" if "look" in src.stem else "away"
    capture = cv2.VideoCapture(str(src))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {}
    counts = dict(frames=0, masked=0)
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODELS["landmarker"])),
        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    last = None
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if not writers:
                writers["d"] = cv2.VideoWriter(str(out / f"{src.stem}_masked.mp4"), fourcc, fps,
                                               (width, height))
                if shrink_to:
                    for metres, _ in METRES:
                        writers[metres] = cv2.VideoWriter(
                            str(out / f"m{metres}_{kind}_masked.mp4"), fourcc, fps, (1920, 1080))
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(image, int(counts["frames"] * 1000 / fps))
            if result.face_landmarks:
                last = [(p.x * width, p.y * height) for p in result.face_landmarks[0]]
            if last is not None:  # a missed frame keeps the previous frame's mask
                paint_mask(frame, last, rng)
                counts["masked"] += 1
            counts["frames"] += 1
            writers["d"].write(frame)
            for metres, face in METRES if shrink_to else ():
                writers[metres].write(shrink(frame, FACE_1M[kind] / face, rng))
    for writer in writers.values():
        writer.release()
    capture.release()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shrink", action="store_true",
                        help="also write the 3/5/7/10 m copies (1 m clips named *_look / *_away)")
    args = parser.parse_args(argv)
    for clip in args.clips:
        counts = mask_clip(clip, args.out, args.shrink)
        print(f"{clip.name}: masked {counts['masked']}/{counts['frames']} frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
