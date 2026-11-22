"""Measure HeadEstimator on recorded clips: speed, hits with the face hidden, false positives.

    .venv/bin/python tools/measure_head_fallback.py CLIP.mp4 [...] [--every 10]

Each sampled frame is letterboxed to 4:3 and scaled to 640x480 (the turret camera's size).
BlazeFace (back model) finds the face; the face is then blurred out (a turned-away or
masked peeper) and HeadEstimator runs. A hit is a head candidate whose centre lies within
the face box grown by half its size on every side. Negatives are the top corners of each
frame (building, no person) cropped to 4:3 and scaled to 640x480.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medusa.body_detector import HeadEstimator  # noqa: E402
from medusa.tflite_detector import TFLiteFaceDetector  # noqa: E402


def to_640(frame):
    h, w = frame.shape[:2]
    tall = w * 3 // 4
    pad = max(0, (tall - h) // 2)
    frame = cv2.copyMakeBorder(frame, pad, max(0, tall - h - pad), 0, 0, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)


def blur_out(frame, box):
    l, t = max(0, int(box.left - 0.1 * (box.right - box.left))), max(0, int(box.top - 0.1 * (box.bottom - box.top)))
    r, b = int(box.right + 0.1 * (box.right - box.left)), int(box.bottom + 0.1 * (box.bottom - box.top))
    out = frame.copy()
    region = out[t:b, l:r]
    if region.size:
        out[t:b, l:r] = cv2.blur(region, (max(3, (r - l) // 2) | 1, max(3, (b - t) // 2) | 1))
    return out


def near(head, face) -> bool:
    cx, cy = (head.left + head.right) / 2, (head.top + head.bottom) / 2
    w, h = face.right - face.left, face.bottom - face.top
    return face.left - w / 2 <= cx <= face.right + w / 2 and face.top - h / 2 <= cy <= face.bottom + h / 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--every", type=int, default=10)
    args = parser.parse_args()
    faces = TFLiteFaceDetector("qnx/face_detector.tflite")
    est = HeadEstimator()
    times: list[float] = []
    positives = hidden = hits = still_face = 0
    hit_kind, wrong_kind = Counter(), Counter()
    negatives, neg_fp = 0, Counter()
    no_face_frames = no_face_any = 0
    for path in args.clips:
        cap = cv2.VideoCapture(path)
        index = 0
        while True:
            ok, raw = cap.read()
            if not ok:
                break
            index += 1
            if index % args.every:
                continue
            frame = to_640(raw)
            found = faces.detect(frame)
            if not found:
                no_face_frames += 1
                no_face_any += bool(est.detect(frame))
            for face in found[:1]:
                positives += 1
                masked = blur_out(frame, face)
                if faces.detect(masked):
                    still_face += 1
                hidden += 1
                start = time.perf_counter()
                heads = est.detect(masked)
                times.append((time.perf_counter() - start) * 1000)
                good = [h for h in heads if near(h, face)]
                hits += bool(good)
                for h in good:
                    hit_kind[h.kind] += 1
                for h in heads:
                    if not near(h, face):
                        wrong_kind[h.kind] += 1
            h, w = raw.shape[:2]
            cw, ch = int(w * 0.32), int(w * 0.32 * 3 / 4)
            for x0 in (0, w - cw):
                crop = cv2.resize(raw[0:ch, x0:x0 + cw], (640, 480), interpolation=cv2.INTER_AREA)
                negatives += 1
                for head in est.detect(crop):
                    neg_fp[head.kind] += 1
    print(f"detectors: {est.kinds}")
    if times:
        times.sort()
        print(f"HeadEstimator.detect on 640x480: mean {statistics.mean(times):.1f} ms, "
              f"p95 {times[int(0.95 * (len(times) - 1))]:.1f} ms, max {times[-1]:.1f} ms (n={len(times)})")
    print(f"face frames: {positives}; face blurred out: BlazeFace still finds one in {still_face}")
    print(f"head found near the hidden face: {hits}/{hidden} "
          f"({100 * hits / max(1, hidden):.0f}%) by kind {dict(hit_kind)}")
    print(f"candidates away from the hidden face (other heads or false): {dict(wrong_kind)}")
    print(f"frames where BlazeFace found nothing: {no_face_frames}, any head candidate there: {no_face_any}")
    print(f"negatives (no-person corner crops): {negatives}, false heads {sum(neg_fp.values())} {dict(neg_fp)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
