"""Score laptop-detector pipeline variants on recorded clips (e.g. d3_look.mp4, d3_away.mp4).

Each clip runs through the same path as the live loop: LandmarkPipeline (--zoom off / far /
all) -> landmarker_faces -> PeeperTracker, with the clip's own frame times as the clock.
The subject is the largest face that is not the tracker's user (the largest, sticky face).

Per clip and variant it prints:
  found   fraction of frames with a subject face
  raw     fraction of frames the subject's own per-frame gaze says looking
  track   fraction of frames the subject's track is looking (smoothed, hysteresis)
  active  fraction of frames the detector is ACTIVE (what the Pi would get)
  face/eye  median subject face width and eye width in pixels
  yawerr  with --ref-dir: median |head yaw - the reference clip's head yaw| over frames where
          both have a subject; the reference is <ref-dir>/<stem minus _masked>.mp4, same variant
  fps     frames per second of the loop's work: pipeline, tracker, and the preview's JPEG
          encode (camera read excluded)
A *_look clip wants found/track/active high; a *_away clip wants track/active low.

  .venv/bin/python tools/eval_distance.py clips/d*.mp4 --zoom off,far
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from laptop_detector import (  # noqa: E402
    MODELS, ZOOM_FINDERS, detection, landmarker_faces, small_face_defaults,
)
from medusa.gaze import LEFT_EYE, RIGHT_EYE, GazeConfig  # noqa: E402
from medusa.peeper_tracker import PeeperTracker, TrackerConfig  # noqa: E402
from medusa.zoom import FINDERS, ZOOM_MODES, LandmarkPipeline, parse_enhance  # noqa: E402


def eye_px(landmarks, width: int, height: int) -> float | None:
    if len(landmarks) < 478:
        return None
    sizes = []
    for eye in (RIGHT_EYE, LEFT_EYE):
        a, b = landmarks[eye["corner_a"]], landmarks[eye["corner_b"]]
        sizes.append(((a.x - b.x) * width) ** 2 + ((a.y - b.y) * height) ** 2)
    return statistics.mean(s ** 0.5 for s in sizes)


def run_clip(path: Path, zoom: str, gaze: GazeConfig, tracking: TrackerConfig, *,
             enhance: tuple[str, ...] = (), max_frames: int | None = None,
             finder: str = "yunet") -> dict:
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps_in = capture.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = PeeperTracker(tracking)
    counts = dict(frames=0, found=0, raw=0, track=0, active=0)
    faces_px, eyes_px, yaws = [], [], []
    busy = 0.0
    with LandmarkPipeline(str(MODELS["landmarker"]), str(ZOOM_FINDERS[finder]), zoom=zoom,
                          enhance=enhance, finder=finder) as pipeline:
        while max_frames is None or counts["frames"] < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            now = (counts["frames"] + 1) / fps_in
            height, width = frame.shape[:2]
            start = time.perf_counter()
            raw = pipeline(frame, int(now * 1000))
            faces = landmarker_faces(raw, width, height, gaze)
            result = tracker.update([detection(f, width, height) for f in faces], now,
                                    aspect=height / width)
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])  # the preview's cost
            busy += time.perf_counter() - start
            counts["frames"] += 1
            counts["active"] += result.decision.active
            others = [(f, t) for f, t in zip(faces, result.tracks)
                      if t.id != result.user_id or tracking.min_faces <= 1]
            if not others:
                yaws.append(None)
                continue
            face, track = max(others, key=lambda pair: pair[0].box[2])
            counts["found"] += 1
            counts["raw"] += bool(face.observation.screen_facing)
            counts["track"] += track.looking
            yaws.append(face.gaze.head_yaw if face.gaze is not None else None)
            faces_px.append(face.box[2])
            eye = eye_px(face.raw, width, height)
            if eye is not None:
                eyes_px.append(eye)
    capture.release()
    n = max(1, counts["frames"])
    return {
        "frames": counts["frames"],
        **{key: counts[key] / n for key in ("found", "raw", "track", "active")},
        "face_px": statistics.median(faces_px) if faces_px else None,
        "eye_px": statistics.median(eyes_px) if eyes_px else None,
        "fps": counts["frames"] / busy if busy else None,
        "yaws": yaws,
    }


def yaw_error(yaws, ref_yaws) -> float | None:
    """Median |yaw - reference yaw| over the frames where both found the subject."""
    diffs = [abs(a - b) for a, b in zip(yaws, ref_yaws) if a is not None and b is not None]
    return statistics.median(diffs) if diffs else None


def _fmt(value, spec):
    return "-" if value is None else format(value, spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("clips", nargs="+", type=Path)
    parser.add_argument("--zoom", default="off,far",
                        help=f"comma list of variants from {', '.join(ZOOM_MODES)} (default: off,far)")
    parser.add_argument("--small-rules", choices=("on", "off", "both"), default="both",
                        help="size-aware trust for small faces (the detector's defaults) on, off or both")
    parser.add_argument("--enhance", default="none",
                        help="comma list of crop-enhance variants, each a preset or steps joined "
                             "by + (e.g. none,grey-clahe,bilateral+unsharp)")
    parser.add_argument("--finder", default="yunet",
                        help=f"comma list of small-face finders from {', '.join(FINDERS)}")
    parser.add_argument("--min-faces", type=int, default=2,
                        help="1 = a lone face is the subject (clips with nobody else in them)")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--ref-dir", type=Path,
                        help="unmasked reference clips: adds yawerr for each *_masked clip")
    args = parser.parse_args(argv)
    zooms = [z.strip() for z in args.zoom.split(",") if z.strip()]
    if any(z not in ZOOM_MODES for z in zooms):
        parser.error(f"--zoom takes {', '.join(ZOOM_MODES)}")
    rules = {"on": [True], "off": [False], "both": [False, True]}[args.small_rules]
    try:
        enhances = [(name.strip(), parse_enhance(name)) for name in args.enhance.split(",")]
    except ValueError as error:
        parser.error(str(error))
    base_gaze, base_tracking = GazeConfig(), TrackerConfig(min_faces=args.min_faces)
    finders = [f.strip() for f in args.finder.split(",") if f.strip()]
    if any(f not in FINDERS for f in finders):
        parser.error(f"--finder takes {', '.join(FINDERS)}")
    print(f"{'clip':14} {'zoom':4} {'finder':9} {'enhance':18} {'small':5} {'found':>5} {'raw':>5} {'track':>5} {'active':>6} "
          f"{'face':>5} {'eye':>5} {'yawerr':>6} {'fps':>5}")
    for clip in args.clips:
        for zoom, finder, (enhance_name, enhance), small in (
                (z, f, e, r) for z in zooms for f in finders for e in enhances for r in rules):
            if True:
                gaze, tracking = base_gaze, base_tracking
                if small:
                    gaze_extra, tracking_extra = small_face_defaults()
                    gaze = replace(gaze, **gaze_extra)
                    tracking = replace(tracking, **tracking_extra)
                r = run_clip(clip, zoom, gaze, tracking, enhance=enhance,
                             max_frames=args.max_frames, finder=finder)
                err = None
                if args.ref_dir is not None:
                    ref = args.ref_dir / f"{clip.stem.removesuffix('_masked')}{clip.suffix}"
                    err = yaw_error(r["yaws"], run_clip(
                        ref, zoom, gaze, tracking, enhance=enhance, max_frames=args.max_frames,
                        finder=finder)["yaws"])
                print(f"{clip.stem[:14]:14} {zoom:4} {finder:9} {enhance_name[:18]:18} {'on' if small else 'off':5} "
                      f"{r['found']:5.2f} {r['raw']:5.2f} {r['track']:5.2f} {r['active']:6.2f} "
                      f"{_fmt(r['face_px'], '5.0f')} {_fmt(r['eye_px'], '5.1f')} {_fmt(err, '6.1f')} "
                      f"{_fmt(r['fps'], '5.1f')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
