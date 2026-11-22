"""BlazeFace back-model support, the turret face debounce, and the AI-loop frame rate.

Runs on plain python3 (no NumPy/TFLite): the anchor generator, head ordering and
box decoder are pure Python; the NumPy path is checked against the real models
on the bench (see the PR notes), not here.
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import qnx_ai_service
from medusa import trigger as trigger_module
from medusa.tflite_detector import (
    MODEL_SPECS, FaceBox, TFLiteFaceDetector, model_spec_for_input, order_output_heads, ssd_anchors,
)
from medusa.trigger import (
    CONFIRMING_REASON, HELD_REASON, NO_FACE_REASON, FaceConfirm, FrameRate, LaptopTrigger, PanTracker,
    decide_aim,
)

REPO = Path(__file__).resolve().parent.parent
LAPTOP = ("172.20.10.4", 50000)


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now


class AnchorTests(unittest.TestCase):
    def test_back_model_anchors_follow_mediapipe_ssd_layout(self) -> None:
        spec = MODEL_SPECS[256]
        anchors = ssd_anchors(spec.input_size, spec.strides)
        self.assertEqual((spec.name, len(anchors)), ("back", 896))
        # Stride 16: a 16x16 grid, 2 anchors per cell (one aspect ratio + the interpolated scale), row-major.
        self.assertEqual(anchors[0], (0.5 / 16, 0.5 / 16))
        self.assertEqual(anchors[1], (0.5 / 16, 0.5 / 16))
        self.assertEqual(anchors[2], (1.5 / 16, 0.5 / 16))
        self.assertEqual(anchors[32], (0.5 / 16, 1.5 / 16))
        self.assertEqual(anchors[511], (15.5 / 16, 15.5 / 16))
        # Three stride-32 layers share an 8x8 grid: 6 anchors per cell, after all 512 of the 16x16 grid.
        self.assertEqual(anchors[512:518], [(0.5 / 8, 0.5 / 8)] * 6)
        self.assertEqual(anchors[518], (1.5 / 8, 0.5 / 8))
        self.assertEqual(anchors[512 + 8 * 6], (0.5 / 8, 1.5 / 8))
        self.assertEqual(anchors[-1], (7.5 / 8, 7.5 / 8))

    def test_short_range_anchors_unchanged(self) -> None:
        legacy = []  # The hand-rolled generator this replaced.
        for stride, repeats in ((8, 2), (16, 6)):
            for y in range(0, 128, stride):
                for x in range(0, 128, stride):
                    legacy.extend([((x + stride / 2) / 128, (y + stride / 2) / 128)] * repeats)
        self.assertEqual(ssd_anchors(128, MODEL_SPECS[128].strides), legacy)

    def test_model_is_picked_by_input_size(self) -> None:
        self.assertEqual(model_spec_for_input((1, 256, 256, 3)).name, "back")
        self.assertEqual(model_spec_for_input((1, 128, 128, 3)).name, "short")
        for shape in ((1, 192, 192, 3), (1, 256, 128, 3), (2, 256, 256, 3), (1, 256, 256, 1), (256, 256, 3)):
            self.assertIsNone(model_spec_for_input(shape), shape)

    def test_output_heads_are_ordered_512_then_384(self) -> None:
        # The checked-in back model lists its outputs as 512x16, 384x16, 512x1, 384x1 (by tensor index
        # the scores come first); order by shape, never by position.
        shapes = [(1, 384, 16), (1, 512, 1), (1, 512, 16), (1, 384, 1)]
        self.assertEqual(order_output_heads(shapes), ([2, 0], [1, 3]))
        self.assertEqual(order_output_heads([(1, 896, 16), (1, 896, 1)]), ([0], [1]))

    def test_checked_in_models_exist(self) -> None:
        for path in qnx_ai_service.FACE_MODELS.values():
            self.assertTrue((REPO / path).is_file(), path)


class BackModelDecodeTests(unittest.TestCase):
    def detector(self, score_threshold=0.55):
        detector = TFLiteFaceDetector.__new__(TFLiteFaceDetector)
        spec = MODEL_SPECS[256]
        detector.input_size, detector.score_threshold = spec.input_size, score_threshold
        detector.anchors = ssd_anchors(spec.input_size, spec.strides)
        return detector

    def test_decode_uses_256_pixel_offsets_letterbox_and_nose_keypoint(self) -> None:
        # 640x480 letterboxed into 256: scale 0.4 -> 256x192, 32 px of padding above and below.
        # Anchor 680 is cell (4, 3) of the 8x8 grid: centre (144, 112) in model pixels.
        index = 512 + (3 * 8 + 4) * 6
        raw = [(0.0,) * 16] * 896
        raw[index] = (4, 8, 20, 20, 0, 0, 0, 0, 2, -2, 0, 0, 0, 0, 0, 0)  # nose = keypoint 2
        logits = [-10.0] * 896
        logits[index] = 2.0
        boxes = self.detector()._decode(raw, logits, 640, 480, 0, 32, 256, 192)

        self.assertEqual(len(boxes), 1)
        box = boxes[0]
        # Centre (4 + 144, 8 + 112 - 32) = (148, 88) model px, x2.5 to photo px, 20x20 -> 50x50.
        self.assertEqual((box.left, box.top, box.right, box.bottom), (345, 195, 395, 245))
        self.assertAlmostEqual(box.score, 0.88079708)
        self.assertEqual(box.nose, (365, 195))

    def test_score_threshold_is_configurable(self) -> None:
        raw = [(0, 0, 20, 20) + (0,) * 12]
        anchors = [(0.5, 0.5)]
        logit = 0.4  # sigmoid 0.599
        self.assertEqual(len(self.detector(0.55)._decode(raw, [logit], 256, 256, 0, 0, 256, 256,
                                                         anchors=anchors)), 1)
        self.assertEqual(self.detector(0.65)._decode(raw, [logit], 256, 256, 0, 0, 256, 256,
                                                     anchors=anchors), [])


def pick(x, y=0.5, width=640, height=480):
    box = FaceBox(x * width - 20, y * height - 20, x * width + 20, y * height + 20, 0.9)
    return box, x, y


class FaceConfirmTests(unittest.TestCase):
    def test_two_consecutive_frames_near_the_same_spot_confirm(self) -> None:
        confirm = FaceConfirm()
        self.assertIsNone(confirm.update(pick(0.40), 640, 480))
        self.assertEqual(confirm.streak, 1)
        self.assertEqual(confirm.update(pick(0.50), 640, 480)[1], 0.50)  # moved 0.10 <= 0.15: confirmed
        self.assertFalse(confirm.held)

    def test_a_jump_restarts_the_confirmation(self) -> None:
        confirm = FaceConfirm()
        self.assertIsNone(confirm.update(pick(0.20), 640, 480))
        self.assertIsNone(confirm.update(pick(0.40), 640, 480))  # 0.20 away: a new candidate
        self.assertEqual(confirm.streak, 1)
        self.assertIsNotNone(confirm.update(pick(0.45), 640, 480))
        # Vertical distance counts in frame widths too: 0.25 of the height is 0.1875 widths.
        confirm = FaceConfirm()
        confirm.update(pick(0.5, 0.25), 640, 480)
        self.assertIsNone(confirm.update(pick(0.5, 0.50), 640, 480))

    def test_a_single_frame_flash_never_aims(self) -> None:
        confirm = FaceConfirm()
        self.assertIsNone(confirm.update(pick(0.5), 640, 480))
        self.assertIsNone(confirm.update(None, 640, 480))
        self.assertIsNone(confirm.update(pick(0.5), 640, 480))  # The streak broke on the empty frame.

    def test_two_frame_dropout_is_held_then_dropped(self) -> None:
        confirm = FaceConfirm()
        confirm.update(pick(0.5), 640, 480)
        target = confirm.update(pick(0.52), 640, 480)
        self.assertIs(confirm.update(None, 640, 480), target)
        self.assertTrue(confirm.held)
        self.assertIs(confirm.update(None, 640, 480), target)
        self.assertIsNone(confirm.update(None, 640, 480))
        self.assertFalse(confirm.held)
        self.assertIsNone(confirm.update(pick(0.52), 640, 480))  # Lost: confirm again from scratch.

    def test_face_returning_within_the_hold_is_tracked_at_once(self) -> None:
        confirm = FaceConfirm()
        confirm.update(pick(0.5), 640, 480)
        confirm.update(pick(0.5), 640, 480)
        confirm.update(None, 640, 480)
        self.assertEqual(confirm.update(pick(0.6), 640, 480)[1], 0.6)
        self.assertFalse(confirm.held)

    def test_a_distant_new_face_holds_the_old_target_until_it_confirms(self) -> None:
        confirm = FaceConfirm()
        confirm.update(pick(0.2), 640, 480)
        old = confirm.update(pick(0.2), 640, 480)
        self.assertIs(confirm.update(pick(0.8), 640, 480), old)
        self.assertTrue(confirm.held)
        self.assertEqual(confirm.update(pick(0.8), 640, 480)[1], 0.8)

    def test_one_and_zero_disable_the_debounce(self) -> None:
        confirm = FaceConfirm(confirm_frames=1, dropout_frames=0)
        self.assertEqual(confirm.update(pick(0.2), 640, 480)[1], 0.2)
        self.assertEqual(confirm.update(pick(0.9), 640, 480)[1], 0.9)
        self.assertIsNone(confirm.update(None, 640, 480))


class DecideAimConfirmTests(unittest.TestCase):
    def test_reasons_and_held_flag(self) -> None:
        confirm = FaceConfirm()
        face = pick(0.5)[0]
        aim = decide_aim(True, "", [face], 640, 480, confirm=confirm)
        self.assertEqual((aim.active, aim.reason), (False, CONFIRMING_REASON))
        aim = decide_aim(True, "", [face], 640, 480, confirm=confirm)
        self.assertTrue(aim.active)
        self.assertFalse(aim.held)
        aim = decide_aim(True, "", [], 640, 480, confirm=confirm)
        self.assertEqual((aim.active, aim.held, aim.reason, aim.target_x), (True, True, HELD_REASON, 0.5))
        decide_aim(True, "", [], 640, 480, confirm=confirm)
        aim = decide_aim(True, "", [], 640, 480, confirm=confirm)
        self.assertEqual((aim.active, aim.reason), (False, NO_FACE_REASON))

    def test_frames_before_the_laptop_fires_count_toward_confirmation(self) -> None:
        confirm = FaceConfirm()
        face = pick(0.5)[0]
        self.assertFalse(decide_aim(False, "laptop sees no peeper", [face], 640, 480, confirm=confirm).active)
        self.assertTrue(decide_aim(True, "", [face], 640, 480, confirm=confirm).active)

    def test_inference_failure_is_inactive_and_counts_as_a_miss(self) -> None:
        confirm = FaceConfirm()
        face = pick(0.5)[0]
        decide_aim(True, "", [face], 640, 480, confirm=confirm)
        aim = decide_aim(True, "", None, 640, 480, confirm=confirm)
        self.assertEqual((aim.active, aim.reason), (False, "AI inference unavailable"))
        self.assertEqual(confirm.streak, 0)


class FrameRateTests(unittest.TestCase):
    def test_rolling_two_second_window(self) -> None:
        rate = FrameRate()
        self.assertIsNone(rate.fps())
        rate.tick(10.0)
        self.assertIsNone(rate.fps())
        for i in range(1, 31):
            rate.tick(10.0 + i * 0.1)  # 10 fps for 3 s
        self.assertEqual(rate.fps(), 10.0)
        self.assertEqual(len(rate.times), 21)  # Only the last 2 s are kept.
        for i in range(1, 11):
            rate.tick(13.0 + i * 0.2)  # Slows to 5 fps for 2 s: the old rate has aged out.
        self.assertEqual(rate.fps(), 5.0)

    def test_same_instant_has_no_rate(self) -> None:
        rate = FrameRate()
        rate.tick(1.0)
        rate.tick(1.0)
        self.assertIsNone(rate.fps())


class FaceFlagTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = qnx_ai_service.parse_args(["--trigger-udp-port", "8767"])
        self.assertEqual((args.face_model, args.face_detector), ("back", Path("qnx/face_detector.tflite")))
        self.assertEqual((args.face_score, args.tflite_threads), (0.55, 4))
        self.assertEqual((args.face_confirm_frames, args.face_dropout_frames), (2, 2))
        args = qnx_ai_service.parse_args(["--event-host", "127.0.0.1"])
        self.assertEqual((args.face_model, args.face_detector),
                         ("short", Path("qnx/blaze_face_short_range.tflite")))

    def test_overrides(self) -> None:
        args = qnx_ai_service.parse_args([
            "--trigger-udp-port", "8767", "--face-model", "short", "--face-score", "0.7", "--tflite-threads", "2",
        ])
        self.assertEqual((args.face_model, args.face_detector, args.face_score, args.tflite_threads),
                         ("short", Path("qnx/blaze_face_short_range.tflite"), 0.7, 2))
        args = qnx_ai_service.parse_args(["--trigger-udp-port", "8767", "--face-detector", "/m.tflite"])
        self.assertEqual(args.face_detector, Path("/m.tflite"))

    def test_invalid_values_are_refused(self) -> None:
        for extra in (["--face-score", "0"], ["--face-score", "1"], ["--tflite-threads", "0"],
                      ["--face-confirm-frames", "0"], ["--face-dropout-frames", "-1"], ["--face-model", "full"]):
            with patch("sys.stderr"), self.assertRaises(SystemExit):
                qnx_ai_service.parse_args(["--trigger-udp-port", "8767", *extra])

    def test_main_passes_score_and_threads_to_the_detector(self) -> None:
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = KeyboardInterrupt
        cv2 = SimpleNamespace(VideoCapture=Mock(return_value=camera), error=RuntimeError)
        with (
            patch.dict(sys.modules, {"cv2": cv2}),
            patch.object(sys, "argv", ["qnx_ai_service.py", "--preview-port", "0", "--trigger-udp-port", "8767",
                                       "--face-score", "0.6", "--tflite-threads", "3"]),
            patch.object(qnx_ai_service, "TFLiteFaceGazeDetector") as detector_class,
            patch.object(qnx_ai_service.LaptopTrigger, "bind", return_value=LaptopTrigger(MagicMock())),
        ):
            self.assertEqual(qnx_ai_service.main(), 0)
        detector_class.assert_called_once_with(
            str(Path("qnx/face_detector.tflite")), str(Path("qnx/face_landmark_detector.tflite")),
            score_threshold=0.6, num_threads=3,
        )


class TriggerLoopDebounceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        patcher = patch.object(trigger_module, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_default_loop_confirms_holds_reports_fps_and_never_feeds_pan_a_held_target(self) -> None:
        face = FaceBox(300, 220, 340, 260, 0.9)  # image centre
        script = [[face], [face], [], [], [], [face]]
        frame = SimpleNamespace(shape=(480, 640, 3))
        index, pending, pan_inputs = {"i": -1}, [], []

        def read():
            index["i"] += 1
            if index["i"] == len(script):
                raise KeyboardInterrupt
            self.clock.now += 0.1
            pending.append((json.dumps({"type": "medusa-peeper", "active": True, "target_x": 0.5}).encode(),
                            LAPTOP))
            return True, frame

        def recvfrom(_size):
            if pending:
                return pending.pop(0)
            raise BlockingIOError

        real_update = PanTracker.update

        def update(pan, target_x, *args, **kwargs):
            pan_inputs.append(target_x)
            return real_update(pan, target_x, *args, **kwargs)

        trigger_sock = MagicMock()
        trigger_sock.recvfrom.side_effect = recvfrom
        camera = Mock()
        camera.read.side_effect = read
        detector = Mock()
        detector.face_detector.detect.side_effect = lambda _frame: script[index["i"]]
        preview = Mock()
        args = qnx_ai_service.parse_args(["--trigger-udp-port", "8767", "--camera-on-turret", "--no-search"])
        with (
            patch.dict(sys.modules, {"cv2": SimpleNamespace(error=RuntimeError)}),
            patch.object(qnx_ai_service, "draw_aim_boxes"),
            patch.object(qnx_ai_service.LaptopTrigger, "bind", return_value=LaptopTrigger(trigger_sock)),
            patch.object(PanTracker, "update", update),
        ):
            self.assertEqual(qnx_ai_service.run_laptop_trigger(args, camera, detector, preview, None), 0)

        statuses = [c.kwargs["extra"] | {"active": c.args[0], "reason": c.args[1]}
                    for c in preview.state.update_status.call_args_list]
        self.assertEqual([s["active"] for s in statuses], [False, True, True, True, False, False])
        self.assertEqual(statuses[0]["reason"], CONFIRMING_REASON)
        self.assertEqual(statuses[2]["reason"], HELD_REASON)
        self.assertEqual(statuses[4]["reason"], NO_FACE_REASON)
        self.assertEqual(statuses[5]["reason"], CONFIRMING_REASON)  # Lost: confirm again.
        self.assertEqual([s["fps"] for s in statuses], [None, 10.0, 10.0, 10.0, 10.0, 10.0])
        # Pan sees only fresh confirmed faces; held frames coast on the tracker's own prediction.
        self.assertEqual(pan_inputs, [None, 0.5, None, None, None, None])


if __name__ == "__main__":
    unittest.main()
