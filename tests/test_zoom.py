"""Find-small-then-zoom: crop/map-back math, the zoomer, crop clean-up, size-aware rules, flags."""

import io
import sys
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import laptop_detector
from medusa.gaze import GazeConfig, estimate_gaze, iris_trust
from medusa.peeper_tracker import Detection, PeeperTracker, TrackerConfig
from medusa.tflite_detector import FaceBox
from medusa.yunet_detector import YuNetFaceDetector
from medusa.zoom import (
    PRESETS, CropEnhancer, FaceZoomer, Window, auto_gamma_lut, covered, cut, grid_schedule,
    map_landmarks, merge_boxes, parse_enhance, square_window, tile_windows,
)
from tests.gaze_fixtures import landmarks, pose

W, H = 1920, 1080


class WindowMathTests(unittest.TestCase):
    def test_square_window_is_centred_and_margin_times_the_longer_side(self):
        window = square_window((100, 200, 140, 250), margin=2.0)  # 40 x 50 box, centre 120,225
        self.assertEqual(window.side, 100)
        self.assertEqual((window.left, window.top), (70, 175))

    def test_map_landmarks_takes_crop_corners_and_centre_to_the_window(self):
        window = Window(960, 540, 200)
        points = [SimpleNamespace(x=0, y=0, z=0), SimpleNamespace(x=1, y=1, z=0),
                  SimpleNamespace(x=0.5, y=0.25, z=0.1)]
        mapped = map_landmarks(points, window, W, H)
        self.assertAlmostEqual(mapped[0].x, 960 / W)
        self.assertAlmostEqual(mapped[0].y, 540 / H)
        self.assertAlmostEqual(mapped[1].x, 1160 / W)
        self.assertAlmostEqual(mapped[1].y, 740 / H)
        self.assertAlmostEqual(mapped[2].x, 1060 / W)
        self.assertAlmostEqual(mapped[2].y, 590 / H)
        self.assertAlmostEqual(mapped[2].z, 0.1 * 200 / W)  # crop widths -> frame widths

    def test_cut_upscales_and_pads_outside_the_frame_black(self):
        frame = np.full((H, W, 3), 200, np.uint8)
        crop = cut(frame, Window(-50, -50, 100), 256)
        self.assertEqual(crop.shape, (256, 256, 3))
        self.assertEqual(int(crop[10, 10, 0]), 0)        # top-left quarter is off-frame
        self.assertEqual(int(crop[200, 200, 0]), 200)    # bottom-right quarter is the frame

    def test_cut_then_map_back_lands_on_the_same_pixel(self):
        frame = np.zeros((H, W, 3), np.uint8)
        frame[600:604, 1300:1304] = 255  # a dot at ~(1302, 602)
        window = square_window((1280, 580, 1320, 630), margin=1.8)
        crop = cut(frame, window, 256)
        ys, xs = np.nonzero(crop[..., 0] > 128)
        point = SimpleNamespace(x=xs.mean() / 256, y=ys.mean() / 256)
        back = map_landmarks([point], window, W, H)[0]
        self.assertAlmostEqual(back.x * W, 1302, delta=1.0)
        self.assertAlmostEqual(back.y * H, 602, delta=1.0)

    def test_tiles_cover_the_frame_with_overlap(self):
        tiles = tile_windows(W, H, (2, 1), overlap=0.1)
        self.assertEqual(tiles, [(0, 0, 1056, 1080), (864, 0, 1920, 1080)])
        self.assertEqual(len(grid_schedule(W, H, ((2, 1), (4, 2)))), 10)

    def test_merge_and_covered(self):
        boxes = merge_boxes([(0, 0, 10, 10, 0.9), (1, 1, 11, 11, 0.8), (50, 50, 60, 60, 0.7)])
        self.assertEqual([b[4] for b in boxes], [0.9, 0.7])
        self.assertTrue(covered((2, 2, 6, 6), [(0, 0, 10, 10)]))
        self.assertFalse(covered((20, 20, 30, 30), [(0, 0, 10, 10)]))


class EnhanceTests(unittest.TestCase):
    def test_presets_and_step_lists(self):
        self.assertEqual(parse_enhance("none"), ())
        self.assertEqual(parse_enhance("unsharp+grey"), ("grey", "unsharp"))  # pipeline order
        self.assertEqual(parse_enhance("grey-clahe"), PRESETS["grey-clahe"])
        with self.assertRaises(ValueError):
            parse_enhance("grey+sparkle")

    def test_none_is_identity_and_grey_is_three_equal_channels(self):
        crop = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        self.assertIs(CropEnhancer(())(crop), crop)
        grey = CropEnhancer(("grey", "clahe", "unsharp"))(crop)
        self.assertEqual(grey.shape, crop.shape)
        self.assertTrue((grey[..., 0] == grey[..., 1]).all() and (grey[..., 1] == grey[..., 2]).all())

    def test_auto_gamma_brightens_dark_and_darkens_bright(self):
        self.assertGreater(int(auto_gamma_lut(40)[40]), 40)
        self.assertLess(int(auto_gamma_lut(220)[220]), 220)


class FakeLandmarker:
    """Returns one face covering the middle of whatever crop it is given."""

    def __init__(self, found=True):
        self.found, self.calls = found, 0

    def detect(self, image):
        self.calls += 1
        if not self.found:
            return SimpleNamespace(face_landmarks=[], facial_transformation_matrixes=[])
        points = [SimpleNamespace(x=x, y=y, z=0.0) for x, y in ((0.3, 0.3), (0.7, 0.7))]
        return SimpleNamespace(face_landmarks=[points], facial_transformation_matrixes=[pose()])


class FakeDetector:
    def __init__(self, boxes):
        self.boxes, self.calls = boxes, []

    def detect(self, image):
        self.calls.append(image.shape[:2])
        return list(self.boxes)


def fake_mediapipe():
    mp = MagicMock()
    mp.Image = lambda image_format, data: data
    return mp


class FaceZoomerTests(unittest.TestCase):
    def zoomer(self, boxes, found=True, **kwargs):
        kwargs.setdefault("grids", ((1, 1),))
        kwargs.setdefault("tiles_per_frame", None)
        return FaceZoomer(FakeLandmarker(found), FakeDetector(boxes), **kwargs)

    def test_a_missed_small_face_is_zoomed_and_mapped_back(self):
        zoomer = self.zoomer([FaceBox(1000, 500, 1040, 550, 0.9)])
        frame = np.zeros((H, W, 3), np.uint8)
        with patch.dict(sys.modules, {"mediapipe": fake_mediapipe()}):
            result = zoomer(frame)
        self.assertEqual(result.zoomed, 1)
        window = square_window((1000, 500, 1040, 550))
        first = result.face_landmarks[0][0]
        self.assertAlmostEqual(first.x, (window.left + 0.3 * window.side) / W)
        self.assertAlmostEqual(first.y, (window.top + 0.3 * window.side) / H)

    def test_a_face_the_full_frame_landmarker_has_is_not_zoomed_again(self):
        zoomer = self.zoomer([FaceBox(1000, 500, 1040, 550, 0.9)])
        full = SimpleNamespace(face_landmarks=[[SimpleNamespace(x=900 / W, y=450 / H),
                                                SimpleNamespace(x=1100 / W, y=650 / H)]],
                               facial_transformation_matrixes=[pose()])
        with patch.dict(sys.modules, {"mediapipe": fake_mediapipe()}):
            result = zoomer(np.zeros((H, W, 3), np.uint8), full)
        self.assertEqual((result.zoomed, len(result.face_landmarks)), (0, 1))
        self.assertEqual(zoomer.crop_landmarker.calls, 0)

    def test_a_zoomed_face_is_followed_without_a_new_detection(self):
        zoomer = self.zoomer([FaceBox(1000, 500, 1040, 550, 0.9)])
        frame = np.zeros((H, W, 3), np.uint8)
        with patch.dict(sys.modules, {"mediapipe": fake_mediapipe()}):
            zoomer(frame)
            zoomer.detector.boxes = []  # the finder loses it
            result = zoomer(frame)
        self.assertEqual(result.zoomed, 1)

    def test_no_landmarks_in_the_crop_means_no_face(self):
        zoomer = self.zoomer([FaceBox(1000, 500, 1040, 550, 0.9)], found=False)
        with patch.dict(sys.modules, {"mediapipe": fake_mediapipe()}):
            result = zoomer(np.zeros((H, W, 3), np.uint8))
        self.assertEqual(result.face_landmarks, [])

    def test_a_big_face_the_full_frame_landmarker_missed_is_zoomed_too(self):
        # A masked face close up: no size limit on what the finder hands the crop landmarker.
        zoomer = self.zoomer([FaceBox(700, 200, 1100, 700, 0.4)])
        with patch.dict(sys.modules, {"mediapipe": fake_mediapipe()}):
            result = zoomer(np.zeros((H, W, 3), np.uint8))
        self.assertEqual(result.zoomed, 1)
        self.assertGreater(result.face_landmarks[0][1].x - result.face_landmarks[0][0].x, 0.1)

    def test_crop_tries_a_frame_are_capped(self):
        boxes = [FaceBox(100 * i, 50, 100 * i + 40, 100, 0.35) for i in range(10)]
        zoomer = self.zoomer(boxes, found=False, max_tries=4)
        with patch.dict(sys.modules, {"mediapipe": fake_mediapipe()}):
            zoomer(np.zeros((H, W, 3), np.uint8))
        self.assertEqual(zoomer.crop_landmarker.calls, 4)

    def test_an_unsure_box_with_no_face_waits_before_it_is_tried_again(self):
        zoomer = self.zoomer([FaceBox(1000, 500, 1040, 550, 0.35)], found=False, fail_hold=3)
        with patch.dict(sys.modules, {"mediapipe": fake_mediapipe()}):
            for _ in range(4):
                zoomer(np.zeros((H, W, 3), np.uint8))
        self.assertEqual(zoomer.crop_landmarker.calls, 2)  # frames 1 and 4

    def test_a_sure_box_is_tried_every_frame(self):
        zoomer = self.zoomer([FaceBox(1000, 500, 1040, 550, 0.9)], found=False, fail_hold=3)
        with patch.dict(sys.modules, {"mediapipe": fake_mediapipe()}):
            for _ in range(4):
                zoomer(np.zeros((H, W, 3), np.uint8))
        self.assertEqual(zoomer.crop_landmarker.calls, 4)

    def test_round_robin_runs_tiles_per_frame_tiles(self):
        zoomer = self.zoomer([], grids=((2, 1), (4, 2)), tiles_per_frame=3)
        with patch.dict(sys.modules, {"mediapipe": fake_mediapipe()}):
            for _ in range(4):
                zoomer(np.zeros((H, W, 3), np.uint8))
        self.assertEqual(len(zoomer.detector.calls), 12)  # 10 tiles swept, then 2 more

    def test_temporal_keeps_a_copy_not_the_drawn_on_frame(self):
        zoomer = self.zoomer([], enhance=("temporal",))
        frame = np.zeros((H, W, 3), np.uint8)
        with patch.dict(sys.modules, {"mediapipe": fake_mediapipe()}):
            zoomer(frame)
        frame[:] = 255  # the preview overlay drawn afterwards
        self.assertEqual(int(zoomer._previous[0].max()), 0)


class YuNetTests(unittest.TestCase):
    def test_boxes_and_points_scale_back_to_the_full_frame(self):
        detector = YuNetFaceDetector.__new__(YuNetFaceDetector)
        import cv2
        detector.cv2, detector.scale, detector._size = cv2, 0.5, None
        row = [100, 50, 12, 14] + [103, 55, 109, 55, 106, 58, 104, 61, 108, 61] + [0.9]
        detector.model = MagicMock()
        detector.model.detect.return_value = (1, np.array([row], np.float32))
        faces = detector.faces(np.zeros((1080, 1920, 3), np.uint8))
        detector.model.setInputSize.assert_called_once_with((960, 540))
        box = faces[0].box
        self.assertEqual((box.left, box.top, box.right, box.bottom), (200, 100, 224, 128))
        self.assertEqual(box.nose, (212, 116))
        self.assertAlmostEqual(box.score, 0.9, places=5)

    def test_the_checked_in_model_loads_and_finds_nothing_in_a_blank_frame(self):
        detector = YuNetFaceDetector(str(laptop_detector.ZOOM_FINDERS["yunet"]))
        self.assertEqual(detector.detect(np.zeros((1080, 1920, 3), np.uint8)), [])


class SizeAwareGazeTests(unittest.TestCase):
    def test_iris_trust_ramps_between_the_two_widths(self):
        config = GazeConfig(eye_px_min=5, eye_px_full=9)
        self.assertEqual(iris_trust(4, config), 0.0)
        self.assertEqual(iris_trust(7, config), 0.5)
        self.assertEqual(iris_trust(12, config), 1.0)
        self.assertEqual(iris_trust(1, GazeConfig()), 1.0)  # default: always trusted

    def test_a_few_pixel_eye_falls_back_to_head_pose(self):
        # 0.2 of a 1920 frame wide face: eyes ~96 px; 0.01: eyes ~5 px.
        config = GazeConfig(eye_px_min=6, eye_px_full=9)
        big = estimate_gaze(landmarks(eye_yaw=20, size=0.2), pose(), config, W, H)
        tiny = estimate_gaze(landmarks(eye_yaw=20, size=0.01), pose(), config, W, H)
        self.assertEqual(big.method, "eyes")
        self.assertAlmostEqual(big.eye_yaw, 20, delta=0.5)
        self.assertEqual(tiny.method, "head")
        self.assertEqual((tiny.gaze_yaw, tiny.eye_yaw), (0.0, 0.0))

    def test_default_config_keeps_the_iris_on_tiny_faces(self):
        tiny = estimate_gaze(landmarks(eye_yaw=20, size=0.01), pose(), GazeConfig(), W, H)
        self.assertEqual(tiny.method, "eyes")


def gaze(yaw):
    return estimate_gaze(landmarks(eye_yaw=yaw, size=0.2), pose(), GazeConfig())


class SmallFaceTrackerTests(unittest.TestCase):
    def run_track(self, config, width):
        tracker = PeeperTracker(config)
        tracker.update([Detection(0.5, 0.5, width, width, True, gaze(0))], 0.0)
        result = tracker.update([Detection(0.5, 0.5, width, width, True, gaze(20))], 0.1)
        return result.tracks[0].gaze_yaw

    def test_small_faces_smooth_longer_and_big_ones_are_unchanged(self):
        config = TrackerConfig(small_width=0.05, small_tau_s=0.3)
        self.assertLess(self.run_track(config, 0.02), self.run_track(config, 0.2))
        self.assertAlmostEqual(self.run_track(config, 0.2), self.run_track(TrackerConfig(), 0.2))

    def test_small_faces_can_need_a_longer_look(self):
        config = TrackerConfig(min_faces=1, small_width=0.05, small_enter_s=1.0)
        tracker = PeeperTracker(config)
        active_at = None
        for i in range(60):
            now = i / 30
            result = tracker.update([Detection(0.5, 0.5, 0.02, 0.02, True, gaze(0))], now)
            if result.decision.active and active_at is None:
                active_at = now
        self.assertGreaterEqual(active_at, 1.0)


class FlagTests(unittest.TestCase):
    def run_main(self, argv):
        camera = MagicMock()
        camera.isOpened.return_value = True
        camera.read.side_effect = KeyboardInterrupt
        modules = {"cv2": SimpleNamespace(CAP_AVFOUNDATION=1200,
                                          VideoCapture=MagicMock(return_value=camera)),
                   "mediapipe": MagicMock()}
        with patch.dict(sys.modules, modules), \
                patch.object(laptop_detector, "PreviewServer"), \
                patch.object(laptop_detector, "PrivacyAction"), \
                patch.object(laptop_detector, "LaptopDetector") as made, \
                patch.object(laptop_detector, "LandmarkPipeline") as pipeline:
            self.assertEqual(laptop_detector.main(["--no-pi"] + argv), 0)
        return pipeline, made

    def test_default_zooms_far_faces_with_yunet_and_small_face_rules(self):
        pipeline, made = self.run_main([])
        kwargs = pipeline.call_args.kwargs
        self.assertEqual((kwargs["zoom"], kwargs["finder"], kwargs["enhance"]), ("far", "yunet", ()))
        self.assertTrue(pipeline.call_args.args[1].endswith("face_detection_yunet_2023mar.onnx"))
        tracking = made.call_args.kwargs["tracker"].config
        self.assertEqual(tracking.small_tau_s, 0.3)
        gaze_config = made.call_args.kwargs["extract"].keywords["config"]
        self.assertEqual((gaze_config.eye_px_min, gaze_config.eye_px_full), (5.0, 9.0))

    def test_zoom_off_and_blazeface_finder(self):
        pipeline, _ = self.run_main(["--zoom-faces", "off"])
        pipeline.assert_not_called()
        pipeline, _ = self.run_main(["--zoom-finder", "blazeface", "--zoom-enhance", "grey-clahe"])
        kwargs = pipeline.call_args.kwargs
        self.assertEqual((kwargs["finder"], kwargs["enhance"]), ("blazeface", ("grey", "clahe")))
        self.assertTrue(pipeline.call_args.args[1].endswith("face_detector.tflite"))

    def test_no_small_faces_keeps_the_old_configs(self):
        _, made = self.run_main(["--no-small-faces"])
        self.assertEqual(made.call_args.kwargs["tracker"].config.small_width, 0.0)
        gaze_config = made.call_args.kwargs["extract"].keywords["config"]
        self.assertEqual(gaze_config.eye_px_full, 0.0)

    def test_a_bad_enhance_step_is_refused(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            laptop_detector.main(["--no-pi", "--zoom-enhance", "sparkle"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
