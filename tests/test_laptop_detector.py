import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import laptop_detector
from laptop_detector import LaptopDetector, PiLink, landmarker_faces
from tests.gaze_fixtures import landmarks, pose, result

# Frames at 30 fps (the fake clock starts at 1/30 s) until a face looking from the first
# frame has looked for the 0.5 s --enter-s.
ENTER_FRAMES = 16
AWAY_FRAMES = 25  # > the 0.7 s --exit-s


class Clock:
    def __init__(self, step):
        self.now, self.step = 0.0, step

    def __call__(self):
        self.now += self.step
        return self.now


def face(left, width, *, score=0.9, facing=True, frame_w=1280, frame_h=720, top=100):
    """A BlazeFace-shaped detection: eyes then nose, normalized to the full frame."""
    eye_y = (top + width * 0.4) / frame_h
    eye_a = (left + width * 0.3) / frame_w
    eye_b = (left + width * 0.7) / frame_w
    nose = (left + width * (0.5 if facing else 0.75)) / frame_w
    return SimpleNamespace(
        bounding_box=SimpleNamespace(origin_x=left, origin_y=top, width=width, height=width),
        keypoints=[SimpleNamespace(x=eye_a, y=eye_y), SimpleNamespace(x=eye_b, y=eye_y),
                   SimpleNamespace(x=nose, y=eye_y + 0.05)],
        categories=[SimpleNamespace(score=score)],
    )


USER = face(400, 400)
PEEPER = face(1000, 120)


class FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def sendto(self, data, address):
        self.sent.append((json.loads(data), address))

    def close(self):
        self.closed = True


def make_detector(link=True):
    sock = FakeSocket()
    action = MagicMock()
    preview = MagicMock()
    pi = PiLink("10.0.0.2", 8767, sock=sock) if link else None
    return LaptopDetector(action, pi, preview, clock=Clock(1 / 30)), sock, action, preview


class HeartbeatTests(unittest.TestCase):
    def test_every_frame_sends_one_packet_with_the_documented_shape(self):
        detector, sock, _, _ = make_detector()
        frames = ENTER_FRAMES + 2
        for _ in range(frames):
            detector.step([USER, PEEPER], 1280, 720)
        self.assertEqual(len(sock.sent), frames)
        first, address = sock.sent[0]
        self.assertEqual(address, ("10.0.0.2", 8767))
        self.assertEqual(first, {"type": "medusa-peeper", "active": False,
                                 "target_x": None, "target_y": None, "seq": 1,
                                 "user_x": (400 + 200) / 1280})
        last, _ = sock.sent[-1]
        self.assertEqual(set(last), {"type", "active", "target_x", "target_y", "seq", "user_x",
                                     "peeper_id"})
        self.assertTrue(last["active"])
        self.assertEqual(last["peeper_id"], detector.tracker.peeper_id)  # only sent while active
        self.assertEqual(last["seq"], frames)
        self.assertAlmostEqual(last["target_x"], (1000 + 60) / 1280)
        self.assertAlmostEqual(last["target_y"], (100 + 60) / 720)
        self.assertEqual([packet["seq"] for packet, _ in sock.sent], list(range(1, frames + 1)))

    def test_no_faces_still_sends_inactive_heartbeats(self):
        detector, sock, _, _ = make_detector()
        detector.step([], 1280, 720)
        detector.step([], 1280, 720)
        self.assertEqual([p["active"] for p, _ in sock.sent], [False, False])
        self.assertNotIn("user_x", sock.sent[-1][0])  # optional: only sent with a user face

    def test_user_x_is_the_largest_face_wherever_it_sits(self):
        detector, sock, _, _ = make_detector()
        detector.step([PEEPER, USER], 1280, 720)  # order in the frame does not matter
        self.assertAlmostEqual(sock.sent[-1][0]["user_x"], (400 + 200) / 1280)
        alone, sock, _, _ = make_detector()
        alone.step([PEEPER], 1280, 720)  # alone, the small face is the user
        self.assertAlmostEqual(sock.sent[-1][0]["user_x"], (1000 + 60) / 1280)

    def test_send_errors_do_not_raise(self):
        sock = MagicMock()
        sock.sendto.side_effect = OSError("Network is unreachable")
        link = PiLink("10.0.0.2", 8767, sock=sock)
        with self.assertLogs("laptop_detector", "WARNING") as logs:
            link.send(laptop_detector.SnooperDecision(False))
            link.send(laptop_detector.SnooperDecision(False))
        self.assertEqual(len(logs.output), 1)

    def test_no_pi_sends_nothing(self):
        detector, sock, _, _ = make_detector(link=False)
        for _ in range(10):
            detector.step([USER, PEEPER], 1280, 720)
        detector.shutdown()
        self.assertEqual(sock.sent, [])


class DecisionTests(unittest.TestCase):
    def test_active_after_hold_frames_and_peeper_is_the_smaller_face(self):
        detector, _, _, _ = make_detector()
        results = [detector.step([USER, PEEPER], 1280, 720) for _ in range(ENTER_FRAMES)]
        self.assertEqual([d.active for d, _ in results], [False] * (ENTER_FRAMES - 1) + [True])
        self.assertIs(results[-1][1].raw, PEEPER)
        self.assertIsNone(results[0][1])

    def test_rising_edge_triggers_docs_once(self):
        detector, _, _, _ = make_detector()
        detector.docs_always = True
        detector.action = laptop_detector.PrivacyAction(enabled=True)
        with patch("medusa.actions._switch_to_docs", return_value="focused chrome") as switch, \
                patch("medusa.actions.webbrowser.open_new_tab") as browser:
            for _ in range(20):
                detector.step([USER, PEEPER], 1280, 720)
            if detector.action._pending is not None:
                detector.action._pending.result(timeout=2)
            self.assertEqual(switch.call_count + browser.call_count, 1)
            for _ in range(3):  # 0.1 s away is inside --exit-s: no new trigger
                detector.step([USER], 1280, 720)
            for _ in range(20):
                detector.step([USER, PEEPER], 1280, 720)
            if detector.action._pending is not None:
                detector.action._pending.result(timeout=2)
            self.assertEqual(switch.call_count + browser.call_count, 1)
            for _ in range(AWAY_FRAMES):
                detector.step([USER], 1280, 720)
            for _ in range(ENTER_FRAMES):
                detector.step([USER, PEEPER], 1280, 720)
            if detector.action._pending is not None:
                detector.action._pending.result(timeout=2)
        self.assertEqual(switch.call_count + browser.call_count, 2)

    def test_no_browser_never_switches(self):
        detector, _, _, _ = make_detector()
        detector.action = laptop_detector.PrivacyAction(enabled=False)
        with patch("medusa.actions._switch_to_docs") as switch, \
                patch("medusa.actions.webbrowser.open_new_tab") as browser:
            for _ in range(20):
                detector.step([USER, PEEPER], 1280, 720)
        switch.assert_not_called()
        browser.assert_not_called()

    def test_logs_once_per_state_change(self):
        detector, _, _, preview = make_detector()
        with self.assertLogs("laptop_detector", "INFO") as logs:
            for _ in range(ENTER_FRAMES + 4):
                detector.step([USER, PEEPER], 1280, 720)
            for _ in range(AWAY_FRAMES):
                detector.step([USER], 1280, 720)
        self.assertEqual(len(logs.output), 2)
        self.assertIn("ACTIVE: peeper 2", logs.output[0])
        self.assertIn("inactive: fewer than 2 faces", logs.output[1])
        args, kwargs = preview.update_status.call_args
        self.assertEqual(args, (False, "fewer than 2 faces"))
        self.assertEqual((kwargs["face_count"], kwargs["snooper_count"]), (1, 0))

    def test_preview_status_reports_the_peeper(self):
        detector, _, _, preview = make_detector()
        for _ in range(ENTER_FRAMES):
            detector.step([USER, PEEPER], 1280, 720)
        args, kwargs = preview.update_status.call_args
        self.assertEqual(args, (True, "peeper 2 looking"))
        self.assertEqual((kwargs["face_count"], kwargs["snooper_count"]), (2, 1))
        extra = kwargs["extra"]
        self.assertEqual(extra["peeper_id"], 2)
        self.assertEqual([face["id"] for face in extra["faces"]], [1, 2])
        self.assertEqual([face["method"] for face in extra["faces"]], ["head", "head"])
        self.assertEqual(extra["faces"][1]["looking_s"], 0.5)
        self.assertIsNone(extra["faces"][0]["gaze_yaw"])

    def test_turned_away_face_does_not_count(self):
        detector, _, _, _ = make_detector()
        away = face(1000, 120, facing=False)
        for _ in range(ENTER_FRAMES + 4):
            decision, _ = detector.step([USER, away], 1280, 720)
        self.assertFalse(decision.active)

    def test_shutdown_sends_a_final_inactive_packet(self):
        detector, sock, _, _ = make_detector()
        for _ in range(ENTER_FRAMES):
            detector.step([USER, PEEPER], 1280, 720)
        detector.shutdown()
        self.assertTrue(sock.sent[-2][0]["active"])
        self.assertEqual(sock.sent[-1][0], {"type": "medusa-peeper", "active": False,
                                            "target_x": None, "target_y": None,
                                            "seq": ENTER_FRAMES + 1})


# Landmarker scene: the big face is the laptop user looking straight in; the small face
# behind has its head turned 40 degrees away but its eyes back on the screen.
GAZE_USER = (landmarks(cx=0.4, cy=0.55, size=0.3), pose(0, -10))
SIDE_EYE = (landmarks(-30, 0, cx=0.8, cy=0.3, size=0.1), pose(40, 0))
HEAD_AWAY = (landmarks(0, 0, cx=0.8, cy=0.3, size=0.1), pose(40, 0))


def gaze_detector(**kwargs):
    action, preview = MagicMock(), MagicMock()
    sock = FakeSocket()
    kwargs.setdefault("clock", Clock(1 / 30))
    detector = LaptopDetector(action, PiLink("10.0.0.2", 8767, sock=sock), preview,
                              extract=landmarker_faces, **kwargs)
    return detector, sock, preview


class GazeDetectorTests(unittest.TestCase):
    def test_side_eyeing_second_face_becomes_the_peeper_after_the_dwell(self):
        detector, sock, _ = gaze_detector()
        frame = result(GAZE_USER, SIDE_EYE)
        steps = [detector.step(frame, 1280, 720) for _ in range(ENTER_FRAMES)]
        self.assertEqual([d.active for d, _ in steps], [False] * (ENTER_FRAMES - 1) + [True])
        decision, peeper = steps[-1]
        self.assertIs(peeper.raw, SIDE_EYE[0])
        self.assertEqual(peeper.gaze.method, "eyes")
        self.assertAlmostEqual(peeper.gaze.gaze_yaw, 10, places=6)
        self.assertAlmostEqual(decision.target_x, 0.8)
        self.assertAlmostEqual(decision.target_y, 0.3)
        self.assertTrue(sock.sent[-1][0]["active"])

    def test_head_turned_away_with_eyes_straight_never_triggers(self):
        detector, _, _ = gaze_detector()
        for _ in range(ENTER_FRAMES + 5):
            decision, peeper = detector.step(result(GAZE_USER, HEAD_AWAY), 1280, 720)
        self.assertFalse(decision.active)
        self.assertIsNone(peeper)
        self.assertEqual([face.observation.screen_facing for face in detector.faces], [True, False])

    def test_a_glance_shorter_than_the_dwell_does_not_trigger(self):
        detector, _, _ = gaze_detector()
        actives = []
        for _ in range(3):
            for _ in range(ENTER_FRAMES - 2):  # 0.47 s
                actives.append(detector.step(result(GAZE_USER, SIDE_EYE), 1280, 720)[0].active)
            for _ in range(AWAY_FRAMES):
                actives.append(detector.step(result(GAZE_USER, HEAD_AWAY), 1280, 720)[0].active)
        self.assertFalse(any(actives))

    def test_one_away_frame_does_not_reset_the_dwell(self):
        detector, _, _ = gaze_detector()
        for _ in range(ENTER_FRAMES - 6):
            detector.step(result(GAZE_USER, SIDE_EYE), 1280, 720)
        detector.step(result(GAZE_USER, HEAD_AWAY), 1280, 720)
        steps = [detector.step(result(GAZE_USER, SIDE_EYE), 1280, 720)[0].active
                 for _ in range(6)]
        self.assertEqual(steps, [False] * 4 + [True] * 2)

    def test_status_json_lists_every_face_and_the_fps(self):
        detector, _, preview = gaze_detector(clock=Clock(1 / 30))
        for _ in range(ENTER_FRAMES):
            detector.step(result(GAZE_USER, HEAD_AWAY), 1280, 720)
        _, kwargs = preview.update_status.call_args
        extra = kwargs["extra"]
        self.assertEqual(kwargs["face_count"], 2)
        self.assertEqual(extra["fps"], 30.0)
        self.assertEqual(extra["faces"], [
            {"id": 1, "x": 0.4, "y": 0.55, "looking": True, "looking_s": 0.5, "gaze_yaw": 0.0,
             "gaze_pitch": -10.0, "method": "eyes"},
            {"id": 2, "x": 0.8, "y": 0.3, "looking": False, "looking_s": 0.0, "gaze_yaw": 40.0,
             "gaze_pitch": 0.0, "method": "eyes"},
        ])
        self.assertIsNone(extra["peeper_id"])
        json.dumps(extra)  # what /status.json serves must be plain JSON

    def test_fps_is_none_until_two_frames(self):
        detector, _, preview = gaze_detector(clock=Clock(0.05))
        detector.step(result(), 1280, 720)
        self.assertIsNone(preview.update_status.call_args[1]["extra"]["fps"])
        detector.step(result(), 1280, 720)
        self.assertEqual(preview.update_status.call_args[1]["extra"]["fps"], 20.0)

    def test_face_without_a_pose_matrix_is_skipped(self):
        frame = SimpleNamespace(face_landmarks=[GAZE_USER[0], SIDE_EYE[0]],
                                facial_transformation_matrixes=[GAZE_USER[1]])
        self.assertEqual(len(landmarker_faces(frame, 1280, 720)), 1)

    def test_box_and_nose_are_in_pixels(self):
        face = landmarker_faces(result(GAZE_USER), 1000, 500)[0]
        self.assertEqual(face.box, (250, 185, 300, 180))
        self.assertEqual(face.nose, (400, round((0.55 + 0.3 * 0.05) * 500)))
        self.assertAlmostEqual(face.observation.relative_size, 0.3)
        self.assertEqual(face.observation.confidence, 1.0)

    def test_debug_gaze_logs_each_face_once_per_second(self):
        detector, _, _ = gaze_detector(debug_gaze=True, clock=Clock(0.1))
        with self.assertLogs("laptop_detector", "INFO") as logs:
            for _ in range(25):  # 0.1 .. 2.5 s
                detector.step(result(GAZE_USER, SIDE_EYE), 1280, 720)
        lines = [line for line in logs.output if "gaze[" in line]
        self.assertEqual(len(lines), 6)  # t = 0.1, 1.1, 2.1, two faces each
        self.assertIn("gaze[id 2 other]", lines[1])
        self.assertIn("raw +10/+0 smooth +10/+0", lines[1])
        self.assertIn("eyes LOOKING", lines[1])
        self.assertIn("gaze[id 1 user]", lines[0])

    def test_debug_gaze_off_logs_no_gaze(self):
        detector, _, _ = gaze_detector(clock=Clock(0.5))
        with self.assertLogs("laptop_detector", "INFO") as logs:
            for _ in range(ENTER_FRAMES):
                detector.step(result(GAZE_USER, SIDE_EYE), 1280, 720)
        self.assertFalse([line for line in logs.output if "gaze[" in line])


class MainTests(unittest.TestCase):
    def fake_modules(self, opened):
        camera = MagicMock()
        camera.isOpened.return_value = opened
        cv2 = SimpleNamespace(CAP_AVFOUNDATION=1200, VideoCapture=MagicMock(return_value=camera))
        return {"cv2": cv2, "mediapipe": MagicMock()}, camera

    def test_camera_open_failure_exits_2_with_permission_hint(self):
        modules, camera = self.fake_modules(opened=False)
        stderr = io.StringIO()
        with patch.dict(sys.modules, modules), \
                patch.object(laptop_detector, "PiLink") as link, \
                patch.object(laptop_detector, "PreviewServer") as preview, \
                redirect_stderr(stderr):
            self.assertEqual(laptop_detector.main(["--pi-host", "10.0.0.2"]), 2)
        modules["cv2"].VideoCapture.assert_called_once_with(0, 1200)
        camera.release.assert_called_once()
        link.assert_not_called()
        preview.assert_not_called()
        self.assertIn("Privacy & Security > Camera", stderr.getvalue())
        self.assertIn("Terminal", stderr.getvalue())

    def test_pi_host_required_unless_no_pi(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            laptop_detector.main([])
        self.assertEqual(raised.exception.code, 2)

    def test_ctrl_c_sends_final_packet_and_cleans_up(self):
        modules, camera = self.fake_modules(opened=True)
        camera.read.side_effect = KeyboardInterrupt
        sock = FakeSocket()
        with patch.dict(sys.modules, modules), \
                patch.object(laptop_detector, "PiLink", lambda host, port: PiLink(host, port, sock=sock)), \
                patch.object(laptop_detector, "PreviewServer") as preview, \
                patch.object(laptop_detector, "PrivacyAction"):
            self.assertEqual(laptop_detector.main(["--pi-host", "10.0.0.2", "--camera", "1"]), 0)
        modules["cv2"].VideoCapture.assert_called_once_with(1, 1200)
        self.assertEqual(sock.sent, [({"type": "medusa-peeper", "active": False, "target_x": None,
                                       "target_y": None, "seq": 1}, ("10.0.0.2", 8767))])
        self.assertTrue(sock.closed)
        camera.release.assert_called_once()
        preview.return_value.stop.assert_called_once()

    def test_no_pi_opens_no_socket(self):
        modules, camera = self.fake_modules(opened=True)
        camera.read.side_effect = KeyboardInterrupt
        with patch.dict(sys.modules, modules), \
                patch.object(laptop_detector, "PiLink") as make_link, \
                patch.object(laptop_detector, "PreviewServer"), \
                patch.object(laptop_detector, "PrivacyAction"):
            self.assertEqual(laptop_detector.main(["--no-pi"]), 0)
        make_link.assert_not_called()


    def run_two_frames(self, argv):
        modules, camera = self.fake_modules(opened=True)
        frame = MagicMock()
        frame.shape = (720, 1280, 3)
        camera.read.side_effect = [(True, frame), (True, frame), KeyboardInterrupt]
        modules["cv2"].cvtColor = MagicMock(return_value="rgb")
        modules["cv2"].COLOR_BGR2RGB = 4
        modules["mediapipe"].tasks.vision.FaceLandmarker.create_from_options.return_value \
            .__enter__.return_value.detect_for_video.return_value = result()
        with patch.dict(sys.modules, modules), \
                patch.object(laptop_detector, "PreviewServer"), \
                patch.object(laptop_detector, "PrivacyAction"), \
                patch.object(laptop_detector, "LaptopDetector", wraps=LaptopDetector) as made, \
                patch.object(laptop_detector, "detect_faces", return_value=[]):
            self.assertEqual(laptop_detector.main(["--no-pi"] + argv), 0)
        self.tasks = modules["mediapipe"].tasks
        return modules["mediapipe"].tasks.vision, made

    def test_landmarker_is_the_default_in_video_mode_with_matrices(self):
        vision, made = self.run_two_frames([])
        options = vision.FaceLandmarkerOptions.call_args.kwargs
        self.assertEqual(options["num_faces"], 4)
        self.assertTrue(options["output_facial_transformation_matrixes"])
        self.assertIs(options["running_mode"], vision.RunningMode.VIDEO)
        vision.FaceDetector.create_from_options.assert_not_called()
        model = self.tasks.BaseOptions.call_args.kwargs["model_asset_path"]
        self.assertTrue(model.endswith("qnx/face_landmarker.task"))
        landmarker = vision.FaceLandmarker.create_from_options.return_value.__enter__.return_value
        stamps = [call.args[1] for call in landmarker.detect_for_video.call_args_list]
        self.assertEqual(len(stamps), 2)
        self.assertLess(stamps[0], stamps[1])
        config = made.call_args.kwargs["tracker"].config
        self.assertEqual((config.enter_s, config.exit_s, config.min_faces), (0.5, 0.7, 2))
        self.assertEqual((config.yaw_max, config.yaw_margin, config.pitch_margin), (25, 7, 7))

    def test_gaze_flags_reach_the_config(self):
        _, made = self.run_two_frames(["--gaze-yaw-max", "30", "--gaze-pitch-range", "-40", "5",
                                       "--eye-gain", "50", "--enter-s", "1", "--exit-s", "2",
                                       "--gaze-yaw-margin", "5", "--gaze-pitch-margin", "4",
                                       "--min-faces", "1", "--debug-gaze"])
        kwargs = made.call_args.kwargs
        config = kwargs["extract"].keywords["config"]
        self.assertEqual((config.yaw_max_deg, config.pitch_min_deg, config.pitch_max_deg,
                          config.eye_gain_deg), (30, -40, 5, 50))
        tracking = kwargs["tracker"].config
        self.assertEqual((tracking.yaw_max, tracking.pitch_min, tracking.pitch_max,
                          tracking.yaw_margin, tracking.pitch_margin), (30, -40, 5, 5, 4))
        self.assertEqual((tracking.enter_s, tracking.exit_s, tracking.min_faces), (1, 2, 1))
        self.assertTrue(kwargs["debug_gaze"])

    def test_blazeface_flag_keeps_the_old_path(self):
        vision, made = self.run_two_frames(["--detector", "blazeface"])
        vision.FaceLandmarker.create_from_options.assert_not_called()
        vision.FaceDetector.create_from_options.assert_called_once()
        model = self.tasks.BaseOptions.call_args.kwargs["model_asset_path"]
        self.assertTrue(model.endswith("blaze_face_short_range.tflite"))
        self.assertIs(made.call_args.kwargs["extract"], laptop_detector.face_observations)

    def test_negative_dwell_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            laptop_detector.main(["--no-pi", "--enter-s", "-1"])
        self.assertEqual(raised.exception.code, 2)

    def test_inverted_pitch_range_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            laptop_detector.main(["--no-pi", "--gaze-pitch-range", "10", "-35"])
        self.assertEqual(raised.exception.code, 2)

    def run_until_ctrl_c(self, argv):
        modules, camera = self.fake_modules(opened=True)
        camera.read.side_effect = KeyboardInterrupt
        with patch.dict(sys.modules, modules), \
                patch.object(laptop_detector, "PiLink"), \
                patch.object(laptop_detector, "PreviewServer"), \
                patch.object(laptop_detector, "PrivacyAction"), \
                patch.object(laptop_detector, "TurretPoller") as poller, \
                patch.object(laptop_detector, "LaptopDetector", wraps=LaptopDetector) as made:
            self.assertEqual(laptop_detector.main(argv), 0)
        return poller, made

    def test_turret_is_polled_on_the_pi_by_default_and_gates_docs(self):
        poller, made = self.run_until_ctrl_c(["--pi-host", "10.0.0.2"])
        poller.assert_called_once_with("http://10.0.0.2:8082/control.json")
        poller.return_value.start.assert_called_once()
        poller.return_value.stop.assert_called_once()
        self.assertIs(made.call_args.kwargs["turret"], poller.return_value)
        self.assertFalse(made.call_args.kwargs["docs_always"])

    def test_control_url_and_docs_always_flags(self):
        poller, made = self.run_until_ctrl_c(["--no-pi", "--docs-always",
                                              "--control-url", "http://pi.local:9/c.json"])
        poller.assert_called_once_with("http://pi.local:9/c.json")
        self.assertTrue(made.call_args.kwargs["docs_always"])

    def test_no_pi_without_control_url_polls_nothing(self):
        poller, made = self.run_until_ctrl_c(["--no-pi"])
        poller.assert_not_called()
        self.assertIsNone(made.call_args.kwargs["turret"])


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def control(link_ok=True, armed=True, **gate):
    return {"state": "SAFE", "arduino": {"link_ok": link_ok, "armed": armed, "pan": 90, **gate}}


class TurretPollerTests(unittest.TestCase):
    def poller(self, *answers):
        """answers: dicts (served as JSON), bytes (served raw) or exceptions (raised)."""
        self.now = 0.0
        self.calls = []
        queue = list(answers)

        def opener(url, timeout):
            self.calls.append((url, timeout))
            answer = queue.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return FakeResponse(answer if isinstance(answer, bytes) else json.dumps(answer).encode())

        return laptop_detector.TurretPoller("http://pi:8082/control.json", opener=opener,
                                            clock=lambda: self.now)

    def test_armed_with_gate_closed_is_on(self):
        poller = self.poller(control(gate_closed=True))
        self.assertTrue(poller.poll_once())
        self.assertEqual(poller.status(), (True, "armed, gate closed"))
        self.assertEqual(self.calls, [("http://pi:8082/control.json", 1.0)])

    def test_gate_open_disarmed_or_link_down_is_off(self):
        poller = self.poller(control(gate_closed=False), control(armed=False, gate_closed=True),
                             control(link_ok=False, gate_closed=True))
        for reason in ("gate open", "disarmed", "Arduino link down"):
            self.assertFalse(poller.poll_once())
            self.assertEqual(poller.status(), (False, reason))

    def test_missing_or_null_gate_counts_as_closed_only_when_armed(self):
        poller = self.poller(control(), control(gate_closed=None), control(armed=False),
                             control(armed=None))
        self.assertEqual([poller.poll_once() for _ in range(4)], [True, True, False, False])

    def test_unreachable_or_malformed_turns_it_off_at_once(self):
        import urllib.error
        poller = self.poller(control(gate_closed=True), urllib.error.URLError("timed out"),
                             control(gate_closed=True), b"<html>", control(True, True)["arduino"],
                             control(gate_closed=True), OSError("refused"))
        results = []
        for _ in range(7):
            poller.poll_once()
            results.append(poller.status()[0])
        self.assertEqual(results, [True, False, True, False, False, True, False])
        self.assertIn("refused", poller.status()[1])

    def test_a_good_read_older_than_two_seconds_is_off(self):
        poller = self.poller(control(gate_closed=True))
        poller.poll_once()
        self.now = 2.0
        self.assertTrue(poller.status()[0])
        self.now = 2.01
        self.assertEqual(poller.status(), (False, "control.json stale"))

    def test_off_before_the_first_read(self):
        self.assertEqual(self.poller().status(), (False, "no control.json read yet"))

    def test_daemon_thread_polls_until_stopped(self):
        import threading
        polled = threading.Event()
        answers = [control(gate_closed=True)] * 1000
        poller = self.poller(*answers)
        poller.interval_s = 0.001
        original = poller.poll_once

        def poll():
            result = original()
            if len(self.calls) >= 3:
                polled.set()
            return result

        poller.poll_once = poll
        poller.start()
        self.assertTrue(poller._thread.daemon)
        self.assertTrue(polled.wait(2))
        poller.stop()
        poller._thread.join(2)
        self.assertFalse(poller._thread.is_alive())
        self.assertTrue(poller.status()[0])


class FakeTurret:
    def __init__(self, on=False):
        self.on = on

    def status(self):
        return self.on, "armed, gate closed" if self.on else "disarmed"


class DocsGateTests(unittest.TestCase):
    def setUp(self):
        self.turret = FakeTurret()
        self.detector, _, _, self.preview = make_detector()
        self.detector.turret = self.turret
        self.detector.action = laptop_detector.PrivacyAction(enabled=True)
        switch = patch("medusa.actions._switch_to_docs", return_value="focused chrome")
        browser = patch("medusa.actions.webbrowser.open_new_tab")
        self.switch, self.browser = switch.start(), browser.start()
        self.addCleanup(switch.stop)
        self.addCleanup(browser.stop)

    def frames(self, n, peeper=True):
        for _ in range(n):
            self.detector.step([USER, PEEPER] if peeper else [USER], 1280, 720)
        pending = getattr(self.detector.action, "_pending", None)
        if pending is not None:
            pending.result(timeout=2)
        return self.switch.call_count + self.browser.call_count

    def test_turret_on_then_peeper_switches_once(self):
        self.turret.on = True
        self.assertEqual(self.frames(ENTER_FRAMES - 1), 0)
        self.assertEqual(self.frames(1), 1)
        self.assertEqual(self.frames(10), 1)

    def test_peeper_while_off_does_not_switch_until_the_turret_turns_on(self):
        self.assertEqual(self.frames(ENTER_FRAMES + 10), 0)
        self.assertTrue(self.detector.active)
        self.turret.on = True
        self.assertEqual(self.frames(1), 1)
        self.assertEqual(self.frames(10), 1)

    def test_turret_flicker_mid_encounter_does_not_switch_again(self):
        self.turret.on = True
        self.assertEqual(self.frames(ENTER_FRAMES), 1)
        self.turret.on = False
        self.frames(3)
        self.turret.on = True
        self.assertEqual(self.frames(10), 1)

    def test_turret_on_after_the_peeper_left_waits_for_the_next_encounter(self):
        self.assertEqual(self.frames(ENTER_FRAMES), 0)
        self.assertEqual(self.frames(AWAY_FRAMES, peeper=False), 0)
        self.assertFalse(self.detector.active)
        self.turret.on = True
        self.assertEqual(self.frames(5, peeper=False), 0)
        self.assertEqual(self.frames(ENTER_FRAMES), 1)
        self.assertEqual(self.frames(AWAY_FRAMES, peeper=False), 1)
        self.assertEqual(self.frames(ENTER_FRAMES), 2)  # a new encounter switches again

    def test_no_turret_poll_means_off(self):
        self.detector.turret = None
        self.assertEqual(self.frames(ENTER_FRAMES + 5), 0)

    def test_docs_always_bypasses_the_gate(self):
        self.detector.docs_always = True
        self.assertEqual(self.frames(ENTER_FRAMES), 1)
        self.assertEqual(self.preview.update_status.call_args[1]["extra"]["docs_enabled"], True)
        self.assertEqual(self.preview.update_status.call_args[1]["extra"]["turret_on"], False)

    def test_no_browser_wins_over_docs_always_and_turret_on(self):
        self.detector.action = laptop_detector.PrivacyAction(enabled=False)
        self.detector.docs_always = True
        self.turret.on = True
        self.assertEqual(self.frames(ENTER_FRAMES + 5), 0)
        self.assertFalse(self.preview.update_status.call_args[1]["extra"]["docs_enabled"])

    def test_detection_and_heartbeats_run_while_the_turret_is_off(self):
        detector, sock, _, _ = make_detector()
        detector.turret = self.turret
        for _ in range(ENTER_FRAMES):
            detector.step([USER, PEEPER], 1280, 720)
        self.assertEqual(len(sock.sent), ENTER_FRAMES)
        self.assertTrue(sock.sent[-1][0]["active"])

    def test_status_reports_turret_and_docs_and_logs_each_change(self):
        with self.assertLogs("laptop_detector", "INFO") as logs:
            self.frames(2, peeper=False)
            extra = self.preview.update_status.call_args[1]["extra"]
            self.assertEqual((extra["turret_on"], extra["docs_enabled"]), (False, False))
            self.turret.on = True
            self.frames(3, peeper=False)
            extra = self.preview.update_status.call_args[1]["extra"]
            self.assertEqual((extra["turret_on"], extra["docs_enabled"]), (True, True))
            self.turret.on = False
            self.frames(3, peeper=False)
        lines = [line for line in logs.output if "turret" in line]
        self.assertEqual(len(lines), 2)
        self.assertIn("turret ON (armed, gate closed): docs switching enabled", lines[0])
        self.assertIn("turret OFF (disarmed): docs switching paused", lines[1])
        json.dumps(extra)


if __name__ == "__main__":
    unittest.main()
