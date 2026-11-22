import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from medusa import actions, mac_docs


def fake_run(running="", chrome="none", safari="none", fail=None):
    """Stand-in for subprocess.run that answers each script by which one it is."""
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if fail is not None:
            if isinstance(fail, BaseException):
                raise fail
            return subprocess.CompletedProcess(command, 1, "", fail)
        if command[0] == "open":
            return subprocess.CompletedProcess(command, 0, "", "")
        script = command[2]
        if script == mac_docs._RUNNING_SCRIPT:
            out = running
        elif script == mac_docs._CHROME_SCRIPT:
            out = chrome
        elif script == mac_docs._SAFARI_SCRIPT:
            out = safari
        else:
            raise AssertionError(f"unexpected script {script!r}")
        return subprocess.CompletedProcess(command, 0, out + "\n", "")

    return run, calls


def scripts(calls):
    names = {mac_docs._RUNNING_SCRIPT: "running", mac_docs._CHROME_SCRIPT: "chrome",
             mac_docs._SAFARI_SCRIPT: "safari"}
    return [names[c[2]] if c[0] == "osascript" else c for c in calls]


class FocusOrOpenDocsTests(unittest.TestCase):
    def focus(self, run):
        with (
            patch.object(mac_docs.subprocess, "run", side_effect=run),
            patch.object(mac_docs.webbrowser, "open_new_tab") as browser,
        ):
            outcome = mac_docs.focus_or_open_docs()
        return outcome, browser

    def test_chrome_docs_tab_is_focused_without_opening_a_new_doc(self):
        run, calls = fake_run(running="chrome safari", chrome="focused")
        outcome, browser = self.focus(run)
        self.assertEqual(outcome, "focused chrome")
        self.assertEqual(scripts(calls), ["running", "chrome"])
        browser.assert_not_called()

    def test_safari_is_tried_when_chrome_has_no_docs_tab(self):
        run, calls = fake_run(running="chrome safari", safari="focused")
        outcome, browser = self.focus(run)
        self.assertEqual(outcome, "focused safari")
        self.assertEqual(scripts(calls), ["running", "chrome", "safari"])
        browser.assert_not_called()

    def test_browsers_that_are_not_running_are_never_told_anything(self):
        run, calls = fake_run(running="")
        outcome, browser = self.focus(run)
        self.assertEqual(outcome, "opened docs.new")
        self.assertEqual(scripts(calls), ["running", ["open", "https://docs.new"]])
        browser.assert_not_called()

    def test_no_docs_tab_opens_docs_new_in_running_chrome(self):
        run, calls = fake_run(running="chrome safari")
        outcome, browser = self.focus(run)
        self.assertEqual(outcome, "opened docs.new in chrome")
        self.assertEqual(scripts(calls)[-1], ["open", "-a", "Google Chrome", "https://docs.new"])
        browser.assert_not_called()

    def test_apple_events_denied_falls_back_to_webbrowser_and_logs(self):
        denied = "execution error: Not authorized to send Apple events to Google Chrome. (-1743)"
        run, _ = fake_run(fail=denied)
        with self.assertLogs(mac_docs._LOG, "WARNING") as logs:
            outcome, browser = self.focus(run)
        browser.assert_called_once_with("https://docs.new")
        self.assertTrue(outcome.startswith("fallback opened docs.new (not authorized"))
        self.assertEqual(len(logs.output), 1)
        self.assertIn("Automation", logs.output[0])

    def test_timeout_and_missing_osascript_fall_back(self):
        for error, reason in [
            (subprocess.TimeoutExpired("osascript", 3), "timed out"),
            (FileNotFoundError("osascript"), "osascript not found"),
        ]:
            with self.subTest(reason=reason):
                run, _ = fake_run(fail=error)
                with self.assertLogs(mac_docs._LOG, "WARNING"):
                    outcome, browser = self.focus(run)
                self.assertIn(reason, outcome)
                browser.assert_called_once_with("https://docs.new")


class PrivacyActionTests(unittest.TestCase):
    def test_mac_runs_once_per_rising_edge_off_the_caller_thread(self):
        threads = []

        def focus():
            threads.append(threading.current_thread())
            return "focused chrome"

        with (
            patch.object(sys, "platform", "darwin"),
            patch.object(actions.mac_docs, "focus_or_open_docs", side_effect=focus),
            patch.object(actions.webbrowser, "open_new_tab") as browser,
        ):
            action = actions.PrivacyAction()
            first = action.update(True)
            self.assertEqual(first.result(timeout=2), "focused chrome")
            self.assertIsNone(action.update(True))
            action.update(False)
            action.update(True).result(timeout=2)
        self.assertEqual(len(threads), 2)
        self.assertNotIn(threading.current_thread(), threads)
        browser.assert_not_called()

    def test_mac_skips_a_new_edge_while_one_is_in_flight(self):
        release = threading.Event()
        with (
            patch.object(sys, "platform", "darwin"),
            patch.object(actions.mac_docs, "focus_or_open_docs",
                         side_effect=lambda: release.wait(2) and "focused chrome") as focus,
        ):
            action = actions.PrivacyAction()
            pending = action.update(True)
            action.update(False)
            self.assertIsNone(action.update(True))
            release.set()
            pending.result(timeout=2)
        focus.assert_called_once()

    def test_other_platforms_keep_opening_docs_new_directly(self):
        for platform in ("win32", "linux"):
            with (
                self.subTest(platform=platform),
                patch.object(sys, "platform", platform),
                patch.object(actions.mac_docs, "focus_or_open_docs") as focus,
                patch.object(actions.webbrowser, "open_new_tab") as browser,
            ):
                action = actions.PrivacyAction()
                self.assertIsNone(action.update(True))
                action.update(True)
                focus.assert_not_called()
                browser.assert_called_once_with("https://docs.new")

    def test_disabled_does_nothing(self):
        with (
            patch.object(sys, "platform", "darwin"),
            patch.object(actions.mac_docs, "focus_or_open_docs") as focus,
            patch.object(actions.webbrowser, "open_new_tab") as browser,
        ):
            action = actions.PrivacyAction(enabled=False)
            action.update(True)
        focus.assert_not_called()
        browser.assert_not_called()


if __name__ == "__main__":
    unittest.main()
