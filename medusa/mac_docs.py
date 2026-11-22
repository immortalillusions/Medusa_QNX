"""On macOS, bring an open Google Docs tab to the front, or open a blank doc."""

from __future__ import annotations

import logging
import subprocess
import webbrowser


_LOG = logging.getLogger(__name__)
DOCS_URL = "https://docs.new"
TIMEOUT_S = 3.0

# Reads only the `running` property, so it never launches a browser or asks for Automation access.
_RUNNING_SCRIPT = """
set found to ""
try
	if application "Google Chrome" is running then set found to found & "chrome "
end try
try
	if application "Safari" is running then set found to found & "safari"
end try
return found
"""

# Windows are listed front to back, so the first match is in the most recently used window.
_CHROME_SCRIPT = """
if application "Google Chrome" is running then
	tell application "Google Chrome"
		repeat with w in windows
			set tabURLs to URL of tabs of w
			repeat with i from 1 to count tabURLs
				if item i of tabURLs contains "docs.google.com/document" then
					set active tab index of w to i
					set index of w to 1
					activate
					return "focused"
				end if
			end repeat
		end repeat
	end tell
end if
return "none"
"""

_SAFARI_SCRIPT = """
if application "Safari" is running then
	tell application "Safari"
		repeat with w in windows
			try
				set tabURLs to URL of tabs of w
			on error
				set tabURLs to {}
			end try
			repeat with i from 1 to count tabURLs
				set u to item i of tabURLs
				if u is not missing value and u contains "docs.google.com/document" then
					set current tab of w to tab i of w
					set index of w to 1
					activate
					return "focused"
				end if
			end repeat
		end repeat
	end tell
end if
return "none"
"""


class _ScriptError(RuntimeError):
    pass


def _osascript(script: str) -> str:
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True, timeout=TIMEOUT_S, check=False,
    )
    if result.returncode != 0:
        raise _ScriptError(result.stderr.strip() or f"osascript exited {result.returncode}")
    return result.stdout.strip()


def _describe(error: Exception) -> str:
    if isinstance(error, subprocess.TimeoutExpired):
        return f"AppleScript timed out after {TIMEOUT_S:g} s"
    if isinstance(error, FileNotFoundError):
        return "osascript not found"
    if "-1743" in str(error):
        return ("not authorized to send Apple events; allow it in System Settings > "
                "Privacy & Security > Automation")
    return str(error).splitlines()[0] if str(error) else type(error).__name__


def focus_or_open_docs() -> str:
    """Focus a Docs tab in Chrome, then Safari; else open docs.new. Returns the outcome."""
    try:
        running = _osascript(_RUNNING_SCRIPT).split()
        if "chrome" in running and _osascript(_CHROME_SCRIPT) == "focused":
            return "focused chrome"
        if "safari" in running and _osascript(_SAFARI_SCRIPT) == "focused":
            return "focused safari"
        if "chrome" in running:
            command, outcome = ["open", "-a", "Google Chrome", DOCS_URL], "opened docs.new in chrome"
        else:
            command, outcome = ["open", DOCS_URL], "opened docs.new"
        subprocess.run(command, capture_output=True, timeout=TIMEOUT_S, check=True)
        return outcome
    except (OSError, subprocess.SubprocessError, _ScriptError) as error:
        reason = _describe(error)
        _LOG.warning("Docs tab switch failed (%s); opening docs.new instead.", reason)
        webbrowser.open_new_tab(DOCS_URL)
        return f"fallback opened docs.new ({reason})"
