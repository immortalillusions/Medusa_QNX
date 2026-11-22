from __future__ import annotations

import logging
import sys
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor

from . import mac_docs


_LOG = logging.getLogger(__name__)


class PrivacyAction:
    """Shows a Google Doc once per snooper event.

    On macOS it switches to an open Docs tab (or opens docs.new) on a single background
    thread, so a slow AppleScript never blocks the listener. Elsewhere it opens docs.new.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._was_active = False
        self._pending: Future[str] | None = None
        self._worker: ThreadPoolExecutor | None = None
        if enabled and sys.platform == "darwin":
            self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="medusa-docs")

    def update(self, active: bool) -> Future[str] | None:
        rising = active and not self._was_active and self.enabled
        self._was_active = active
        if not rising:
            return None
        if self._worker is None:
            webbrowser.open_new_tab(mac_docs.DOCS_URL)
            return None
        if self._pending is not None and not self._pending.done():
            _LOG.info("Docs switch skipped because the previous one is still running.")
            return None
        self._pending = self._worker.submit(_switch_to_docs)
        return self._pending


def _switch_to_docs() -> str:
    try:
        outcome = mac_docs.focus_or_open_docs()
    except Exception as error:  # keep the worker alive whatever happens
        outcome = f"failed: {error}"
    _LOG.info("Docs: %s", outcome)
    return outcome
