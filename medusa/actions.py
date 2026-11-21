from __future__ import annotations

import webbrowser


class PrivacyAction:
    """Opens a blank Google Doc once per snooper event."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._was_active = False

    def update(self, active: bool) -> None:
        if active and not self._was_active and self.enabled:
            webbrowser.open_new_tab("https://docs.new")
        self._was_active = active

