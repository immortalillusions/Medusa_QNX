"""Activate a pinned Windows app once per confirmed snooper encounter."""

from __future__ import annotations

import ctypes
import logging
import sys
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum

from .decision import SCENE_CLEAR_FRAMES


_LOG = logging.getLogger(__name__)
TASKBAR_SLOTS = range(1, 11)

_INPUT_KEYBOARD = 1
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002
_VK_LWIN = 0x5B
_VK_LCONTROL = 0xA2
_VK_SHIFT = 0x10
_VK_MENU = 0x12  # Alt


class _KeyboardInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_uint16),
        ("wScan", ctypes.c_uint16),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_int32),
        ("dy", ctypes.c_int32),
        ("mouseData", ctypes.c_uint32),
        ("dwFlags", ctypes.c_uint32),
        ("time", ctypes.c_uint32),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_uint32),
        ("wParamL", ctypes.c_uint16),
        ("wParamH", ctypes.c_uint16),
    ]


class _InputData(ctypes.Union):
    # The full union gives INPUT the size and alignment SendInput expects.
    _fields_ = [("ki", _KeyboardInput), ("mi", _MouseInput), ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("data", _InputData)]


def _send_keyboard_events(send_input, events: list[tuple[int, int]]) -> int:
    inputs = (_Input * len(events))(
        *(_Input(_INPUT_KEYBOARD, _InputData(ki=_KeyboardInput(key, 0, flags, 0, 0)))
          for key, flags in events)
    )
    return send_input(len(inputs), inputs, ctypes.sizeof(_Input))


def send_taskbar_shortcut(slot: int) -> None:
    """Send Win+Ctrl+number to the Windows shell."""
    if slot not in TASKBAR_SLOTS:
        raise ValueError("Taskbar slot must be from 1 to 10.")
    if sys.platform != "win32":
        raise RuntimeError("Desktop switching requires Windows.")

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    send_input = user32.SendInput
    send_input.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Input), ctypes.c_int]
    send_input.restype = ctypes.c_uint32
    key_state = user32.GetAsyncKeyState
    key_state.argtypes = [ctypes.c_int]
    key_state.restype = ctypes.c_int16

    digit = ord("0") + slot % 10
    if any(key_state(key) & 0x8000 for key in (_VK_SHIFT, _VK_MENU, digit)):
        raise RuntimeError("Shift, Alt, or the slot key is held down. Release it before testing.")

    # Preserve modifiers the player is already holding and release our own in reverse order.
    presses = [
        (key, flags)
        for key, flags in ((_VK_LWIN, _KEYEVENTF_EXTENDEDKEY), (_VK_LCONTROL, 0))
        if not key_state(key) & 0x8000
    ]
    presses.append((digit, 0))
    releases = [(key, flags | _KEYEVENTF_KEYUP) for key, flags in reversed(presses)]
    events = presses + releases
    ctypes.set_last_error(0)
    sent = _send_keyboard_events(send_input, events)
    if sent == len(events):
        return

    error = ctypes.get_last_error()
    # Clean up synthetic keys left held if Windows accepted only part of the shortcut.
    held = {}
    for key, flags in events[:sent]:
        if flags & _KEYEVENTF_KEYUP:
            held.pop(key)
        else:
            held[key] = flags
    cleanup = [(key, flags | _KEYEVENTF_KEYUP) for key, flags in reversed(held.items())]
    if cleanup and _send_keyboard_events(send_input, cleanup) != len(cleanup):
        _LOG.warning("Windows did not accept every synthetic key release.")
    raise RuntimeError(f"SendInput accepted {sent}/{len(events)} key events (Win32 error {error}).")


class SnooperSignal(Enum):
    CONFIRMED = "confirmed"
    CLEAR = "clear"
    UNCONFIRMED = "unconfirmed"


class DesktopSwitchTrigger:
    """Accept confirmation and clear signals from the detector or a test."""

    def __init__(
        self, slot: int = 1, *, clear_frames: int = SCENE_CLEAR_FRAMES,
        switch_action: Callable[[int], None] = send_taskbar_shortcut,
    ) -> None:
        if slot not in TASKBAR_SLOTS:
            raise ValueError("Taskbar slot must be from 1 to 10.")
        if clear_frames < 1:
            raise ValueError("Clear frames must be positive.")
        self._slot = slot
        self._clear_frames = clear_frames
        self._switch_action = switch_action
        self._encounter_handled = False
        self._clear_streak = 0
        self._pending: Future[bool] | None = None
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="medusa-desktop")

    def update(self, signal: SnooperSignal, reason: str) -> Future[bool] | None:
        if signal is SnooperSignal.CLEAR:
            if self._encounter_handled:
                self._clear_streak += 1
                if self._clear_streak == self._clear_frames:
                    self._encounter_handled = False
                    self._clear_streak = 0
                    _LOG.info("Desktop switching re-armed after the scene cleared.")
            _LOG.debug("Desktop switch skipped because the scene is clear (%s).", reason)
            return None

        self._clear_streak = 0
        if signal is not SnooperSignal.CONFIRMED:
            _LOG.debug("Desktop switch skipped while detection is unconfirmed (%s).", reason)
            return None
        if self._encounter_handled:
            _LOG.debug("Desktop switch skipped because this encounter was already handled.")
            return None

        # Latch before dispatch so a slow or failed shortcut gets one attempt per encounter.
        self._encounter_handled = True
        if self._pending is not None and not self._pending.done():
            _LOG.info("Desktop switch skipped because the previous shortcut is still pending.")
            return None
        try:
            self._pending = self._worker.submit(self._attempt_switch, reason)
        except RuntimeError as error:
            _LOG.error("Desktop switch failed to start: %s", error)
            return None
        return self._pending

    def _attempt_switch(self, reason: str) -> bool:
        _LOG.info("Attempting desktop switch to taskbar slot %d because %s.", self._slot, reason)
        try:
            self._switch_action(self._slot)
        except Exception as error:
            _LOG.error("Desktop switch to taskbar slot %d failed: %s", self._slot, error)
            return False
        _LOG.info("Win+Ctrl+%d sent. Foreground focus was not verified.", self._slot % 10)
        return True

    def close(self) -> None:
        self._worker.shutdown()
