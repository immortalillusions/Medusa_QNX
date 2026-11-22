"""Read BGR frames from the QNX Sensor Framework capture library."""

from __future__ import annotations

import ctypes
import time
from pathlib import Path

import cv2
import numpy as np


class _Frame(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
    ]


# These values are the pixel formats exported by qnx/camera_capture.c.
_COLOR_CONVERSIONS = {
    1: (2, cv2.COLOR_YUV2BGR_YUY2),
    2: (2, cv2.COLOR_YUV2BGR_UYVY),
    3: (4, cv2.COLOR_RGBA2BGR),
    4: (4, cv2.COLOR_BGRA2BGR),
}


class QnxCamera:
    def __init__(self, unit: int) -> None:
        if not 0 < unit < 2**31:
            raise ValueError("QNX camera unit must be a positive integer in the C int range")
        library_path = Path(__file__).resolve().parents[1] / "qnx" / "libmedusa_camera.so"
        try:
            self._library = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise RuntimeError(
                f"Could not load {library_path}. Build the QNX camera library on the Pi. {error}"
            ) from error
        self._library.medusa_camera_open.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        self._library.medusa_camera_open.restype = ctypes.c_void_p
        self._library.medusa_camera_read.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Frame)]
        self._library.medusa_camera_read.restype = ctypes.c_int
        self._library.medusa_camera_close.argtypes = [ctypes.c_void_p]
        self._library.medusa_camera_close.restype = None

        error_code = ctypes.c_int()
        self._handle = self._library.medusa_camera_open(unit, ctypes.byref(error_code))
        if not self._handle:
            raise RuntimeError(f"QNX camera unit {unit} failed to open or start (error {error_code.value})")
        self._frame = _Frame()

    def read(self):
        if self._handle is None:
            return False, None
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if self._library.medusa_camera_read(self._handle, ctypes.byref(self._frame)):
                channels, code = _COLOR_CONVERSIONS[self._frame.format]
                shape = (self._frame.height, self._frame.width, channels)
                pixels = np.ctypeslib.as_array(self._frame.data, shape=shape)
                return True, cv2.cvtColor(pixels, code)
            time.sleep(0.01)
        return False, None

    def release(self) -> None:
        if self._handle is not None:
            self._library.medusa_camera_close(self._handle)
            self._handle = None
