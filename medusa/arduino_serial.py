"""QNX POSIX serial transport for the Arduino safety controller."""

from __future__ import annotations

import os
import termios

from .servo import ServoMapper


def open_serial(device: str, baud: int = 115200, *, read: bool = False) -> int:
    """Open a raw 8N1 non-blocking serial port; write-only unless ``read``.

    Shared by ArduinoController (direct mode, write-only) and the control
    process (medusa/control.py), which also reads the firmware's S, status lines.
    """
    fd = os.open(device, (os.O_RDWR if read else os.O_WRONLY) | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = getattr(termios, f"B{baud}")
        attrs[5] = getattr(termios, f"B{baud}")
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
    except Exception:
        os.close(fd)
        raise
    return fd


class ArduinoController:
    def __init__(
        self,
        device: str,
        mapper: ServoMapper,
        baud: int = 115200,
        tilt_mapper: ServoMapper | None = None,
    ) -> None:
        self.mapper = mapper
        self.tilt_mapper = tilt_mapper or ServoMapper()
        self.sequence = 0
        self.fd = open_serial(device, baud)

    def send(self, active: bool, target_x: float | None, target_y: float | None = None) -> None:
        """Write M,<yaw>,<pitch>,<led>,<sequence>; tilt goes home without a target_y."""
        yaw, led = self.mapper.command(active, target_x)
        pitch, _ = self.tilt_mapper.command(active, target_y)
        self.sequence += 1
        message = f"M,{yaw},{pitch},{int(led)},{self.sequence}\n".encode("ascii")
        os.write(self.fd, message)

    def send_angles(self, active: bool, yaw: float, pitch: float) -> None:
        """Write exact angles (no smoothing), each clamped to its mapper's safe range."""
        yaw = round(max(self.mapper.minimum, min(self.mapper.maximum, yaw)))
        pitch = round(max(self.tilt_mapper.minimum, min(self.tilt_mapper.maximum, pitch)))
        self.sequence += 1
        os.write(self.fd, f"M,{yaw},{pitch},{int(active)},{self.sequence}\n".encode("ascii"))

    def safe_stop(self) -> None:
        self.send(False, None)

    def poll_actual(self):
        """Direct mode has no status feedback (the port is write-only): the tracker uses its model."""
        return None

    def close(self) -> None:
        os.close(self.fd)

