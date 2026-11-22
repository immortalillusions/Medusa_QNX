"""QNX POSIX serial transport for the Arduino safety controller."""

from __future__ import annotations

import os
import termios

from .servo import ServoMapper


class ArduinoController:
    def __init__(self, device: str, mapper: ServoMapper, baud: int = 115200) -> None:
        self.mapper = mapper
        self.sequence = 0
        self.fd = os.open(device, os.O_WRONLY | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            attrs = termios.tcgetattr(self.fd)
            attrs[0] = 0
            attrs[1] = 0
            attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
            attrs[3] = 0
            attrs[4] = getattr(termios, f"B{baud}")
            attrs[5] = getattr(termios, f"B{baud}")
            termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        except Exception:
            os.close(self.fd)
            raise

    def send(self, active: bool, target_x: float | None) -> None:
        yaw, led = self.mapper.command(active, target_x)
        self.sequence += 1
        message = f"M,{yaw},{int(led)},{self.sequence}\n".encode("ascii")
        os.write(self.fd, message)

    def safe_stop(self) -> None:
        self.send(False, None)

    def close(self) -> None:
        os.close(self.fd)

