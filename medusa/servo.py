"""Map a normalized snooper position to a safe, smooth servo command."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ServoMapper:
    minimum: int = 35
    maximum: int = 145
    home: int = 90
    reversed: bool = False
    smoothing: float = 0.35
    _last_yaw: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.minimum <= self.maximum <= 180:
            raise ValueError("servo limits must be ordered angles from 0 through 180")
        if not self.minimum <= self.home <= self.maximum:
            raise ValueError("servo home must be inside the safe servo range")
        if not 0 < self.smoothing <= 1:
            raise ValueError("servo smoothing must be greater than 0 and at most 1")

    def command(self, active: bool, target_x: float | None) -> tuple[int, bool]:
        if not active or target_x is None:
            desired = self.home
        else:
            position = max(0.0, min(1.0, target_x))
            if self.reversed:
                position = 1.0 - position
            desired = self.minimum + position * (self.maximum - self.minimum)

        previous = self.home if self._last_yaw is None else self._last_yaw
        self._last_yaw = previous + self.smoothing * (desired - previous)
        yaw = round(max(self.minimum, min(self.maximum, self._last_yaw)))
        return yaw, active

