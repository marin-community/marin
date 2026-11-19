"""Config-defined progress display settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressSettings:
    mode: str
    min_interval_seconds: float
    heartbeat_seconds: float
    percent_step: float
    count_step: float

    @classmethod
    def from_config(cls, config) -> "ProgressSettings":
        """Read the display settings for a native progress reporter."""
        return cls(
            mode=str(config.mode),
            min_interval_seconds=float(config.min_interval_seconds),
            heartbeat_seconds=float(config.heartbeat_seconds),
            percent_step=float(config.percent_step),
            count_step=float(config.count_step),
        )
