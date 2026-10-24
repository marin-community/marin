"""No-op Rigging surface for installations without the telemetry extra."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EventBody:
    fields: Mapping[str, str | int | float | bool]


class _Instrument:
    def add(self, value: float = 1.0, *, attributes: Mapping[str, str] | None = None) -> None:
        pass

    def set(self, value: float, *, attributes: Mapping[str, str] | None = None) -> None:
        pass

    def record(self, value: float, *, attributes: Mapping[str, str] | None = None) -> None:
        pass


@dataclass(frozen=True)
class _Status:
    configured: bool = False
    queued_records: int = 0
    lost_records: int = 0


_instrument = _Instrument()
_status = _Status()


def counter(name: str, *, unit: str = "") -> _Instrument:
    return _instrument


def gauge(name: str, *, unit: str = "") -> _Instrument:
    return _instrument


def histogram(name: str, *, unit: str = "") -> _Instrument:
    return _instrument


def configure(*, endpoint: str, service: str, attributes: Mapping[str, str]) -> None:
    pass


def event(name: str, body: EventBody, *, attributes: Mapping[str, str] | None = None) -> None:
    pass


def runtime_status() -> _Status:
    return _status


def shutdown(timeout: float = 5.0) -> None:
    pass
