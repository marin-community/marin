import sys
from types import SimpleNamespace

from harbor.telemetry.observer import TelemetryRuntime, _bound_text, record_job_terminal


class _Instrument:
    def record(self, *args, **kwargs) -> None:
        pass

    def add(self, *args, **kwargs) -> None:
        pass


class _Telemetry:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def histogram(self, *args, **kwargs) -> _Instrument:
        return _Instrument()

    def counter(self, *args, **kwargs) -> _Instrument:
        return _Instrument()

    def event(self, name: str, body: dict[str, object], **kwargs) -> None:
        self.events.append((name, body))


def test_job_terminal_telemetry_preserves_capability_url(monkeypatch) -> None:
    endpoint = "https://iris.example/proxy/t/scoped-token/eval/v1"
    telemetry = _Telemetry()
    monkeypatch.setitem(sys.modules, "rigging", SimpleNamespace(telemetry=telemetry))

    record_job_terminal(
        TelemetryRuntime(job_uid="job-1", started_monotonic=0.0),
        RuntimeError(f"request to {endpoint} failed"),
    )

    assert telemetry.events[-1][1]["message"] == f"request to {endpoint} failed"


def test_error_message_with_surrogate_is_escaped_not_fatal():
    """Exception messages embedding agent output can carry unpaired
    surrogates; bounding them must never crash event emission."""
    message = "agent exited 1: output \ud842 tail"
    bounded = _bound_text(message)
    assert "\\ud842" in bounded
