from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from cloud.iris import export_hf_checkpoint
from skyrl_train.hf_export_schema import HFExportStatus


@dataclass(frozen=True)
class _Request:
    export_path: str = "/tmp/export"
    step: int = 10
    status: HFExportStatus = HFExportStatus.PENDING
    hf_hub_repo_id: str | None = None

    def with_status(self, status: HFExportStatus, **_updates) -> "_Request":
        return replace(self, status=status)


def test_completed_export_is_recorded_without_conversion(monkeypatch):
    written = []
    monkeypatch.setattr(export_hf_checkpoint, "_has_completed_hf_model_export", lambda _path: True)
    monkeypatch.setattr(export_hf_checkpoint, "_write_hf_export_request", written.append)
    monkeypatch.setattr(
        export_hf_checkpoint, "_run_export", lambda *_args: pytest.fail("verified export was converted again")
    )

    export_hf_checkpoint.submit_requested_export(SimpleNamespace(request=_Request(), timeout=60), ["convert"])

    assert [request.status for request in written] == [HFExportStatus.COMPLETE]


def test_conversion_without_a_completed_export_returns_request_to_pending(monkeypatch):
    written = []
    monkeypatch.setattr(export_hf_checkpoint, "_has_completed_hf_model_export", lambda _path: False)
    monkeypatch.setattr(export_hf_checkpoint, "_write_hf_export_request", written.append)
    monkeypatch.setattr(export_hf_checkpoint, "_run_export", lambda *_args: None)

    with pytest.raises(RuntimeError, match="without a completed HF model"):
        export_hf_checkpoint.submit_requested_export(SimpleNamespace(request=_Request(), timeout=60), ["convert"])

    assert [request.status for request in written] == [HFExportStatus.IN_PROGRESS, HFExportStatus.PENDING]


def test_hub_request_runs_publication_even_when_local_export_is_complete(monkeypatch):
    written = []
    conversions = []
    monkeypatch.setattr(export_hf_checkpoint, "_has_completed_hf_model_export", lambda _path: True)
    monkeypatch.setattr(export_hf_checkpoint, "_write_hf_export_request", written.append)
    monkeypatch.setattr(export_hf_checkpoint, "_run_export", lambda *_args: conversions.append(True))

    request = _Request(hf_hub_repo_id="org/model")
    export_hf_checkpoint.submit_requested_export(SimpleNamespace(request=request, timeout=60), ["convert"])

    assert conversions == [True]
    assert [entry.status for entry in written] == [HFExportStatus.IN_PROGRESS, HFExportStatus.COMPLETE]
