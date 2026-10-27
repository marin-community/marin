"""pending_export_steps: which committed checkpoints still owe an HF conversion."""

from types import SimpleNamespace

from cloud.iris import export_hf_checkpoint
from skyrl_train.hf_export_schema import HFExportStatus

ROOT = "s3://bucket/run/2026.10.27.3"


class _DetailListdirStorage:
    """The guarded remote-I/O filesystem: directory entries arrive as detail dicts.

    Regression: pending_export_steps sorted() raw listdir entries, which under
    this filesystem are dicts and raised TypeError — after the model had
    finished training, at the run's export phase.
    """

    def __init__(self, directories: list[str]):
        self.directories = directories

    def exists(self, path: str) -> bool:
        return path == ROOT

    def ls(self, path: str, detail: bool) -> list[dict]:
        assert detail is True
        return [{"name": f"{ROOT}/{name}/"} for name in self.directories]


def _config():
    return SimpleNamespace(artifacts=SimpleNamespace(checkpoint_root=ROOT))


def _patch(monkeypatch, storage, requests):
    monkeypatch.setattr(export_hf_checkpoint, "fs_and_path", lambda _uri: (storage, ROOT))
    monkeypatch.setattr(export_hf_checkpoint, "_read_hf_export_request", requests.get)


def test_pending_steps_read_detail_dicts_and_sort_numerically(monkeypatch):
    storage = _DetailListdirStorage(["logs", "global_step_30", "global_step_10", "global_step_50"])
    requests = {
        f"{ROOT}/global_step_{step}": SimpleNamespace(status=HFExportStatus.PENDING) for step in (10, 30, 50)
    }
    requests[f"{ROOT}/logs"] = None
    _patch(monkeypatch, storage, requests)
    assert export_hf_checkpoint.pending_export_steps(_config()) == [
        (10, f"{ROOT}/global_step_10"),
        (30, f"{ROOT}/global_step_30"),
        (50, f"{ROOT}/global_step_50"),
    ]


def test_completed_exports_and_missing_roots_are_skipped(monkeypatch):
    storage = _DetailListdirStorage(["global_step_10", "global_step_30"])
    requests = {
        f"{ROOT}/global_step_10": SimpleNamespace(status=HFExportStatus.COMPLETE),
        f"{ROOT}/global_step_30": SimpleNamespace(status=HFExportStatus.PENDING),
    }
    _patch(monkeypatch, storage, requests)
    assert export_hf_checkpoint.pending_export_steps(_config()) == [(30, f"{ROOT}/global_step_30")]


def test_a_root_without_pending_requests_is_an_error(monkeypatch):
    storage = _DetailListdirStorage(["global_step_10"])
    requests = {f"{ROOT}/global_step_10": SimpleNamespace(status=HFExportStatus.COMPLETE)}
    _patch(monkeypatch, storage, requests)
    try:
        export_hf_checkpoint.pending_export_steps(_config())
    except ValueError as error:
        assert "no exportable checkpoint" in str(error)
    else:
        raise AssertionError("expected ValueError for a root with nothing pending")


def test_a_missing_checkpoint_root_is_an_error(monkeypatch):
    storage = _DetailListdirStorage([])

    def exists(_path):
        return False

    storage.exists = exists
    _patch(monkeypatch, storage, {})
    try:
        export_hf_checkpoint.pending_export_steps(_config())
    except ValueError as error:
        assert "did not commit" in str(error)
    else:
        raise AssertionError("expected ValueError for a missing checkpoint root")
