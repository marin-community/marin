import pytest

from scripts.capture_task_images import (
    _sanitize_legacy_receipt,
    _validate_archive_receipt,
)


def valid_receipt():
    return {
        "snapshot": "candidate",
        "source_env": {"DAYTONA_API_KEY": "must-not-survive"},
        "raw": "must-not-survive",
        "ok": True,
        "object_bytes": 123,
        "sha256": "a" * 64,
        "capture": {
            "compressed_bytes": 123,
            "sha256": "a" * 64,
            "tar_exit": 0,
            "gzip_exit": 0,
            "tar_stderr_tail": "",
            "etags": [[1, "secret-presigned-result"]],
        },
    }


def test_capture_receipt_uses_allowlists():
    sanitized = _sanitize_legacy_receipt(valid_receipt())
    assert "source_env" not in sanitized
    assert "raw" not in sanitized
    assert "etags" not in sanitized["capture"]


def test_archive_receipt_checks_processes_bytes_and_hashes():
    assert _validate_archive_receipt(valid_receipt())["tar_exit_class"] == "clean"
    document = valid_receipt()
    document["capture"]["gzip_exit"] = 1
    with pytest.raises(RuntimeError, match="gzip"):
        _validate_archive_receipt(document)
    document = valid_receipt()
    document["object_bytes"] = 122
    with pytest.raises(RuntimeError, match="inconsistent"):
        _validate_archive_receipt(document)


def test_tar_exit_one_requires_a_separate_reviewed_disposition():
    document = valid_receipt()
    document["capture"].update(
        {"tar_exit": 1, "tar_stderr_tail": "tar: ./var/log/x: file changed as we read it"}
    )
    with pytest.raises(RuntimeError, match="tar process failed"):
        _validate_archive_receipt(document)
