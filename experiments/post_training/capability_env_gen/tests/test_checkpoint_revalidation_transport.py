"""The large checkpoint input travels by digest, without an Iris file copy."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "checkpoint_revalidation_transport", Path(__file__).resolve().parents[1] / "scripts/checkpoint_revalidation_transport.py"
)
assert spec and spec.loader
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pack_restore_full_checkpoint_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    (source / "deep").mkdir(parents=True)
    (source / "manifest.json").write_text("manifest\n")
    (source / "request.json").write_text("request\n")
    (source / "deep" / "large.bin").write_bytes(b"a" * (26 * 1024 * 1024))
    (source / "deep" / "tool.sh").write_text("#!/bin/sh\nexit 0\n")
    (source / "deep" / "tool.sh").chmod(0o755)
    manifest_sha, request_sha = _sha(source / "manifest.json"), _sha(source / "request.json")

    def validate(root: Path, *, expected_manifest_sha256: str, expected_request_sha256: str) -> None:
        assert _sha(root / "manifest.json") == expected_manifest_sha256
        assert _sha(root / "request.json") == expected_request_sha256
        assert (root / "deep" / "large.bin").stat().st_size == 26 * 1024 * 1024

    monkeypatch.setattr(transport, "validate_checkpoint_bundle", validate)
    archive, receipt = tmp_path / "bundle.tar.gz", tmp_path / "receipt.json"
    transport.pack(source, archive, receipt, manifest_sha, request_sha)
    restored = tmp_path / "restored"
    transport.restore(archive, receipt, restored, manifest_sha, request_sha)
    assert (restored / "deep" / "large.bin").read_bytes() == (source / "deep" / "large.bin").read_bytes()
    assert (restored / "deep" / "tool.sh").stat().st_mode & 0o777 == 0o755
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(ValueError, match="another request"):
        transport.restore(archive, receipt, tmp_path / "wrong", manifest_sha, "0" * 64)
    archive.write_bytes(archive.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="archive changed"):
        transport.restore(archive, receipt, tmp_path / "wrong", manifest_sha, request_sha)


def test_s3_receipt_binds_exact_blob_and_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFS:
        def __init__(self) -> None:
            self.root = tmp_path / "object-store"

        def exists(self, key: str) -> bool:
            return (self.root / key).exists()

        def open(self, key: str, mode: str):
            path = self.root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.open(mode)

    fs = FakeFS()
    monkeypatch.setattr(transport, "_cloud", lambda uri: (fs, uri.removeprefix("s3://")))
    archive = tmp_path / "bundle.tar.gz"
    archive.write_bytes(b"opaque archive bytes")
    digest = _sha(archive)
    manifest_sha, request_sha = "a" * 64, "b" * 64
    receipt = tmp_path / "transport.json"
    receipt.write_text(json.dumps({"schema_version": transport.SCHEMA, "archive_sha256": digest,
                                   "archive_bytes": archive.stat().st_size,
                                   "bundle_manifest_sha256": manifest_sha,
                                   "request_sha256": request_sha, "member_count": 1}))
    remote = tmp_path / "blob.json"
    transport.upload(archive, receipt, "s3://bucket/prefix", remote)
    downloaded = tmp_path / "downloaded.tar.gz"
    transport.download(remote, downloaded, manifest_sha, request_sha)
    assert downloaded.read_bytes() == archive.read_bytes()
    with pytest.raises(ValueError, match="frozen request"):
        transport.download(remote, tmp_path / "wrong.tar.gz", manifest_sha, "0" * 64)
