import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "continuation_seed_transport",
    Path(__file__).resolve().parents[1] / "scripts/continuation_seed_transport.py",
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FS:
    def __init__(self):
        self.objects = {}

    def exists(self, key):
        return key in self.objects

    def cat(self, key):
        return self.objects[key]

    def pipe(self, key, data):
        self.objects[key] = data


def test_upload_and_download_bind_exact_digest_and_bytes(tmp_path):
    archive = tmp_path / "restore-seed.tar.gz"
    archive.write_bytes(b"seed-bytes")
    uri, receipt = MOD.receipt_for("s3://bucket/root", archive, "a" * 64)
    assert receipt["sha256"] == hashlib.sha256(b"seed-bytes").hexdigest()
    assert uri.endswith(f"/{receipt['sha256']}/restore-seed.tar.gz")
    fs = FS()
    MOD.upload(
        fs,
        "bucket/root/_continuation_seed_blobs/"
        + receipt["sha256"]
        + "/restore-seed.tar.gz",
        archive,
        receipt,
    )
    destination = tmp_path / "seed" / "restore-seed.tar.gz"
    MOD.download(
        fs,
        "bucket/root/_continuation_seed_blobs/"
        + receipt["sha256"]
        + "/restore-seed.tar.gz",
        destination,
        MOD.validate_receipt(receipt),
    )
    assert destination.read_bytes() == b"seed-bytes"


def test_existing_conflicting_object_is_rejected(tmp_path):
    archive = tmp_path / "restore-seed.tar.gz"
    archive.write_bytes(b"seed-bytes")
    _, receipt = MOD.receipt_for("s3://bucket/root", archive, "a" * 64)
    key = (
        "bucket/root/_continuation_seed_blobs/"
        + receipt["sha256"]
        + "/restore-seed.tar.gz"
    )
    fs = FS()
    fs.objects[key] = b"different"
    with pytest.raises(ValueError, match="conflicting"):
        MOD.upload(fs, key, archive, receipt)


def test_receipt_rejects_unbound_uri_and_download_never_overwrites(tmp_path):
    archive = tmp_path / "restore-seed.tar.gz"
    archive.write_bytes(b"seed-bytes")
    _, receipt = MOD.receipt_for("s3://bucket/root", archive, "a" * 64)
    receipt["uri"] = "s3://bucket/root/not-the-digest"
    with pytest.raises(ValueError, match="invalid"):
        MOD.validate_receipt(receipt)
    _, good = MOD.receipt_for("s3://bucket/root", archive, "a" * 64)
    destination = tmp_path / "existing.tar.gz"
    destination.write_bytes(b"do-not-overwrite")
    with pytest.raises(ValueError, match="already exists"):
        MOD.download(FS(), "unused", destination, MOD.validate_receipt(good))


def test_receipt_is_bound_to_the_exact_source_manifest_and_dangling_symlink(tmp_path):
    archive = tmp_path / "restore-seed.tar.gz"
    archive.write_bytes(b"seed-bytes")
    _, receipt = MOD.receipt_for("s3://bucket/root", archive, "a" * 64)
    with pytest.raises(ValueError, match="different source manifest"):
        MOD.require_source_manifest(MOD.validate_receipt(receipt), "b" * 64)
    fs = FS()
    key = (
        "bucket/root/_continuation_seed_blobs/"
        + receipt["sha256"]
        + "/restore-seed.tar.gz"
    )
    fs.objects[key] = b"seed-bytes"
    destination = tmp_path / "dangling.tar.gz"
    destination.symlink_to(tmp_path / "not-present")
    with pytest.raises(ValueError, match="already exists"):
        MOD.download(fs, key, destination, MOD.validate_receipt(receipt))
