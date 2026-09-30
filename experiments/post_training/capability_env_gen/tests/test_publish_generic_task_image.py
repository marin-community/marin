import hashlib
import io
import json

import pytest

from scripts import publish_generic_task_image as publisher


def _receipt(path, payload: bytes, *, object_name="s3://bucket/users/task/layer.tar.gz"):
    digest = hashlib.sha256(payload).hexdigest()
    bucket, key = object_name.removeprefix("s3://").split("/", 1)
    path.write_text(json.dumps({
        "schema_version": "capability-rootfs-capture-v1",
        "state": "captured_pending_privacy_and_publication",
        "capture": {
            "ok": True,
            "object": object_name,
            "object_key": key,
            "object_bytes": len(payload),
            "sha256": digest,
            "capture": {
                "compressed_bytes": len(payload),
                "sha256": digest,
                "tar_exit": 0,
                "gzip_exit": 0,
            },
        },
    }))
    return bucket, key, digest


class _Signer:
    def __init__(self, payload: bytes, *, length=None):
        self.payload = payload
        self.length = len(payload) if length is None else length
        self.c = self
        self.calls = []

    def head(self, bucket, key):
        self.calls.append(("head", bucket, key))
        return {"ContentLength": self.length}

    def get_object(self, *, Bucket, Key):
        self.calls.append(("get", Bucket, Key))
        return {"Body": io.BytesIO(self.payload)}


def test_downloaded_layer_is_streamed_and_bound_to_sanitized_receipt(tmp_path):
    payload = b"compressed-layer"
    receipt = tmp_path / "capture.json"
    bucket, key, digest = _receipt(receipt, payload)
    tools = tmp_path / "tools"
    tools.mkdir()
    signer = _Signer(payload)
    layer, transport = publisher.download_captured_layer(
        capture_receipt=receipt,
        download_dir=tmp_path / "layers",
        capture_tools=tools,
        presigner_factory=lambda: signer,
    )
    assert layer.read_bytes() == payload
    assert layer.name == digest + ".tar.gz"
    assert transport == {
        "schema_version": "capability-captured-layer-transport-v1",
        "capture_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "object": f"s3://{bucket}/{key}",
        "object_bucket": bucket,
        "object_key": key,
        "expected_bytes": len(payload),
        "expected_sha256": digest,
        "downloaded_bytes": len(payload),
        "downloaded_sha256": digest,
        "local_filename": digest + ".tar.gz",
    }
    assert signer.calls == [("head", bucket, key), ("get", bucket, key)]


@pytest.mark.parametrize("payload,length", [(b"changed", None), (b"layer", 7)])
def test_downloaded_layer_rejects_s3_byte_or_hash_drift(tmp_path, payload, length):
    receipt = tmp_path / "capture.json"
    _receipt(receipt, b"layer")
    tools = tmp_path / "tools"
    tools.mkdir()
    signer = _Signer(payload, length=length)
    with pytest.raises(ValueError, match="size differs|byte count or SHA"):
        publisher.download_captured_layer(
            capture_receipt=receipt,
            download_dir=tmp_path / "layers",
            capture_tools=tools,
            presigner_factory=lambda: signer,
        )
    assert not list((tmp_path / "layers").glob("*.part"))
    assert not list((tmp_path / "layers").glob("*.tar.gz"))


def test_capture_object_rejects_untrusted_or_malformed_object_locator(tmp_path):
    receipt = tmp_path / "capture.json"
    _receipt(receipt, b"layer", object_name="https://bucket/users/task/layer.tar.gz")
    with pytest.raises(ValueError, match="object identity is unsafe"):
        publisher._capture_object(receipt)


def test_remote_download_mode_requires_explicit_trusted_execution(tmp_path):
    with pytest.raises(ValueError, match="requires trusted publisher --execute"):
        publisher.main([
            "--plan", str(tmp_path / "plan.json"),
            "--workspace", str(tmp_path / "workspace"),
            "--capture-tools", str(tmp_path / "tools"),
            "--approval", str(tmp_path / "approval.json"),
            "--capture-receipt", str(tmp_path / "capture.json"),
            "--download-layer-dir", str(tmp_path / "layers"),
            "--max-uncompressed-bytes", "1",
            "--output", str(tmp_path / "publication.json"),
        ])
