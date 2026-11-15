# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from rigging.fsutil.verified_copy import verified_copy_prefix

from experiments.post_training.long_loop.exp3_recipe import verified_initializer_model
from experiments.post_training.long_loop.sft import BASE_MODEL, BASE_REVISION


def test_verified_hub_snapshot_resolves_regional_model_and_rejects_changed_copy(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    weight = blobs / "weight"
    weight.write_bytes(b"unchanged pretrained tensor bytes")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(weight)
    destination = tmp_path / "regional"
    copied = verified_copy_prefix(str(snapshot), str(destination), workers=1)
    manifest_path = destination / ".verified-copy-manifest.json"
    manifest_raw = manifest_path.read_bytes()
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    receipt = {
        "repo": BASE_MODEL,
        "revision": BASE_REVISION,
        "model_ref": str(destination),
        "verified_copy_manifest": copied.manifest_url,
        "verified_copy_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "files": [
            {"path": "model.safetensors", "size": weight.stat().st_size, "sha256": digest, "lfs": {"sha256": digest}}
        ],
    }
    receipt_path = tmp_path / "source.json"
    raw = json.dumps(receipt).encode()
    receipt_path.write_bytes(raw)
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    assert verified_initializer_model(str(receipt_path), receipt_sha256) == str(destination)
    assert (destination / "model.safetensors").read_bytes() == weight.read_bytes()
    manifest = json.loads(manifest_raw)
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="completion manifest SHA256 mismatch"):
        verified_initializer_model(str(receipt_path), receipt_sha256)
