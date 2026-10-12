# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from levanter.model_cache import DEFAULT_COMPLETE_MARKER, _cache_slug

from experiments.evaluation import prewarm_uploaded_model


@pytest.mark.parametrize("source_payload", [b"published weights", b"stale parent weights"])
def test_uploaded_weights_must_match_hf_before_cache_is_complete(tmp_path, monkeypatch, source_payload):
    revision = "a" * 40
    expected = b"published weights"
    digest = hashlib.sha256(expected).hexdigest()
    source = tmp_path / "export"
    source.mkdir()
    (source / "model.safetensors").write_bytes(source_payload)
    metadata = tmp_path / "hf-metadata"
    metadata.mkdir()
    (metadata / "config.json").write_text('{"model_type":"grug_moe"}')
    (metadata / ".cache_metadata.json").write_text('{"source_revision":"parent"}')
    manifest = metadata / "upload-manifest.json"
    manifest.write_text(
        json.dumps({"checkpoint": str(source), "objects": [{"name": "model.safetensors", "sha256": digest}]})
    )
    config = tmp_path / "model.yaml"
    config.write_text(f"name: merged\nlocation: org/merged\nrevision: {revision}\n")
    cache = tmp_path / "cache" / _cache_slug(f"org/merged@{revision}")
    siblings = [
        SimpleNamespace(rfilename="model.safetensors", lfs=SimpleNamespace(sha256=digest)),
        SimpleNamespace(rfilename="config.json", lfs=None),
        SimpleNamespace(rfilename=".cache_metadata.json", lfs=None),
    ]

    def download(_repo, filename, *, revision, **_kwargs):
        assert revision == "a" * 40
        return str(metadata / filename)

    monkeypatch.setattr(prewarm_uploaded_model, "hf_hub_download", download)
    monkeypatch.setattr(
        prewarm_uploaded_model,
        "model_info",
        lambda *_args, **_kwargs: SimpleNamespace(sha=revision, siblings=siblings),
    )
    marker = Path(f"{cache}{DEFAULT_COMPLETE_MARKER}")
    if source_payload != expected:
        with pytest.raises(ValueError, match="checksum mismatch"):
            prewarm_uploaded_model.warm_uploaded_model(config, str(cache), "upload-manifest.json", 2)
        assert not marker.exists()
        assert not (cache / "model.safetensors").exists()
        return

    prewarm_uploaded_model.warm_uploaded_model(config, str(cache), "upload-manifest.json", 2)

    assert (cache / "model.safetensors").read_bytes() == expected
    assert json.loads((cache / ".cache_metadata.json").read_text())["source_revision"] == "parent"
    assert json.loads(marker.read_text())["source_revision"] == revision
