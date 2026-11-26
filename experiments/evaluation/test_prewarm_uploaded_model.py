# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace

import pytest
from levanter.model_cache import DEFAULT_COMPLETE_MARKER, _cache_slug

from experiments.evaluation import prewarm_uploaded_model


@pytest.mark.parametrize("source_payload", [b"published weights", b"stale parent weights"])
@pytest.mark.parametrize("manifest_format", ["upload", "producer"])
def test_uploaded_weights_must_match_hf_before_cache_is_complete(tmp_path, monkeypatch, source_payload, manifest_format):
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
    document = (
        {"checkpoint": str(source), "objects": [{"name": "model.safetensors", "sha256": digest}]}
        if manifest_format == "upload"
        else {"files": [{"path": "model.safetensors", "sha256": digest}]}
    )
    manifest.write_text(json.dumps(document))
    source_override = str(source) if manifest_format == "producer" else None
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
    marker = cache / DEFAULT_COMPLETE_MARKER
    if source_payload != expected:
        with pytest.raises(ValueError, match="checksum mismatch"):
            prewarm_uploaded_model.warm_uploaded_model(config, str(cache), "upload-manifest.json", 2, source_override)
        assert not marker.exists()
        assert not (cache / "model.safetensors").exists()
        return

    prewarm_uploaded_model.warm_uploaded_model(config, str(cache), "upload-manifest.json", 2, source_override)

    assert (cache / "model.safetensors").read_bytes() == expected
    assert json.loads(marker.read_text())["source_revision"] == revision
