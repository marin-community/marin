# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from experiments.post_training.task_curation import registry


def test_registry_loads_the_pinned_file_and_rejects_corrupted_cache_bytes(tmp_path, monkeypatch):
    payload = b'{"source": {"count": 3}}\n'
    cached = tmp_path / "cached.json"
    cached.write_bytes(payload)
    reference = {
        "repo": "open-athena/registry-fixture",
        "revision": "a" * 40,
        "path": "receipts/part-000.json",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))

    def download(**kwargs):
        assert kwargs == {
            "repo_id": reference["repo"],
            "repo_type": "dataset",
            "revision": reference["revision"],
            "filename": reference["path"],
        }
        return str(cached)

    monkeypatch.setattr(registry, "hf_hub_download", download)
    assert registry.read_registry_json(path) == {"source": {"count": 3}}
    cached.write_bytes(b'{"source": {"count": 4}}\n')
    with pytest.raises(ValueError, match="Registry bytes changed"):
        registry.read_registry_json(path)


def test_registry_does_not_replace_an_unavailable_pin_with_another_revision(tmp_path, monkeypatch):
    payload = b'{"source": {"count": 3}}\n'
    latest = tmp_path / "latest.json"
    latest.write_bytes(payload)
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "repo": "open-athena/registry-fixture",
                "revision": "a" * 40,
                "path": "data.json",
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    )

    def unavailable(**kwargs):
        if kwargs["revision"] == "a" * 40:
            raise FileNotFoundError("Pinned fixture revision is unavailable")
        return str(latest)

    monkeypatch.setattr(registry, "hf_hub_download", unavailable)
    with pytest.raises(FileNotFoundError, match="Pinned fixture revision"):
        registry.read_registry_json(path)
