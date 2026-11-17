# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest

from experiments.post_training.task_curation.datasets.facet import registry


def test_public_package_is_refused_before_any_image_transfer(tmp_path, monkeypatch):
    transfers = []
    monkeypatch.setattr(registry.subprocess, "run", lambda *args, **kwargs: transfers.append(args))
    with pytest.raises(ValueError, match="internal or private"):
        registry.publish_image(
            tmp_path, "ghcr.io/fixture/facet", "image", "sha256:" + "a" * 64, Path("auth.json"), lambda: "public"
        )
    assert not transfers


@pytest.mark.parametrize("visibility", ["internal", "private"])
def test_nonpublic_package_preserves_and_verifies_the_image_digest(tmp_path, monkeypatch, visibility):
    manifest = b'{"schemaVersion":2}'
    digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    transfers, inspections = [], []
    monkeypatch.setattr(registry.subprocess, "run", lambda command, **kwargs: transfers.append(command))

    def inspect(command):
        inspections.append(command)
        return manifest

    monkeypatch.setattr(registry.subprocess, "check_output", inspect)
    observed = registry.publish_image(
        tmp_path, "ghcr.io/fixture/facet", "image", digest, tmp_path / "auth.json", lambda: visibility
    )
    assert observed == visibility
    assert len(transfers) == 1 and "--preserve-digests" in transfers[0]
    assert inspections[0][-1] == "docker://ghcr.io/fixture/facet@" + digest


def test_registry_digest_change_aborts_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(registry.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(registry.subprocess, "check_output", lambda command: b"changed manifest")
    with pytest.raises(ValueError, match="preserved digest"):
        registry.publish_image(
            tmp_path, "ghcr.io/fixture/facet", "image", "sha256:" + "a" * 64, tmp_path / "auth.json", lambda: "internal"
        )
