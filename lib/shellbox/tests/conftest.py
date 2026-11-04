# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline OCI images shared by image preparation and real local machine tests."""

import hashlib
import json
from collections.abc import Callable
from itertools import count
from pathlib import Path

import pytest
from shellbox.image import PreparedImage, RegistryImage


def pytest_addoption(parser):
    parser.addoption("--local-cgroup-parent", help="Writable delegated cgroup v2 parent for manual local-limit tests")


@pytest.fixture
def oci_image(tmp_path: Path) -> Callable[[tuple[bytes, ...], dict[str, object]], PreparedImage]:
    sequence = count()

    def build(layers: tuple[bytes, ...], process: dict[str, object]) -> PreparedImage:
        layout = tmp_path / f"oci-{next(sequence)}"
        blobs = layout / "blobs/sha256"
        blobs.mkdir(parents=True)

        def blob(data: bytes, media_type: str) -> dict[str, object]:
            digest = hashlib.sha256(data).hexdigest()
            (blobs / digest).write_bytes(data)
            return {"digest": f"sha256:{digest}", "size": len(data), "mediaType": media_type}

        descriptors = tuple(blob(layer, "application/vnd.oci.image.layer.v1.tar") for layer in layers)
        config = {
            "os": "linux",
            "architecture": "amd64",
            "config": process,
            "rootfs": {"type": "layers", "diff_ids": [layer["digest"] for layer in descriptors]},
        }
        descriptor = blob(json.dumps(config).encode(), "application/vnd.oci.image.config.v1+json")
        manifest = blob(
            json.dumps({"schemaVersion": 2, "config": descriptor, "layers": descriptors}).encode(),
            "application/vnd.oci.image.manifest.v1+json",
        )
        manifest["annotations"] = {"org.opencontainers.image.ref.name": "image"}
        (layout / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [manifest]}))
        (layout / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}))
        return PreparedImage(layout, str(manifest["digest"]), RegistryImage("offline-fixture"))

    return build
