# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepared OCI fixtures for offline environment tests."""

import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import pytest
from shellbox.image import PreparedImage, RegistryImage


@pytest.fixture
def prepared_busybox_image(tmp_path: Path) -> PreparedImage:
    binary = shutil.which("busybox")
    if binary is None:
        if os.environ.get("CI"):
            pytest.fail("The prepared local runtime tests require busybox-static")
        pytest.skip("The prepared local runtime tests require busybox-static")
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w") as archive:
        content = Path(binary).read_bytes()
        member = tarfile.TarInfo("bin/busybox")
        member.mode, member.size = 0o755, len(content)
        archive.addfile(member, io.BytesIO(content))
        for applet in ("sh", "sleep", "mkdir", "cat", "tar", "httpd", "wget", "dirname", "basename", "id", "chmod"):
            member = tarfile.TarInfo(f"bin/{applet}")
            member.type, member.linkname = tarfile.SYMTYPE, "busybox"
            archive.addfile(member)
        for name, mode in (("etc", 0o755), ("tmp", 0o1777)):
            member = tarfile.TarInfo(name)
            member.type, member.mode = tarfile.DIRTYPE, mode
            archive.addfile(member)
    layout = tmp_path / "oci-busybox"
    blobs = layout / "blobs" / "sha256"
    blobs.mkdir(parents=True)

    def blob(content: bytes, media_type: str) -> dict:
        digest = hashlib.sha256(content).hexdigest()
        (blobs / digest).write_bytes(content)
        return {"digest": f"sha256:{digest}", "size": len(content), "mediaType": media_type}

    layer_descriptor = blob(layer.getvalue(), "application/vnd.oci.image.layer.v1.tar")
    config = blob(
        json.dumps(
            {
                "os": "linux",
                "architecture": "amd64",
                "rootfs": {"type": "layers", "diff_ids": [layer_descriptor["digest"]]},
                "config": {
                    "Env": ["PATH=/bin", "IMAGE_VALUE=from-image"],
                    "WorkingDir": "/app",
                    "User": "0:0",
                    "Entrypoint": ["/bin/sh"],
                    "Cmd": [
                        "-c",
                        "mkdir -p /logs/agent; printf '%s' \"$IMAGE_VALUE\" >index.html; "
                        "printf '{\"answer\":42}' >/logs/agent/runtime_state.json; "
                        "exec httpd -f -p 8123 -h .",
                    ],
                },
            }
        ).encode(),
        "application/vnd.oci.image.config.v1+json",
    )
    manifest = blob(
        json.dumps({"schemaVersion": 2, "config": config, "layers": [layer_descriptor]}).encode(),
        "application/vnd.oci.image.manifest.v1+json",
    )
    manifest["annotations"] = {"org.opencontainers.image.ref.name": "image"}
    (layout / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [manifest]}))
    (layout / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}))
    reference = "offline.test/busybox@" + manifest["digest"]
    return PreparedImage(layout, manifest["digest"], RegistryImage(reference))
