# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Image build reuse across temporary task directories."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from shellbox.image import DockerfileSource, ImageCache


def test_identical_contexts_share_a_build_and_content_changes_rebuild(tmp_path, monkeypatch):
    contexts = [tmp_path / name for name in ("first", "second")]
    contexts[0].mkdir()
    (contexts[0] / "Dockerfile").write_text("FROM fixture\nCOPY payload /payload\n")
    (contexts[0] / "payload").write_bytes(b"first")
    shutil.copytree(contexts[0], contexts[1])
    builds = []

    def run(args, **_kwargs):
        if args[:2] == ("docker", "build"):
            builds.append((Path(args[-1]) / "payload").read_bytes())
        elif args[0] == "skopeo":
            layout = Path(args[-1].removeprefix("oci:").removesuffix(":image"))
            blobs = layout / "blobs/sha256"
            blobs.mkdir(parents=True)
            config = json.dumps({"os": "linux", "architecture": "amd64", "payload": builds[-1].decode()}).encode()
            config_digest = hashlib.sha256(config).hexdigest()
            (blobs / config_digest).write_bytes(config)
            manifest = json.dumps({"config": {"digest": f"sha256:{config_digest}"}}).encode()
            manifest_digest = hashlib.sha256(manifest).hexdigest()
            (blobs / manifest_digest).write_bytes(manifest)
            (layout / "index.json").write_text(
                json.dumps(
                    {
                        "manifests": [
                            {
                                "digest": f"sha256:{manifest_digest}",
                                "annotations": {"org.opencontainers.image.ref.name": "image"},
                            }
                        ]
                    }
                )
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    cache = ImageCache(tmp_path / "cache", skopeo=Path("skopeo"))
    first = cache.prepare(DockerfileSource(contexts[0], contexts[0] / "Dockerfile"))
    shutil.rmtree(contexts[0])
    second = cache.prepare(DockerfileSource(contexts[1], contexts[1] / "Dockerfile"))
    assert builds == [b"first"]
    assert first.manifest_digest == second.manifest_digest
    (contexts[1] / "payload").write_bytes(b"changed")
    changed = cache.prepare(DockerfileSource(contexts[1], contexts[1] / "Dockerfile"))
    assert builds == [b"first", b"changed"]
    assert changed.manifest_digest != first.manifest_digest
