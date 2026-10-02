# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Image build reuse across temporary task directories."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from shellbox.backends.qemu.image import QemuAssets, stage_qemu_image
from shellbox.image import DockerfileSource, ImageCache, PreparedImage, RegistryImage, prepared_image_workdir


@pytest.mark.parametrize("working_directory,expected", [("/task workspace", "/task workspace"), ("", "/")])
def test_image_working_directory(tmp_path, working_directory, expected):
    blobs = tmp_path / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    config = json.dumps({"config": {"WorkingDir": working_directory}}).encode()
    config_digest = hashlib.sha256(config).hexdigest()
    config_path = blobs / config_digest
    config_path.write_bytes(config)
    manifest = json.dumps({"config": {"digest": f"sha256:{config_digest}"}}).encode()
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    (blobs / manifest_digest).write_bytes(manifest)
    image = PreparedImage(tmp_path, f"sha256:{manifest_digest}", RegistryImage("fixture"))
    assert prepared_image_workdir(image) == expected
    config_path.write_bytes(json.dumps({"config": {"WorkingDir": "/changed"}}).encode())
    with pytest.raises(ValueError):
        prepared_image_workdir(image)


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
            manifest = json.dumps({"config": {"digest": f"sha256:{config_digest}"}, "layers": []}).encode()
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
        elif args[0] == "umoci":
            layout = Path(args[args.index("--image") + 1].removesuffix(":image"))
            descriptor = json.loads((layout / "index.json").read_text())["manifests"][0]
            bundle = Path(args[-1])
            (bundle / "rootfs/bin").mkdir(parents=True)
            (bundle / "rootfs/bin/sh").write_bytes(b"shell fixture")
            (bundle / "config.json").write_text(json.dumps({"process": {"user": {"uid": 0, "gid": 0}}}))
            (bundle / "umoci.json").write_text(json.dumps({"from_descriptor_path": {"descriptor_walk": [descriptor]}}))
        elif args[0] == "cc":
            Path(args[args.index("-o") + 1]).write_bytes(b"agent fixture")
        elif args[0] == "cpio":
            return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    cache = ImageCache(tmp_path / "cache", skopeo=Path("skopeo"))
    first = cache.prepare(DockerfileSource(contexts[0], contexts[0] / "Dockerfile"))
    shutil.rmtree(contexts[0])
    second = cache.prepare(DockerfileSource(contexts[1], contexts[1] / "Dockerfile"))
    assert builds == [b"first"]
    assert first.manifest_digest == second.manifest_digest
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    for name in ("qemu", "kernel", "busybox"):
        (assets_dir / name).write_bytes(b"runtime fixture")
    (assets_dir / "firmware").mkdir()
    (assets_dir / "firmware/bios-microvm.bin").write_bytes(b"firmware fixture")
    (assets_dir / "libraries").mkdir()
    assets = QemuAssets(
        qemu=assets_dir / "qemu",
        kernel=assets_dir / "kernel",
        busybox=assets_dir / "busybox",
        firmware=assets_dir / "firmware",
        libraries=assets_dir / "libraries",
        umoci=Path("umoci"),
        disk_size_mb=1,
        runtime_id="fixture",
    )
    bundle = stage_qemu_image(second, assets, tmp_path / "bundles")
    metadata = json.loads((bundle / "image.json").read_text())
    assert metadata["manifest_digest"] == second.manifest_digest
    assert metadata["dockerfile_sha256"] == hashlib.sha256((contexts[1] / "Dockerfile").read_bytes()).hexdigest()
    (contexts[1] / "payload").write_bytes(b"changed")
    changed = cache.prepare(DockerfileSource(contexts[1], contexts[1] / "Dockerfile"))
    assert builds == [b"first", b"changed"]
    assert changed.manifest_digest != first.manifest_digest
