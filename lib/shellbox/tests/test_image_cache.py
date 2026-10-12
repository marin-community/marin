# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Image build reuse across temporary task directories."""

import hashlib
import io
import json
import multiprocessing
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest
from shellbox.backends.qemu.image import QemuAssets, stage_qemu_image
from shellbox.image import DockerfileSource, ImageCache, PreparedImage, RegistryImage, load_docker_image


@pytest.fixture
def registry_transport(tmp_path):
    layout = tmp_path / "registry-image"
    blobs = layout / "blobs/sha256"
    blobs.mkdir(parents=True)
    config = json.dumps({"os": "linux", "architecture": "amd64", "rootfs": {"type": "layers", "diff_ids": []}}).encode()
    config_hash = hashlib.sha256(config).hexdigest()
    manifest = json.dumps({"config": {"digest": f"sha256:{config_hash}"}, "layers": []}).encode()
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    (blobs / config_hash).write_bytes(config)
    (blobs / manifest_hash).write_bytes(manifest)
    (layout / "index.json").write_text(
        json.dumps(
            {
                "manifests": [
                    {
                        "digest": f"sha256:{manifest_hash}",
                        "annotations": {"org.opencontainers.image.ref.name": "image"},
                    }
                ]
            }
        )
    )
    contact_log = tmp_path / "registry-requests.jsonl"
    skopeo = tmp_path / "skopeo"
    skopeo.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, shutil, sys\n"
        f"with pathlib.Path({str(contact_log)!r}).open('a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:])+'\\n')\n"
        "if '--policy' in sys.argv:\n"
        "    policy = json.loads(pathlib.Path(sys.argv[sys.argv.index('--policy')+1]).read_text())\n"
        "    if policy['default'][0]['type'] == 'reject':\n"
        "        sys.exit('signature policy rejected image')\n"
        "destination = sys.argv[-1].removeprefix('oci:').removesuffix(':image')\n"
        f"shutil.copytree({str(layout)!r}, destination)\n"
    )
    skopeo.chmod(0o755)
    return skopeo, contact_log, f"sha256:{manifest_hash}"


def _prepare_registry_process(directory, skopeo, source, start, results):
    start.wait(timeout=20)
    image = ImageCache(directory, skopeo=skopeo).prepare(source)
    results.put((str(image.layout), image.manifest_digest))


def test_pinned_registry_image_is_available_to_a_second_process_without_registry_access(tmp_path, registry_transport):
    skopeo, contact_log, selected_digest = registry_transport
    source = RegistryImage(f"registry.invalid/image@sha256:{'a' * 64}")
    directory = tmp_path / "cache"
    first = ImageCache(directory, skopeo=skopeo).prepare(source)
    assert first.manifest_digest == selected_digest
    # Registry and selected OCI manifests can differ after platform selection.
    assert not source.reference.endswith(selected_digest)
    skopeo.write_text("#!/bin/sh\nexit 99\n")
    context = multiprocessing.get_context("spawn")
    start, results = context.Barrier(2), context.Queue()
    process = context.Process(target=_prepare_registry_process, args=(directory, skopeo, source, start, results))
    process.start()
    start.wait(timeout=20)
    process.join(timeout=20)
    assert process.exitcode == 0
    assert results.get(timeout=5) == (str(first.layout), selected_digest)
    assert len(contact_log.read_text().splitlines()) == 1


def test_concurrent_processes_prepare_one_pinned_registry_layout(tmp_path, registry_transport):
    skopeo, contact_log, selected_digest = registry_transport
    source = RegistryImage(f"registry.invalid/image@sha256:{'b' * 64}")
    directory = tmp_path / "cache"
    context = multiprocessing.get_context("spawn")
    start, results = context.Barrier(5), context.Queue()
    processes = [
        context.Process(target=_prepare_registry_process, args=(directory, skopeo, source, start, results))
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    start.wait(timeout=20)
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    prepared = [results.get(timeout=5) for _ in processes]
    assert len(set(prepared)) == 1
    assert prepared[0][1] == selected_digest
    assert len(contact_log.read_text().splitlines()) == 1


def test_mutable_references_are_rechecked_between_cache_instances(tmp_path, registry_transport):
    skopeo, contact_log, _ = registry_transport
    source = RegistryImage("registry.invalid/image:mutable")
    first = ImageCache(tmp_path / "cache", skopeo=skopeo).prepare(source)
    second = ImageCache(tmp_path / "cache", skopeo=skopeo).prepare(source)
    assert first.manifest_digest == second.manifest_digest
    assert len(contact_log.read_text().splitlines()) == 2


def test_pinned_registry_cache_does_not_bypass_an_explicit_signature_policy(tmp_path, registry_transport):
    skopeo, contact_log, _ = registry_transport
    source = RegistryImage(f"registry.invalid/image@sha256:{'c' * 64}")
    ImageCache(tmp_path / "cache", skopeo=skopeo).prepare(source)
    policy = tmp_path / "policy.json"
    policy.write_text('{"default": [{"type": "reject"}]}')
    with pytest.raises(RuntimeError, match="signature policy rejected image"):
        ImageCache(tmp_path / "cache", skopeo=skopeo, policy=policy).prepare(source)
    assert len(contact_log.read_text().splitlines()) == 2


def test_pinned_registry_cache_rejects_a_changed_configuration_blob(tmp_path, registry_transport):
    skopeo, _, _ = registry_transport
    source = RegistryImage(f"registry.invalid/image@sha256:{'d' * 64}")
    cache = tmp_path / "cache"
    image = ImageCache(cache, skopeo=skopeo).prepare(source)
    manifest = json.loads((image.layout / "blobs/sha256" / image.manifest_digest.split(":")[1]).read_text())
    config = image.layout / "blobs/sha256" / manifest["config"]["digest"].split(":")[1]
    config.write_bytes(config.read_bytes() + b" ")
    with pytest.raises(ValueError, match="OCI configuration digest mismatch"):
        ImageCache(cache, skopeo=skopeo).prepare(source)


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


@pytest.fixture
def skopeo() -> Path:
    executable = shutil.which("skopeo")
    if executable is None:
        pytest.skip("Image transport checks require Skopeo")
    return Path(executable)


@pytest.fixture(params=["zstd", "gzip"])
def compressed_image(tmp_path, request, skopeo):
    layout = tmp_path / "uncompressed"
    blobs = layout / "blobs/sha256"
    blobs.mkdir(parents=True)
    marker = b"image transport payload\x00\xff"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("payload")
        member.size = len(marker)
        archive.addfile(member, io.BytesIO(marker))
    layer = stream.getvalue()
    layer_digest = hashlib.sha256(layer).hexdigest()
    config = json.dumps(
        {
            "os": "linux",
            "architecture": "amd64",
            "rootfs": {"type": "layers", "diff_ids": [f"sha256:{layer_digest}"]},
            "config": {"Labels": {"pytest-image": str(tmp_path)}, "Healthcheck": {"Test": ["CMD", "/unused"]}},
        }
    ).encode()
    config_digest = hashlib.sha256(config).hexdigest()
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "digest": f"sha256:{config_digest}",
                "size": len(config),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.oci.image.layer.v1.tar",
                    "digest": f"sha256:{layer_digest}",
                    "size": len(layer),
                }
            ],
        }
    ).encode()
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    for digest, contents in ((layer_digest, layer), (config_digest, config), (manifest_digest, manifest)):
        (blobs / digest).write_bytes(contents)
    (layout / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}))
    (layout / "index.json").write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "manifests": [
                    {
                        "mediaType": "application/vnd.oci.image.manifest.v1+json",
                        "digest": f"sha256:{manifest_digest}",
                        "size": len(manifest),
                        "annotations": {"org.opencontainers.image.ref.name": "image"},
                    }
                ],
            }
        )
    )
    compressed = tmp_path / "compressed"
    subprocess.run(
        (
            str(skopeo),
            "--insecure-policy",
            "copy",
            "--dest-compress-format",
            request.param,
            f"oci:{layout}:image",
            f"oci:{compressed}:image",
        ),
        check=True,
        capture_output=True,
    )
    digest = json.loads((compressed / "index.json").read_text())["manifests"][0]["digest"]
    image = PreparedImage(compressed, digest, RegistryImage(f"fixture.invalid/image@{digest}"))
    tag = f"harbor-prepared:sha256-{digest.split(':')[1]}"
    try:
        yield image, json.loads(config), marker
    finally:
        subprocess.run(("docker", "image", "rm", tag), capture_output=True)


@pytest.mark.docker
def test_compressed_oci_load_preserves_image_identity_payload_and_pinned_cache(compressed_image, skopeo):
    image, config, marker = compressed_image
    original = {path.relative_to(image.layout): path.read_bytes() for path in image.layout.rglob("*") if path.is_file()}
    tag = load_docker_image(image, skopeo=skopeo)
    actual = json.loads(subprocess.check_output(("docker", "image", "inspect", tag)))[0]
    assert actual["RootFS"]["Layers"] == config["rootfs"]["diff_ids"]
    assert actual["Config"]["Labels"] == config["config"]["Labels"]
    assert actual["Config"]["Healthcheck"]["Test"] == config["config"]["Healthcheck"]["Test"]
    container = subprocess.check_output(("docker", "create", tag, "/unused"), text=True).strip()
    try:
        exported = subprocess.check_output(("docker", "export", container))
        with tarfile.open(fileobj=io.BytesIO(exported)) as archive:
            payload = archive.extractfile("payload")
            assert payload is not None
            assert payload.read() == marker
    finally:
        subprocess.run(("docker", "rm", container), check=True, capture_output=True)
    assert load_docker_image(image, skopeo=skopeo) == tag
    assert {
        path.relative_to(image.layout): path.read_bytes() for path in image.layout.rglob("*") if path.is_file()
    } == original


@pytest.mark.docker
def test_cached_docker_tag_with_different_contents_is_rejected(compressed_image, skopeo):
    image, _, _ = compressed_image
    tag = load_docker_image(image, skopeo=skopeo)
    original_id = json.loads(subprocess.check_output(("docker", "image", "inspect", tag)))[0]["Id"]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        member = tarfile.TarInfo("wrong-payload")
        member.size = 3
        archive.addfile(member, io.BytesIO(b"bad"))
    try:
        subprocess.run(("docker", "import", "-", tag), input=stream.getvalue(), check=True, capture_output=True)
        with pytest.raises(ValueError):
            load_docker_image(image, skopeo=skopeo)
    finally:
        subprocess.run(("docker", "image", "rm", original_id), capture_output=True)
