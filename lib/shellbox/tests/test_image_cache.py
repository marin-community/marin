# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Image build reuse across temporary task directories."""

import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from shellbox.backends.local.images import prepare_local_image
from shellbox.backends.qemu.image import QemuAssets, stage_qemu_image
from shellbox.image import DockerfileSource, ImageCache, PreparedImage, RegistryImage, prepared_image_workdir


@dataclass(frozen=True)
class RegistryFixture:
    source: RegistryImage
    image: PreparedImage
    manifests: dict[str, bytes]
    source_configuration: bytes


@pytest.fixture
def converted_registry_image(layered_oci_image, monkeypatch):
    image = layered_oci_image
    blobs = image.layout / "blobs/sha256"
    manifest = json.loads((blobs / image.manifest_digest.removeprefix("sha256:")).read_bytes())
    manifest["mediaType"] = "application/vnd.oci.image.manifest.v1+json"
    for layer in manifest["layers"]:
        compressed = gzip.compress((blobs / layer["digest"].removeprefix("sha256:")).read_bytes(), mtime=0)
        digest = hashlib.sha256(compressed).hexdigest()
        (blobs / digest).write_bytes(compressed)
        layer.update(
            digest=f"sha256:{digest}", size=len(compressed), mediaType="application/vnd.oci.image.layer.v1.tar+gzip"
        )
    encoded = json.dumps(manifest).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    (blobs / digest).write_bytes(encoded)
    index = json.loads((image.layout / "index.json").read_bytes())
    index["manifests"][0].update(digest=f"sha256:{digest}", size=len(encoded))
    (image.layout / "index.json").write_text(json.dumps(index))
    image = replace(image, manifest_digest=f"sha256:{digest}")
    manifest["mediaType"] = "application/vnd.docker.distribution.manifest.v2+json"
    manifest["config"]["mediaType"] = "application/vnd.docker.container.image.v1+json"
    configuration = json.loads((blobs / manifest["config"]["digest"].removeprefix("sha256:")).read_bytes())
    source_configuration = json.dumps(configuration, indent=2, sort_keys=True).encode()
    manifest["config"].update(
        digest=f"sha256:{hashlib.sha256(source_configuration).hexdigest()}", size=len(source_configuration)
    )
    for layer in manifest["layers"]:
        layer["mediaType"] = "application/vnd.docker.image.rootfs.diff.tar.gzip"
    original = json.dumps(manifest).encode()
    original_digest = f"sha256:{hashlib.sha256(original).hexdigest()}"
    manifests = {original_digest: original}

    def image_tool(args, **_kwargs):
        if "inspect" in args:
            output = source_configuration if "--config" in args else manifests[args[-1].rsplit("@", 1)[1]]
            return subprocess.CompletedProcess(args, 0, stdout=output, stderr=b"")
        destination = Path(args[-1].removeprefix("oci:").removesuffix(":image"))
        shutil.copytree(image.layout, destination)
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", image_tool)
    return RegistryFixture(
        RegistryImage(f"registry.invalid/task@{original_digest}"), image, manifests, source_configuration
    )


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
    (tmp_path / "index.json").write_text(
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
    image = PreparedImage(tmp_path, f"sha256:{manifest_digest}", RegistryImage("fixture"))
    assert prepared_image_workdir(image) == expected
    config_path.write_bytes(json.dumps({"config": {"WorkingDir": "/changed"}}).encode())
    with pytest.raises(ValueError):
        prepared_image_workdir(image)


@pytest.fixture
def layered_oci_image(oci_image):
    layers = []
    for files in ({"data/payload": b"original", "data/deleted": b"lower"}, {"data/.wh.deleted": b""}):
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w") as output:
            for name, contents in files.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(contents)
                entry.mode = 0o644
                output.addfile(entry, io.BytesIO(contents))
        layers.append(archive.getvalue())
    return oci_image(
        tuple(layers),
        {
            "Env": ["PATH=/bin", "FIXTURE=from=image"],
            "WorkingDir": "/data",
            "Entrypoint": ["/bin/service", "--foreground"],
            "Cmd": ["--port", "8000"],
            "User": "0:0",
            "AttachStdin": False,
        },
    )


def test_local_image_unpack_applies_layers_and_keeps_each_member_private(tmp_path, layered_oci_image):
    umoci = shutil.which("umoci")
    if umoci is None:
        pytest.skip("OCI preparation requires umoci; the Shellbox CI lane installs it")
    first = prepare_local_image(layered_oci_image, tmp_path / "first", umoci=Path(umoci))
    second = prepare_local_image(layered_oci_image, tmp_path / "second", umoci=Path(umoci))
    assert (first.root / "data/payload").read_bytes() == b"original"
    assert not (first.root / "data/deleted").exists()
    (first.root / "data/payload").write_bytes(b"changed")
    assert (second.root / "data/payload").read_bytes() == b"original"
    assert first.environment == {"PATH": "/bin", "FIXTURE": "from=image"}
    assert first.workdir == "/data"
    assert first.entrypoint + first.arguments == ("/bin/service", "--foreground", "--port", "8000")
    assert first.user == "0:0"


@pytest.mark.parametrize("tampering", ["config", "tag", "manifest-path"])
def test_local_image_rejects_changed_or_invalid_identity_before_unpack(tmp_path, layered_oci_image, tampering):
    image = layered_oci_image
    if tampering == "config":
        manifest = json.loads((image.layout / "blobs/sha256" / image.manifest_digest.split(":")[1]).read_bytes())
        (image.layout / "blobs/sha256" / manifest["config"]["digest"].split(":")[1]).write_bytes(b"{}")
    elif tampering == "tag":
        index = json.loads((image.layout / "index.json").read_text())
        index["manifests"][0]["digest"] = f"sha256:{'0' * 64}"
        (image.layout / "index.json").write_text(json.dumps(index))
    else:
        invalid_digest = "sha256:../../outside"
        image = replace(image, manifest_digest=invalid_digest)
        index = json.loads((image.layout / "index.json").read_text())
        index["manifests"][0]["digest"] = invalid_digest
        (image.layout / "index.json").write_text(json.dumps(index))
    destination = tmp_path / "unpacked"
    with pytest.raises(ValueError):
        prepare_local_image(image, destination, umoci=tmp_path / "must-not-run")
    assert not destination.exists()


def test_pinned_registry_image_reuses_verified_local_layout_offline(tmp_path, layered_oci_image, monkeypatch):
    image = layered_oci_image
    cache_dir = tmp_path / "cache"
    destination = cache_dir / image.manifest_digest.replace(":", "-")
    shutil.copytree(image.layout, destination)

    def unavailable(*_args, **_kwargs):
        pytest.fail("An immutable local cache hit must not invoke an external image tool")

    monkeypatch.setattr(subprocess, "run", unavailable)
    source = RegistryImage(f"registry.invalid/task@{image.manifest_digest}")
    prepared = ImageCache(cache_dir).prepare(source)
    assert prepared.manifest_digest == image.manifest_digest
    assert prepared.layout == destination
    assert prepared_image_workdir(prepared) == "/data"


@pytest.mark.parametrize("tampering", ["config", "layer", "tag"])
@pytest.mark.parametrize("already_loaded", [False, True])
def test_pinned_registry_cache_rejects_corruption(tmp_path, layered_oci_image, monkeypatch, tampering, already_loaded):
    image = layered_oci_image
    cache_dir = tmp_path / "cache"
    destination = cache_dir / image.manifest_digest.replace(":", "-")
    shutil.copytree(image.layout, destination)
    cache = ImageCache(cache_dir)
    source = RegistryImage(f"registry.invalid/task@{image.manifest_digest}")
    if already_loaded:
        cache.prepare(source)
    manifest = json.loads((destination / "blobs/sha256" / image.manifest_digest.split(":")[1]).read_bytes())
    if tampering == "tag":
        index = json.loads((destination / "index.json").read_text())
        index["manifests"][0]["digest"] = manifest["config"]["digest"]
        (destination / "index.json").write_text(json.dumps(index))
    else:
        descriptor = manifest["config"] if tampering == "config" else manifest["layers"][0]
        (destination / "blobs/sha256" / descriptor["digest"].split(":")[1]).write_bytes(b"corrupt")

    def unavailable(*_args, **_kwargs):
        pytest.fail("Corrupt local cache entries must fail closed")

    monkeypatch.setattr(subprocess, "run", unavailable)
    with pytest.raises(ValueError):
        cache.prepare(source)


def test_pinned_registry_cache_still_requires_configured_signature_verification(
    tmp_path, layered_oci_image, monkeypatch
):
    image = layered_oci_image
    cache_dir = tmp_path / "cache"
    shutil.copytree(image.layout, cache_dir / image.manifest_digest.replace(":", "-"))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"default": [{"type": "reject"}]}))

    def rejected(args, **_kwargs):
        return subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"signature verification rejected")

    monkeypatch.setattr(subprocess, "run", rejected)
    with pytest.raises(RuntimeError, match="signature verification rejected"):
        ImageCache(cache_dir, policy=policy).prepare(RegistryImage(f"registry.invalid/task@{image.manifest_digest}"))


def test_registry_index_pin_retains_verified_platform_chain_offline(tmp_path, converted_registry_image, monkeypatch):
    fixture = converted_registry_image
    leaf = fixture.source.reference.rsplit("@", 1)[1]
    index = json.dumps(
        {"schemaVersion": 2, "manifests": [{"digest": leaf, "platform": {"os": "linux", "architecture": "amd64"}}]}
    ).encode()
    digest = f"sha256:{hashlib.sha256(index).hexdigest()}"
    fixture.manifests[digest] = index
    source = RegistryImage(f"registry.invalid/task@{digest}")
    cache = tmp_path / "cache"
    prepared = ImageCache(cache).prepare(source)

    def unavailable(*_args, **_kwargs):
        pytest.fail("A verified source index must resolve offline in a new cache instance")

    monkeypatch.setattr(subprocess, "run", unavailable)
    reloaded = ImageCache(cache).prepare(source)
    assert reloaded.manifest_digest == prepared.manifest_digest == fixture.image.manifest_digest
    assert reloaded.source == source
    assert prepared_image_workdir(reloaded) == "/data"
    binding_path = cache / "registry" / f"{digest.removeprefix('sha256:')}.json"
    binding = json.loads(binding_path.read_bytes())
    binding["source_manifests"][1] = fixture.image.manifest_digest
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="source index"):
        ImageCache(cache).prepare(source)


def test_converted_manifest_reloads_in_a_new_process_without_image_tools(tmp_path, converted_registry_image):
    fixture = converted_registry_image
    cache = tmp_path / "cache"
    prepared = ImageCache(cache).prepare(fixture.source)
    assert prepared.manifest_digest != fixture.source.reference.rsplit("@", 1)[1]
    program = (
        "import json,sys; from pathlib import Path; "
        "from shellbox.image import ImageCache,RegistryImage,prepared_image_workdir; "
        "image=ImageCache(Path(sys.argv[1])).prepare(RegistryImage(sys.argv[2])); "
        "print(json.dumps([image.manifest_digest,image.source.reference,prepared_image_workdir(image)]))"
    )
    with subprocess.Popen(
        [sys.executable, "-c", program, str(cache), fixture.source.reference],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PATH": ""},
    ) as child:
        stdout, stderr = child.communicate(timeout=20)
        assert child.returncode == 0, stderr.decode()
    assert json.loads(stdout) == [fixture.image.manifest_digest, fixture.source.reference, "/data"]


@pytest.mark.parametrize("tampering", ["source_manifest", "source_binding", "source_config", "config", "tag"])
def test_converted_registry_binding_rejects_corruption(tmp_path, converted_registry_image, tampering):
    fixture = converted_registry_image
    cache = tmp_path / "cache"
    prepared = ImageCache(cache).prepare(fixture.source)
    original = fixture.source.reference.rsplit("@", 1)[1].removeprefix("sha256:")
    if tampering == "source_manifest":
        (prepared.layout / "blobs/sha256" / original).write_bytes(b"changed")
    elif tampering == "source_binding":
        path = cache / "registry" / f"{original}.json"
        binding = json.loads(path.read_bytes())
        binding["source_manifests"][0] = f"sha256:{'0' * 64}"
        path.write_text(json.dumps(binding))
    elif tampering == "source_config":
        original_manifest = json.loads(fixture.manifests[fixture.source.reference.rsplit("@", 1)[1]])
        (prepared.layout / "blobs/sha256" / original_manifest["config"]["digest"].removeprefix("sha256:")).write_bytes(
            b"changed"
        )
    elif tampering == "config":
        manifest = json.loads(
            (prepared.layout / "blobs/sha256" / prepared.manifest_digest.removeprefix("sha256:")).read_bytes()
        )
        (prepared.layout / "blobs/sha256" / manifest["config"]["digest"].removeprefix("sha256:")).write_bytes(b"changed")
    else:
        index = json.loads((prepared.layout / "index.json").read_bytes())
        index["manifests"][0]["digest"] = f"sha256:{'0' * 64}"
        (prepared.layout / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        ImageCache(cache).prepare(fixture.source)


def test_concurrent_registry_conversion_publishes_one_complete_binding(tmp_path, converted_registry_image, monkeypatch):
    fixture = converted_registry_image
    image_tool = subprocess.run
    barrier = threading.Barrier(2)

    def concurrent(args, **kwargs):
        result = image_tool(args, **kwargs)
        if "copy" in args:
            barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(subprocess, "run", concurrent)
    cache = tmp_path / "cache"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ImageCache(cache).prepare, fixture.source) for _ in range(2)]
        prepared = [future.result() for future in futures]
    assert prepared[0].layout == prepared[1].layout
    assert ImageCache(cache).prepare(fixture.source).manifest_digest == fixture.image.manifest_digest


def test_registry_binding_cannot_select_an_unrelated_valid_image(tmp_path, converted_registry_image, oci_image):
    fixture = converted_registry_image
    cache = tmp_path / "cache"
    ImageCache(cache).prepare(fixture.source)
    unrelated = oci_image((), {"WorkingDir": "/elsewhere"})
    destination = cache / unrelated.manifest_digest.replace(":", "-")
    shutil.copytree(unrelated.layout, destination)
    original = fixture.source.reference.rsplit("@", 1)[1]
    (destination / "blobs/sha256" / original.removeprefix("sha256:")).write_bytes(fixture.manifests[original])
    original_manifest = json.loads(fixture.manifests[original])
    (destination / "blobs/sha256" / original_manifest["config"]["digest"].removeprefix("sha256:")).write_bytes(
        fixture.source_configuration
    )
    binding_path = cache / "registry" / f"{original.removeprefix('sha256:')}.json"
    binding = json.loads(binding_path.read_bytes())
    binding["prepared_manifest_digest"] = unrelated.manifest_digest
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="payload differs"):
        ImageCache(cache).prepare(fixture.source)


@pytest.mark.parametrize("mismatch", ["source_digest", "layer_order", "configuration_value", "configuration_type"])
def test_registry_conversion_requires_the_pinned_source_payload(tmp_path, converted_registry_image, mismatch):
    fixture = converted_registry_image
    if mismatch == "source_digest":
        fixture.manifests[fixture.source.reference.rsplit("@", 1)[1]] = b"{}"
    else:
        image = fixture.image
        blobs = image.layout / "blobs/sha256"
        manifest = json.loads((blobs / image.manifest_digest.removeprefix("sha256:")).read_bytes())
        if mismatch == "layer_order":
            manifest["layers"].reverse()
        else:
            configuration = json.loads((blobs / manifest["config"]["digest"].removeprefix("sha256:")).read_bytes())
            if mismatch == "configuration_value":
                configuration["config"]["WorkingDir"] = "/different"
            else:
                configuration["config"]["AttachStdin"] = 0
            raw = json.dumps(configuration).encode()
            digest = hashlib.sha256(raw).hexdigest()
            (blobs / digest).write_bytes(raw)
            manifest["config"].update(digest=f"sha256:{digest}", size=len(raw))
        raw = json.dumps(manifest).encode()
        digest = hashlib.sha256(raw).hexdigest()
        (blobs / digest).write_bytes(raw)
        index = json.loads((image.layout / "index.json").read_bytes())
        index["manifests"][0].update(digest=f"sha256:{digest}", size=len(raw))
        (image.layout / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError):
        ImageCache(tmp_path / "cache").prepare(fixture.source)


def test_converted_registry_binding_does_not_bypass_signature_policy(tmp_path, converted_registry_image, monkeypatch):
    fixture = converted_registry_image
    cache = tmp_path / "cache"
    ImageCache(cache).prepare(fixture.source)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"default": [{"type": "reject"}]}))

    def rejected(args, **_kwargs):
        return subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"signature verification rejected")

    monkeypatch.setattr(subprocess, "run", rejected)
    with pytest.raises(RuntimeError, match="signature verification rejected"):
        ImageCache(cache, policy=policy).prepare(fixture.source)


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
            config = json.dumps(
                {
                    "os": "linux",
                    "architecture": "amd64",
                    "config": {"WorkingDir": "/task workspace"},
                    "payload": builds[-1].decode(),
                }
            ).encode()
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
    cache = ImageCache(tmp_path / "cache")
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
    assert metadata["cwd"] == "/task workspace"
    manifest = json.loads((second.layout / "blobs/sha256" / second.manifest_digest.split(":")[1]).read_bytes())
    config_path = second.layout / "blobs/sha256" / manifest["config"]["digest"].split(":")[1]
    original_config = config_path.read_bytes()
    config_path.write_bytes(b"{}")
    with pytest.raises(ValueError):
        stage_qemu_image(second, assets, tmp_path / "bundles")
    config_path.write_bytes(original_config)
    assert metadata["dockerfile_sha256"] == hashlib.sha256((contexts[1] / "Dockerfile").read_bytes()).hexdigest()
    (contexts[1] / "payload").write_bytes(b"changed")
    changed = cache.prepare(DockerfileSource(contexts[1], contexts[1] / "Dockerfile"))
    assert builds == [b"first", b"changed"]
    assert changed.manifest_digest != first.manifest_digest
