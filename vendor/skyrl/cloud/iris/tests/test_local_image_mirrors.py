import hashlib
import json
import shutil

import pytest
from rigging.filesystem.cluster_config import BucketSpec, StoreType
import rigging.filesystem.cluster_config as cluster_config
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from rigging.filesystem import transfer
from rigging.fsutil.verified_copy import COMPLETION_MANIFEST
from shellbox.image import ImageCache, RegistryImage

from marinskyrl.local_images import LocalImageMirror, LocalMachineSettings, cached_mirror_image, local_machine_factory
import marinskyrl.local_images as local_images
import shellbox.image as image_runtime


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    layout = tmp_path / "mirror"
    blobs = layout / "blobs" / "sha256"
    blobs.mkdir(parents=True)

    def blob(content):
        digest = hashlib.sha256(content).hexdigest()
        (blobs / digest).write_bytes(content)
        return "sha256:" + digest

    config = blob(json.dumps({"os": "linux", "architecture": "amd64"}).encode())
    layer = blob(b"immutable source layer")
    manifest = blob(
        json.dumps({"schemaVersion": 2, "config": {"digest": config}, "layers": [{"digest": layer}]}).encode()
    )
    (layout / "index.json").write_text(
        json.dumps({"manifests": [{"digest": manifest, "annotations": {"org.opencontainers.image.ref.name": "image"}}]})
    )
    (layout / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    # Real regional producers append this marker; accepting their prefix regresses without it.
    (layout / COMPLETION_MANIFEST).write_text('{"completed":true}')
    settings = LocalImageMirror(image="registry.fixture/private/image@" + manifest, uri="s3://mirror-bucket/oci")
    monkeypatch.setattr(
        cluster_config,
        "data_buckets",
        lambda: {"mirror-bucket": BucketSpec("mirror-bucket", StoreType.COREWEAVE, "US-EAST-02A")},
    )
    original_filesystem_for = transfer.filesystem_for

    def mirror_filesystem(url, *args, **kwargs):
        return original_filesystem_for(str(layout) if url == settings.uri else url, *args, **kwargs)

    # The external S3 filesystem is local; transfer and cryptographic verification stay real.
    monkeypatch.setattr(transfer, "filesystem_for", mirror_filesystem)

    def registry_unavailable(*args):
        pytest.fail("The registry is unavailable; the pinned OCI mirror must serve the image")

    monkeypatch.setattr(image_runtime, "_run", registry_unavailable)
    return settings, layout


def test_regional_mirror_preserves_registry_identity_and_reuses_verified_cache(tmp_path, mirror):
    settings, layout = mirror
    cache = ImageCache(tmp_path / "cache")
    prepared = cached_mirror_image(cache, settings, region="US-EAST-02A")
    assert prepared.source == RegistryImage(settings.image)
    assert prepared.manifest_digest == settings.image.rsplit("@", 1)[1]
    expected = {str(p.relative_to(layout)): p.read_bytes() for p in layout.rglob("*") if p.is_file()}
    assert {
        str(p.relative_to(prepared.layout)): p.read_bytes() for p in prepared.layout.rglob("*") if p.is_file()
    } == expected
    shutil.rmtree(layout)
    assert cached_mirror_image(cache, settings, region="US-EAST-02A") == prepared


def test_corrupt_mirror_never_publishes_a_ready_image(tmp_path, mirror):
    settings, layout = mirror
    manifest = json.loads((layout / "blobs" / "sha256" / settings.image.rsplit(":", 1)[1]).read_text())
    layer = manifest["layers"][0]["digest"]
    (layout / "blobs" / "sha256" / layer.removeprefix("sha256:")).write_bytes(b"changed image layer")
    cache = ImageCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        cached_mirror_image(cache, settings, region="US-EAST-02A")
    assert not (cache.directory / settings.image.rsplit("@", 1)[1].replace(":", "-")).exists()


def test_cross_region_mirror_is_rejected_before_transfer(tmp_path, mirror):
    settings, _ = mirror
    cache = ImageCache(tmp_path / "cache")
    with pytest.raises(ValueError):
        cached_mirror_image(cache, settings, region="US-EAST-08A")
    assert not cache.directory.exists()


@pytest.mark.parametrize("worker_region", ["US-EAST-08A", None])
def test_mirror_configuration_cannot_override_actual_worker_placement(monkeypatch, mirror, worker_region):
    settings, _ = mirror
    info = JobInfo(task_id=JobName.from_wire("/fixture/worker/0"), worker_region=worker_region)
    monkeypatch.setattr(local_images, "get_job_info", lambda: info)
    with pytest.raises(ValueError):
        local_machine_factory(LocalMachineSettings(region="US-EAST-02A", image_mirrors=(settings,)))
