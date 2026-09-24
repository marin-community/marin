# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage prepared OCI images as cached QEMU guest bundles."""

import fcntl
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from harbor_qemu.backends.qemu.bundle import guest_code_id, stage_bundle
from harbor_qemu.image import OCI_TAG, DockerfileSource, PreparedImage, RegistryImage


@dataclass(frozen=True)
class QemuAssets:
    """Host assets needed to turn an OCI image into a QEMU guest bundle."""

    qemu: Path
    kernel: Path
    busybox: Path
    firmware: Path
    libraries: Path
    umoci: Path
    disk_size_mb: int
    runtime_id: str


def stage_qemu_image(image: PreparedImage, assets: QemuAssets, cache: Path) -> Path:
    """Stage a prepared image as a reusable QEMU bundle."""
    source_id = (
        image.source.reference
        if isinstance(image.source, RegistryImage)
        else hashlib.sha256(image.source.dockerfile.read_bytes()).hexdigest()
    )
    key = hashlib.sha256(
        f"{image.manifest_digest}:{assets.runtime_id}:{assets.disk_size_mb}:{source_id}:{guest_code_id()}".encode()
    ).hexdigest()
    bundle = cache / key
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / f".{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if bundle.is_dir():
            metadata = json.loads((bundle / "image.json").read_text())
            if metadata["manifest_digest"] != image.manifest_digest:
                raise ValueError("Cached QEMU bundle image digest differs from prepared image")
            return bundle
        with tempfile.TemporaryDirectory(prefix="bundle-", dir=cache) as temporary:
            staged = Path(temporary) / "bundle"
            stage_bundle(
                assets.qemu,
                assets.kernel,
                assets.busybox,
                assets.firmware,
                assets.libraries,
                staged,
                oci_layout=image.layout,
                oci_tag=OCI_TAG,
                umoci=assets.umoci,
                disk_size_mb=assets.disk_size_mb,
                task_dockerfile=image.source.dockerfile if isinstance(image.source, DockerfileSource) else None,
                image_reference=image.source.reference if isinstance(image.source, RegistryImage) else None,
            )
            metadata = json.loads((staged / "image.json").read_text())
            if metadata["manifest_digest"] != image.manifest_digest:
                raise ValueError("QEMU bundle image digest differs from prepared image")
            staged.rename(bundle)
    return bundle
