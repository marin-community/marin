# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve registry images and Dockerfiles into local, pinned OCI layouts."""

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any

OCI_TAG = "image"


class ImageArchitecture(StrEnum):
    """The OCI ``architecture`` of a Linux image; Shellbox prepares no other operating system."""

    AMD64 = "amd64"
    ARM64 = "arm64"

    @property
    def platform(self) -> str:
        """The OCI platform string, such as ``linux/arm64``, that registries and builders select by."""
        return f"linux/{self.value}"


HOST_MACHINE_ARCHITECTURES = {
    "x86_64": ImageArchitecture.AMD64,
    "aarch64": ImageArchitecture.ARM64,
    "arm64": ImageArchitecture.ARM64,
}
"""``platform.machine()`` values mapped to the native architecture: Linux reports x86_64 or aarch64, macOS arm64."""


class ImageArchitectureMismatch(ValueError):
    """An image's architecture differs from the one its consumer runs."""


def host_architecture() -> ImageArchitecture:
    """Return the image architecture this host runs natively, failing on an unknown machine type."""
    machine = platform.machine()
    try:
        return HOST_MACHINE_ARCHITECTURES[machine]
    except KeyError:
        raise ValueError(f"Shellbox does not support host machine type {machine!r}") from None


@dataclass(frozen=True)
class RegistryImage:
    """An OCI registry reference such as ghcr.io/org/image:tag."""

    reference: str


@dataclass(frozen=True)
class DockerfileSource:
    """A Dockerfile and its build context."""

    context: Path
    dockerfile: Path


@dataclass(frozen=True)
class PreparedImage:
    """One Linux image resolved to a content-addressed OCI layout of a single platform."""

    layout: Path
    manifest_digest: str
    source: RegistryImage | DockerfileSource


def _oci_blob(layout: Path, digest: str) -> Path:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("Only sha256 OCI digests are supported")
    path = layout / "blobs" / "sha256" / digest.removeprefix("sha256:")
    with path.open("rb") as blob:
        actual = hashlib.file_digest(blob, "sha256").hexdigest()
    if f"sha256:{actual}" != digest:
        raise ValueError("Prepared image blob digest mismatch")
    return path


def _oci_json_blob(layout: Path, digest: str) -> dict[str, Any]:
    return json.loads(_oci_blob(layout, digest).read_bytes())


def _tagged_manifest_digest(layout: Path) -> str:
    index = json.loads((layout / "index.json").read_text())
    matching = [
        entry
        for entry in index["manifests"]
        if entry.get("annotations", {}).get("org.opencontainers.image.ref.name") == OCI_TAG
    ]
    if len(matching) != 1:
        raise ValueError("OCI layout must contain one image tag")
    return matching[0]["digest"]


def prepared_image_config(image: PreparedImage) -> dict[str, Any]:
    """Read the digest-verified image config, requiring its OCI tag to remain pinned."""
    if _tagged_manifest_digest(image.layout) != image.manifest_digest:
        raise ValueError("Prepared image tag no longer selects its pinned manifest")
    manifest = _oci_json_blob(image.layout, image.manifest_digest)
    return _oci_json_blob(image.layout, manifest["config"]["digest"])


def check_image_architecture(config: dict[str, Any], architecture: ImageArchitecture, image: str) -> None:
    """Raise ``ImageArchitectureMismatch`` unless an OCI image config is Linux on ``architecture``."""
    actual = f"{config.get('os')}/{config.get('architecture')}"
    if actual != architecture.platform:
        raise ImageArchitectureMismatch(
            f"Image {image} is {actual}, but this consumer runs only {architecture.platform}"
        )


def prepared_image_workdir(image: PreparedImage) -> str:
    """Read the selected image's initial working directory from its verified OCI config."""
    config = prepared_image_config(image)
    return config.get("config", {}).get("WorkingDir") or "/"


def image_source_key(source: RegistryImage | DockerfileSource) -> str:
    """Identify a registry reference or the current contents of a build context."""
    if isinstance(source, RegistryImage):
        return f"registry:{source.reference}"
    context = source.context.resolve()
    dockerfile = source.dockerfile.resolve()
    digest = hashlib.sha256()
    dockerfile_name = str(dockerfile.relative_to(context)) if dockerfile.is_relative_to(context) else "<external>"
    digest.update(f"{dockerfile_name}\0".encode())
    for path in sorted(context.rglob("*")):
        relative = path.relative_to(context)
        digest.update(f"{relative}\0".encode())
        stat = path.lstat()
        digest.update(f"{stat.st_mode}\0".encode())
        if path.is_symlink():
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            digest.update(path.read_bytes())
    if not dockerfile.is_relative_to(context):
        digest.update(dockerfile.read_bytes())
    return f"dockerfile:{digest.hexdigest()}"


class ImageCache:
    """Reuse prepared sources within one Python process.

    Without a signature policy, registry references are process-lifetime snapshots.
    Dockerfile contexts are fingerprinted on each call so edits cause a new build.
    One source prepared for two architectures yields two entries.
    """

    def __init__(self, directory: Path, *, authfile: Path | None = None, policy: Path | None = None):
        self.directory = directory
        self.authfile = authfile
        self.policy = policy
        self._images: dict[str, PreparedImage] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def prepare(self, source: RegistryImage | DockerfileSource, architecture: ImageArchitecture) -> PreparedImage:
        """Serialize preparation per source and architecture, reusing content unless a policy requires verification."""
        key = f"{architecture}:{image_source_key(source)}"
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            image = self._images.get(key)
            if image is not None and self.policy is None:
                _manifest_digest(image.layout, architecture, image.source, expected=image.manifest_digest)
                return replace(image, source=source)
            image = prepare_image(
                source,
                self.directory,
                architecture=architecture,
                authfile=self.authfile,
                policy=self.policy,
            )
            self._images[key] = image
            return image


@cache
def process_image_cache(directory: Path, authfile: Path | None, policy: Path | None) -> ImageCache:
    return ImageCache(directory.resolve(), authfile=authfile, policy=policy)


def _run(*args: str, context: str = "") -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args[:2])} failed{context}: {result.stderr[-2000:]}")


def source_description(source: RegistryImage | DockerfileSource) -> str:
    """Name an image source in error messages."""
    return source.reference if isinstance(source, RegistryImage) else f"built from {source.dockerfile}"


def _manifest_digest(
    layout: Path,
    architecture: ImageArchitecture,
    source: RegistryImage | DockerfileSource,
    *,
    expected: str | None = None,
) -> str:
    digest = _tagged_manifest_digest(layout)
    if expected is not None and digest != expected:
        raise ValueError("Prepared image tag no longer selects its pinned manifest")
    manifest = _oci_json_blob(layout, digest)
    config = _oci_json_blob(layout, manifest["config"]["digest"])
    check_image_architecture(config, architecture, source_description(source))
    for layer in manifest["layers"]:
        _oci_blob(layout, layer["digest"])
    return digest


def _index_alias(cache: Path, index_digest: str, architecture: ImageArchitecture) -> Path:
    """The link naming the layout an index digest selected for ``architecture``; an index has no layout of its own."""
    return cache / f"{index_digest.replace(':', '-')}-{architecture}"


def _cached_pin(
    cache: Path, pinned_digest: str, architecture: ImageArchitecture, source: RegistryImage
) -> PreparedImage | None:
    """Return the verified cached layout for a manifest pin, or for an index pin's recorded selection."""
    destination = cache / pinned_digest.replace(":", "-")
    if destination.exists():
        _manifest_digest(destination, architecture, source, expected=pinned_digest)
        return PreparedImage(destination, pinned_digest, source)
    alias = _index_alias(cache, pinned_digest, architecture)
    if not alias.exists():
        return None
    selected = "sha256:" + os.readlink(alias).removeprefix("sha256-")
    _manifest_digest(alias, architecture, source, expected=selected)
    return PreparedImage(cache / selected.replace(":", "-"), selected, source)


def _record_index_selection(cache: Path, index_digest: str, architecture: ImageArchitecture, layout: Path) -> None:
    try:
        _index_alias(cache, index_digest, architecture).symlink_to(layout.name)
    except FileExistsError:
        pass  # Another process recorded the same immutable selection.


def prepare_image(
    source: RegistryImage | DockerfileSource,
    cache: Path,
    *,
    architecture: ImageArchitecture,
    authfile: Path | None = None,
    policy: Path | None = None,
) -> PreparedImage:
    """Copy or build an image for ``linux/<architecture>`` and retain that platform by manifest digest.

    A reference to a multi-platform index selects the ``architecture`` manifest; a link named for
    the index digest and architecture lets later processes reuse that layout offline. A reference
    to a single-platform manifest of another architecture raises ``ImageArchitectureMismatch``,
    as does a cached layout of another architecture. A Dockerfile builds for ``architecture``;
    on a host of another architecture that needs the Docker builder's emulation.

    The default Skopeo policy accepts unsigned images. TLS verification remains
    enabled. Digest-pinned platform manifests already present in the cache are
    verified locally. A signature policy requires a fresh verified copy instead.
    """
    cache.mkdir(parents=True, exist_ok=True)
    if isinstance(source, RegistryImage) and ("://" in source.reference or not source.reference.strip()):
        raise ValueError("RegistryImage requires a container image reference without a URL scheme")
    pinned_digest = None
    if isinstance(source, RegistryImage):
        match = re.fullmatch(r".+@(sha256:[0-9a-f]{64})", source.reference)
        if match is not None:
            pinned_digest = match[1]
    if pinned_digest is not None and policy is None:
        cached = _cached_pin(cache, pinned_digest, architecture, source)
        if cached is not None:
            return cached
    with tempfile.TemporaryDirectory(prefix="image-", dir=cache) as temporary:
        layout = Path(temporary) / "oci"
        skopeo_args = ["skopeo"]
        skopeo_args.extend(("--policy", str(policy)) if policy is not None else ("--insecure-policy",))
        skopeo_args.extend(("--override-os", "linux", "--override-arch", str(architecture), "copy"))
        if authfile is not None:
            skopeo_args.extend(("--src-authfile", str(authfile)))
        if isinstance(source, RegistryImage):
            source_ref = f"docker://{source.reference}"
        else:
            tag = f"harbor-image-build:{uuid.uuid4().hex}"
            try:
                _run(
                    "docker",
                    "build",
                    "--platform",
                    architecture.platform,
                    "-f",
                    str(source.dockerfile),
                    "-t",
                    tag,
                    str(source.context),
                )
                source_ref = f"docker-daemon:{tag}"
                _run(*skopeo_args, source_ref, f"oci:{layout}:{OCI_TAG}")
            finally:
                subprocess.run(("docker", "image", "rm", tag), capture_output=True)
        if isinstance(source, RegistryImage):
            _run(
                *skopeo_args,
                source_ref,
                f"oci:{layout}:{OCI_TAG}",
                context=f" for {source.reference} on {architecture.platform}",
            )
        digest = _manifest_digest(layout, architecture, source)
        destination = cache / digest.replace(":", "-")
        if not destination.exists():
            try:
                layout.rename(destination)
            except FileExistsError:
                pass  # Another process published this content-addressed layout.
        _manifest_digest(destination, architecture, source, expected=digest)
        if pinned_digest is not None and digest != pinned_digest:
            _record_index_selection(cache, pinned_digest, architecture, destination)
        return PreparedImage(destination, digest, source)
