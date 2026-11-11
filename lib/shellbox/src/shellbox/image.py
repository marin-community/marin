# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve registry images and Dockerfiles into local, pinned OCI layouts."""

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

OCI_TAG = "image"


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
    """One Linux amd64 image resolved to a content-addressed OCI layout."""

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
    """

    def __init__(self, directory: Path, *, authfile: Path | None = None, policy: Path | None = None):
        self.directory = directory
        self.authfile = authfile
        self.policy = policy
        self._images: dict[str, PreparedImage] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def prepare(self, source: RegistryImage | DockerfileSource) -> PreparedImage:
        """Serialize preparation per source, reusing content unless a policy requires verification."""
        key = image_source_key(source)
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            image = self._images.get(key)
            if image is not None and self.policy is None:
                _manifest_digest(image.layout, expected=image.manifest_digest)
                return replace(image, source=source)
            image = prepare_image(
                source,
                self.directory,
                authfile=self.authfile,
                policy=self.policy,
            )
            self._images[key] = image
            return image


@cache
def process_image_cache(directory: Path, authfile: Path | None, policy: Path | None) -> ImageCache:
    return ImageCache(directory.resolve(), authfile=authfile, policy=policy)


def _run(*args: str) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args[:2])} failed: {result.stderr[-2000:]}")


def _manifest_digest(layout: Path, *, expected: str | None = None) -> str:
    digest = _tagged_manifest_digest(layout)
    if expected is not None and digest != expected:
        raise ValueError("Prepared image tag no longer selects its pinned manifest")
    manifest = _oci_json_blob(layout, digest)
    config = _oci_json_blob(layout, manifest["config"]["digest"])
    if config["os"] != "linux" or config["architecture"] != "amd64":
        raise ValueError("Only Linux amd64 images are supported")
    for layer in manifest["layers"]:
        _oci_blob(layout, layer["digest"])
    return digest


def prepare_image(
    source: RegistryImage | DockerfileSource,
    cache: Path,
    *,
    authfile: Path | None = None,
    policy: Path | None = None,
) -> PreparedImage:
    """Copy or build an image and retain its selected platform by manifest digest.

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
        destination = cache / pinned_digest.replace(":", "-")
        if destination.exists():
            _manifest_digest(destination, expected=pinned_digest)
            return PreparedImage(destination, pinned_digest, source)
    with tempfile.TemporaryDirectory(prefix="image-", dir=cache) as temporary:
        layout = Path(temporary) / "oci"
        skopeo_args = ["skopeo"]
        skopeo_args.extend(("--policy", str(policy)) if policy is not None else ("--insecure-policy",))
        skopeo_args.extend(("--override-os", "linux", "--override-arch", "amd64", "copy"))
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
                    "linux/amd64",
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
            _run(*skopeo_args, source_ref, f"oci:{layout}:{OCI_TAG}")
        digest = _manifest_digest(layout)
        destination = cache / digest.replace(":", "-")
        if not destination.exists():
            try:
                layout.rename(destination)
            except FileExistsError:
                pass  # Another process published this content-addressed layout.
        _manifest_digest(destination, expected=digest)
        return PreparedImage(destination, digest, source)
