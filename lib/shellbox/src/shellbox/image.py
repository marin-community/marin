# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve registry images and Dockerfiles into local, pinned OCI layouts."""

import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from functools import cache
from pathlib import Path

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


def _source_key(source: RegistryImage | DockerfileSource) -> str:
    if isinstance(source, RegistryImage):
        return f"registry:{source.reference}"
    context = source.context.resolve()
    dockerfile = source.dockerfile.resolve()
    digest = hashlib.sha256()
    digest.update(f"{context}\0{dockerfile}\0".encode())
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

    Registry references are snapshots for the lifetime of this cache. Dockerfile
    contexts are fingerprinted on each call so edits cause a new build.
    """

    def __init__(self, directory: Path, *, skopeo: Path, authfile: Path | None = None, policy: Path | None = None):
        self.directory = directory
        self.skopeo = skopeo
        self.authfile = authfile
        self.policy = policy
        self._images: dict[str, PreparedImage] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def prepare(self, source: RegistryImage | DockerfileSource) -> PreparedImage:
        """Prepare a source once per cache key, including concurrent callers."""
        key = _source_key(source)
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            image = self._images.get(key)
            if image is not None:
                return image
            image = prepare_image(
                source,
                self.directory,
                skopeo=self.skopeo,
                authfile=self.authfile,
                policy=self.policy,
            )
            self._images[key] = image
            return image


@cache
def process_image_cache(directory: Path, skopeo: Path, authfile: Path | None, policy: Path | None) -> ImageCache:
    """Share preparation across machine factories created by Harbor trials."""
    return ImageCache(directory.resolve(), skopeo=skopeo, authfile=authfile, policy=policy)


def _run(*args: str) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args[:2])} failed: {result.stderr[-2000:]}")


def _manifest_digest(layout: Path) -> str:
    index = json.loads((layout / "index.json").read_text())
    matching = [
        entry
        for entry in index["manifests"]
        if entry.get("annotations", {}).get("org.opencontainers.image.ref.name") == OCI_TAG
    ]
    if len(matching) != 1:
        raise ValueError("OCI layout must contain one image tag")
    digest = matching[0]["digest"]
    algorithm, value = digest.split(":", 1)
    if algorithm != "sha256" or len(value) != 64:
        raise ValueError(f"Unsupported image digest: {digest}")
    manifest_path = layout / "blobs" / algorithm / value
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != value:
        raise ValueError("OCI manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    config_algorithm, config_hash = manifest["config"]["digest"].split(":", 1)
    config = json.loads((layout / "blobs" / config_algorithm / config_hash).read_text())
    if config["os"] != "linux" or config["architecture"] != "amd64":
        raise ValueError("Only Linux amd64 images are supported")
    return digest


def prepare_image(
    source: RegistryImage | DockerfileSource,
    cache: Path,
    *,
    skopeo: Path,
    authfile: Path | None = None,
    policy: Path | None = None,
) -> PreparedImage:
    """Copy or build an image and retain its selected platform by manifest digest.

    The default Skopeo policy accepts unsigned images. TLS verification remains
    enabled. Pass a policy file when signature verification is required.
    """
    cache.mkdir(parents=True, exist_ok=True)
    if isinstance(source, RegistryImage) and ("://" in source.reference or not source.reference.strip()):
        raise ValueError("RegistryImage requires a container image reference without a URL scheme")
    with tempfile.TemporaryDirectory(prefix="image-", dir=cache) as temporary:
        layout = Path(temporary) / "oci"
        skopeo_args = [str(skopeo)]
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
        if destination.exists():
            if _manifest_digest(destination) != digest:
                raise ValueError("Cached OCI image digest differs from prepared image")
        else:
            try:
                layout.rename(destination)
            except FileExistsError:
                if _manifest_digest(destination) != digest:
                    raise ValueError("Cached OCI image digest differs from prepared image") from None
        return PreparedImage(destination, digest, source)


def load_docker_image(image: PreparedImage, *, skopeo: Path, policy: Path | None = None) -> str:
    """Load a prepared OCI layout into Docker under a digest-derived local tag."""
    tag = f"harbor-prepared:sha256-{image.manifest_digest.split(':', 1)[1]}"
    lock_path = image.layout.parent / f".{image.manifest_digest.replace(':', '-')}.docker.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inspect = subprocess.run(("docker", "image", "inspect", tag), capture_output=True)
        if inspect.returncode:
            args = [str(skopeo)]
            args.extend(("--policy", str(policy)) if policy is not None else ("--insecure-policy",))
            _run(*args, "copy", f"oci:{image.layout}:{OCI_TAG}", f"docker-daemon:{tag}")
    return tag
