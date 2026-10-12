# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve registry images and Dockerfiles into local, pinned OCI layouts."""

import fcntl
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

    Mutable registry references are snapshots for the lifetime of this cache.
    Digest-pinned references also share their prepared layout across processes.
    Dockerfile contexts are fingerprinted on each call so edits cause a new build.
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
        key = image_source_key(source)
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            image = self._images.get(key)
            if image is not None:
                return replace(image, source=source)
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
    config_bytes = (layout / "blobs" / config_algorithm / config_hash).read_bytes()
    if config_algorithm != "sha256" or hashlib.sha256(config_bytes).hexdigest() != config_hash:
        raise ValueError("OCI configuration digest mismatch")
    config = json.loads(config_bytes)
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
    # Explicit signature policies must be evaluated by Skopeo rather than bypassed
    # by a persistent cache hit. Tags remain process-local snapshots.
    if isinstance(source, RegistryImage) and re.search(r"@sha256:[0-9a-f]{64}$", source.reference) and policy is None:
        key = hashlib.sha256(source.reference.encode()).hexdigest()
        reference = cache / f".registry-{key}.json"
        with (cache / f".registry-{key}.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if reference.exists():
                recorded = json.loads(reference.read_text())
                if recorded["source"] != source.reference:
                    raise ValueError("Cached registry reference differs from requested image")
                digest = recorded["manifest_digest"]
                if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
                    raise ValueError(f"Unsupported cached OCI digest: {digest}")
                layout = cache / digest.replace(":", "-")
                if _manifest_digest(layout) != digest:
                    raise ValueError("Cached registry layout differs from recorded OCI digest")
                return PreparedImage(layout, digest, source)
            image = _prepare_image(source, cache, skopeo=skopeo, authfile=authfile, policy=policy)
            # The selected OCI manifest can differ from the registry's pinned
            # manifest after platform selection and media-type conversion.
            with tempfile.TemporaryDirectory(prefix="reference-", dir=cache) as temporary:
                pending = Path(temporary) / "reference.json"
                pending.write_text(json.dumps({"source": source.reference, "manifest_digest": image.manifest_digest}))
                pending.replace(reference)
            return image
    return _prepare_image(source, cache, skopeo=skopeo, authfile=authfile, policy=policy)


def _prepare_image(
    source: RegistryImage | DockerfileSource,
    cache: Path,
    *,
    skopeo: Path,
    authfile: Path | None,
    policy: Path | None,
) -> PreparedImage:
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
    if _manifest_digest(image.layout) != image.manifest_digest:
        raise ValueError("Prepared OCI image digest differs from its cached manifest")
    algorithm, digest = image.manifest_digest.split(":", 1)
    manifest = json.loads((image.layout / "blobs" / algorithm / digest).read_bytes())
    config_digest = manifest["config"]["digest"]
    config_algorithm, config_hash = config_digest.split(":", 1)
    config_bytes = (image.layout / "blobs" / config_algorithm / config_hash).read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != config_hash:
        raise ValueError("Prepared OCI image configuration digest mismatch")
    config = json.loads(config_bytes)
    tag = f"harbor-prepared:sha256-{image.manifest_digest.split(':', 1)[1]}"
    lock_path = image.layout.parent / f".{image.manifest_digest.replace(':', '-')}.docker.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inspect_args = ("docker", "image", "inspect", tag)
        inspect = subprocess.run(inspect_args, capture_output=True, text=True)
        if inspect.returncode:
            args = [str(skopeo)]
            args.extend(("--policy", str(policy)) if policy is not None else ("--insecure-policy",))
            source = f"oci:{image.layout}:{OCI_TAG}"
            if any("zstd" in layer["mediaType"] for layer in manifest["layers"]):
                # Docker's manifest conversion rejects zstd before Skopeo can recompress it.
                # Keep the pinned cache unchanged and bridge through a gzip OCI transport.
                with tempfile.TemporaryDirectory(prefix="docker-", dir=image.layout.parent) as temporary:
                    compressed = f"oci:{Path(temporary) / 'oci'}:{OCI_TAG}"
                    _run(
                        *args, "copy", "--dest-compress-format", "gzip", "--dest-compress-level", "1", source, compressed
                    )
                    _run(*args, "copy", compressed, f"docker-daemon:{tag}")
            else:
                _run(*args, "copy", source, f"docker-daemon:{tag}")
            inspect = subprocess.run(inspect_args, capture_output=True, text=True, check=True)
        loaded = json.loads(inspect.stdout)[0]
        # Docker normalizes configuration JSON, so its image ID can change even when
        # the uncompressed layers and declared runtime configuration are unchanged.
        expected = (config["os"], config["architecture"], config["rootfs"]["type"], config["rootfs"]["diff_ids"])
        actual = (loaded["Os"], loaded["Architecture"], loaded["RootFS"]["Type"], loaded["RootFS"]["Layers"])
        if actual != expected or not _runtime_config_matches(config.get("config", {}), loaded["Config"]):
            raise ValueError(f"Loaded Docker image {tag} differs from its prepared OCI image")
    return tag


def _runtime_config_matches(declared: dict, loaded: dict) -> bool:
    # Docker fills omitted nested defaults, notably zero-valued Healthcheck fields.
    for key, value in declared.items():
        actual = loaded.get(key)
        if isinstance(value, dict):
            if not isinstance(actual, dict) or not _runtime_config_matches(value, actual):
                return False
        elif actual != value:
            return False
    return True
