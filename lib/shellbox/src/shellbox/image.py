# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve registry images and Dockerfiles into local, pinned OCI layouts."""

import errno
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
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

OCI_TAG = "image"
SHA256_PATTERN = r"sha256:[0-9a-f]{64}"
Sha256Digest = Annotated[str, Field(pattern=f"^{SHA256_PATTERN}$")]


class _RegistryBinding(BaseModel):
    """A source-manifest proof for a separately addressed prepared layout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prepared_manifest_digest: Sha256Digest
    source_manifests: tuple[Sha256Digest, ...] = Field(min_length=1)


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


def _digest_hex(digest: str) -> str:
    if re.fullmatch(SHA256_PATTERN, digest) is None:
        raise ValueError("Only sha256 OCI digests are supported")
    return digest.removeprefix("sha256:")


def _oci_blob(layout: Path, digest: str) -> Path:
    path = layout / "blobs" / "sha256" / _digest_hex(digest)
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
    """Reuse prepared images in memory and verified pinned layouts on disk.

    Without a signature policy, mutable registry references are process-lifetime snapshots.
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


def _run(*args: str) -> bytes:
    result = subprocess.run(args, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{' '.join(args[:2])} failed: {result.stderr.decode(errors='replace')[-2000:]}")
    return result.stdout


def _layer_digests(manifest: dict[str, Any]) -> tuple[str, ...]:
    return tuple(layer["digest"] for layer in manifest["layers"])


def _configuration_json(configuration: dict[str, Any]) -> str:
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _registry_manifest_chain(
    repository: str, digest: str, configuration_digest: str, layers: tuple[str, ...], authfile: Path | None
) -> tuple[bytes, ...] | None:
    _digest_hex(digest)
    args = ["skopeo", "inspect", "--raw"]
    if authfile is not None:
        args.extend(("--authfile", str(authfile)))
    raw = _run(*args, f"docker://{repository}@{digest}")
    if f"sha256:{hashlib.sha256(raw).hexdigest()}" != digest:
        raise ValueError("Registry manifest differs from the source pin")
    manifest = json.loads(raw)
    if "manifests" not in manifest:
        if manifest["config"]["digest"] == configuration_digest and _layer_digests(manifest) == layers:
            return (raw,)
        return None
    for descriptor in manifest["manifests"]:
        platform = descriptor.get("platform", {})
        if platform.get("os") not in (None, "linux") or platform.get("architecture") not in (None, "amd64"):
            continue
        chain = _registry_manifest_chain(repository, descriptor["digest"], configuration_digest, layers, authfile)
        if chain is not None:
            return (raw, *chain)
    return None


def _require_registry_binding(layout: Path, binding: _RegistryBinding, source_digest: str) -> None:
    if binding.source_manifests[0] != source_digest:
        raise ValueError("Cached registry binding belongs to another source pin")
    for parent, child in pairwise(binding.source_manifests):
        manifest = _oci_json_blob(layout, parent)
        if child not in {descriptor["digest"] for descriptor in manifest["manifests"]}:
            raise ValueError("Cached registry manifest chain differs from its source index")
    original = _oci_json_blob(layout, binding.source_manifests[-1])
    prepared = _oci_json_blob(layout, binding.prepared_manifest_digest)
    original_configuration = _oci_json_blob(layout, original["config"]["digest"])
    prepared_configuration = _oci_json_blob(layout, prepared["config"]["digest"])
    if _configuration_json(original_configuration) != _configuration_json(prepared_configuration) or _layer_digests(
        original
    ) != _layer_digests(prepared):
        raise ValueError("Cached registry image payload differs from its source manifest")
    _manifest_digest(layout, expected=binding.prepared_manifest_digest)


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
    verified locally. Converted manifests and source indexes retain a digest-verified
    source chain with identical configuration and ordered layers. A signature policy
    requires a fresh verified copy instead.
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
        binding_path = cache / "registry" / f"{_digest_hex(pinned_digest)}.json"
        if binding_path.exists():
            binding = _RegistryBinding.model_validate_json(binding_path.read_bytes())
            destination = cache / binding.prepared_manifest_digest.replace(":", "-")
            _require_registry_binding(destination, binding, pinned_digest)
            return PreparedImage(destination, binding.prepared_manifest_digest, source)
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
        source_blobs: tuple[bytes, ...] = ()
        binding = None
        if isinstance(source, RegistryImage) and pinned_digest is not None and pinned_digest != digest:
            repository = source.reference.rsplit("@", 1)[0]
            config_args = [
                "skopeo",
                "--override-os",
                "linux",
                "--override-arch",
                "amd64",
                "inspect",
                "--raw",
                "--config",
            ]
            if authfile is not None:
                config_args.extend(("--authfile", str(authfile)))
            source_configuration = _run(*config_args, source_ref)
            prepared = _oci_json_blob(layout, digest)
            if _configuration_json(json.loads(source_configuration)) != _configuration_json(
                _oci_json_blob(layout, prepared["config"]["digest"])
            ):
                raise ValueError("Prepared image configuration differs from its pinned registry source")
            configuration_digest = f"sha256:{hashlib.sha256(source_configuration).hexdigest()}"
            chain = _registry_manifest_chain(
                repository, pinned_digest, configuration_digest, _layer_digests(prepared), authfile
            )
            if chain is None:
                raise ValueError("Prepared image payload differs from its pinned registry source")
            source_blobs = (*chain, source_configuration)
            binding = _RegistryBinding(
                prepared_manifest_digest=digest,
                source_manifests=tuple(f"sha256:{hashlib.sha256(raw).hexdigest()}" for raw in chain),
            )
        destination = cache / digest.replace(":", "-")
        if not destination.exists():
            try:
                layout.rename(destination)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
        _manifest_digest(destination, expected=digest)
        if binding is not None:
            assert pinned_digest is not None
            for raw in source_blobs:
                blob_digest = hashlib.sha256(raw).hexdigest()
                pending = Path(temporary) / blob_digest
                pending.write_bytes(raw)
                try:
                    os.link(pending, destination / "blobs" / "sha256" / blob_digest)
                except FileExistsError:
                    pass
            _require_registry_binding(destination, binding, pinned_digest)
            binding_path = cache / "registry" / f"{_digest_hex(pinned_digest)}.json"
            binding_path.parent.mkdir(parents=True, exist_ok=True)
            pending_binding = Path(temporary) / "binding.json"
            pending_binding.write_text(binding.model_dump_json() + "\n")
            pending_binding.replace(binding_path)
        return PreparedImage(destination, digest, source)
