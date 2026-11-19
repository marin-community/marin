# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline metadata checks shared by release promotion and serving."""

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

GPU_RELEASE_REPOSITORY = "marin-community/vllm"


@dataclass(frozen=True)
class VllmGpuWheel:
    """One architecture's wheel from a promoted vLLM GPU release."""

    architecture: str
    sm_targets: tuple[str, ...]
    url: str
    sha256: str


@dataclass(frozen=True)
class VllmGpuRelease:
    """Artifacts and ABI for a promoted vLLM GPU release."""

    release_tag: str
    source_commit: str
    version: str
    torch_backend: str
    torch_version: str
    wheels: tuple[VllmGpuWheel, ...]


def gpu_release_url_prefix(release_tag: str) -> str:
    return f"https://github.com/{GPU_RELEASE_REPOSITORY}/releases/download/{release_tag}/"


def validate_vllm_gpu_release(release: VllmGpuRelease, path: Path) -> None:
    """Check a release descriptor's identities, wheel URLs and CUDA metadata."""
    source_commit = release.source_commit
    backend = release.torch_backend
    torch_version = release.torch_version
    version = release.version
    release_tag = release.release_tag
    wheels = release.wheels
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError(f"{path}: expected a full vLLM source commit, found {source_commit!r}")
    if not re.fullmatch(r"cu[0-9]+", backend):
        raise ValueError(f"{path}: expected a CUDA torch backend, found {backend!r}")
    if not torch_version.endswith(f"+{backend}"):
        raise ValueError(f"{path}: torch version {torch_version!r} does not match {backend}")
    architectures = tuple(wheel.architecture for wheel in wheels)
    if not architectures or len(set(architectures)) != len(architectures):
        raise ValueError(f"{path}: expected one or more unique wheel architectures, found {architectures!r}")
    prefix = gpu_release_url_prefix(release_tag)
    filename_prefix = f"vllm-{quote(version, safe='')}-"
    urls = tuple(wheel.url for wheel in wheels)
    if len(set(urls)) != len(urls):
        raise ValueError(f"{path}: expected a unique asset URL for each wheel")
    for wheel in wheels:
        architecture, url = wheel.architecture, wheel.url
        if not wheel.sm_targets:
            raise ValueError(f"{path}: {architecture} must declare at least one SM target")
        if not url.startswith(prefix) or "#" in url:
            raise ValueError(f"{path}: {architecture} wheel URL is not an asset from {release_tag}")
        if f"/{filename_prefix}" not in url or not url.endswith(f"_{architecture}.whl"):
            raise ValueError(f"{path}: {architecture} wheel URL does not match vLLM version {version}")
        if re.fullmatch(r"[0-9a-f]{64}", wheel.sha256) is None:
            raise ValueError(f"{path}: {architecture} wheel has invalid SHA-256 {wheel.sha256!r}")
