# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select and describe verified vLLM GPU release wheels."""

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import draccus

from marin.external_dependencies import VllmGpuRelease, VllmGpuWheel


def load_vllm_gpu_release(path: Path | None) -> VllmGpuRelease | None:
    """Read a per-deployment promoted-release pin without changing shared defaults."""
    if path is None:
        return None
    record = json.loads(path.read_text())
    if "release" in record:
        release = record["release"]
        if (
            release["repository"] != "marin-community/vllm"
            or release["status"] != "released"
            or record["validation"]["status"] != "passed"
        ):
            raise ValueError("A GPU override requires a validated published Marin release")
        tag = release["tag"]
        record = {
            "release_tag": tag,
            "source_commit": record["source"]["fork_commit"],
            "version": record["distribution"]["version"],
            "torch_backend": record["abi"]["cuda_variant"],
            "torch_version": record["abi"]["torch_version"],
            "wheels": [
                {
                    "architecture": platform["architecture"],
                    "sm_targets": platform["sm_targets"],
                    "url": (
                        f"https://github.com/marin-community/vllm/releases/download/{tag}/"
                        + quote(platform["wheel"]["filename"], safe="")
                    ),
                    "sha256": platform["wheel"]["sha256"],
                }
                for platform in record["platforms"]
            ],
        }
    return draccus.decode(VllmGpuRelease, record)


@dataclass(frozen=True)
class VllmGpuWheelProvenance:
    """Expected wheel identity passed to the isolated startup verifier."""

    release_tag: str
    sm_targets: tuple[str, ...]
    source_commit: str
    version: str
    wheel_sha256: str
    wheel_url: str


def vllm_gpu_wheel_for_architecture(release: VllmGpuRelease, architecture: str) -> VllmGpuWheel:
    """Return the verified wheel for a worker architecture."""
    for wheel in release.wheels:
        if wheel.architecture == architecture:
            return wheel
    raise ValueError(f"No verified Marin vLLM GPU wheel for architecture {architecture}")


def current_vllm_gpu_wheel(release: VllmGpuRelease) -> VllmGpuWheel:
    """Select by ``platform.machine()`` for heterogeneous Iris GPU workers."""
    return vllm_gpu_wheel_for_architecture(release, platform.machine())


def vllm_gpu_wheel_requirement(wheel: VllmGpuWheel) -> str:
    """Return the direct wheel requirement, tagged with the promoted digest.

    ``uv tool run`` records neither the fragment digest nor an ``archive_info`` hash for a remote
    direct reference and does not check it, so the digest documents the promoted asset rather than
    gating the install. Startup verification compares the installed URL, version, and ABI instead.
    """
    return f"vllm @ {wheel.url}#sha256={wheel.sha256}"


def vllm_gpu_wheel_provenance(release: VllmGpuRelease, wheel: VllmGpuWheel) -> VllmGpuWheelProvenance:
    """Return the expected runtime identity for a release wheel."""
    return VllmGpuWheelProvenance(
        release_tag=release.release_tag,
        sm_targets=wheel.sm_targets,
        source_commit=release.source_commit,
        version=release.version,
        wheel_sha256=wheel.sha256,
        wheel_url=wheel.url,
    )
