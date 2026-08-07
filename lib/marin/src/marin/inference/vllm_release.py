# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select and describe verified vLLM GPU release wheels."""

import platform
from dataclasses import dataclass

from marin.external_dependencies import VllmGpuRelease, VllmGpuWheel


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
    """Resolve the verified wheel on the current worker."""
    return vllm_gpu_wheel_for_architecture(release, platform.machine())


def vllm_gpu_wheel_requirement(wheel: VllmGpuWheel) -> str:
    """Return the hash-verified direct wheel requirement."""
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
