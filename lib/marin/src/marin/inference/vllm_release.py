# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verified Marin vLLM GPU release consumed by isolated CUDA launchers."""

import platform
from dataclasses import dataclass

_RELEASE_TAG = "marin-vllm-gpu-20260805-fa50698a9a30"
_RELEASE_VERSION = "0.0.0.dev20260805+marin.fa50698a9a30.cu129"
_RELEASE_ASSET_PREFIX = (
    f"https://github.com/marin-community/vllm/releases/download/{_RELEASE_TAG}/"
    "vllm-0.0.0.dev20260805%2Bmarin.fa50698a9a30.cu129-cp38-abi3-"
)


@dataclass(frozen=True)
class VllmGpuWheel:
    """One architecture-specific wheel from a promoted GPU release."""

    architecture: str
    sm_targets: tuple[str, ...]
    url: str
    sha256: str

    def requirement(self) -> str:
        """Return the hash-verified direct wheel requirement."""
        return f"vllm @ {self.url}#sha256={self.sha256}"


@dataclass(frozen=True)
class VllmGpuWheelProvenance:
    """Expected wheel identity passed to the isolated startup verifier."""

    release_tag: str
    sm_targets: tuple[str, ...]
    source_commit: str
    version: str
    wheel_sha256: str
    wheel_url: str


@dataclass(frozen=True)
class VllmGpuRelease:
    """Promoted GPU release and its supported worker architectures."""

    release_tag: str
    source_commit: str
    version: str
    torch_backend: str
    wheels: tuple[VllmGpuWheel, ...]

    def wheel_for_architecture(self, architecture: str) -> VllmGpuWheel:
        """Return the verified wheel for a worker architecture."""
        for wheel in self.wheels:
            if wheel.architecture == architecture:
                return wheel
        raise ValueError(f"No verified Marin vLLM GPU wheel for architecture {architecture}")

    def wheel_for_current_platform(self) -> VllmGpuWheel:
        """Resolve the wheel on the worker where vLLM will run."""
        return self.wheel_for_architecture(platform.machine())

    def provenance(self, wheel: VllmGpuWheel) -> VllmGpuWheelProvenance:
        """Return the expected runtime identity for a release wheel."""
        return VllmGpuWheelProvenance(
            release_tag=self.release_tag,
            sm_targets=wheel.sm_targets,
            source_commit=self.source_commit,
            version=self.version,
            wheel_sha256=wheel.sha256,
            wheel_url=wheel.url,
        )


# Update these fields together from the promoted release manifest after both Iris GPU lanes pass:
# https://github.com/marin-community/vllm/releases/download/marin-vllm-gpu-20260805-fa50698a9a30/marin-vllm-gpu-manifest.json
MARIN_VLLM_GPU_RELEASE = VllmGpuRelease(
    release_tag=_RELEASE_TAG,
    source_commit="fa50698a9a303f7282aa0e969f35717703de4911",
    version=_RELEASE_VERSION,
    torch_backend="cu129",
    wheels=(
        VllmGpuWheel(
            architecture="x86_64",
            sm_targets=("9.0",),
            url=f"{_RELEASE_ASSET_PREFIX}manylinux_2_28_x86_64.whl",
            sha256="d4e5d6e19da49c0f1dd030bd14d3ab795a10b8f1185c55162ae5daf6745c98eb",
        ),
        VllmGpuWheel(
            architecture="aarch64",
            sm_targets=("10.0",),
            url=f"{_RELEASE_ASSET_PREFIX}manylinux_2_28_aarch64.whl",
            sha256="a248408d444425906a0e878513c8f31a43582ddf6e82e32a1573b6c714b677f9",
        ),
    ),
)
