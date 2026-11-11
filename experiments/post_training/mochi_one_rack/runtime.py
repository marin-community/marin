# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use the smoke's experimental wheel with Marin's ordinary CUDA serving launcher."""

import tomllib
from pathlib import Path

from marin.external_dependencies import VllmGpuRelease, VllmGpuWheel
from marin.inference.vllm_server import IsolatedCudaVllm, VllmType


def vllm_backend() -> IsolatedCudaVllm:
    path = Path(__file__).resolve().parents[3] / "config/external/vllm/mochi-gb200.toml"
    config = tomllib.loads(path.read_text())["release"]
    wheels = tuple(
        VllmGpuWheel(
            architecture=wheel["architecture"],
            sm_targets=tuple(wheel["sm_targets"]),
            url=wheel["url"],
            sha256=wheel["sha256"],
        )
        for wheel in config.pop("wheels")
    )
    return IsolatedCudaVllm(source=VllmType.MARIN_FORK, gpu_release=VllmGpuRelease(**config, wheels=wheels))
