# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the two fixed serving releases without changing Marin's shared adoption."""

import dataclasses
import hashlib
from pathlib import Path

from marin.external_dependencies import VllmGpuRelease, VllmGpuWheel

from experiments.evaluation.agentx.recipe import load_series, series_path

HERO_CONSTRAINTS = Path(__file__).with_name("requirements-vllm-hero.txt")


def gpu_release(profile: str) -> VllmGpuRelease:
    series = load_series(series_path(profile))
    if (
        profile == "gb200"
        and hashlib.sha256(HERO_CONSTRAINTS.read_bytes()).hexdigest() != series["vllm_constraints_sha256"]
    ):
        raise ValueError("Hero runtime constraints differ from the pinned recipe")
    pin = series["vllm_release"]
    return VllmGpuRelease(
        release_tag=pin["release_tag"],
        source_commit=pin["source_commit"],
        version=pin["version"],
        torch_backend=pin["torch_backend"],
        torch_version=pin["torch_version"],
        wheels=tuple(
            VllmGpuWheel(
                architecture=wheel["architecture"],
                sm_targets=tuple(wheel["sm_targets"]),
                url=wheel["url"],
                sha256=wheel["sha256"],
            )
            for wheel in pin["wheels"]
        ),
    )


def runtime_metadata(profile: str) -> dict:
    metadata = dataclasses.asdict(gpu_release(profile))
    if profile == "gb200":
        metadata["constraints_sha256"] = load_series(series_path(profile))["vllm_constraints_sha256"]
    return metadata
