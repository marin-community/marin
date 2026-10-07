# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra mopd swe pivot len40k nebius swe rebench v2 source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import swe_repo as ultra_swe_repo
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import swe_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2",
        name="nemotron_ultra_mopd_swe_pivot_len40k_nebius_swe_rebench_v2",
        version="nemotron_ultra_mopd_swe_pivot_len40k_nebius_swe_rebench_v2-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="mopd/swe_pivot_len40k/nebius/SWE-rebench-V2",
        split="train",
        inputs=partial(swe_inputs, "mopd", "swe_pivot_len40k", "swe_pivot_len40k/nebius/SWE-rebench-V2"),
        policy=ultra_swe_repo.policy(
            "swe_pivot_len40k",
            "nemotron_ultra_mopd_swe_pivot_len40k_nebius_swe_rebench_v2-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
