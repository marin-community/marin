# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra rlvr1 ultra sft step3200 swe pivot len40k swe gym swe gym source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import swe_repo as ultra_swe_repo
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import swe_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        name="nemotron_ultra_rlvr1_ultra_sft_step3200_swe_pivot_len40k_swe_gym_swe_gym",
        version="nemotron_ultra_rlvr1_ultra_sft_step3200_swe_pivot_len40k_swe_gym_swe_gym-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="rlvr1/ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        split="train",
        inputs=partial(
            swe_inputs,
            "rlvr1",
            "ultra_sft_step3200_swe_pivot_len40k",
            "ultra_sft_step3200_swe_pivot_len40k/SWE-Gym/SWE-Gym",
        ),
        policy=ultra_swe_repo.policy(
            "ultra_sft_step3200_swe_pivot_len40k",
            "nemotron_ultra_rlvr1_ultra_sft_step3200_swe_pivot_len40k_swe_gym_swe_gym-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
