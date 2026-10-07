# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra rlvr1 ultra sft step3200 multichallenge len40k source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import instruction_following as ultra_instruction_following
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_multichallenge_len40k",
        name="nemotron_ultra_rlvr1_ultra_sft_step3200_multichallenge_len40k",
        version="nemotron_ultra_rlvr1_ultra_sft_step3200_multichallenge_len40k-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="rlvr1/ultra_sft_step3200_multichallenge_len40k",
        split="train",
        inputs=partial(blend_inputs, "rlvr1", "ultra_sft_step3200_multichallenge_len40k"),
        policy=ultra_instruction_following.policy(
            "ultra_sft_step3200_multichallenge_len40k",
            "nemotron_ultra_rlvr1_ultra_sft_step3200_multichallenge_len40k-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
