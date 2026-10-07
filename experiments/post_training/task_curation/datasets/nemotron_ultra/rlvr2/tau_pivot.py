# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra rlvr2 ultra sft step3200 tau pivot source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import tool_use as ultra_tool_use
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_tau_pivot",
        name="nemotron_ultra_rlvr2_ultra_sft_step3200_tau_pivot",
        version="nemotron_ultra_rlvr2_ultra_sft_step3200_tau_pivot-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="rlvr2/ultra_sft_step3200_tau_pivot",
        split="train",
        inputs=partial(blend_inputs, "rlvr2", "ultra_sft_step3200_tau_pivot"),
        policy=ultra_tool_use.policy(
            "ultra_sft_step3200_tau_pivot",
            "nemotron_ultra_rlvr2_ultra_sft_step3200_tau_pivot-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
