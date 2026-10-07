# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra rlvr1 ultra sft step3200 abstention source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import qa_abstention as ultra_qa_abstention
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_abstention",
        name="nemotron_ultra_rlvr1_ultra_sft_step3200_abstention",
        version="nemotron_ultra_rlvr1_ultra_sft_step3200_abstention-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="rlvr1/ultra_sft_step3200_abstention",
        split="train",
        inputs=partial(blend_inputs, "rlvr1", "ultra_sft_step3200_abstention"),
        policy=ultra_qa_abstention.policy(
            "ultra_sft_step3200_abstention",
            "nemotron_ultra_rlvr1_ultra_sft_step3200_abstention-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
