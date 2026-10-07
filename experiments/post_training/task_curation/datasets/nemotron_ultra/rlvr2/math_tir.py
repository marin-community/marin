# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra rlvr2 ultra sft step3200 math tir source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import math_answer as ultra_math_answer
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import math_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_math_tir",
        name="nemotron_ultra_rlvr2_ultra_sft_step3200_math_tir",
        version="nemotron_ultra_rlvr2_ultra_sft_step3200_math_tir-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="rlvr2/ultra_sft_step3200_math_tir",
        split="train",
        inputs=partial(math_inputs, "rlvr2", "ultra_sft_step3200_math_tir"),
        policy=ultra_math_answer.policy(
            "ultra_sft_step3200_math_tir",
            "nemotron_ultra_rlvr2_ultra_sft_step3200_math_tir-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
