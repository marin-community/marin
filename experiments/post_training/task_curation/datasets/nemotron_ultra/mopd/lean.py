# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra mopd ultra sft step3200 lean source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import math_proof as ultra_math_proof
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_lean",
        name="nemotron_ultra_mopd_ultra_sft_step3200_lean",
        version="nemotron_ultra_mopd_ultra_sft_step3200_lean-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="mopd/ultra_sft_step3200_lean",
        split="train",
        inputs=partial(blend_inputs, "mopd", "ultra_sft_step3200_lean"),
        policy=ultra_math_proof.policy(
            "ultra_sft_step3200_lean",
            "nemotron_ultra_mopd_ultra_sft_step3200_lean-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
