# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra rlvr2 ultra sft step3200 reasoning gym source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import reasoning_gym as ultra_reasoning_gym
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.reasoning_gym import binding as reasoning_gym_binding
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_reasoning_gym",
        runtime_binding=partial(
            bind_private_grader,
            binder=partial(
                reasoning_gym_binding.bind,
                contract=reasoning_gym_binding.ReasoningContract.ULTRA,
                package_path="/opt/reasoning-gym-ultra",
            ),
        ),
        name="nemotron_ultra_rlvr2_ultra_sft_step3200_reasoning_gym",
        version="nemotron_ultra_rlvr2_ultra_sft_step3200_reasoning_gym-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="rlvr2/ultra_sft_step3200_reasoning_gym",
        split="train",
        inputs=partial(blend_inputs, "rlvr2", "ultra_sft_step3200_reasoning_gym"),
        policy=ultra_reasoning_gym.policy(
            "ultra_sft_step3200_reasoning_gym",
            "nemotron_ultra_rlvr2_ultra_sft_step3200_reasoning_gym-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
