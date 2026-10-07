# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra mopd ultra sft step3200 stem mcqa source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import qa_multiple_choice as ultra_qa_multiple_choice
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading import (
    multiple_choice_binding as ultra_multiple_choice_binding,
)
from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_stem_mcqa",
        runtime_binding=partial(bind_private_grader, binder=ultra_multiple_choice_binding.bind),
        name="nemotron_ultra_mopd_ultra_sft_step3200_stem_mcqa",
        version="nemotron_ultra_mopd_ultra_sft_step3200_stem_mcqa-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="mopd/ultra_sft_step3200_stem_mcqa",
        split="train",
        inputs=partial(blend_inputs, "mopd", "ultra_sft_step3200_stem_mcqa"),
        policy=ultra_qa_multiple_choice.policy(
            "ultra_sft_step3200_stem_mcqa",
            "nemotron_ultra_mopd_ultra_sft_step3200_stem_mcqa-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
