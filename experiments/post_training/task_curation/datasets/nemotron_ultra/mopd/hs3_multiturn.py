# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra mopd hs3 multiturn source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import preference as ultra_preference
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_mopd/hs3_multiturn",
        name="nemotron_ultra_mopd_hs3_multiturn",
        version="nemotron_ultra_mopd_hs3_multiturn-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="mopd/hs3_multiturn",
        split="train",
        inputs=partial(blend_inputs, "mopd", "hs3_multiturn"),
        policy=ultra_preference.policy("hs3_multiturn", "nemotron_ultra_mopd_hs3_multiturn-answerability"),
        intended_use=IntendedUse.TRAIN,
    )
