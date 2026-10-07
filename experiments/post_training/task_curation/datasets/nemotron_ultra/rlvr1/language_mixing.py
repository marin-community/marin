# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra rlvr1 language mixing hs3 ultra genrm fmt source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import preference as ultra_preference
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_rlvr1/language_mixing_hs3_ultra_genrm_fmt",
        name="nemotron_ultra_rlvr1_language_mixing_hs3_ultra_genrm_fmt",
        version="nemotron_ultra_rlvr1_language_mixing_hs3_ultra_genrm_fmt-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="rlvr1/language_mixing_hs3_ultra_genrm_fmt",
        split="train",
        inputs=partial(blend_inputs, "rlvr1", "language_mixing_hs3_ultra_genrm_fmt"),
        policy=ultra_preference.policy(
            "language_mixing_hs3_ultra_genrm_fmt",
            "nemotron_ultra_rlvr1_language_mixing_hs3_ultra_genrm_fmt-answerability",
        ),
        intended_use=IntendedUse.TRAIN,
    )
