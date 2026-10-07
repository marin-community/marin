# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra mopd makeshn ultra v3 ipi train source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import agentic_safety as ultra_agentic_safety
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train",
        name="nemotron_ultra_mopd_makeshn_ultra_v3_ipi_train",
        version="nemotron_ultra_mopd_makeshn_ultra_v3_ipi_train-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="mopd/makeshn_ultra_v3_ipi_train",
        split="train",
        inputs=partial(blend_inputs, "mopd", "makeshn_ultra_v3_ipi_train"),
        policy=ultra_agentic_safety.policy(
            "makeshn_ultra_v3_ipi_train",
            "nemotron_ultra_mopd_makeshn_ultra_v3_ipi_train-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
