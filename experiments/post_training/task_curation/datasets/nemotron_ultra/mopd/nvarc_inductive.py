# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra mopd ultra sft step3200 nvarc inductive source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import arc_agi as ultra_arc_agi
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.arc import binding as arc_binding
from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_nvarc_inductive",
        runtime_binding=partial(
            bind_private_grader,
            binder=partial(arc_binding.bind, source=arc_binding.ArcSource.ULTRA),
            timeout=120.0,
            memory_mb=4096,
        ),
        name="nemotron_ultra_mopd_ultra_sft_step3200_nvarc_inductive",
        version="nemotron_ultra_mopd_ultra_sft_step3200_nvarc_inductive-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="mopd/ultra_sft_step3200_nvarc_inductive",
        split="train",
        inputs=partial(blend_inputs, "mopd", "ultra_sft_step3200_nvarc_inductive"),
        policy=ultra_arc_agi.policy(
            "ultra_sft_step3200_nvarc_inductive",
            "nemotron_ultra_mopd_ultra_sft_step3200_nvarc_inductive-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
