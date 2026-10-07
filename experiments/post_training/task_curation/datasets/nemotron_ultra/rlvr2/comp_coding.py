# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra rlvr2 ultra sft step3200 comp coding source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import competitive_programming as ultra_competitive_programming
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading import code_binding
from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.datasets.skyrl.code_sql import binding as code_sql_binding
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_comp_coding",
        runtime_binding=partial(
            bind_private_grader,
            binder=code_binding.bind,
            timeout=code_sql_binding.CONTROL_TIMEOUT,
            memory_mb=code_sql_binding.CONTROL_MEMORY_MB,
        ),
        name="nemotron_ultra_rlvr2_ultra_sft_step3200_comp_coding",
        version="nemotron_ultra_rlvr2_ultra_sft_step3200_comp_coding-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="rlvr2/ultra_sft_step3200_comp_coding",
        split="train",
        inputs=partial(blend_inputs, "rlvr2", "ultra_sft_step3200_comp_coding"),
        policy=ultra_competitive_programming.policy(
            "ultra_sft_step3200_comp_coding",
            "nemotron_ultra_rlvr2_ultra_sft_step3200_comp_coding-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
