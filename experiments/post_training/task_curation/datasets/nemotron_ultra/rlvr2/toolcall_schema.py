# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron ultra rlvr2 ultra sft step3200 toolcall schema source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import tool_use as ultra_tool_use
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading import binding as ultra_grader_binding
from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import blend_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_toolcall_schema",
        runtime_binding=partial(bind_private_grader, binder=ultra_grader_binding.bind_tool_action),
        name="nemotron_ultra_rlvr2_ultra_sft_step3200_toolcall_schema",
        version="nemotron_ultra_rlvr2_ultra_sft_step3200_toolcall_schema-v1",
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="rlvr2/ultra_sft_step3200_toolcall_schema",
        split="train",
        inputs=partial(blend_inputs, "rlvr2", "ultra_sft_step3200_toolcall_schema"),
        policy=ultra_tool_use.policy(
            "ultra_sft_step3200_toolcall_schema",
            "nemotron_ultra_rlvr2_ultra_sft_step3200_toolcall_schema-quality",
        ),
        intended_use=IntendedUse.TRAIN,
    )
