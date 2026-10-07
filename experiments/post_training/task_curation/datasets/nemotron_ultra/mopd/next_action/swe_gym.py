# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Nemotron Ultra source declaration."""

from functools import partial

from taskcompendium.datasets.nemotron_ultra import swe_repo as ultra_swe_repo
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading import binding as ultra_grader_binding
from experiments.post_training.task_curation.datasets.nemotron_ultra.inputs import swe_inputs
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_ultra_mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym",
        runtime_binding=partial(bind_private_grader, binder=ultra_grader_binding.bind_tool_action),
        name=("nemotron_ultra_mopd_agent_swe_pivot_single_step_tool_use_with_argument_comparison_agent_swe_gym_swe_gym"),
        version=(
            "nemotron_ultra_mopd_agent_swe_pivot_single_step_tool_use_with_argument_comparison_ag"
            "ent_swe_gym_swe_gym-v1"
        ),
        hf_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        revision="482392c14c6418e26804ea2e5d10359df9877df4",
        config="mopd/agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym",
        split="train",
        inputs=partial(
            swe_inputs,
            "mopd",
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent",
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent/SWE-Gym/SWE-Gym",
        ),
        policy=ultra_swe_repo.policy(
            "agent:swe_pivot_single_step_tool_use_with_argument_comparison_agent",
            (
                "nemotron_ultra_mopd_agent_swe_pivot_single_step_tool_use_with_argument_compariso"
                "n_agent_swe_gym_swe_gym-quality"
            ),
        ),
        intended_use=IntendedUse.TRAIN,
    )
