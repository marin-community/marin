# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned reasoning gym source declaration."""

from functools import partial

from taskcompendium.datasets import reasoning_tasks
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.reasoning_gym import binding as reasoning_gym_binding
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:laion__nemotron-gym-reasoning-gym-v2",
        runtime_binding=partial(
            bind_private_grader,
            binder=partial(
                reasoning_gym_binding.bind,
                contract=reasoning_gym_binding.ReasoningContract.TASKTROVE,
                package_path="/opt/reasoning-gym-tasktrove",
            ),
        ),
        name="tasktrove-reasoning-gym",
        version="tasktrove-reasoning-gym-v2",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="laion__nemotron-gym-reasoning-gym-v2",
        split="train",
        files=tasktrove_files("laion__nemotron-gym-reasoning-gym-v2"),
        policy=reasoning_tasks.reasoning_policy(),
        intended_use=IntendedUse.TRAIN,
    )
