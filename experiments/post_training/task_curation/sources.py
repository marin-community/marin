# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The RL data catalog: every dataset declaration, keyed by name."""

from experiments.post_training.task_curation.datasets.arc import arc
from experiments.post_training.task_curation.datasets.nemotron_ultra import components as nemotron_ultra
from experiments.post_training.task_curation.datasets.reasoning_gym import tasks as reasoning_gym
from experiments.post_training.task_curation.datasets.skyrl import code as skyrl_code
from experiments.post_training.task_curation.datasets.skyrl import ifeval as skyrl_ifeval
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.datasets.skyrl import mcq as skyrl_mcq
from experiments.post_training.task_curation.datasets.skyrl import preference as skyrl_preference
from experiments.post_training.task_curation.datasets.tasktrove import calendar as tasktrove_calendar
from experiments.post_training.task_curation.datasets.tasktrove import code as tasktrove_code
from experiments.post_training.task_curation.datasets.tasktrove import instruction_following as tasktrove_instructions
from experiments.post_training.task_curation.datasets.tasktrove import judged as tasktrove_judged
from experiments.post_training.task_curation.datasets.tasktrove import math as tasktrove_math
from experiments.post_training.task_curation.datasets.tasktrove import multichallenge as tasktrove_multichallenge
from experiments.post_training.task_curation.datasets.tasktrove import nl2bash as tasktrove_nl2bash
from experiments.post_training.task_curation.datasets.tasktrove import puzzles as tasktrove_puzzles
from experiments.post_training.task_curation.datasets.tasktrove import python_tests as tasktrove_python_tests
from experiments.post_training.task_curation.datasets.tasktrove import qa as tasktrove_qa
from experiments.post_training.task_curation.datasets.tasktrove import repositories as tasktrove_repositories
from experiments.post_training.task_curation.datasets.tasktrove import structured_outputs as tasktrove_structured_outputs
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def all_pipelines() -> dict[str, RlDataPipeline]:
    pipelines = [
        *skyrl_math.pipelines(),
        *skyrl_code.pipelines(),
        *skyrl_ifeval.pipelines(),
        *skyrl_mcq.pipelines(),
        *skyrl_preference.pipelines(),
        *tasktrove_code.pipelines(),
        *tasktrove_python_tests.pipelines(),
        *tasktrove_nl2bash.pipelines(),
        *tasktrove_structured_outputs.pipelines(),
        *tasktrove_repositories.pipelines(),
        *tasktrove_math.pipelines(),
        *tasktrove_judged.pipelines(),
        *tasktrove_qa.pipelines(),
        *tasktrove_calendar.pipelines(),
        *tasktrove_instructions.pipelines(),
        *tasktrove_multichallenge.pipelines(),
        *tasktrove_puzzles.pipelines(),
        *arc.pipelines(),
        *reasoning_gym.pipelines(),
        *nemotron_ultra.pipelines(),
    ]
    names = [pipeline.name for pipeline in pipelines]
    assert len(set(names)) == len(names), sorted(name for name in names if names.count(name) > 1)
    return {pipeline.name: pipeline for pipeline in pipelines}
