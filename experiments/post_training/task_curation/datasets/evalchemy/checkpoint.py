# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect checkpoint-menu variants declared by their dataset families."""

from experiments.post_training.task_curation.datasets.evalchemy.code import checkpoint_sources as code_sources
from experiments.post_training.task_curation.datasets.evalchemy.commonsense import (
    checkpoint_sources as commonsense_sources,
)
from experiments.post_training.task_curation.datasets.evalchemy.harness import checkpoint_sources as short_answer_sources
from experiments.post_training.task_curation.datasets.evalchemy.math import checkpoint_sources as math_sources
from experiments.post_training.task_curation.datasets.evalchemy.reasoning import checkpoint_sources as reasoning_sources
from experiments.post_training.task_curation.evaluation import EvalDataset
from experiments.post_training.task_curation.source import RlDataSource


def sources() -> list[RlDataSource[EvalDataset]]:
    return [*commonsense_sources(), *reasoning_sources(), *math_sources(), *short_answer_sources(), *code_sources()]
