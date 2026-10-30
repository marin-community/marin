# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation data adopted from Evalchemy into the canonical task catalog."""

from experiments.post_training.task_curation.datasets.evalchemy import (
    code,
    datasets,
    harness,
    instructions,
    jee,
    judged,
    long_context,
    math,
    numeric,
    reasoning,
)
from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet
from experiments.post_training.task_curation.source import RlDataSource


def sources() -> list[RlDataSource[EvalDataSet]]:
    return [
        *datasets.sources(),
        *harness.sources(),
        *math.sources(),
        *reasoning.sources(),
        *code.sources(),
        *instructions.sources(),
        *jee.sources(),
        *judged.sources(),
        *long_context.sources(),
        *numeric.sources(),
    ]
