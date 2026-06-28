# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin-specific *content* for the experiment helpers.

``EvalSuite`` lives in ``marin.experiment``; the choice of *which* eval suite (and
which default validation sets) is a marin convention, so it lives here, next to the
experiments that use it.
"""

from marin.execution.lazy import Dataset
from marin.experiment import EvalSuite

from experiments.evals.task_configs import CORE_TASKS
from experiments.evals.uncheatable import uncheatable_validation
from experiments.llama import llama3_tokenizer
from experiments.paloma import paloma_validation


def core_tasks(*, every: int = 10000) -> EvalSuite:
    """The marin CORE harness suite (DCLM-paper subset), run every ``every`` steps."""
    return EvalSuite(CORE_TASKS, every)


def default_validation(*, tokenizer: str = llama3_tokenizer) -> list[Dataset]:
    """The marin convention's default validation handles: Paloma + Uncheatable Eval."""
    return [*paloma_validation(tokenizer=tokenizer), *uncheatable_validation(tokenizer=tokenizer)]
