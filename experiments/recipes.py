# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin-specific *content* for the experiment helpers.

``EvalSuite`` lives in ``marin.experiment``; the choices of *which* eval suite and
*which* validation sets are marin conventions, so they live here, next to the
experiments that use them.
"""

from marin.experiment import EvalSuite
from marin.processing.tokenize import TokenizerStep

from experiments.defaults import default_validation_sets
from experiments.evals.task_configs import CORE_TASKS


def core_tasks(*, every: int = 10000) -> EvalSuite:
    """The marin CORE harness suite (DCLM-paper subset), run every ``every`` steps."""
    return EvalSuite(CORE_TASKS, every)


def marin_validation(tokenizer: str) -> dict[str, TokenizerStep]:
    """The marin default validation sets (Paloma + uncheatable) for a tokenizer."""
    return default_validation_sets(tokenizer=tokenizer)
