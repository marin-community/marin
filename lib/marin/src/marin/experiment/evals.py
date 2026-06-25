# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-harness plumbing for the experiment helpers.

``EvalSuite`` pairs a set of harness tasks with how often to run them. The
*content* — which tasks make up a named suite like the marin CORE set — is an
experiment-level decision and lives in ``experiments/`` (see ``experiments.recipes``).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from marin.evaluation.evaluation_config import EvalTaskConfig


@dataclass(frozen=True)
class EvalSuite:
    """A set of harness tasks plus the step interval at which to run them."""

    tasks: tuple[EvalTaskConfig, ...]
    every: int

    def __init__(self, tasks: Sequence[EvalTaskConfig], every: int):
        object.__setattr__(self, "tasks", tuple(tasks))
        object.__setattr__(self, "every", every)
