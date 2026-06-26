# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin-specific *content* for the experiment helpers.

``EvalSuite`` lives in ``marin.experiment``; the choice of *which* eval suite is a
marin convention, so it lives here, next to the experiments that use it.
"""

from marin.experiment import EvalSuite

from experiments.evals.task_configs import CORE_TASKS


def core_tasks(*, every: int = 10000) -> EvalSuite:
    """The marin CORE harness suite (DCLM-paper subset), run every ``every`` steps."""
    return EvalSuite(CORE_TASKS, every)
