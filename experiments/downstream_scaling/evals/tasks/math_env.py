# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vendored from marin.rl.environments.tinker_environments.math_env (deleted on
main by #8063), trimmed to `safe_grade` for math500 grading."""

import logging
import math

from experiments.downstream_scaling.evals.tasks.math_grading import (
    grade_answer,
    grade_answer_math_verify,
    run_with_timeout_signal,
)

logger = logging.getLogger(__name__)


def safe_grade(given_answer: str, ground_truth: str, grader: str = "sympy", timeout: float = 1.0):
    if grader == "sympy":
        grader_func = grade_answer
    elif grader == "math_verify":
        grader_func = grade_answer_math_verify
    else:
        raise ValueError(f"Invalid grader: {grader}")
    out = run_with_timeout_signal(grader_func, args=(given_answer, ground_truth), timeout_seconds=math.ceil(timeout))
    if out is None:
        logger.warning(f"Timeout grading {given_answer} against {ground_truth}")
        return False
    return out
