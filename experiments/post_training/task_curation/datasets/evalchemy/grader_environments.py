# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared pinned environments for source-owned evaluation scorers."""

from experiments.post_training.task_curation.environment import Environment

BASIC_GRADER = Environment(requirements=("tomli-w==1.2.0",))
MATH_GRADER = Environment(
    requirements=("tomli-w==1.2.0", "math-verify==0.9.0", "sympy==1.13.3", "antlr4-python3-runtime==4.11.0")
)
MATH_JUDGE_GRADER = Environment(requirements=(*MATH_GRADER.requirements, "openai==2.24.0"))
