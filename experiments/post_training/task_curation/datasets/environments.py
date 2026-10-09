# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grader environments shared by the catalog's declarations."""

from dataclasses import replace
from pathlib import Path

from experiments.post_training.task_curation.environment import Environment

HERE = Path(__file__).resolve().parent
VERIFYIT_PACKAGE = HERE.parents[3] / "lib/verifyit"

GRADER_PACKAGES = Environment(lock=HERE / "grader.lock", data=("nltk:punkt_tab", "nltk:wordnet"))
"""The packages every grade script in the catalog imports, compiled from ``grader.in``, and their NLTK data."""

COMPILER_GRADER_PACKAGES = replace(GRADER_PACKAGES, apt=("build-essential",))
"""The grader packages with a C++ toolchain, for graders that compile submissions."""
