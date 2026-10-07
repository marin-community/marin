# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""qa-abstention contracts and normalization policies."""

from taskcompendium.datasets.nemotron_ultra.source import quality_policy
from taskcompendium.pipeline.models import TaskPolicy

CRITERIA = (
    "Check whether the actual question is answerable, and whether the private answer is correct.",
    (
        "Abstention policy and any [IDK] output requirements belong to the source contract; do not "
        "substitute exact-only matching for its semantic evaluator."
    ),
)


def policy(selector: str, rubric_id: str) -> TaskPolicy:
    return quality_policy("qa-abstention", selector, rubric_id, CRITERIA, rubric_version="2")
