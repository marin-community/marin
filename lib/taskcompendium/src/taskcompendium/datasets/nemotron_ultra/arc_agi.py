# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""arc-agi contracts and normalization policies."""

from taskcompendium.datasets.nemotron_ultra.source import quality_policy
from taskcompendium.pipeline.models import TaskPolicy

CRITERIA = (
    (
        "Inspect the original stateless Python scorer's final-stdout acceptance: a valid grid can receive "
        "reward even after a nonzero candidate exit. Report this source grading weakness when relevant; "
        "do not silently replace the original reward with a stricter process-exit policy."
    ),
    (
        "Check that all training grids, public test inputs, and private expected_output match the "
        "stated grid transformation and dimensions."
    ),
    (
        "Inductive variants require producing a reusable transformation program; transductive "
        "variants request the output grid. Do not replace one grading contract with the other or "
        "expose hidden outputs."
    ),
)


def policy(selector: str, rubric_id: str) -> TaskPolicy:
    return quality_policy("arc-agi", selector, rubric_id, CRITERIA, rubric_version="2")
