# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

from curate_compact_hierarchy import curated_hierarchy  # noqa: E402
from glm_hierarchical_labels import Hierarchy, LeafBucket  # noqa: E402
from glm_semantic_labels import OTHER_BUCKET_ID, Bucket  # noqa: E402


def test_curated_hierarchy_removes_only_invalid_form_rule() -> None:
    parents = [
        *[Bucket(f"PARENT_{index}", f"Parent {index}", "Domain", [], []) for index in range(8)],
        Bucket(OTHER_BUCKET_ID, "Other", "Other", [], []),
    ]
    leaves = [
        *[LeafBucket(f"LEAF_{index}", f"PARENT_{index % 8}", f"Leaf {index}", "Domain", [], []) for index in range(18)],
        LeafBucket(OTHER_BUCKET_ID, OTHER_BUCKET_ID, "Other", "Other", [], []),
    ]
    hierarchy = Hierarchy(
        parents,
        leaves,
        ["Classify forms under FORMS_TEMPLATES.", "Classify central domains under LEAF_0."],
    )

    curated, removed = curated_hierarchy(asdict(hierarchy))

    assert removed == "Classify forms under FORMS_TEMPLATES."
    assert curated.precedence_rules == ["Classify central domains under LEAF_0."]


def test_curated_hierarchy_requires_exactly_one_invalid_rule() -> None:
    payload = {"parents": [], "leaves": [], "precedence_rules": []}

    with pytest.raises(ValueError, match="Expected one FORMS_TEMPLATES"):
        curated_hierarchy(payload)
