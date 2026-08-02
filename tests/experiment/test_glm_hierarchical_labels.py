# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents" / "projects" / "luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import glm_hierarchical_labels as hierarchical_labels  # noqa: E402
from glm_hierarchical_labels import (  # noqa: E402
    FORMS,
    Hierarchy,
    LeafBucket,
    Variant,
    assign_document,
    validate_hierarchy,
)
from glm_semantic_labels import OTHER_BUCKET_ID, Bucket, SampleDocument  # noqa: E402


def hierarchy() -> tuple[Hierarchy, Variant]:
    parents = [
        Bucket("SCIENCE", "Science", "Science", [], []),
        Bucket("ARTS", "Arts", "Arts", [], []),
        Bucket(OTHER_BUCKET_ID, "Other", "Other", [], []),
    ]
    leaves = [
        LeafBucket("BIOLOGY", "SCIENCE", "Biology", "Biology", [], []),
        LeafBucket("FICTION", "ARTS", "Fiction", "Fiction", [], []),
        LeafBucket(OTHER_BUCKET_ID, OTHER_BUCKET_ID, "Other", "Other", [], []),
    ]
    return Hierarchy(parents, leaves, ["Use the central purpose."]), Variant("test", 2, 2, 2, 2)


def test_validate_hierarchy_accepts_linked_parent_and_leaf_ids() -> None:
    value, variant = hierarchy()

    validate_hierarchy(value, variant)


def test_validate_hierarchy_rejects_leaf_with_unknown_parent() -> None:
    value, variant = hierarchy()
    value = Hierarchy(
        value.parents,
        [*value.leaves[:-1], LeafBucket(OTHER_BUCKET_ID, "MISSING", "Other", "Other", [], [])],
        value.precedence_rules,
    )

    with pytest.raises(ValueError, match="unknown parent"):
        validate_hierarchy(value, variant)


def test_assign_document_checks_primary_leaf_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    value, _ = hierarchy()
    payload = {
        "primary_parent_id": "SCIENCE",
        "secondary_parent_ids": [],
        "primary_leaf_id": "FICTION",
        "secondary_leaf_ids": [],
        "form_id": FORMS[0].bucket_id,
        "confidence": 0.9,
        "rationale": "Test",
    }
    monkeypatch.setattr(hierarchical_labels, "completion", lambda *args, **kwargs: payload)
    document = SampleDocument(0, "hash", "hidden", "standard", 0, "Text")

    with pytest.raises(ValueError, match="wrong parent"):
        assign_document("http://server", document, value, 0)


def test_run_waits_for_queued_server_without_client_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    waits = []

    class Job:
        def wait(self, timeout: float, raise_on_failure: bool) -> None:
            waits.append((timeout, raise_on_failure))

    class Context:
        client = object()

    monkeypatch.setattr(hierarchical_labels, "iris_ctx", Context)
    monkeypatch.setattr(hierarchical_labels, "submit_glm52", lambda *args, **kwargs: Job())

    hierarchical_labels.run("run", [], batch_size=50, concurrency=1)

    assert waits == [(float("inf"), True)]
