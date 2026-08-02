# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import asdict
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
    build_hierarchy,
    hierarchy_launch_config,
    validate_hierarchy,
)
from glm_semantic_labels import OTHER_BUCKET_ID, Bucket, SampleDocument  # noqa: E402
from rigging.filesystem import StoragePath  # noqa: E402


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


def test_validate_hierarchy_rejects_precedence_rule_with_unknown_bucket() -> None:
    value, variant = hierarchy()
    value = Hierarchy(value.parents, value.leaves, ["Classify forms under FORMS_TEMPLATES."])

    with pytest.raises(ValueError, match=r"unknown bucket IDs.*FORMS_TEMPLATES"):
        validate_hierarchy(value, variant)


def test_build_hierarchy_gives_validation_error_to_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    value, variant = hierarchy()
    invalid = asdict(Hierarchy(value.parents, value.leaves, ["Classify forms under FORMS_TEMPLATES."]))
    corrected = asdict(value)
    calls = []

    def completion(*args, **kwargs):
        calls.append(args[1].copy())
        return invalid if len(calls) == 1 else corrected

    monkeypatch.setattr(hierarchical_labels, "completion", completion)
    (tmp_path / variant.name).mkdir()

    result = build_hierarchy("http://server", [], variant, StoragePath(str(tmp_path)))

    assert result == value
    assert len(calls) == 2
    assert calls[1][-2] == {"role": "assistant", "content": hierarchical_labels.json.dumps(invalid)}
    assert "FORMS_TEMPLATES" in calls[1][-1]["content"]


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


def test_assign_document_gives_validation_error_to_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    value, _ = hierarchy()
    invalid = {
        "primary_parent_id": "SCIENCE",
        "secondary_parent_ids": [],
        "primary_leaf_id": "FICTION",
        "secondary_leaf_ids": [],
        "form_id": FORMS[0].bucket_id,
        "confidence": 0.9,
        "rationale": "Test",
    }
    corrected = invalid | {"primary_leaf_id": "BIOLOGY"}
    calls = []

    def completion(*args, **kwargs):
        calls.append(args[1].copy())
        return invalid if len(calls) == 1 else corrected

    monkeypatch.setattr(hierarchical_labels, "completion", completion)
    document = SampleDocument(0, "hash", "hidden", "standard", 0, "Text")

    assignment = assign_document("http://server", document, value, 0)

    assert assignment.primary_leaf_id == "BIOLOGY"
    assert len(calls) == 2
    assert calls[1][-2] == {"role": "assistant", "content": hierarchical_labels.json.dumps(invalid)}
    assert "primary leaf has the wrong parent" in calls[1][-1]["content"]


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


def test_hierarchy_launch_config_keeps_callback_in_server_head() -> None:
    _, variant = hierarchy()

    launch = hierarchy_launch_config(
        "run",
        [variant],
        batch_size=10,
        concurrency=2,
        tensor_parallel_size=4,
        max_model_len=16_384,
        max_num_seqs=4,
    )

    assert launch.client is not None
    assert launch.tensor_parallel_size == 4
    assert launch.server.max_model_len == 16_384
    assert launch.server.max_num_seqs == 4
    assert launch.priority_band == hierarchical_labels.job_pb2.PRIORITY_BAND_INTERACTIVE
