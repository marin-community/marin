# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity ignores row order and keeps overlap flags advisory."""

import hashlib
import json
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.source_fingerprints import DuplicateLabel, SourceFingerprint

from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey, identity_survey_payload
from experiments.post_training.task_curation.duplicate_review import review_pairs
from experiments.post_training.task_curation.source_duplicate_ci import changed_source_declarations
from experiments.post_training.task_curation.source_fingerprints import (
    candidate_pairs,
    duplicate_groups,
    fingerprint_source,
    pair_similarity,
)
from experiments.post_training.task_curation.source_neighbor_survey import cache_population, read_population
from experiments.post_training.task_curation.task_neighbors import (
    PopulationIndex,
    TaskContent,
    compare_populations,
    content_rows,
    exact_reference_integrity,
)
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id


def task_document(identifier: str, instruction: str) -> dict:
    task = TaskSpec(
        id=identifier,
        source=Source(dataset="fixture", revision="pin", row=identifier, importer_revision="v1"),
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="Fixture has no executable reference", contract={}),
    )
    return task.model_dump(mode="json")


def source_input_declaration(identifier: str, revision: str = "pin") -> dict:
    return {
        "id": identifier,
        "dataset_id": "fixture/tasks",
        "dataset_revision": revision,
        "pipeline": {"name": "fixture", "version": "1", "source": "fixture/tasks", "revision": revision, "files": []},
        "actor_census_input": {
            "dataset_id": "fixture/tasks",
            "dataset_revision": revision,
            "converter_version": "1",
            "converter_revision": "converter-v1",
            "measured_tasks": 10,
            "task_specs_sha256": "a" * 64,
            "evidence_url": "s3://fixture/actor-census.json",
        },
    }


def test_source_fingerprint_census_is_order_independent_and_counts_duplicate_contracts():
    rows = [task_document(str(index), f"Compute the square of {index}.") for index in range(30)]
    rows.append(task_document("copied-row", "Compute the square of 2."))
    first = fingerprint_source(rows, {})
    second = fingerprint_source(reversed(rows), {})
    assert first == second
    assert first.task_count == 31 and first.within_source_duplicates == 1
    assert len(first.sample) == 20


def test_measured_review_acknowledgement_is_bound_to_content_evidence(tmp_path):
    rows = [task_document(str(index), f"Compute the square of {index}.") for index in range(25)]
    populations = {key: PopulationIndex(list(content_rows(rows))) for key in ("left", "right")}
    fingerprints = {key: fingerprint_source(rows, {}) for key in populations}
    pairs = candidate_pairs(fingerprints, {"left"}, 1)
    receipts = StoragePath(str(tmp_path / "responses"))
    first = next(review_pairs(pairs, populations, {}, receipts))
    assert first.verdict.label is DuplicateLabel.DUPLICATE
    assert first.acknowledgement is None
    # Write a different receipt directory so an ACK changes no immutable bytes.
    acknowledged = next(
        review_pairs(
            pairs,
            populations,
            {first.prompt_sha256: "Retain both histories"},
            StoragePath(str(tmp_path / "acknowledged")),
        )
    )
    assert acknowledged.acknowledgement == "Retain both histories"
    assert acknowledged.prompt_sha256 == first.prompt_sha256
    assert len(duplicate_groups([acknowledged])) == 1
    changed_rows = [task_document(str(index), f"Name chemical element {index}.") for index in range(25)]
    changed = {**populations, "right": PopulationIndex(list(content_rows(changed_rows)))}
    reviewed = next(
        review_pairs(
            pairs, changed, {first.prompt_sha256: "Retain both histories"}, StoragePath(str(tmp_path / "changed"))
        )
    )
    assert reviewed.verdict.label is DuplicateLabel.DISTINCT
    assert reviewed.acknowledgement is None


def test_source_change_detection_includes_added_and_modified_declarations():
    before = {
        "same": source_input_declaration("same", "a"),
        "changed": source_input_declaration("changed", "old"),
        "removed": source_input_declaration("removed", "a"),
    }
    after = {
        "same": source_input_declaration("same", "a"),
        "changed": source_input_declaration("changed", "new"),
        "added": source_input_declaration("added", "b"),
    }
    assert changed_source_declarations(before, after) == {"changed", "added"}


@pytest.mark.parametrize("field", ["converter_revision", "task_specs_sha256"])
def test_source_content_changes_require_fingerprints_while_measurement_metadata_reuses_them(field):
    before = {"fixture": source_input_declaration("fixture")}
    after = {"fixture": source_input_declaration("fixture")}
    after["fixture"].update(notes="New display notes", task_count=8, lexical_contamination={"measured_tasks": 10})
    after["fixture"]["actor_census_input"]["evidence_url"] = "s3://fixture/copied-census.json"
    assert changed_source_declarations(before, after) == set()
    after["fixture"]["actor_census_input"][field] = "changed"
    assert changed_source_declarations(before, after) == {"fixture"}


def test_known_codeforces_rebuild_ranks_before_distinct_nl2bash():
    fixture = Path(__file__).parent / "fixtures/source-overlap-fingerprints.json"
    fingerprints = {
        key: SourceFingerprint.model_validate(value) for key, value in json.loads(fixture.read_text()).items()
    }
    pairs = candidate_pairs(fingerprints, {"TaskTrove:tasktrove-codeforces-rebuilt-v1"}, 1)
    assert [(pair.left, pair.right) for pair in pairs] == [
        ("TaskTrove:tasktrove-codeforces-rebuilt-v1", "TaskTrove:laion__codeforces-v3")
    ]


def test_alias_only_declaration_change_reuses_published_fingerprints():
    old = "Campaign Release:tasktrove-codeforces-rebuilt-v1"
    current = canonical_source_id(old)
    before = {old: source_input_declaration(old)}
    after = {current: source_input_declaration(current)}
    assert changed_source_declarations(before, after) == set()
    after[current]["pipeline"]["version"] = "changed"
    assert changed_source_declarations(before, after) == {current}


def test_frozen_survey_digest_survives_canonical_source_projection():
    raw = {
        "version": 1,
        "catalog_root": "fixture",
        "snapshot_id": "frozen",
        "generated_at": "2026-11-12T00:00:00Z",
        "fingerprints": {"MarinSkyRL:aime24": None, "MarinSkyRL:svamp": None},
        "reviews": [],
        "groups": [],
        "judge": None,
    }
    survey = canonical_identity_survey(raw)
    assert set(survey.fingerprints) == {"HuggingFaceH4:aime24", "ChilleD:svamp"}
    assert canonical_sha256(identity_survey_payload(survey)) == canonical_sha256(raw)


@pytest.mark.parametrize(
    "cohort,expected,fraction",
    [("permuted", "duplicate", 1.0), ("disjoint", "distinct", 0.0), ("subset", "partial_overlap", 0.5)],
)
def test_content_neighbors_ignore_permuted_ids_and_measure_both_populations(cohort, expected, fraction):
    left = [task_document(str(index), f"Compute the square of {index}.") for index in range(100)]
    if cohort == "disjoint":
        right = [
            task_document(str(index), f"Name the chemical element with atomic number {index}.") for index in range(100)
        ]
    else:
        right = [
            task_document(str(99 - index), f"Compute the square of {index}.")
            for index in range(50 if cohort == "subset" else 100)
        ]
    fingerprints = {"left": fingerprint_source(left, {}), "right": fingerprint_source(right, {})}
    pair = pair_similarity("left", fingerprints["left"], "right", fingerprints["right"])
    review = compare_populations(
        pair, PopulationIndex(list(content_rows(left))), PopulationIndex(list(content_rows(right)))
    )
    assert review.verdict.label == expected
    assert review.verdict.overlap_fraction == pytest.approx(fraction)
    assert review.measurement is not None
    assert review.measurement.left_to_right.exact_fraction == pytest.approx(fraction)
    if cohort == "permuted":
        assert all(row.task_id != row.partner_task_id for row in review.measurement.left_to_right.neighbors)
    if cohort == "subset":
        assert review.measurement.right_to_left.fraction == 1


def test_neighbor_integrity_reports_answer_swap_even_with_identical_id_sets():
    rows = [TaskContent(str(index), str(index), f"Question {index}", f"answer-{index}", "test") for index in range(10)]
    permuted = [
        TaskContent(str(9 - index), row.instruction_hash, row.text, row.reference_hash, row.tests_hash)
        for index, row in enumerate(rows)
    ]
    original, copy = PopulationIndex(rows), PopulationIndex(permuted)
    assert exact_reference_integrity(original, copy)["mismatched"] == 0
    swapped = [
        TaskContent(row.task_id, row.instruction_hash, row.text, f"answer-{9 - index}", row.tests_hash)
        for index, row in enumerate(permuted)
    ]
    result = exact_reference_integrity(original, PopulationIndex(swapped))
    assert result["checked"] == result["mismatched"] == 10
    assert result["examples"][0]["reference_sha256"] != result["examples"][0]["partner_reference_sha256"]


def test_lsh_matches_full_population_after_shared_wrapper_is_stripped():
    wrapper = (
        "Follow these instructions carefully, explain all your reasoning, "
        "then give the final answer in the required format."
    )
    question = " ".join(f"substance{index}" for index in range(100))
    distractors = [
        TaskContent(
            str(index),
            f"hash-{index}",
            f"{wrapper}\nDescribe the unrelated chemical element with atomic number {index}",
            None,
            "test",
        )
        for index in range(20)
    ]
    target = PopulationIndex(
        [*distractors, TaskContent("hidden-partner", "other", f"{wrapper}\n{question}", "reference", "test")]
    )
    query = TaskContent("query", "new", question.replace("substance50", "modified50"), "reference", "test")
    neighbor = target.neighbors([query])[0]
    assert neighbor.partner_task_id == "hidden-partner"
    assert neighbor.method == "lsh" and neighbor.similarity >= 0.9
    assert neighbor.reference_match and neighbor.tests_match


def test_population_cache_verifies_pinned_bytes_and_retains_complete_task_content(tmp_path, monkeypatch):
    documents = [task_document(str(index), f"Compute the square of {index}.") for index in range(30)]
    shard = tmp_path / "population.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": json.dumps(row)} for row in documents]), shard)
    fingerprint = fingerprint_source(
        documents, {"s3://marin-us-east-02a/fixture/specs.parquet": hashlib.sha256(shard.read_bytes()).hexdigest()}
    )
    monkeypatch.setattr(
        "experiments.post_training.task_curation.source_neighbor_survey.filesystem_for",
        lambda uri: (fsspec.filesystem("file"), str(shard)),
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    population = read_population(cache_population("fixture", fingerprint, cache))
    assert len(population.rows) == 30
    assert {row.task_id for row in population.rows} == {str(index) for index in range(30)}
    assert population.neighbors([population.rows[-1]])[0].partner_task_id == "29"
    changed = fingerprint.model_copy(update={"inputs": {"s3://marin-us-east-02a/fixture/specs.parquet": "0" * 64}})
    with pytest.raises(ValueError, match="Input digest differs"):
        cache_population("fixture", changed, cache)
