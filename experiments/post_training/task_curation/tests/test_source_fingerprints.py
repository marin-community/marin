# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity ignores row order and keeps overlap flags advisory."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationInput,
    ConversationToolCall,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
    ToolResult,
)
from taskcompendium.source_fingerprints import DuplicateLabel, SourceFingerprint

from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey, identity_survey_payload
from experiments.post_training.task_curation.duplicate_review import review_pairs
from experiments.post_training.task_curation.source_duplicate_ci import changed_source_declarations
from experiments.post_training.task_curation.source_duplicates import fingerprint_manifest
from experiments.post_training.task_curation.source_fingerprints import (
    candidate_pairs,
    duplicate_groups,
    fingerprint_source,
    pair_similarity,
)
from experiments.post_training.task_curation.source_neighbor_survey import measure_target
from experiments.post_training.task_curation.source_population_cache import (
    PopulationLocations,
    PreparedPopulation,
    cache_population,
    read_normalized_population,
    read_population,
)
from experiments.post_training.task_curation.task_neighbors import (
    PopulationIndex,
    TaskContent,
    compare_populations,
    content_rows,
    exact_reference_integrity,
    measured_review,
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
    populations = {key: PopulationIndex(list(content_rows(rows)), {}) for key in ("left", "right")}
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
    changed = {**populations, "right": PopulationIndex(list(content_rows(rows)), {"shard": "changed-input-digest"})}
    reviewed = next(
        review_pairs(
            pairs, changed, {first.prompt_sha256: "Retain both histories"}, StoragePath(str(tmp_path / "changed"))
        )
    )
    assert reviewed.verdict.label is DuplicateLabel.DUPLICATE
    assert reviewed.prompt_sha256 != first.prompt_sha256
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
        "reviews": [
            {
                "similarity": {
                    "left": "MarinSkyRL:aime24",
                    "right": "MarinSkyRL:svamp",
                    "rank_score": 0.0,
                    "instruction_similarity": 0.0,
                    "instruction_overlap": 0.0,
                    "reference_overlap": 0.0,
                    "test_overlap": 0.0,
                    "task_id_overlap": 0.0,
                    "url_overlap": 0.0,
                    "shared_lineage": False,
                },
                "verdict": {"label": "distinct", "overlap_fraction": 0.0, "reason": "Archived opinion"},
                "prompt_sha256": "receipt",
                "model": "archived",
                "completion_id": "old",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "acknowledgement": None,
            }
        ],
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
        pair, PopulationIndex(list(content_rows(left)), {}), PopulationIndex(list(content_rows(right)), {})
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
    rows = [
        TaskContent(str(index), str(index), f"Question {index}", f"answer-{index}", "test", "state")
        for index in range(10)
    ]
    permuted = [
        TaskContent(str(9 - index), row.instruction_hash, row.text, row.reference_hash, row.tests_hash, "state")
        for index, row in enumerate(rows)
    ]
    original, copy = PopulationIndex(rows, {}), PopulationIndex(permuted, {})
    assert exact_reference_integrity(original, copy)["mismatched"] == 0
    swapped = [
        TaskContent(row.task_id, row.instruction_hash, row.text, f"answer-{9 - index}", row.tests_hash, "state")
        for index, row in enumerate(permuted)
    ]
    result = exact_reference_integrity(original, PopulationIndex(swapped, {}))
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
            "state",
        )
        for index in range(20)
    ]
    target = PopulationIndex(
        [*distractors, TaskContent("hidden-partner", "other", f"{wrapper}\n{question}", "reference", "test", "state")],
        {},
    )
    query = TaskContent("query", "new", question.replace("substance50", "modified50"), "reference", "test", "state")
    neighbor = target.neighbors([query])[0]
    assert neighbor.partner_task_id == "hidden-partner"
    assert neighbor.method == "lsh" and neighbor.similarity >= 0.9
    assert neighbor.reference_match and neighbor.tests_match


def test_population_cache_verifies_pinned_bytes_and_retains_complete_task_content(tmp_path, monkeypatch):
    documents = [task_document(str(index), f"Compute the square of {index}.") for index in range(30)]
    shard = tmp_path / "population.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": json.dumps(row)} for row in documents]), shard)
    survey = fingerprint_manifest(
        {
            "catalog_root": "fixture",
            "snapshot_id": "frozen",
            "source_ids": ["fixture"],
            "inputs": [
                {
                    "source_id": "fixture",
                    "paths": [str(shard)],
                    "input_uris": ["s3://marin-us-east-02a/fixture/specs.parquet"],
                    "column": "task_spec",
                    "expected_tasks": 30,
                }
            ],
        }
    )
    fingerprint = survey.fingerprints["fixture"]
    assert fingerprint is not None
    assert list(fingerprint.inputs) == ["s3://marin-us-east-02a/fixture/specs.parquet"]
    digest = next(iter(fingerprint.inputs.values()))
    fingerprint = fingerprint.model_copy(update={"inputs": {"0:tasks.parquet": digest}})
    locations = PopulationLocations(files={digest: "s3://marin-us-east-02a/fixture/specs.parquet"})
    monkeypatch.setattr(
        "experiments.post_training.task_curation.source_population_cache.filesystem_for",
        lambda uri: (fsspec.filesystem("file"), str(shard)),
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    population = read_population(
        cache_population("fixture", fingerprint, cache, locations, {}, source_bucket="marin-us-east-02a"),
        fingerprint.inputs,
    )
    assert len(population.rows) == 30
    assert {row.task_id for row in population.rows} == {str(index) for index in range(30)}
    assert population.neighbors([population.rows[-1]])[0].partner_task_id == "29"
    assert fingerprint.inputs == {"0:tasks.parquet": digest}
    changed = fingerprint.model_copy(update={"inputs": {"0:tasks.parquet": "0" * 64}})
    with pytest.raises(ValueError, match="Input digest differs"):
        cache_population(
            "fixture",
            changed,
            cache,
            PopulationLocations(files={"0" * 64: "s3://marin-us-east-02a/fixture/specs.parquet"}),
            {},
            source_bucket="marin-us-east-02a",
        )


def test_published_measurement_digest_survives_truncation_and_output_location():
    documents = [task_document(str(index), f"Compute the square of {index}.") for index in range(30)]
    fingerprint = fingerprint_source(documents, {})
    pair = pair_similarity("left", fingerprint, "right", fingerprint)
    index = PopulationIndex(list(content_rows(documents)), {"shard": "digest"})
    full = compare_populations(pair, index, index)
    measurement = full.measurement
    assert measurement is not None
    left = measurement.left_to_right.model_copy(
        update={
            "neighbors": measurement.left_to_right.neighbors[:2],
            "neighbor_receipts_uri": "s3://elsewhere/evidence.json",
        }
    )
    right = measurement.right_to_left.model_copy(
        update={
            "neighbors": measurement.right_to_left.neighbors[:2],
            "neighbor_receipts_uri": "s3://another/evidence.json",
        }
    )
    published = measured_review(
        pair.model_copy(update={"rank_score": 0}),
        left,
        right,
        measurement.sample_size,
        index.inputs_sha256,
        index.inputs_sha256,
    )
    assert published.prompt_sha256 == full.prompt_sha256


@pytest.mark.parametrize("state_kind", ["assistant", "tool"])
def test_content_matching_preserves_completion_and_tool_state(state_kind):
    left, right = [], []
    for index in range(10):
        for rows, content in [(left, "("), (right, "{")]:
            document = task_document(str(index), "Complete the unfinished structure.")
            task = TaskSpec.model_validate(document)
            if state_kind == "assistant":
                extra = (TextMessage(role="assistant", content=content),)
            else:
                extra = (
                    AssistantToolCalls(calls=(ConversationToolCall(call_id="state", name="observe", arguments={}),)),
                    ToolResult(call_id="state", content=content),
                )
            task = task.model_copy(update={"context": ConversationInput(events=(*task.context.events, *extra))})
            rows.append(task.model_dump(mode="json"))
    fingerprints = [fingerprint_source(rows, {}) for rows in (left, right)]
    pair = pair_similarity("left", fingerprints[0], "right", fingerprints[1])
    review = compare_populations(
        pair, PopulationIndex(list(content_rows(left)), {}), PopulationIndex(list(content_rows(right)), {})
    )
    assert review.verdict.label is DuplicateLabel.DISTINCT
    assert review.measurement.left_to_right.fraction == 0


def test_failed_prepared_population_is_rejected_again_on_retry(tmp_path, monkeypatch):
    documents = [task_document(str(index), f"Compute the square of {index}.") for index in range(2)]
    fingerprint = fingerprint_source(documents, {"opaque": "original-input-digest"})
    prepared = tmp_path / "wrong-count.parquet"
    pq.write_table(pa.Table.from_pylist([asdict(next(iter(content_rows(documents))))]), prepared)
    payload = prepared.read_bytes()
    locations = PopulationLocations(
        prepared={
            "fixture": PreparedPopulation(
                uri="s3://marin-us-east-02a/fixture/prepared.parquet",
                sha256=hashlib.sha256(payload).hexdigest(),
                inputs_sha256=canonical_sha256(fingerprint.inputs),
                task_count=2,
            )
        }
    )
    monkeypatch.setattr(StoragePath, "read_bytes", lambda path: payload)
    cache = tmp_path / "cache"
    cache.mkdir()
    for _ in range(2):
        with pytest.raises(ValueError, match="Prepared population count differs"):
            cache_population("fixture", fingerprint, cache, locations, {}, source_bucket="marin-us-east-02a")


def test_normalized_population_cache_preserves_neighbors_with_repeated_task_ids(tmp_path):
    wrapper = "Follow these instructions carefully and explain your reasoning before giving the final answer."
    left = [
        TaskContent(
            "reused-id",
            f"hash-{index}",
            f"{wrapper}\n" + " ".join(f"substance{index}_{word}" for word in range(60)) + " detail.",
            "ref",
            "test",
            "state",
        )
        for index in range(30)
    ]
    right = [
        TaskContent("partner", f"other-{index}", row.text.replace("detail.", "detail!"), "ref", "test", "state")
        for index, row in enumerate(left)
    ]
    paths = {}
    for name, rows in [("left", left), ("right", right)]:
        paths[name] = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist([asdict(row) for row in rows]), paths[name])
    pair = pair_similarity(
        "left",
        fingerprint_source([task_document("a", "a")], {}),
        "right",
        fingerprint_source([task_document("b", "b")], {}),
    )
    expected = compare_populations(pair, PopulationIndex(left, {}), PopulationIndex(right, {}), 30)
    assert expected.measurement.left_to_right.matched == 30
    for _ in range(2):
        actual = compare_populations(
            pair, read_normalized_population(paths["left"], {}), read_normalized_population(paths["right"], {}), 30
        )
        assert actual.model_dump(mode="json") == expected.model_dump(mode="json")
    directions, _ = measure_target(
        "right", ["left"], paths, {"left": {}, "right": {}}, 30, str(tmp_path / "prior"), None
    )
    resumed, _ = measure_target(
        "right",
        ["left"],
        {**paths, "left": tmp_path / "not-downloaded.parquet"},
        {"left": {}, "right": {}},
        30,
        str(tmp_path / "resumed"),
        str(tmp_path / "prior"),
    )
    assert resumed["left", "right"].model_dump(exclude={"neighbor_receipts_uri"}) == directions[
        "left", "right"
    ].model_dump(exclude={"neighbor_receipts_uri"})
    assert resumed["left", "right"].neighbors_sha256 == expected.measurement.left_to_right.neighbors_sha256
