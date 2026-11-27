# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source identity ignores row order and keeps overlap flags advisory."""

import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

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
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    ToolResult,
    VerifyitGrader,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.source_fingerprints import (
    DirectionalOverlap,
    DuplicateLabel,
    DuplicateReview,
    SourceFingerprint,
    SourceIdentitySurvey,
)

from experiments.post_training.task_curation import native_source_survey
from experiments.post_training.task_curation.catalog_identity import canonical_identity_survey, identity_survey_payload
from experiments.post_training.task_curation.duplicate_review import review_pairs
from experiments.post_training.task_curation.source_duplicate_ci import changed_source_declarations
from experiments.post_training.task_curation.source_duplicates import fingerprint_manifest
from experiments.post_training.task_curation.source_fingerprints import (
    SourceStanding,
    candidate_pairs,
    duplicate_groups,
    fingerprint_source,
    pair_similarity,
)
from experiments.post_training.task_curation.source_neighbor_survey import (
    SourcePair,
    measure_target,
    read_pair_shortlist,
    shortlist_pairs,
    write_pair_shortlist,
)
from experiments.post_training.task_curation.source_population_cache import (
    PopulationLocations,
    PreparedPopulation,
    cache_population,
    population_preflight,
    read_hash_population,
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
from infra.marina.apps.rl_data_catalog.server.duplicate_folding import partial_overlap_notes
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


def test_native_survey_replaces_stale_prepared_locations_before_neighbor_preflight(tmp_path, monkeypatch):
    empty = fingerprint_source([], {"old": "old-digest"})
    measured = {}
    for source in ("first", "second", "retained"):
        uri = f"s3://marin-us-east-02a/{source}/specs.parquet"
        inputs = {uri: source + "-digest"}
        fingerprint = fingerprint_source([task_document(source, f"Solve {source}.")], inputs)
        prepared = PreparedPopulation(
            uri=f"s3://marin-us-east-02a/{source}/prepared.parquet",
            sha256=source + "-prepared-digest",
            inputs_sha256=canonical_sha256(inputs),
            task_count=1,
        )
        measured[source] = (fingerprint, inputs, {source + "-digest": uri}, prepared)
    stale = PreparedPopulation(
        uri="s3://marin-us-east-02a/old/prepared.parquet",
        sha256="old-prepared-digest",
        inputs_sha256=canonical_sha256(empty.inputs),
        task_count=0,
    )
    locations = PopulationLocations(
        files={"retained-digest": measured["retained"][2]["retained-digest"]},
        prepared={"first": stale, "second": stale, "retained": measured["retained"][3]},
    )
    prior = SourceIdentitySurvey(
        catalog_root="fixture",
        snapshot_id="frozen",
        generated_at="frozen",
        fingerprints={"first": empty, "second": empty, "retained": measured["retained"][0]},
        reviews=[],
        groups=[],
        judge=None,
    )
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(prior.model_dump_json())
    monkeypatch.setattr(native_source_survey, "measure_population", lambda source, *args: measured[source])
    monkeypatch.setattr(native_source_survey, "read_locator_manifest", lambda path: locations)

    def run_neighbors(survey, output, work, workers, sample, only_openqa, cpu, current_locations, **kwargs):
        result = population_preflight(survey, current_locations, "marin-us-east-02a")
        assert result["prepared_sources"] == 3
        assert current_locations.prepared["retained"] == locations.prepared["retained"]
        assert locations.prepared["first"] == stale
        return {
            "survey_uri": str(prior_path),
            "survey_sha256": "digest",
            "pairs": 1,
            "labels": {"distinct": 1},
            "changed_pairs": 1,
        }

    monkeypatch.setattr(native_source_survey, "rerun", run_neighbors)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native-source-survey",
            "--source",
            "first=1",
            "--source",
            "second=1",
            "--population-version",
            "new",
            "--survey-root",
            str(tmp_path / "output"),
            "--work-dir",
            str(tmp_path / "work"),
            "--served-survey-uri",
            str(prior_path),
            "--served-survey-sha256",
            hashlib.sha256(prior_path.read_bytes()).hexdigest(),
            "--allocated-cpus",
            "2",
        ],
    )
    native_source_survey.main()


def test_empty_release_fingerprints_without_overlap_candidates():
    empty = fingerprint_source([], {"empty-release.parquet": "digest"})
    populated = fingerprint_source([task_document("task", "Solve this task")], {})

    assert empty.task_count == 0
    assert empty.inputs == {"empty-release.parquet": "digest"}
    assert empty.sample_sha256 == canonical_sha256([])
    assert pair_similarity("empty", empty, "populated", populated).rank_score == 0
    assert pair_similarity("empty", empty, "also-empty", empty).instruction_similarity == 0
    assert candidate_pairs({"empty": empty, "populated": populated}, {"empty", "populated"}, 3) == []
    with pytest.raises(ValueError, match="Empty populations"):
        SourceFingerprint.model_validate({**empty.model_dump(), "sample": populated.sample})


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
    assert len(duplicate_groups([acknowledged], {key: SourceStanding(25) for key in populations})) == 1
    changed = {**populations, "right": PopulationIndex(list(content_rows(rows)), {"shard": "changed-input-digest"})}
    reviewed = next(
        review_pairs(
            pairs, changed, {first.prompt_sha256: "Retain both histories"}, StoragePath(str(tmp_path / "changed"))
        )
    )
    assert reviewed.verdict.label is DuplicateLabel.DUPLICATE
    assert reviewed.prompt_sha256 != first.prompt_sha256
    assert reviewed.acknowledgement is None


def test_measured_pair_receipt_is_sealed_against_retry_outcomes(tmp_path):
    """A pair's outcome is its first complete sealed receipt; a retry cannot replace it."""
    rows = [task_document(str(index), f"Compute the square of {index}.") for index in range(25)]
    populations = {key: PopulationIndex(list(content_rows(rows)), {}) for key in ("left", "right")}
    fingerprints = {key: fingerprint_source(rows, {}) for key in populations}
    pairs = candidate_pairs(fingerprints, {"left"}, 1)
    receipts = StoragePath(str(tmp_path / "responses"))
    first = next(review_pairs(pairs, populations, {}, receipts))
    sealed = receipts / f"{first.prompt_sha256}.json"
    assert sealed.exists()
    with sealed.open("wb") as stream:
        stream.write(b'{"outcome": "a different retry result"}')
    with pytest.raises(ValueError, match="Measured receipt already differs"):
        next(review_pairs(pairs, populations, {}, receipts))


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


def benchmark_fixture_documents(filename: str) -> dict[str, list[dict]]:
    fixture = json.loads((Path(__file__).parent / "fixtures" / filename).read_text())
    populations = {}
    for name, rows in fixture["sources"].items():
        documents = []
        for row in rows:
            document = task_document(row["task_id"], row["text"])
            if row.get("grader") in {"boxed_math", "gpqa"}:
                document["grader"] = ScriptGrader(
                    argv=("python3", "-m", "source_grader.main", row["grader"]),
                    environment=EnvironmentRequirements(command_semantics="linux_process", packages_lock="fixture.lock"),
                ).model_dump(mode="json")
                document["resources"] = ResourceGroups(
                    verifier=(inline_resource("config.json", json.dumps({"expected": row["expected"]}).encode()),)
                ).model_dump(mode="json")
            else:
                parameters = row.get("parameters", {"expected": row["expected"]})
                if row.get("grader") == "numeric":
                    parameters.update(tolerance_abs=0.0, tolerance_rel=0.0)
                elif row.get("grader") == "mcq":
                    parameters["options"] = 4
                document["grader"] = VerifyitGrader(
                    mode=row.get("grader", "exact"),
                    parameters=parameters,
                ).model_dump(mode="json")
            documents.append(document)
        populations[name] = documents
    return populations


def test_real_aime24_wrappers_formatting_and_missing_diagrams_match_every_problem(tmp_path):
    documents = benchmark_fixture_documents("aime24_overlap.json")
    pair = pair_similarity(
        "eval", fingerprint_source(documents["eval"], {}), "marin", fingerprint_source(documents["marin"], {})
    )
    populations = [PopulationIndex(list(content_rows(documents[name])), {}) for name in ("eval", "marin")]
    direct = compare_populations(pair, *populations)
    assert direct.verdict.label is DuplicateLabel.DUPLICATE
    assert direct.measurement.left_to_right.matched == direct.measurement.right_to_left.matched == 30
    statement_matches = [row for row in direct.measurement.left_to_right.neighbors if row.method == "statement"]
    assert len(statement_matches) == 3
    assert all(row.reference_match for row in statement_matches)
    paths = []
    for name, population in zip(("eval", "marin"), populations, strict=True):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist([asdict(row) for row in population.raw_rows]), path)
        paths.append(path)
    cached = compare_populations(pair, *(read_hash_population(path, {}) for path in paths))
    assert cached == direct


def test_real_mmlu_subjects_with_shared_answer_letters_remain_distinct():
    documents = benchmark_fixture_documents("mmlu_overlap_negatives.json")
    populations = [PopulationIndex(list(content_rows(rows)), {}) for rows in documents.values()]
    pair = pair_similarity("algebra", fingerprint_source([], {}), "anatomy", fingerprint_source([], {}))
    review = compare_populations(pair, *populations)
    assert {row.reference_hash for row in populations[0].raw_rows} & {
        row.reference_hash for row in populations[1].raw_rows
    }
    assert review.verdict.label is DuplicateLabel.DISTINCT
    assert review.measurement.left_to_right.matched == review.measurement.right_to_left.matched == 0


def test_real_gpqa_wrappers_and_shuffled_choices_preserve_questions_and_answer_content():
    documents = benchmark_fixture_documents("gpqa_overlap.json")
    populations = [PopulationIndex(list(content_rows(rows)), {}) for rows in documents.values()]
    pair = pair_similarity("eval", fingerprint_source([], {}), "marin", fingerprint_source([], {}))
    review = compare_populations(pair, *populations)
    assert review.verdict.label is DuplicateLabel.DUPLICATE
    assert review.measurement.left_to_right.matched == review.measurement.right_to_left.matched == 8
    assert all(row.reference_match for row in review.measurement.left_to_right.neighbors)


def test_real_math500_math_and_boxed_math_contracts_match_across_wrappers():
    documents = benchmark_fixture_documents("math500_overlap.json")
    populations = [PopulationIndex(list(content_rows(rows)), {}) for rows in documents.values()]
    pair = pair_similarity("eval", fingerprint_source([], {}), "marin", fingerprint_source([], {}))
    review = compare_populations(pair, *populations)
    assert review.verdict.label is DuplicateLabel.DUPLICATE
    assert review.measurement.left_to_right.matched == review.measurement.right_to_left.matched == 6


def test_nonbenchmark_literal_syntax_is_preserved_during_matching():
    populations = [
        PopulationIndex(list(content_rows([task_document("task", text)])), {})
        for text in (r"Emit the literal string '\(x+1\)'.", "Emit the literal string 'x+1'.")
    ]
    pair = pair_similarity("escaped", fingerprint_source([], {}), "plain", fingerprint_source([], {}))
    review = compare_populations(pair, *populations)
    assert review.verdict.label is DuplicateLabel.DISTINCT
    assert review.measurement.left_to_right.matched == review.measurement.right_to_left.matched == 0


@pytest.mark.parametrize("difference", ["answer", "diagram", "number", "state"])
def test_missing_diagram_matching_requires_problem_answer_and_state_agreement(difference):
    documents = benchmark_fixture_documents("aime24_overlap.json")
    left = next(row for row in content_rows(documents["eval"]) if "Rectangles" in row.text)
    right = next(row for row in content_rows(documents["marin"]) if "Rectangles" in row.text)
    if difference == "answer":
        right = replace(right, reference_hash="conflicting-answer")
    elif difference == "diagram":
        # Reusing the prose and answer cannot override two different supplied diagrams.
        right = replace(right, text=right.text + " [asy] " + "draw(other); " * 200 + " [/asy]")
    elif difference == "number":
        right = replace(right, text=right.text.replace("107", "108"))
    else:
        right = replace(right, state_hash="different-prefill")
    query, target = PopulationIndex([left], {}), PopulationIndex([right], {})
    assert target.neighbors([left], query)[0].similarity < 0.9


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


def test_population_cache_verifies_pinned_bytes_and_retains_complete_task_content(tmp_path):
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
    local_paths = {"s3://marin-us-east-02a/fixture/specs.parquet": str(shard)}
    cache = tmp_path / "cache"
    cache.mkdir()
    population = read_population(
        cache_population("fixture", fingerprint, cache, locations, local_paths, source_bucket="marin-us-east-02a"),
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
            local_paths,
            source_bucket="marin-us-east-02a",
        )


def test_legacy_input_fingerprints_name_the_missing_locator(tmp_path, monkeypatch):
    """A pre-uri fingerprint fails with its source named, not with a field-name URI.

    Fingerprints published before inputs became ``{uri: sha256}`` records carry
    provenance fields such as ``dataset_id`` instead. With no prepared locator
    the cache must refuse with the source and the offending key, and a prepared
    locator bound to the legacy inputs' digest still serves the population.
    """
    documents = [task_document(str(index), f"Compute the square of {index}.") for index in range(30)]
    fingerprint = fingerprint_source(documents, {}).model_copy(
        update={
            "inputs": {
                "input_sha256": "a" * 64,
                "dataset_id": "fixture/legacy-cohort",
                "dataset_revision": "b" * 64,
            }
        }
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    with pytest.raises(ValueError, match="Missing co-located population locator for fixture: dataset_id"):
        cache_population("fixture", fingerprint, cache, PopulationLocations(), {}, source_bucket="marin-us-east-02a")

    prepared_payload = pa.Table.from_pylist([asdict(row) for row in content_rows(documents)])
    prepared = tmp_path / "prepared.parquet"
    pq.write_table(prepared_payload, prepared)
    locations = PopulationLocations(
        prepared={
            "fixture": PreparedPopulation(
                uri="s3://marin-us-east-02a/fixture/prepared.parquet",
                sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),
                inputs_sha256=canonical_sha256(fingerprint.inputs),
                task_count=30,
            )
        }
    )
    monkeypatch.setattr(StoragePath, "read_bytes", lambda path: prepared.read_bytes())
    population = read_population(
        cache_population("fixture", fingerprint, cache, locations, {}, source_bucket="marin-us-east-02a"),
        fingerprint.inputs,
    )
    assert len(population.rows) == 30


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
            pair, read_hash_population(paths["left"], {}), read_hash_population(paths["right"], {}), 30
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


def test_lazy_population_direction_combines_exact_matches_and_disjoint_states(tmp_path):
    left = [
        TaskContent(
            str(i), f"hash-{i}", f"Solve chemistry problem {i}.", "ref", "test", "left" if i < 5 else "left-other"
        )
        for i in range(10)
    ]
    right = [
        TaskContent(
            f"partner-{i}",
            f"hash-{i}" if i < 5 else f"different-{i}",
            f"Solve chemistry problem {i}.",
            "ref",
            "test",
            "left" if i < 5 else "right",
        )
        for i in range(10)
    ]
    paths = {}
    for name, rows in [("left", left), ("right", right)]:
        paths[name] = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist([asdict(row) for row in rows]), paths[name])
    directions, _ = measure_target("right", ["left"], paths, {"left": {}, "right": {}}, 10, str(tmp_path / "out"), None)
    result = directions["left", "right"]
    assert result.fraction == result.exact_fraction == 0.5
    assert {row.method for row in result.neighbors} == {"exact", "unmatched"}
    assert {row.partner_task_id for row in result.neighbors if row.method == "exact"} == {
        f"partner-{i}" for i in range(5)
    }


@pytest.mark.parametrize("cache_bytes", [1, 30_000, 1_000_000])
def test_candidate_cache_capacity_preserves_neighbors_on_revisited_questions(cache_bytes):
    common = " ".join(f"molecule{word}" for word in range(80))
    order = [*range(10), *reversed(range(10))]
    source = PopulationIndex(
        [
            TaskContent(f"query-{n}", f"query-{i}", f"{common} unique{i} final.", f"ref-{i}", "test", "state")
            for n, i in enumerate(order)
        ],
        {},
    )
    target = PopulationIndex(
        [
            TaskContent(f"partner-{i}", f"target-{i}", f"{common} unique{i} final!", f"ref-{i}", "test", "state")
            for i in range(10)
        ],
        {},
        shingle_cache_bytes=cache_bytes,
    )
    neighbors = target.neighbors(source.raw_rows, source)
    assert [row.partner_task_id for row in neighbors] == [f"partner-{i}" for i in order]
    assert all(
        row.method == "lsh" and row.similarity >= 0.9 and row.reference_match and row.tests_match for row in neighbors
    )


def test_frozen_shortlist_keeps_population_and_pair_identity(tmp_path):
    fingerprint = fingerprint_source([task_document("task", "Solve this chemistry problem.")], {"shard": "original"})
    pair = pair_similarity("left", fingerprint, "right", fingerprint)
    survey = SourceIdentitySurvey(
        catalog_root="fixture",
        snapshot_id="frozen",
        generated_at="2026-11-14T00:00:00Z",
        fingerprints={"left": fingerprint, "right": fingerprint},
        reviews=[],
        groups=[],
        judge=None,
    )
    path = tmp_path / "shortlist.json"
    digest = write_pair_shortlist(str(path), survey, "fixture", "frozen-survey", [pair])
    restored = read_pair_shortlist(str(path), digest, "frozen-survey", survey)
    assert shortlist_pairs(survey, restored) == [pair]
    assert json.loads(path.read_bytes())["pairs"] == [{"left": "left", "right": "right"}]
    changed = survey.model_copy(
        update={
            "fingerprints": {
                "left": fingerprint.model_copy(update={"inputs": {"shard": "changed"}}),
                "right": fingerprint,
            }
        }
    )
    with pytest.raises(ValueError, match="shortlist input population differs"):
        read_pair_shortlist(str(path), digest, "frozen-survey", changed)
    flipped = restored.model_copy(update={"pairs": [SourcePair(left="right", right="left")]})
    payload = flipped.model_dump_json().encode()
    path.write_bytes(payload)
    replayed = read_pair_shortlist(str(path), hashlib.sha256(payload).hexdigest(), "frozen-survey", survey)
    assert shortlist_pairs(survey, replayed) == [pair]


def overlap_review(left: str, right: str, fraction: float, reverse: float) -> DuplicateReview:
    empty = fingerprint_source([], {})
    directions = [
        DirectionalOverlap(
            population=1000,
            sampled=1000,
            matched=round(value * 1000),
            fraction=value,
            ci_low=value,
            ci_high=value,
            exact_matched=round(value * 1000),
            exact_fraction=value,
            neighbors=[],
            neighbors_sha256=canonical_sha256([]),
        )
        for value in (fraction, reverse)
    ]
    return measured_review(pair_similarity(left, empty, right, empty), *directions, 1000, "left", "right")


def test_duplicate_groups_partial_chain_stays_unfolded_with_directional_notes():
    reviews = [overlap_review("a", "b", 0.23, 0.04), overlap_review("b", "c", 0.7, 0.3)]
    groups = duplicate_groups(reviews, {key: SourceStanding(1000) for key in "abc"})
    assert groups == []
    notes = partial_overlap_notes(reviews, groups)
    assert notes["a"] == [{"other_id": "b", "source_fraction": 0.23, "other_fraction": 0.04}]
    assert notes["b"][0] == {"other_id": "a", "source_fraction": 0.04, "other_fraction": 0.23}


@pytest.mark.parametrize("fraction,folded", [(0.949, False), (0.95, True), (0.97, True)])
def test_duplicate_groups_measured_subset_threshold(fraction, folded):
    reviews = [overlap_review("small", "large", fraction, 0.1)]
    groups = duplicate_groups(reviews, {"small": SourceStanding(100), "large": SourceStanding(1000)})
    assert len(groups) == int(folded)
    if folded:
        assert groups[0].canonical_source_id == "large"
        assert {m.source_id: m.overlap_with_canonical for m in groups[0].members} == {"large": 1, "small": fraction}
        assert partial_overlap_notes(reviews, groups) == {}


@pytest.mark.parametrize(
    "standings,canonical",
    [
        ({"a": SourceStanding(10, "FULL"), "b": SourceStanding(20, "QUICK")}, "b"),
        ({"a": SourceStanding(20, "SAMPLE"), "b": SourceStanding(20, "FULL")}, "b"),
        ({"a": SourceStanding(20, "QUICK"), "b": SourceStanding(20, "SAMPLE")}, "b"),
        ({"a": SourceStanding(20, "FULL"), "b": SourceStanding(20, "FULL")}, "a"),
    ],
)
def test_duplicate_groups_mutual_duplicates_choose_count_quality_then_id(standings, canonical):
    groups = duplicate_groups([overlap_review("a", "b", 1, 1)], standings)
    assert len(groups) == 1
    assert groups[0].canonical_source_id == canonical


def test_duplicate_groups_unmeasured_duplicate_verdict_never_folds():
    review = overlap_review("a", "b", 1, 1).model_copy(update={"measurement": None})
    assert duplicate_groups([review], {key: SourceStanding(100) for key in "ab"}) == []
    assert partial_overlap_notes([review], []) == {}


def test_duplicate_groups_containment_chain_requires_direct_canonical_measurement():
    reviews = [overlap_review("a", "b", 0.97, 0.1), overlap_review("b", "c", 0.97, 0.1)]
    groups = duplicate_groups(
        reviews, {key: SourceStanding(count) for key, count in zip("abc", [10, 100, 1000], strict=True)}
    )
    assert len(groups) == 1
    assert groups[0].canonical_source_id == "c"
    assert {m.source_id for m in groups[0].members} == {"b", "c"}


def test_duplicate_groups_one_way_superset_precedes_inconsistent_population_count():
    groups = duplicate_groups(
        [overlap_review("a", "b", 0.97, 0.1)], {"a": SourceStanding(1000), "b": SourceStanding(100)}
    )
    assert groups[0].canonical_source_id == "b"


def test_duplicate_groups_real_numina_dapo_deepscaler_hendrycks_regression():
    # Frozen served v9.3 survey; neighbor rows omitted, all measured fractions retained.
    document = json.loads((Path(__file__).parent / "fixtures/math_overlap_reviews.json").read_text())
    reviews = [DuplicateReview.model_validate(row) for row in document["reviews"]]
    standings = {row["id"]: SourceStanding(row["task_count"], row["quality"]) for row in document["sources"]}
    groups = duplicate_groups(reviews, standings)
    folded = {
        m.source_id: g.canonical_source_id for g in groups for m in g.members if m.source_id != g.canonical_source_id
    }
    assert "AI-MO:numina_math" not in folded
    assert "BytedTsinghua-SIA:dapo_math" not in folded
    assert folded["EleutherAI:hendrycks_math"] == "agentica-org:deepscaler"
    notes = partial_overlap_notes(reviews, groups)
    assert next(n for n in notes["agentica-org:deepscaler"] if n["other_id"] == "AI-MO:numina_math") == {
        "other_id": "AI-MO:numina_math",
        "source_fraction": 0.7294921875,
        "other_fraction": 0.042236328125,
    }
    assert not any(
        {r.similarity.left, r.similarity.right} == {"AI-MO:numina_math", "BytedTsinghua-SIA:dapo_math"} for r in reviews
    )


def test_real_e2egit_subset_folds_under_directly_measured_larger_mix():
    document = json.loads((Path(__file__).parent / "fixtures/e2egit_overlap_reviews.json").read_text())
    reviews = [DuplicateReview.model_validate(row) for row in document["reviews"]]
    standings = {row["id"]: SourceStanding(row["task_count"]) for row in document["sources"]}
    groups = duplicate_groups(reviews, standings)
    assert len(groups) == 1
    group = groups[0]
    assert group.canonical_source_id == "TaskTrove:laion__mix_h11_single_skill_only-v2"
    assert {member.source_id for member in group.members} == standings.keys()
    e2egit = "TaskTrove:DCAgent__exp_rpt_e2egit-v2"
    h10 = "TaskTrove:laion__mix_h10_reward_proportional-v2"
    review = next(row for row in reviews if {row.similarity.left, row.similarity.right} == {e2egit, h10})
    assert review.verdict.label is DuplicateLabel.PARTIAL_OVERLAP
    assert review.measurement.left_to_right.exact_matched == 475
    assert review.measurement.left_to_right.population == 477
