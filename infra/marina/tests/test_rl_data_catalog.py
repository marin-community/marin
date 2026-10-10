# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError

from infra.marina.applets.rl_data_catalog.server.app import (
    active_sources_with_review_state,
    difficulty_summary,
    migrate,
    refresh_catalog,
    save_snapshot,
    source_with_review,
)
from infra.marina.applets.rl_data_catalog.server.catalog import Snapshot


def write_catalog(path: Path, rows: list[dict]) -> None:
    revision = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(json.dumps({"schema_version": 1, "revision": revision, "sources": rows}))


@pytest.fixture
def catalog_connection(database_url: str, tmp_path: Path) -> Iterator[Connection]:
    engine = create_engine(database_url)
    schema = "catalog_test_" + uuid.uuid4().hex
    with engine.connect() as connection, connection.begin():
        connection.execute(text(f"CREATE SCHEMA {schema}"))
        connection.execute(text(f"SET LOCAL search_path TO {schema}"))
        empty_catalog = tmp_path / "empty.json"
        write_catalog(empty_catalog, [])
        migrate(connection, empty_catalog)
        yield connection
        # Roll back all fixture writes, including the schema.
        connection.rollback()
    engine.dispose()


def test_generated_catalog_reconciles_sources_and_preserves_reviews(
    catalog_connection: Connection, tmp_path: Path
) -> None:
    connection = catalog_connection
    keep = {"id": "MarinSkyRL:math", "origin": "MarinSkyRL", "task_count": 7, "dataset_revision": "data1"}
    save_snapshot(connection, Snapshot("MarinSkyRL", "old", "2026-09-01", [keep, {"id": "MarinSkyRL:retired"}]))
    save_snapshot(connection, Snapshot("Retired origin", "old", "2026-09-01", [{"id": "retired:source"}]))
    connection.execute(
        text("UPDATE catalog_sources SET quality = 'good', traces = 12, review_id = 'saved' WHERE id = :id"),
        {"id": keep["id"]},
    )
    artifact = tmp_path / "catalog.json"
    write_catalog(
        artifact, [{**keep, "task_count": 11}, {"id": "Task Trove:new", "origin": "Task Trove", "task_count": 4}]
    )
    result = refresh_catalog(connection, artifact)
    active = {
        row["id"]: row for row in connection.execute(text("SELECT * FROM catalog_sources WHERE active")).mappings()
    }
    assert set(active) == {"MarinSkyRL:math", "Task Trove:new"}
    assert active[keep["id"]]["payload"]["task_count"] == 11
    assert (active[keep["id"]]["quality"], active[keep["id"]]["traces"], active[keep["id"]]["review_id"]) == (
        "good",
        12,
        "saved",
    )
    assert all(item["changed"] for item in result["results"])
    assert not any(item["changed"] for item in refresh_catalog(connection, artifact)["results"])
    assert all(item["changed"] for item in refresh_catalog(connection, artifact, force=True)["results"])
    assert connection.execute(text("SELECT COUNT(*) FROM catalog_sources")).scalar_one() == 4


@pytest.mark.parametrize("corruption", ["duplicate", "stale_revision"])
def test_malformed_generated_catalog_preserves_saved_inventory(
    catalog_connection: Connection, tmp_path: Path, corruption: str
) -> None:
    connection = catalog_connection
    old = {"id": "MarinSkyRL:math", "origin": "MarinSkyRL", "task_count": 7}
    save_snapshot(connection, Snapshot("MarinSkyRL", "old", "2026-09-01", [old]))
    artifact = tmp_path / "catalog.json"
    write_catalog(artifact, [old, old] if corruption == "duplicate" else [old])
    if corruption == "stale_revision":
        catalog = json.loads(artifact.read_text())
        catalog["sources"][0]["task_count"] = 20
        artifact.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match=r"Duplicate catalog source|Catalog revision does not match"):
        refresh_catalog(connection, artifact)
    assert connection.execute(text("SELECT payload FROM catalog_sources WHERE active")).scalar_one() == old
    assert connection.execute(text("SELECT revision FROM catalog_refreshes")).scalar_one() == "old"


def test_snapshot_replacement_retires_removed_rows_without_affecting_other_origin(
    catalog_connection: Connection,
) -> None:
    connection = catalog_connection
    save_snapshot(connection, Snapshot("MarinSkyRL", "sky1", "2026-09-01", [{"id": "sky:old"}, {"id": "sky:keep"}]))
    save_snapshot(connection, Snapshot("Task Trove", "trove1", "2026-09-01", [{"id": "trove:old"}]))
    save_snapshot(connection, Snapshot("MarinSkyRL", "sky2", "2026-09-28", [{"id": "sky:keep"}, {"id": "sky:new"}]))
    active = set(connection.execute(text("SELECT id FROM catalog_sources WHERE active")).scalars())
    assert active == {"sky:keep", "sky:new", "trove:old"}
    assert connection.execute(text("SELECT COUNT(*) FROM catalog_sources")).scalar_one() == 4


@pytest.mark.parametrize("changed_field", [None, "dataset_revision", "verifier_revision"])
@pytest.mark.parametrize("review_origin", ["database", "source"])
def test_changed_source_preserves_historical_review_but_invalidates_current_rating(changed_field, review_origin) -> None:
    payload = {"id": "MarinSkyRL:math", "dataset_revision": "data1", "verifier_revision": "code1"}
    if changed_field:
        payload[changed_field] = "new-revision"
    review = {
        "quality": "good",
        "difficulty": "32/32",
        "traces": 3,
        "review_id": "review1" if review_origin == "database" else None,
        "review_date": "2026-09-28",
        "review_source_revision": "data1",
        "review_verifier_revision": "code1",
    }
    record = {"payload": payload, "verifier_issues": [], "grading_enrolled": False}
    if review_origin == "source":
        payload.update(review)
        record.update({key: None for key in review})
    else:
        record.update(review)
    row = source_with_review(record)
    assert row["review_id"] == review["review_id"]
    assert row["review_date"] == "2026-09-28"
    assert row["review_stale"] == bool(changed_field)
    assert row["review_applicability"] == ("stale" if changed_field else "current")
    assert row["quality"] == (None if changed_field else "good")
    assert row["difficulty"] == (None if changed_field else "32/32")


@pytest.mark.parametrize("quality,revision", [("good", "data1"), ("some_issues", "data1"), ("good", "data2")])
def test_difficulty_comparison_uses_saved_counts_and_hides_ineligible_measurements(quality, revision) -> None:
    report = {
        "estimated_at": "2026-09-29",
        "sampling": {"task_count": 32, "method": "uniform"},
        "models": [{"size": "large", "model": "model-a", "solved": 17, "verified": 32, "solve_rate": 17 / 32}],
        "protocol_followups": [
            {"state": "complete", "kind": "alternate_checkpoint", "model": "model-b", "solved": 20, "verified": 32},
            {
                "state": "complete",
                "changed_parameter": {"chat_template_kwargs": {"reasoning_effort": "low"}},
                "solved": 7,
                "verified": 32,
            },
            {"state": "pending", "model": "unfinished-model"},
        ],
    }
    row = source_with_review(
        {
            "payload": {"id": "MarinSkyRL:math", "dataset_revision": revision, "verifier_revision": "code1"},
            "quality": quality,
            "difficulty": "Legacy summary: Large 30/32",
            "difficulty_report": json.dumps(report),
            "traces": None,
            "review_id": "review1",
            "review_date": "2026-09-28",
            "review_source_revision": "data1",
            "review_verifier_revision": "code1",
            "verifier_issues": [],
            "grading_enrolled": False,
        }
    )
    if quality != "good" or revision != "data1":
        assert row["difficulty_summary"] is None
        return
    models = row["difficulty_summary"]["models"]
    assert row["difficulty_summary"]["status"] == "historical"
    assert all(model["measurement_status"] == "historical" for model in models)
    assert [(model["size"], model["model"], model["solved"], model["verified"]) for model in models] == [
        ("large", "model-a", 17, 32),
        ("hosted", "model-b", 20, 32),
        ("followup", None, 7, 32),
    ]
    assert models[0]["solve_rate"] == 17 / 32
    assert models[2]["display_name"] == "Generation setting follow-up"
    assert "Legacy summary: Large 30/32" not in row["difficulty"]


@pytest.mark.parametrize(
    "change,expected_status",
    [
        (None, "current"),
        ("awq_as_large", "invalid"),
        ("9b_as_small", "invalid"),
        ("old_output_budget", "invalid"),
        ("unbounded_context", "invalid"),
        ("unbounded_input", "invalid"),
        ("large_thinking_enabled", "invalid"),
        ("large_old_sampling", "invalid"),
        ("hosted_max_effort", "invalid"),
        ("inherited_sampling_defaults", "invalid"),
        ("judge_verifier_protocol", "current"),
        ("judge_verifier_missing_settings", "invalid"),
        ("judge_verifier_thinking_enabled", "invalid"),
        ("checklist_judge_protocol", "current"),
        ("checklist_judge_missing_settings", "invalid"),
        ("checklist_judge_wrong_model", "invalid"),
        ("checklist_judge_wrong_source", "invalid"),
        ("old_protocol", "historical"),
    ],
)
def test_current_difficulty_does_not_accept_legacy_model_roles_or_unmatched_budgets(change, expected_status) -> None:
    parameters = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_tokens": 16384,
    }
    report = {
        "estimated_at": "2026-09-29",
        "sampling": {"task_count": 32, "method": "uniform"},
        "protocol": {
            "id": "atlas-difficulty-v3-65k16k-qwen-recommended-nonthinking",
            "context_window": 65536,
            "max_input_tokens": 49152,
            "max_output_tokens": 16384,
        },
        "models": [
            {
                "size": "small",
                "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "solved": 11,
                "verified": 32,
                "generation_parameters": dict(parameters),
            },
            {
                "size": "large",
                "model": "Qwen/Qwen3.5-122B-A10B",
                "solved": 17,
                "verified": 32,
                "generation_parameters": {
                    **parameters,
                    "top_p": 0.8,
                    "presence_penalty": 1.5,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            },
            {
                "size": "hosted",
                "model": "zai-org/GLM-5.3",
                "solved": 21,
                "verified": 32,
                "generation_parameters": {
                    **parameters,
                    "reasoning_effort": "low",
                },
            },
        ],
    }
    if change == "awq_as_large":
        report["models"][1]["model"] = "cyankiwi/GLM-5.3-AWQ-INT4"
    elif change == "9b_as_small":
        report["models"][0]["model"] = "Qwen/Qwen3.5-9B"
    elif change == "old_output_budget":
        report["models"][1]["generation_parameters"]["max_tokens"] = 8192
    elif change == "unbounded_context":
        report["protocol"]["context_window"] = 131072
    elif change == "unbounded_input":
        report["protocol"]["max_input_tokens"] = 65536
    elif change == "large_thinking_enabled":
        report["models"][1]["generation_parameters"]["chat_template_kwargs"]["enable_thinking"] = True
    elif change == "large_old_sampling":
        report["models"][1]["generation_parameters"]["top_p"] = 0.95
    elif change == "hosted_max_effort":
        report["models"][2]["generation_parameters"]["reasoning_effort"] = "max"
    elif change == "inherited_sampling_defaults":
        report["models"][0]["generation_parameters"].pop("repetition_penalty")
    elif change in ("judge_verifier_protocol", "judge_verifier_missing_settings", "judge_verifier_thinking_enabled"):
        report["protocol"]["id"] = "atlas-difficulty-v4-judge-verifier-nonthinking"
        if change != "judge_verifier_missing_settings":
            report["verifier_configuration"] = {
                "tasktrove_judge": {
                    "model": "Qwen/Qwen3.5-9B",
                    "provider": "Together",
                    "chat_template_kwargs": {"enable_thinking": change == "judge_verifier_thinking_enabled"},
                    "relay_script_sha256": "a" * 64,
                }
            }
    elif change in (
        "checklist_judge_protocol",
        "checklist_judge_missing_settings",
        "checklist_judge_wrong_model",
        "checklist_judge_wrong_source",
    ):
        report["protocol"]["id"] = "atlas-difficulty-v4-checklist-judge"
        report["atlas_id"] = (
            "Task Trove:laion__nemotron-gym-safety-v3"
            if change != "checklist_judge_wrong_source"
            else "Task Trove:laion__unrelated-v1"
        )
        if change != "checklist_judge_missing_settings":
            report["verifier_configuration"] = {
                "tasktrove_judge": {
                    "model": (
                        "deepseek-ai/DeepSeek-V4-Pro-0813"
                        if change != "checklist_judge_wrong_model"
                        else "Qwen/Qwen3.5-9B"
                    ),
                    "provider": "Together",
                    "base_url": "https://api.together.xyz/v1",
                    "api_key_reference": "${TOGETHER_API_KEY}",
                    "chat_template_kwargs": {},
                    "substitution_reason": "OpenAI credits exhausted; the source verifier leaves model blank.",
                }
            }
    elif change == "old_protocol":
        report["protocol"]["id"] = "atlas-difficulty-v2-65k16k"
    row = source_with_review(
        {
            "payload": {
                "id": report.get("atlas_id", "MarinSkyRL:math"),
                "dataset_revision": "data1",
                "verifier_revision": "code1",
            },
            "quality": "good",
            "difficulty": "Small / Large / Hosted comparison",
            "difficulty_report": json.dumps(report),
            "traces": None,
            "review_id": "review1",
            "review_date": "2026-09-29",
            "review_source_revision": "data1",
            "review_verifier_revision": "code1",
            "verifier_issues": [],
            "grading_enrolled": False,
        }
    )
    assert row["difficulty_summary"]["status"] == expected_status
    assert {model["measurement_status"] for model in row["difficulty_summary"]["models"]} == {expected_status}
    assert [model["solved"] for model in row["difficulty_summary"]["models"]] == [11, 17, 21]


def test_difficulty_summary_surfaces_ordering_audit() -> None:
    report = {
        "estimated_at": "2026-09-30",
        "sampling": {"task_count": 32},
        "models": [],
        "protocol": {"artifacts": [{"path": "difficulty/v3/ordering-audit.json"}]},
        "limitations": ["The hosted arm exhausted its output budget while reasoning on 12 tasks."],
    }
    assert difficulty_summary(report)["ordering_warning"] == report["limitations"][0]
    report["protocol"]["artifacts"] = []
    assert difficulty_summary(report)["ordering_warning"] is None


@pytest.mark.parametrize("original_quality", ["good", "bad"])
def test_confirmed_verifier_defect_survives_publication_and_refresh(
    catalog_connection: Connection, original_quality: str
) -> None:
    connection = catalog_connection
    payload = {"id": "MarinSkyRL:math", "dataset_revision": "data1", "verifier_revision": "code1"}
    save_snapshot(connection, Snapshot("MarinSkyRL", "code1", "2026-09-29", [payload]))
    connection.execute(
        text("UPDATE catalog_sources SET quality = :quality, difficulty = '32/32' WHERE id = :id"),
        {"quality": original_quality, "id": payload["id"]},
    )
    connection.execute(
        text(
            """
            INSERT INTO catalog_reviews (id, source_id, collection, updated_at)
            VALUES ('defect-review', :id, '{}'::jsonb, NOW())
        """
        ),
        {"id": payload["id"]},
    )
    connection.execute(
        text(
            """
            INSERT INTO catalog_verifier_issues
                (source_id, issue_url, review_id, status, created_at, updated_at)
            VALUES (:id, 'https://github.com/example/issues/1', 'defect-review', 'open', NOW(), NOW())
        """
        ),
        {"id": payload["id"]},
    )
    expected = "bad" if original_quality == "bad" else "some_issues"
    record = dict(connection.execute(text("SELECT * FROM catalog_sources")).mappings().one())
    assert (record["quality"], record["difficulty"]) == (expected, None)

    # A publisher cannot restore a green rating or difficulty while the defect is open.
    connection.execute(
        text("UPDATE catalog_sources SET quality = :quality, difficulty = '32/32'"),
        {"quality": "good"},
    )
    changed = {**payload, "dataset_revision": "data2", "verifier_revision": "code2"}
    save_snapshot(connection, Snapshot("MarinSkyRL", "code2", "2026-09-30", [changed]))
    record = dict(connection.execute(text("SELECT * FROM catalog_sources")).mappings().one())
    assert (record["quality"], record["difficulty"]) == (expected, None)
    record.update(
        review_id="historical-review",
        review_source_revision="data1",
        review_verifier_revision="code1",
        verifier_issues=[{"issue_url": "https://github.com/example/issues/1", "status": "open"}],
        grading_enrolled=False,
    )
    displayed = source_with_review(record)
    assert displayed["review_stale"]
    assert displayed["quality"] == expected
    assert displayed["difficulty"] is None


@pytest.mark.parametrize("verifier_revision", ["code2", None])
def test_verifier_defect_requires_a_validated_current_review_before_green_restoration(
    catalog_connection: Connection,
    verifier_revision: str | None,
) -> None:
    connection = catalog_connection
    origin = "MarinSkyRL" if verifier_revision else "Task Trove"
    source_id = f"{origin}:math"
    issue_url = "https://github.com/example/issues/1"
    payload = {"id": source_id, "dataset_revision": "data2", "verifier_revision": verifier_revision}
    save_snapshot(connection, Snapshot(origin, "code2", "2026-09-29", [payload]))
    connection.execute(
        text(
            """
            INSERT INTO catalog_reviews (id, source_id, collection, updated_at)
            VALUES ('defect-review', :id, '{}'::jsonb, '2026-09-28'),
                ('new-review', :id, '{}'::jsonb, '2026-09-29')
        """
        ),
        {"id": source_id},
    )
    connection.execute(
        text(
            """
            INSERT INTO catalog_verifier_issues
                (source_id, issue_url, review_id, status, created_at, updated_at)
            VALUES (:id, :issue, 'defect-review', 'open', '2026-09-28', '2026-09-28')
        """
        ),
        {"id": source_id, "issue": issue_url},
    )
    resolve = text("UPDATE catalog_verifier_issues SET status = 'resolved', resolution_review_id = 'new-review'")
    with pytest.raises(DBAPIError, match="fresh native review"), connection.begin_nested():
        connection.execute(resolve)
    collection = {
        "reviews": [
            {
                "method": "runtime_execution",
                "tests_executed": True,
                "attributes": {"verification": {"status": "verified"}},
            },
            {
                "attributes": {
                    "resolved_verifier_issues": [
                        {
                            "issue_url": issue_url,
                            "verifier_revision": verifier_revision,
                            "dataset_revision": "data2",
                            "fix_validated": True,
                        }
                    ]
                }
            },
        ]
    }
    resolution = collection["reviews"][1]["attributes"]["resolved_verifier_issues"][0]
    current_resolution = dict(resolution)
    for invalid_resolution in [
        {key: value for key, value in current_resolution.items() if key != "verifier_revision"},
        {**current_resolution, "verifier_revision": "stale-code"},
        {**current_resolution, "dataset_revision": "stale-data"},
    ]:
        resolution.clear()
        resolution.update(invalid_resolution)
        connection.execute(
            text("UPDATE catalog_reviews SET collection = CAST(:collection AS JSONB) WHERE id = 'new-review'"),
            {"collection": json.dumps(collection)},
        )
        with pytest.raises(DBAPIError, match="fresh native review"), connection.begin_nested():
            connection.execute(resolve)
    resolution.clear()
    resolution.update(current_resolution)
    connection.execute(
        text("UPDATE catalog_reviews SET collection = CAST(:collection AS JSONB) WHERE id = 'new-review'"),
        {"collection": json.dumps(collection)},
    )
    connection.execute(resolve)
    connection.execute(text("UPDATE catalog_sources SET quality = 'good', difficulty = '32/32'"))
    row = connection.execute(text("SELECT quality, difficulty FROM catalog_sources")).one()
    assert tuple(row) == ("good", "32/32")
    assert connection.execute(text("SELECT status FROM catalog_verifier_issues")).scalar_one() == "resolved"


@pytest.mark.parametrize("change", [None, "data", "grader", "proof_hash", "source", "verdict", "malformed"])
def test_grading_equivalence_preserves_ratings_only_for_matching_immutable_evidence(change) -> None:
    claim = {
        "schema_version": 1,
        "equivalent": True,
        "source_id": "MarinSkyRL:math",
        "review_id": "review1",
        "source_revision": "data1",
        "captured_verifier_revision": "old-package",
        "grading_revision": "selected-grader1",
    }
    binding = {**claim, "evidence_sha256": ""}
    payload = {
        "id": "MarinSkyRL:math",
        "dataset_revision": "data1",
        "verifier_revision": "new-package",
        "grading_revision": "selected-grader1",
    }
    if change == "data":
        payload["dataset_revision"] = "data2"
    elif change == "grader":
        payload["grading_revision"] = "selected-grader2"
    elif change == "source":
        claim["source_id"] = "other-source"
    elif change == "verdict":
        claim["equivalent"] = False
    content = "invalid-json" if change == "malformed" else json.dumps(claim)
    digest = hashlib.sha256(content.encode()).hexdigest()
    binding["evidence_sha256"] = digest
    proof = {"content": content, "sha256": "wrong-hash" if change == "proof_hash" else digest}
    record = {
        "payload": payload,
        "quality": "good",
        "difficulty": "32/32",
        "traces": 3,
        "review_id": "review1",
        "review_date": "2026-09-28",
        "review_source_revision": "data1",
        "review_verifier_revision": "old-package",
        "verifier_issues": [],
        "grading_enrolled": True,
        "grading_binding": binding,
        "grading_proof": proof,
    }
    row = source_with_review(record)
    assert row["review_stale"] == (change is not None)
    assert row["quality"] == ("good" if change is None else None)
    assert row["difficulty"] == ("32/32" if change is None else None)
    assert row["review_verifier_revision"] == "old-package"
    assert row["review_id"] == "review1"


def test_grading_migration_preserves_legacy_reviews_then_tracks_only_selected_grader(catalog_connection: Connection):
    connection = catalog_connection
    payload = {
        "id": "MarinSkyRL:math",
        "dataset_revision": "data1",
        "verifier_revision": "raw1",
        "grading_revision": "grader1",
    }
    save_snapshot(connection, Snapshot("MarinSkyRL", "repo1", "2026-10-08", [payload]))
    connection.execute(
        text(
            "UPDATE catalog_sources SET quality='good',difficulty='32/32',review_id='review1',"
            "review_source_revision='data1',review_verifier_revision='raw1'"
        )
    )
    pending = active_sources_with_review_state(connection)[0]
    assert pending["quality"] == "good" and pending["grading_tracking"] == "legacy"
    claim = {
        "schema_version": 1,
        "equivalent": True,
        "source_id": payload["id"],
        "review_id": "review1",
        "source_revision": "data1",
        "captured_verifier_revision": "raw1",
        "grading_revision": "grader1",
    }
    content = json.dumps(claim)
    digest = hashlib.sha256(content.encode()).hexdigest()
    connection.execute(
        text("INSERT INTO review_artifacts VALUES ('review1','proof.json',:content,:sha)"),
        {"content": content, "sha": digest},
    )
    connection.execute(
        text(
            "INSERT INTO catalog_grading_reviews VALUES "
            "(:source,'review1','data1','raw1','grader1','proof.json',:sha)"
        ),
        {"source": payload["id"], "sha": digest},
    )
    payload["verifier_revision"] = "unrelated-package-change"
    save_snapshot(connection, Snapshot("MarinSkyRL", "repo2", "2026-10-08", [payload]))
    enrolled = active_sources_with_review_state(connection)[0]
    assert enrolled["quality"] == "good" and enrolled["grading_tracking"] == "source-specific"
    payload["verifier_revision"] = "raw1"
    payload["grading_revision"] = "grader2"
    save_snapshot(connection, Snapshot("MarinSkyRL", "repo3", "2026-10-08", [payload]))
    changed = active_sources_with_review_state(connection)[0]
    assert changed["review_stale"] and changed["quality"] is None
    del payload["grading_revision"]
    save_snapshot(connection, Snapshot("MarinSkyRL", "repo4", "2026-10-08", [payload]))
    missing = active_sources_with_review_state(connection)[0]
    assert missing["review_stale"] and missing["quality"] is None
    assert missing["review_id"] == "review1"
