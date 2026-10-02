# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import logging
import uuid
from collections.abc import Iterator

import httpx
import pytest
import requests
from google.auth.credentials import AnonymousCredentials
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError

from infra.marina.applets.rl_data_catalog.audit_nemotron import audit_blend
from infra.marina.applets.rl_data_catalog.server import composition, hf_auth
from infra.marina.applets.rl_data_catalog.server.app import (
    difficulty_summary,
    migrate,
    refresh_catalog,
    save_snapshot,
    source_with_review,
)
from infra.marina.applets.rl_data_catalog.server.catalog import (
    Snapshot,
    count_metadata,
    dataset_metadata,
    registry_sources,
    skyrl_snapshot,
    source_row,
    split_count,
    tasktrove_snapshot,
)
from infra.marina.applets.rl_data_catalog.server.composition import canonical_rows, component_rows
from infra.marina.applets.rl_data_catalog.server.hf_auth import HuggingFaceAuth


@pytest.fixture
def catalog_connection(database_url: str) -> Iterator[Connection]:
    engine = create_engine(database_url)
    schema = "catalog_test_" + uuid.uuid4().hex
    with engine.connect() as connection, connection.begin():
        connection.execute(text(f"CREATE SCHEMA {schema}"))
        connection.execute(text(f"SET LOCAL search_path TO {schema}"))
        migrate(connection)
        yield connection
        # Roll back all fixture writes, including the schema.
        connection.rollback()
    engine.dispose()


def test_registry_reads_selected_sources_without_executing_upstream_code(tmp_path) -> None:
    sentinel = tmp_path / "executed"
    source = f"""
from pathlib import Path
Path({str(sentinel)!r}).write_text("untrusted")
DATASET = "org/shared"
def ordinary():
    return Source("math", DATASET, "aime", "train", False, "two_sided", prepare)
def omitted():
    return Source("unused", DATASET, "aime", "test", False, "two_sided", prepare)
def blend(*, name, agents, blend):
    return Source(name, DATASET, "nemotron_ultra", "train", True, "row_selected", prepare)
def selected():
    return blend(name="rlvr2", agents=AGENTS, blend="rlvr2")
SOURCES = {{source.name: source for source in (ordinary(), selected())}}
"""
    rows = registry_sources(source)
    assert [(row["name"], row["dataset_id"], row["env_id"]) for row in rows] == [
        ("math", "org/shared", "aime"),
        ("rlvr2", "org/shared", "nemotron_ultra"),
    ]
    assert not sentinel.exists()


def test_split_count_does_not_double_count_alternative_gsm8k_configs() -> None:
    info = {
        "cardData": {
            "dataset_info": [
                {
                    "config_name": "main",
                    "splits": [{"name": "train", "num_examples": 7473}, {"name": "test", "num_examples": 1319}],
                },
                {"config_name": "socratic", "splits": [{"name": "train", "num_examples": 7473}]},
            ]
        }
    }
    assert split_count({"name": "gsm8k", "env_id": "gsm8k", "split": "train"}, info) == 7473
    assert split_count({"name": "other", "env_id": "aime", "split": "train"}, info) is None


def trove_manifest(count: int = 3) -> dict:
    return {
        "clean_tasks": count,
        "by_source": {"org__math": {"converted": count, "rejected": 2}, "org__old": {"rejected": 8}},
        "source_details": {"org__math": {"modes": {"math": count}, "languages": {}}},
        "source_verdicts": {
            "org__math": {"family": "math-answer", "reason": "Verified", "verdict": "keep"},
            "org__old": {"family": "other", "reason": "No verifier", "verdict": "drop"},
        },
    }


def test_tasktrove_counts_only_released_tasks_and_retains_exclusion_reason() -> None:
    snapshot = tasktrove_snapshot(trove_manifest(), {"sha": "release1", "lastModified": "2026-09-19T15:41:18Z"})
    kept, dropped = snapshot.rows
    assert (kept["task_count"], kept["input_count"], kept["status"]) == (3, 5, "Available")
    assert (dropped["task_count"], dropped["status"], dropped["notes"]) == (0, "Excluded", "No verifier")
    assert kept["turns"] == "Multi-turn"
    assert [(row["environment"], row["type"]) for row in snapshot.rows] == [("Harbor", "Agentic"), ("Harbor", "Agentic")]
    assert "/blob/release1/manifest.json" in kept["provenance_url"]


def test_refresh_failure_preserves_previous_data_and_other_catalog_progress(catalog_connection: Connection) -> None:
    connection = catalog_connection
    save_snapshot(connection, Snapshot("MarinSkyRL", "sky1", "2026-09-01", [{"id": "MarinSkyRL:math", "task_count": 7}]))
    old = tasktrove_snapshot(trove_manifest(), {"sha": "release1", "lastModified": "2026-09-01"})
    save_snapshot(connection, old)
    connection.execute(
        text("UPDATE catalog_sources SET difficulty = 'Hard', traces = 12 WHERE id = 'Task Trove:org__math'")
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return httpx.Response(503, json={"message": "Unavailable"})
        if request.url.path.startswith("/api/datasets/"):
            return httpx.Response(200, json={"sha": "release2", "lastModified": "2026-09-28"})
        return httpx.Response(200, json=trove_manifest(5))

    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        result = refresh_catalog(connection, client, False)
    payloads = {
        row["id"]: row for row in connection.execute(text("SELECT * FROM catalog_sources WHERE active")).mappings()
    }
    assert payloads["MarinSkyRL:math"]["payload"]["task_count"] == 7
    assert payloads["Task Trove:org__math"]["payload"]["task_count"] == 5
    assert (payloads["Task Trove:org__math"]["difficulty"], payloads["Task Trove:org__math"]["traces"]) == ("Hard", 12)
    status = {row["origin"]: row for row in connection.execute(text("SELECT * FROM catalog_refreshes")).mappings()}
    assert status["MarinSkyRL"]["revision"] == "sky1"
    assert status["MarinSkyRL"]["error"]
    assert status["Task Trove"]["revision"] == "release2"
    assert status["Task Trove"]["error"] is None
    assert result["results"][1]["changed"]


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
def test_changed_source_preserves_historical_review_but_invalidates_current_rating(changed_field) -> None:
    payload = {"id": "MarinSkyRL:math", "dataset_revision": "data1", "verifier_revision": "code1"}
    if changed_field:
        payload[changed_field] = "new-revision"
    row = source_with_review(
        {
            "payload": payload,
            "quality": "good",
            "difficulty": "32/32",
            "traces": 3,
            "review_id": "review1",
            "review_date": "2026-09-28",
            "review_source_revision": "data1",
            "review_verifier_revision": "code1",
            "verifier_issues": [],
        }
    )
    assert row["review_id"] == "review1"
    assert row["review_date"] == "2026-09-28"
    assert row["review_stale"] == bool(changed_field)
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
    )
    displayed = source_with_review(record)
    assert displayed["review_stale"]
    assert displayed["quality"] == expected
    assert displayed["difficulty"] is None


def test_verifier_defect_requires_a_validated_current_review_before_green_restoration(
    catalog_connection: Connection,
) -> None:
    connection = catalog_connection
    source_id = "MarinSkyRL:math"
    issue_url = "https://github.com/example/issues/1"
    payload = {"id": source_id, "dataset_revision": "data2", "verifier_revision": "code2"}
    save_snapshot(connection, Snapshot("MarinSkyRL", "code2", "2026-09-29", [payload]))
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
                            "verifier_revision": "code2",
                            "dataset_revision": "data2",
                            "fix_validated": True,
                        }
                    ]
                }
            },
        ]
    }
    connection.execute(
        text("UPDATE catalog_reviews SET collection = CAST(:collection AS JSONB) WHERE id = 'new-review'"),
        {"collection": json.dumps(collection)},
    )
    connection.execute(resolve)
    connection.execute(text("UPDATE catalog_sources SET quality = 'good', difficulty = '32/32'"))
    row = connection.execute(text("SELECT quality, difficulty FROM catalog_sources")).one()
    assert tuple(row) == ("good", "32/32")
    assert connection.execute(text("SELECT status FROM catalog_verifier_issues")).scalar_one() == "resolved"


def test_hf_viewer_metadata_supplies_missing_card_count_without_download() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.host == "huggingface.co" and request.url.path == "/api/datasets/org/math":
            return httpx.Response(200, json={"sha": "dataset1", "cardData": {}})
        if request.url.host == "datasets-server.huggingface.co" and request.url.path == "/size":
            return httpx.Response(
                200,
                json={
                    "size": {
                        "splits": [
                            {"config": "default", "split": "train", "num_rows": 400},
                            {"config": "default", "split": "test", "num_rows": 50},
                        ]
                    }
                },
            )
        raise AssertionError(f"Unexpected task-data request: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        info = dataset_metadata(client, "org/math")
    assert split_count({"name": "math", "env_id": "aime", "split": "train"}, info) == 400


def test_hf_auth_keeps_token_off_github_and_redirect_targets() -> None:
    headers_by_host = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        headers_by_host[request.url.host] = request.headers.get("Authorization")
        if request.url.host == "huggingface.co":
            return httpx.Response(302, headers={"Location": "https://cdn.example.test/manifest.json"})
        return httpx.Response(200, json={})

    with httpx.Client(
        transport=httpx.MockTransport(upstream), auth=HuggingFaceAuth("fixture-token"), follow_redirects=True
    ) as client:
        client.get("https://api.github.com/repos/example/repo")
        client.get("https://huggingface.co/datasets/example/repo/raw/main/manifest.json")
        client.get("https://datasets-server.huggingface.co/size?dataset=example/repo")
    assert headers_by_host == {
        "api.github.com": None,
        "huggingface.co": "Bearer fixture-token",
        "cdn.example.test": None,
        "datasets-server.huggingface.co": "Bearer fixture-token",
    }


def test_hf_rate_limit_retries_with_runtime_readonly_secret(monkeypatch, caplog) -> None:
    # Explicit configuration prevents google-auth from changing global log propagation.
    caplog.set_level(logging.WARNING, logger="google")
    monkeypatch.setattr(hf_auth.google.auth, "default", lambda scopes: (AnonymousCredentials(), "hai-gcp-models"))
    secret_requests = []

    def secret_response(_session, request, **_kwargs):
        secret_requests.append(request.url)
        response = requests.Response()
        response.status_code = (
            200
            if request.url
            == (
                "https://secretmanager.googleapis.com/v1/"
                "projects/hai-gcp-models/secrets/HF_TOKEN_READONLY/versions/1:access"
            )
            else 403
        )
        response._content = json.dumps(
            {"payload": {"data": base64.b64encode(b"runtime-readonly-fixture").decode()}}
        ).encode()
        return response

    monkeypatch.setattr(hf_auth.AuthorizedSession, "send", secret_response)
    hf_requests = []

    def upstream(request: httpx.Request) -> httpx.Response:
        credential = request.headers.get("Authorization")
        hf_requests.append(credential)
        if credential is None:
            return httpx.Response(429)
        assert credential == "Bearer runtime-readonly-fixture"
        return httpx.Response(200, json={"sources": []})

    with httpx.Client(transport=httpx.MockTransport(upstream), auth=HuggingFaceAuth(None)) as client:
        response = client.get("https://huggingface.co/datasets/open-athena/task-trove/raw/main/manifest.json")
    assert response.status_code == 200
    assert response.json() == {"sources": []}
    assert hf_requests == [None, "Bearer runtime-readonly-fixture"]
    assert len(secret_requests) == 1


def test_migration_reclassifies_saved_tasktrove_without_losing_counts_or_curation(
    catalog_connection: Connection,
) -> None:
    connection = catalog_connection
    save_snapshot(
        connection,
        Snapshot(
            "Task Trove",
            "release1",
            "2026-09-19",
            [
                {"id": "Task Trove:math", "type": "RLVR", "task_count": 30},
                {"id": "Task Trove:old", "type": None, "task_count": 0, "status": "Excluded"},
            ],
        ),
    )
    connection.execute(text("UPDATE catalog_sources SET quality = 'Reviewed', traces = 8 WHERE id = 'Task Trove:math'"))
    migrate(connection)
    rows = list(connection.execute(text("SELECT payload, quality, traces FROM catalog_sources ORDER BY id")).mappings())
    assert [(row["payload"]["environment"], row["payload"]["type"]) for row in rows] == [
        ("Harbor", "Agentic"),
        ("Harbor", "Agentic"),
    ]
    assert [row["payload"]["task_count"] for row in rows] == [30, 0]
    assert (rows[0]["quality"], rows[0]["traces"]) == ("Reviewed", 8)


@pytest.mark.parametrize(
    "verifier_date,hf_date,expected",
    [
        ("2026-09-25T12:00:00Z", "2026-09-28T13:00:00Z", "2026-09-28T13:00:00Z"),
        ("2026-09-28T17:00:00Z", "2026-07-17T18:08:18Z", "2026-09-28T17:00:00Z"),
    ],
)
def test_unchanged_git_head_refreshes_hf_counts_and_reports_latest_change(verifier_date, hf_date, expected) -> None:
    head = {"sha": "sky1", "commit": {"committer": {"date": "2026-09-28T17:00:00Z"}}}
    cached = [
        {
            "id": "MarinSkyRL:math",
            "name": "math",
            "origin": "MarinSkyRL",
            "kind": "Dataset",
            "dataset_id": "org/math",
            "environment": "aime",
            "split": "train",
            "revision": "sky1",
            "verifier_revised_at": verifier_date,
            "dataset_revised_at": "2026-01-01T00:00:00Z",
            "task_count": 5,
        }
    ]

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "huggingface.co"
        return httpx.Response(
            200,
            json={
                "sha": "hf2",
                "lastModified": hf_date,
                "cardData": {"dataset_info": {"splits": [{"name": "train", "num_examples": 9}]}},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        snapshot = skyrl_snapshot(client, head, cached)
    row = snapshot.rows[0]
    assert (row["revised_at"], row["task_count"], row["dataset_revision"]) == (expected, 9, "hf2")
    assert (row["verifier_revised_at"], row["dataset_revised_at"]) == (verifier_date, hf_date)
    assert cached[0]["task_count"] == 5


@pytest.mark.parametrize(
    "name,dataset_id,environment,selector,card,expected,display_suffix",
    [
        (
            "apps",
            "codeparrot/apps",
            "apps",
            "train",
            "train: Dataset({ num_rows: 5000 })\ntest: Dataset({ num_rows: 5000 })",
            5000,
            "",
        ),
        (
            "eurus2_code",
            "PRIME-RL/Eurus-2-RL-Data",
            "eurus2",
            "train",
            "| Math | 455261 | 1024 |\n| Coding | 25276 | 1024 |",
            25276,
            " · code",
        ),
        (
            "nemotron_if",
            "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "nemotron",
            "instruction_following",
            "| math | 22,066,397 |\n| instruction following | 56,339 |",
            56339,
            " · RL/instruction_following",
        ),
        (
            "nemotron_ultra_rlvr1",
            "nvidia/Nemotron-RL-Ultra-Training-Blends",
            "nemotron_ultra",
            "train",
            "| rlvr1 | 98,424 | 5.0 GB |\n| rlvr2 | 99,116 | 5.0 GB |\n| mopd | 85,980 | 5.5 GB |",
            98424,
            " · rlvr1",
        ),
        (
            "nemotron_ultra_rlvr2",
            "nvidia/Nemotron-RL-Ultra-Training-Blends",
            "nemotron_ultra",
            "train",
            "| rlvr1 | 98,424 | 5.0 GB |\n| rlvr2 | 99,116 | 5.0 GB |\n| mopd | 85,980 | 5.5 GB |",
            99116,
            " · rlvr2",
        ),
        (
            "nemotron_ultra_mopd",
            "nvidia/Nemotron-RL-Ultra-Training-Blends",
            "nemotron_ultra",
            "train",
            "| rlvr1 | 98,424 | 5.0 GB |\n| rlvr2 | 99,116 | 5.0 GB |\n| mopd | 85,980 | 5.5 GB |",
            85980,
            " · mopd",
        ),
    ],
)
def test_refresh_reads_card_counts_for_selected_population_and_canonical_names(
    name, dataset_id, environment, selector, card, expected, display_suffix
) -> None:
    if environment == "nemotron_ultra":
        blend = name.removeprefix("nemotron_ultra_")
        card += f"\n### {blend}\n| [math](https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2) | 100.00% |\n"
    head = {"sha": "sky1", "commit": {"committer": {"date": "2026-09-28T17:00:00Z"}}}
    cached = [
        {
            "id": f"MarinSkyRL:{name}",
            "origin": "MarinSkyRL",
            "name": name,
            "dataset_id": dataset_id,
            "environment": environment,
            "split": selector,
            "kind": "Dataset",
            "revision": "sky1",
            "verifier_revised_at": "2026-09-25T00:00:00Z",
            "task_count": None,
        }
    ]

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/api/datasets/{dataset_id}/tree/hf2":
            return httpx.Response(200, json=[])
        if request.url.path == f"/api/datasets/{dataset_id}":
            return httpx.Response(
                200,
                json={
                    "sha": "hf2",
                    "lastModified": "2026-09-28T00:00:00Z",
                    "cardData": {"dataset_info": {"splits": [{"name": "train", "num_examples": 999}]}},
                },
            )
        assert request.url.path == f"/datasets/{dataset_id}/raw/hf2/README.md"
        return httpx.Response(200, text=card)

    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        row = skyrl_snapshot(client, head, cached).rows[0]
    assert row["task_count"] == expected
    assert row["count_precision"] == ("estimated" if environment == "nemotron_ultra" else "reported")
    assert row["canonical_source"] == dataset_id + display_suffix
    assert row.get("canonical_id", row["id"]) == f"MarinSkyRL:{name}"
    assert row["count_url"] == f"https://huggingface.co/datasets/{dataset_id}/blob/hf2/README.md"


def test_openscience_partial_viewer_uses_full_estimate_instead_of_preview_rows() -> None:
    source = {"name": "openscience", "dataset_id": "nvidia/OpenScience", "split": "train", "env_id": "mcq"}
    info: dict = {
        "viewer_partial": True,
        "viewer_splits": [
            {"config": "OS-Q2.5-32B-10", "split": "train", "num_rows": 393091, "estimated_num_rows": 2168201},
            {"config": "OS-Q2.5-32B-4", "split": "train", "num_rows": 75720, "estimated_num_rows": None},
            {"config": "OS-Q2.5-72B-10", "split": "train", "num_rows": 415848, "estimated_num_rows": 1917418},
            {"config": "OS-Q3-235B-4", "split": "train", "num_rows": 315579, "estimated_num_rows": None},
        ],
    }
    count = count_metadata(source, info)
    assert (count.task_count, count.count_precision) == (4476918, "estimated")
    for split in info["viewer_splits"]:
        split["estimated_num_rows"] = None
    assert count_metadata(source, info).task_count is None


@pytest.mark.parametrize("revision,expected", [("hf1", 198), ("hf2", None)])
def test_gpqa_gated_viewer_preserves_audited_count_only_for_same_revision(revision, expected) -> None:
    head = {"sha": "sky1", "commit": {"committer": {"date": "2026-09-28T17:00:00Z"}}}
    cached = [
        {
            "id": "MarinSkyRL:gpqa",
            "origin": "MarinSkyRL",
            "name": "gpqa",
            "dataset_id": "Idavidrein/gpqa",
            "environment": "mcq",
            "split": "gpqa_diamond",
            "kind": "Dataset",
            "revision": "sky1",
            "verifier_revised_at": "2026-09-25T00:00:00Z",
            "task_count": 198,
            "dataset_revision": "hf1",
            "count_precision": "exact",
        }
    ]

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.host == "huggingface.co":
            return httpx.Response(200, json={"sha": revision, "lastModified": "2026-09-28T00:00:00Z"})
        return httpx.Response(401, json={"error": "Gated dataset"})

    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        row = skyrl_snapshot(client, head, cached).rows[0]
    assert row["task_count"] == expected
    assert row["count_metadata_error"]
    assert row["display_name"] == "Idavidrein/gpqa · gpqa_diamond"


def test_aime_benchmark_and_audited_family_survive_refresh_without_hf_tag() -> None:
    head = {"sha": "sky1", "commit": {"committer": {"date": "2026-09-28T17:00:00Z"}}}
    cached = [
        {
            "id": "MarinSkyRL:aime_1983_2024",
            "origin": "MarinSkyRL",
            "name": "aime_1983_2024",
            "dataset_id": "di-zhang-fdu/AIME_1983_2024",
            "environment": "aime",
            "split": "train",
            "kind": "Dataset",
            "revision": "sky1",
            "verifier_revised_at": "2026-09-25T00:00:00Z",
            "task_count": 933,
            "is_benchmark": False,
            "family": "",
        }
    ]

    def upstream(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "sha": "hf2",
                "lastModified": "2026-09-28T00:00:00Z",
                "tags": [],
                "cardData": {"dataset_info": {"splits": [{"name": "train", "num_examples": 933}]}},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        row = skyrl_snapshot(client, head, cached).rows[0]
    assert row["is_benchmark"] is True
    assert row["family"] == "math-answer"
    assert row["family_url"].startswith("https://huggingface.co/datasets/di-zhang-fdu/AIME_1983_2024/blob/")


@pytest.mark.parametrize(
    "name,dataset_id,kind,card,expected",
    [
        ("asdiv", "chaochun/nlu-asdiv-dataset", "Dataset", "It contains 2305 english Math Word Problems.", 2305),
        ("reasoning_gym", "open-thought/reasoning-gym", "Generator", "Procedural dataset generators", None),
    ],
)
def test_github_sources_resolve_counts_and_links_without_invalid_hf_requests(
    name, dataset_id, kind, card, expected
) -> None:
    head = {"sha": "sky1", "commit": {"committer": {"date": "2026-09-28T17:00:00Z"}}}
    cached = [
        {
            "id": f"MarinSkyRL:{name}",
            "origin": "MarinSkyRL",
            "name": name,
            "dataset_id": dataset_id,
            "environment": "aime",
            "split": "train",
            "kind": kind,
            "revision": "sky1",
            "verifier_revised_at": "2026-09-25T00:00:00Z",
            "task_count": None,
        }
    ]

    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return httpx.Response(200, json={"sha": "data1", "commit": {"committer": {"date": "2026-09-26T00:00:00Z"}}})
        assert request.url.host == "raw.githubusercontent.com"
        assert request.url.path == f"/{dataset_id}/data1/README.md"
        return httpx.Response(200, text=card)

    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        row = skyrl_snapshot(client, head, cached).rows[0]
    assert row["task_count"] == expected
    assert row["url"] == f"https://github.com/{dataset_id}"
    assert row["dataset_revision"] == "data1"
    assert row["family"]


def test_migration_merges_gym_duplicates_preserving_source_counts_and_curation(
    catalog_connection: Connection,
) -> None:
    revision = "sky1"
    rows = []
    for name, env, count, kind in [
        ("math1", "aime", 10, "Dataset"),
        ("math2", "aime", None, "Dataset"),
        ("gsm8k", "gsm8k", 7473, "Dataset"),
        ("reasoning_gym", "reasoning_gym", None, "Generator"),
    ]:
        row = source_row("MarinSkyRL", name, revision, "2026-09-28")
        row.update(dataset_id=f"org/{name}", environment=env, task_count=count, kind=kind)
        rows.append(row)
    for env, entrypoint in [
        ("aime", "skyrl_gym.envs.aime.env:AIMEEnv"),
        ("gsm8k_multi_turn", "skyrl_gym.envs.gsm8k.multi_turn_env:GSM8kMultiTurnEnv"),
        ("reasoning_gym", "skyrl_gym.envs.reasoning_gym.env:ReasoningGymEnv"),
    ]:
        row = source_row("MarinSkyRL", f"gym/{env}", revision, "2026-09-28")
        row.update(kind="Environment", environment=env, entrypoint=entrypoint)
        rows.append(row)
    save_snapshot(catalog_connection, Snapshot("MarinSkyRL", revision, "2026-09-28", rows))
    catalog_connection.execute(text("UPDATE catalog_sources SET quality = 'Reviewed' WHERE id = 'MarinSkyRL:math1'"))
    migrate(catalog_connection)
    saved = {row["id"]: row for row in catalog_connection.execute(text("SELECT * FROM catalog_sources")).mappings()}
    active = {source_id: row["payload"] for source_id, row in saved.items() if row["active"]}
    assert set(active) == {"MarinSkyRL:math1", "MarinSkyRL:math2", "MarinSkyRL:gsm8k", "MarinSkyRL:reasoning_gym"}
    assert active["MarinSkyRL:math1"]["task_count"] == 10
    assert active["MarinSkyRL:math2"]["task_count"] is None
    assert active["MarinSkyRL:gsm8k"]["task_count"] == 7473
    assert active["MarinSkyRL:math1"]["gym_alias"] == "gym/aime"
    assert active["MarinSkyRL:math2"]["gym_entrypoint"] == "skyrl_gym.envs.aime.env:AIMEEnv"
    assert active["MarinSkyRL:reasoning_gym"]["gym_alias"] == "gym/reasoning_gym"
    assert active["MarinSkyRL:reasoning_gym"]["kind"] == "Generator"
    assert active["MarinSkyRL:reasoning_gym"]["task_count"] is None
    assert saved["MarinSkyRL:math1"]["quality"] == "Reviewed"
    assert all(not row["active"] for source_id, row in saved.items() if source_id.startswith("MarinSkyRL:gym/"))
    assert sum(row["task_count"] or 0 for row in active.values()) == 7483


def test_gym_duplicates_stay_merged_after_dataset_metadata_refresh() -> None:
    head = {"sha": "sky1", "commit": {"committer": {"date": "2026-09-28T17:00:00Z"}}}
    dataset = source_row("MarinSkyRL", "math", "sky1", "2026-09-25T00:00:00Z")
    dataset.update(
        dataset_id="org/math",
        environment="aime",
        split="train",
        task_count=5,
        verifier_revised_at="2026-09-25T00:00:00Z",
    )
    adapter = source_row("MarinSkyRL", "gym/aime", "sky1", "2026-09-25T00:00:00Z")
    adapter.update(
        kind="Environment",
        environment="aime",
        entrypoint="skyrl_gym.envs.aime.env:AIMEEnv",
        verifier_revised_at="2026-09-25T00:00:00Z",
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/datasets/org/math"
        return httpx.Response(
            200,
            json={
                "sha": "hf2",
                "lastModified": "2026-09-28T00:00:00Z",
                "cardData": {"dataset_info": {"splits": [{"name": "train", "num_examples": 9}]}},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(upstream)) as client:
        result = skyrl_snapshot(client, head, [dataset, adapter])
    assert len(result.rows) == 1
    refreshed_dataset = result.rows[0]
    assert refreshed_dataset["task_count"] == 9
    assert refreshed_dataset["gym_alias"] == "gym/aime"
    assert refreshed_dataset["gym_entrypoint"] == "skyrl_gym.envs.aime.env:AIMEEnv"
    assert sum(row["task_count"] or 0 for row in result.rows) == 9


def test_mixed_blend_components_preserve_selections_and_expose_unpublished_counts() -> None:
    parent = source_row("MarinSkyRL", "nemotron_ultra_rlvr1", "registry", "2026-09-28")
    parent.update(
        dataset_id="nvidia/Nemotron-RL-Ultra-Training-Blends",
        display_name="nvidia/Nemotron-RL-Ultra-Training-Blends · rlvr1",
        url="https://huggingface.co/datasets/nvidia/Nemotron-RL-Ultra-Training-Blends",
        task_count=100,
        turns="Mixed",
    )
    swe = "https://huggingface.co/datasets/nebius/SWE-rebench-V2"
    gym = "https://huggingface.co/datasets/SWE-Gym/SWE-Gym"
    card = f"""### rlvr1
| Component | Ratio |
| [math](https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2) | 60.00% |
| [tool](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1) | 20.00% |
| [swe]({swe}) + [gym]({gym}) | 20.00% |
### rlvr2
| [math](https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2) | 100.00% |
"""
    info = {"card_text": card, "card_url": parent["url"] + "/blob/revision/README.md"}
    rows = component_rows(parent, info)
    assert [(row["task_count"], row["type"], row["turns"]) for row in rows] == [
        (60, "RLVR", "Single-turn"),
        (20, "Agentic", "Multi-turn"),
        (None, "Agentic", "Multi-turn"),
        (None, "Agentic", "Multi-turn"),
    ]
    assert len({row["id"] for row in rows}) == 4
    assert all(row["canonical_source"] == parent["display_name"] for row in rows)
    assert rows[0]["count_precision"] == "estimated"
    # Cached refreshes recover the one registry selection, then replace its children.
    restored = canonical_rows(rows)
    assert len(restored) == 1
    assert (restored[0]["id"], restored[0]["task_count"]) == (parent["id"], 100)
    assert component_rows(restored[0], info) == rows


def test_preference_components_use_selected_collection_counts_and_interactions() -> None:
    parent = source_row("MarinSkyRL", "kto_mix", "registry", "2026-09-28")
    parent.update(
        dataset_id="trl-lib/kto-mix-14k",
        display_name="trl-lib/kto-mix-14k",
        url="https://huggingface.co/datasets/trl-lib/kto-mix-14k",
        task_count=10,
    )
    info = {
        "components": {
            "argilla/distilabel-capybara-dpo-7k-binarized": 6,
            "argilla/ultrafeedback-binarized-preferences-cleaned": 4,
        },
        "composition_url": "https://datasets-server.huggingface.co/statistics",
    }
    rows = component_rows(parent, info)
    assert [(row["task_count"], row["turns"]) for row in rows] == [(6, "Multi-turn"), (4, "Single-turn")]
    assert sum(row["task_count"] for row in rows) == 10
    assert all(row["canonical_source"] == "trl-lib/kto-mix-14k" for row in rows)


def test_complete_record_audit_splits_swe_and_math_and_invalidates_changed_revision(tmp_path, monkeypatch) -> None:
    rows = [
        {"dataset": "ultra_sft_step3200_math_cot", "agent_ref": {"name": "math_with_judge_simple_agent"}},
        {"dataset": "ultra_sft_step3200_math_cot", "agent_ref": {"name": "math_with_judge_simple_agent"}},
        {"dataset": "ultra_sft_step3200_math_tir", "agent_ref": {"name": "ns_tools_simple_agent"}},
        {
            "dataset": "ultra_sft_step3200_swe_pivot_len40k",
            "agent_ref": {"name": "swe_agent"},
            "metadata": {"instance_id": "gym-instance"},
        },
        {
            "dataset": "ultra_sft_step3200_swe_pivot_len40k",
            "agent_ref": {"name": "swe_agent"},
            "metadata": {"instance_id": "rebench-instance"},
        },
    ]
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path = tmp_path / "rlvr1.jsonl"
    path.write_bytes(data)
    audit = audit_blend(path, {"gym-instance"})
    assert (audit["total"], audit["sha256"]) == (5, hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(
        composition,
        "NEMOTRON_COUNTS",
        {
            "revision": "audited",
            "blends": {"rlvr1": audit},
            "swe_attribution": "Known fixture origins",
        },
    )
    parent = source_row("MarinSkyRL", "nemotron_ultra_rlvr1", "registry", "2026-09-28")
    parent.update(
        dataset_id=composition.NEMOTRON,
        display_name=composition.NEMOTRON + " · rlvr1",
        url="https://huggingface.co/datasets/" + composition.NEMOTRON,
        dataset_revision="audited",
        family_url="https://huggingface.co/datasets/source",
        task_count=5,
    )
    info = {
        "card_text": "### rlvr1\n| [Math](https://huggingface.co/datasets/nvidia/Nemotron-RL-Math-v2) | 100.00% |",
        "card_url": "https://huggingface.co/datasets/source",
    }
    children = component_rows(parent, info)
    populations = {
        (row["component_selector"], row["display_name"].split(" · ")[0]): (row["task_count"], row["type"], row["turns"])
        for row in children
    }
    assert populations == {
        ("ultra_sft_step3200_math_cot", "nvidia/Nemotron-RL-Math-v2"): (2, "RLVR", "Single-turn"),
        ("ultra_sft_step3200_math_tir", "nvidia/Nemotron-RL-Math-v2"): (1, "Agentic", "Multi-turn"),
        ("ultra_sft_step3200_swe_pivot_len40k", "SWE-Gym/SWE-Gym"): (1, "Agentic", "Multi-turn"),
        ("ultra_sft_step3200_swe_pivot_len40k", "nebius/SWE-rebench-V2"): (1, "Agentic", "Multi-turn"),
    }
    assert sum(row["task_count"] for row in children) == 5
    assert all(row["count_precision"] == "exact" for row in children)
    assert all(row["component_file_sha256"] == audit["sha256"] for row in children)
    parent["dataset_revision"] = "changed"
    info["file_sha256"] = {"rlvr1.jsonl": audit["sha256"]}
    unchanged = component_rows(parent, info)
    assert [(row["id"], row["task_count"], row["turns"]) for row in unchanged] == [
        (row["id"], row["task_count"], row["turns"]) for row in children
    ]
    assert all(row["dataset_revision"] == "changed" for row in unchanged)
    info["file_sha256"]["rlvr1.jsonl"] = "changed-file-content"
    stale = component_rows(parent, info)
    assert len(stale) == 1
    assert stale[0]["count_precision"] == "estimated"
    assert stale[0]["task_count"] == 5


def test_mopd_audit_keeps_unlabeled_swe_and_multiturn_preference_records(tmp_path, monkeypatch) -> None:
    rows = [
        {
            "agent_ref": {"name": "swe_pivot_single_step_tool_use_with_argument_comparison_agent"},
            "metadata": {"instance_id": "gym-instance"},
        },
        {
            "agent_ref": {"name": "swe_pivot_single_step_tool_use_with_argument_comparison_agent"},
            "metadata": {"instance_id": "rebench-instance"},
        },
        {"dataset": "hs3_multiturn", "agent_ref": {"name": "genrm_simple_agent"}},
    ]
    path = tmp_path / "mopd.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    audit = audit_blend(path, {"gym-instance"})
    monkeypatch.setattr(
        composition,
        "NEMOTRON_COUNTS",
        {
            "revision": "audited",
            "blends": {"mopd": audit},
            "swe_attribution": "Known fixture origins",
        },
    )
    parent = source_row("MarinSkyRL", "nemotron_ultra_mopd", "registry", "2026-09-28")
    parent.update(
        dataset_id=composition.NEMOTRON,
        display_name=composition.NEMOTRON + " · mopd",
        url="https://huggingface.co/datasets/" + composition.NEMOTRON,
        dataset_revision="audited",
        family_url="https://huggingface.co/datasets/source",
        task_count=3,
    )
    children = component_rows(parent, {})
    assert len(children) == 3
    assert sum(row["task_count"] for row in children) == 3
    assert all(row["count_precision"] == "exact" and row["turns"] == "Multi-turn" for row in children)
    assert [(row["task_count"], row["type"]) for row in children] == [(1, "Agentic"), (1, "Agentic"), (1, "Alignment")]


def test_unavailable_nemotron_metadata_keeps_the_canonical_population() -> None:
    parent = source_row("MarinSkyRL", "nemotron_ultra_mopd", "registry", "2026-09-28")
    parent.update(
        dataset_id=composition.NEMOTRON, dataset_revision=None, task_count=None, metadata_error="HF metadata unavailable"
    )
    assert component_rows(parent, {"metadata_error": "HF metadata unavailable"}) == [parent]
