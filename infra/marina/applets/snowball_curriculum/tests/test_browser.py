# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from dataclasses import dataclass
from typing import cast

import pytest
from fastapi.testclient import TestClient
from marina.applets import AppletServices
from rigging.filesystem.storage_path import StoragePath
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from experiments.post_training.snowball_curriculum.bundle import file_sha256, write_manifest
from infra.marina.applets.snowball_curriculum.load import load_pool, public_record, read_pool
from infra.marina.applets.snowball_curriculum.server.app import create_api, migrate


@dataclass
class Services:
    database: Engine

    def engine(self) -> Engine:
        return self.database


def selected_row(task_id, source_id="calendar", prompt="Book a meeting", difficulty="medium"):
    return {
        "task_id": task_id,
        "contract_id": task_id + "-contract",
        "public_content_id": task_id + "-public",
        "source_id": source_id,
        "difficulty": difficulty,
        "population": "admitted",
        "task_json": json.dumps(
            {
                "context": {"events": [{"role": "user", "content": prompt}]},
                "answer_type": "text",
                "answer_format": {"kind": "plain_text"},
                "grader": {"kind": "exact", "expected": "PRIVATE_ANSWER"},
                "resources": {"verifier": [{"content": "PRIVATE_FIXTURE"}]},
                "environment_requirements": {"environment_variables": {"TOKEN": "PRIVATE_TOKEN"}},
                "source": {"dataset": "calendar", "revision": "abc123", "row": task_id, "importer_revision": "def456"},
            }
        ),
    }


def test_frozen_pool_projection_retains_lineage_without_grading_evidence(tmp_path):
    rows = [selected_row("a"), selected_row("b")]
    (tmp_path / "tasks.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    (tmp_path / "selection.json").write_text(
        json.dumps(
            {
                "selected_count": 2,
                "target": 100_000,
                "shortfall": 99_998,
                "source_quotas": {"calendar": 2},
            }
        )
    )
    (tmp_path / "recipe.json").write_text(json.dumps({"resolved_source_families": {"calendar": "planning"}}))
    (tmp_path / "selected-aliases.jsonl").write_text(
        json.dumps(
            {
                "contract_id": "a-contract",
                "aliases": [{"source_id": "legacy", "task_id": "old-a"}],
            }
        )
    )
    write_manifest(StoragePath(str(tmp_path)))
    manifest_pin = file_sha256(StoragePath(str(tmp_path / "artifact-manifest.json")))
    metadata, tasks = read_pool(tmp_path, expected_manifest_sha256=manifest_pin)
    detail = json.loads(tasks[0]["detail"])
    assert detail["context"]["events"][0]["content"] == "Book a meeting"
    assert detail["aliases"] == [{"source_id": "legacy", "task_id": "old-a"}]
    assert detail["source"]["revision"] == "abc123"
    assert tasks[0]["family"] == "planning"
    assert metadata["selection"]["shortfall"] == 99_998
    assert all(private not in json.dumps(tasks) for private in ("PRIVATE_ANSWER", "PRIVATE_FIXTURE", "PRIVATE_TOKEN"))
    # A same-count prompt edit must fail before the first serving database write.
    task = json.loads(rows[0]["task_json"])
    task["context"]["events"][0]["content"] = "Altered prompt with stale contract metadata"
    rows[0]["task_json"] = json.dumps(task)
    (tmp_path / "tasks.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    queries = []

    def query(sql, _parameters):
        queries.append(sql)
        raise AssertionError("A corrupted artifact reached the serving database")

    with pytest.raises(ValueError, match="payload differs"):
        load_pool(tmp_path, query, expected_manifest_sha256=manifest_pin)
    assert queries == []
    # Even an internally consistent truncated inventory cannot satisfy selection accounting.
    (tmp_path / "tasks.jsonl").write_text(json.dumps(rows[0]))
    write_manifest(StoragePath(str(tmp_path)))
    truncated_pin = file_sha256(StoragePath(str(tmp_path / "artifact-manifest.json")))
    with pytest.raises(ValueError, match="Task count differs"):
        read_pool(tmp_path, expected_manifest_sha256=truncated_pin)


def test_browser_queries_full_pool_filters_literal_search_and_paging():
    database = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    records = [
        public_record(selected_row("a", prompt="50% availability"), {"calendar": "planning"}, []),
        public_record(selected_row("b", difficulty="hard"), {"calendar": "planning"}, []),
        public_record(selected_row("c", "code", "Sort the input", "unknown"), {"code": "coding"}, []),
    ]
    with database.begin() as connection:
        migrate(connection)
        connection.execute(
            text("INSERT INTO pool_loads VALUES ('frozen', :metadata)"), {"metadata": json.dumps({"selected_count": 3})}
        )
        connection.execute(text("UPDATE pool_current SET load_id = 'frozen'"))
        for record in records:
            connection.execute(
                text(
                    "INSERT INTO pool_tasks (load_id, contract_id, task_id, source_id, family, "
                    "difficulty, preview, search_text, detail) "
                    "VALUES ('frozen', :contract_id, :task_id, :source_id, :family, "
                    ":difficulty, :preview, :search_text, :detail)"
                ),
                record,
            )
    with TestClient(create_api(cast(AppletServices, Services(database)))) as client:
        overview = client.get("/overview").json()
        assert overview["groups"]["source_id"] == [{"value": "calendar", "count": 2}, {"value": "code", "count": 1}]
        first = client.get("/tasks", params={"limit": 1}).json()
        assert first["total"] == 3
        assert first["tasks"][0]["task_id"] == "a"
        assert client.get("/tasks", params={"offset": 1, "limit": 1}).json()["tasks"][0]["task_id"] == "b"
        assert (
            client.get("/tasks", params={"family": "planning", "difficulty": "hard"}).json()["tasks"][0]["task_id"]
            == "b"
        )
        assert client.get("/tasks", params={"q": "%"}).json()["total"] == 1
        assert client.get("/tasks", params={"q": "SORT", "source": "code"}).json()["total"] == 1
        assert client.get("/tasks", params={"source": "calendar' OR 1=1 --"}).json()["total"] == 0
        detail = client.get("/task/a-contract").json()
        assert detail["context"]["events"][0]["content"] == "50% availability"
        assert "grader" not in detail
        assert client.get("/task/missing").status_code == 404
    database.dispose()


def test_native_closure_projection_shows_exact_prefix_and_excludes_expected_continuation():
    row = selected_row("native", source_id="SnowballV01:structure-closure")
    task = json.loads(row["task_json"])
    prefix = '  {"rows": [\n'
    extras = {
        "prefix": prefix,
        "prompt_token_ids": [7, 12, 42],
        "expected": "PRIVATE_CONTINUATION",
        "format": "json",
        "family": "nested",
        "control": "close",
    }
    task["resources"] = {
        "verifier": [
            {
                "path": "config.json",
                "source": {
                    "kind": "inline_file",
                    "content_base64": base64.b64encode(json.dumps({"extras": extras}).encode()).decode(),
                },
            }
        ]
    }
    row.update(
        contract_kind="native-structure_closure",
        task_json=None,
        lowered_task_spec=json.dumps({"task": task, "session": {"task_session": "structure_closure"}}),
        tokenizer_model="pinned-model",
        tokenizer_revision="immutable-revision",
    )
    projection = public_record(row, {row["source_id"]: "structural"}, [])
    detail = json.loads(projection["detail"])
    native = detail["native_contract"]
    assert projection["preview"] == prefix
    assert prefix in projection["search_text"]
    assert native["supplied_prefix"] == prefix
    assert native["prompt_token_ids"] == [7, 12, 42]
    assert native["tokenizer_revision"] == "immutable-revision"
    assert native["format"] == "json"
    assert detail["contract_kind"] == "native-structure_closure"
    assert all(
        secret not in json.dumps(projection)
        for secret in (
            "PRIVATE_CONTINUATION",
            "PRIVATE_ANSWER",
            "PRIVATE_FIXTURE",
            "PRIVATE_TOKEN",
            "content_base64",
        )
    )
