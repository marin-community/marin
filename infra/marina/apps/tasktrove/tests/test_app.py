# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI
from fastapi.testclient import TestClient
from marina.apps import Services
from tasktrove import app as tasktrove


def client(tmp_path) -> TestClient:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    rows = [
        {
            "source": "source-a",
            "path": "codeforces-08503",
            "family": "code",
            "converter": "codeforces",
            "mode": "stdio",
            "dockerfile_id": "python",
            "language": "python",
            "tags": ["contest"],
            "has_solution": True,
            "task_binary": b"first-archive",
        },
        {
            "source": "source-b",
            "path": "calendar-001",
            "family": "tool",
            "converter": "agent_calendar",
            "mode": "script",
            "dockerfile_id": "python",
            "language": None,
            "tags": ["calendar", "agent"],
            "has_solution": False,
            "task_binary": b"second-archive",
        },
        {
            "source": "source-b",
            "path": "calendar-002",
            "family": "tool",
            "converter": "agent_calendar",
            "mode": "script",
            "dockerfile_id": "ubuntu",
            "language": "",
            "tags": ["calendar"],
            "has_solution": False,
            "task_binary": b"third-archive",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), tasks / "part-00000.parquet", row_group_size=2)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "dockerfiles": {
                    "python": {"base_image": "python:3.10-slim"},
                    "ubuntu": {"base_image": "ubuntu:24.04"},
                }
            }
        )
    )
    services = Services(name="tasktrove", data_url=str(tmp_path), database=None)
    api = cast(FastAPI, tasktrove.create_api(services).app)
    return TestClient(api)


def test_task_page_filters_and_returns_exact_total(tmp_path) -> None:
    api = client(tmp_path)

    response = api.get(
        "/tasks",
        params={"source": "source-b", "tag": "calendar", "environment": "python:3.10-slim", "query": "ENDAR"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "rows": [
            {
                "row": 1,
                "source": "source-b",
                "path": "calendar-001",
                "family": "tool",
                "converter": "agent_calendar",
                "mode": "script",
                "dockerfile_id": "python",
                "environment": "python:3.10-slim",
                "language": "",
                "tags": ["calendar", "agent"],
                "has_solution": False,
            }
        ],
        "total": 1,
        "offset": 0,
        "limit": 50,
    }


def test_task_metadata_and_archive_are_served_by_row(tmp_path) -> None:
    api = client(tmp_path)

    task = api.get("/tasks/0")
    archive = api.get("/tasks/0/archive")

    assert task.status_code == 200
    assert task.json()["path"] == "codeforces-08503"
    assert task.json()["row"] == 0
    assert archive.status_code == 200
    assert archive.content == b"first-archive"
    assert archive.headers["content-type"] == "application/gzip"
    assert archive.headers["content-disposition"] == 'attachment; filename="task-0.tar.gz"'


def test_task_api_rejects_rows_outside_the_parquet(tmp_path) -> None:
    api = client(tmp_path)

    assert api.get("/tasks/3").status_code == 404
    assert api.get("/tasks/3/archive").status_code == 404
    assert api.get("/tasks", params={"environment": "missing:latest"}).json()["rows"] == []
