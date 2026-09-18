# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient
from marina.apps import Services
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from plantt import app as plantt


class FakeServices(Services):
    _engine: Engine

    def __init__(self, engine: Engine):
        super().__init__(name="plantt", data_url="memory://plantt", database=None)
        object.__setattr__(self, "_engine", engine)

    def engine(self) -> Engine:
        return self._engine


def client() -> TestClient:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    plantt.migrate(engine)
    plantt.migrate(engine)
    api = cast(FastAPI, plantt.create_api(FakeServices(engine)).app)
    return TestClient(api)


def example_document(title: str = "Release plan") -> dict[str, Any]:
    return {
        "title": title,
        "workstreams": [
            {
                "name": "Build",
                "tasks": [
                    {"name": "Train", "start": ["date", "2026-09-01"], "end": ["weeks", 2]},
                    {
                        "name": "Evaluate",
                        "start": "Train",
                        "end": ["days", 3],
                        "deps": ["Train"],
                    },
                ],
            }
        ],
    }


def test_chart_api_persists_updates_and_rejects_stale_writes() -> None:
    api = client()

    schema = api.get("/schema").json()
    assert set(schema) == {"description", "fields", "task", "milestone"}
    assert set(schema["fields"]) == {"title", "note", "workstreams", "capacity", "annotations"}

    created_response = api.post("/charts", json={"document": example_document()})
    assert created_response.status_code == 201
    created = created_response.json()
    assert created["revision"] == 1

    listed = api.get("/charts").json()
    assert [(chart["id"], chart["title"]) for chart in listed] == [(created["id"], "Release plan")]
    assert api.get(f"/charts/{created['id']}").json()["document"] == example_document()

    updated_document = example_document("Launch plan")
    updated_response = api.put(
        f"/charts/{created['id']}",
        json={"document": updated_document, "revision": created["revision"]},
    )
    assert updated_response.status_code == 200
    updated = updated_response.json()
    assert (updated["title"], updated["revision"], updated["document"]) == ("Launch plan", 2, updated_document)

    stale_response = api.put(
        f"/charts/{created['id']}",
        json={"document": example_document("Stale plan"), "revision": 1},
    )
    assert stale_response.status_code == 409
    assert api.get(f"/charts/{created['id']}").json()["title"] == "Launch plan"

    assert api.request("DELETE", f"/charts/{created['id']}", json={"revision": 2}).status_code == 204
    assert api.get("/charts").json() == []


def test_chart_api_rejects_unknown_dependencies() -> None:
    document = example_document()
    document["workstreams"][0]["tasks"][1]["deps"] = ["Missing"]
    api = client()

    response = api.post("/charts", json={"document": document})

    assert response.status_code == 422
    assert api.get("/charts").json() == []


def test_chart_api_rejects_task_scheduling_cycles() -> None:
    document = example_document()
    document["workstreams"][0]["tasks"][0]["start"] = "Evaluate"
    api = client()

    response = api.post("/charts", json={"document": document})

    assert response.status_code == 422
    assert api.get("/charts").json() == []
