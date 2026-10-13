# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

from marina.applets import AppletServices
from rigging.server_auth import VerifiedIdentity, identity_scope
from sqlalchemy import create_engine
from starlette.testclient import TestClient

from infra.marina.applets.snowball_probes.server.app import create_api, migrate


def test_case_metadata_preserves_nul_evidence_and_searchable_review(database_url):
    engine = create_engine(database_url)
    with engine.begin() as connection:
        migrate(connection)
    services = Mock(spec=AppletServices)
    services.engine.return_value = engine
    owner = VerifiedIdentity(user_id="nul-owner", role="user")
    review = {
        "summary": "A NUL character was emitted.",
        "assessments": [{"model": "small", "score": 0.25}],
        "breakages": [
            {
                "model": "small",
                "category": "format_violation",
                "severity": "major",
                "evidence": [{"quote": "actual\x00NUL and literal \\u0000"}],
            }
        ],
    }
    case = {
        "case_id": "nul-case",
        "run_id": "nul-run",
        "title": "JSON escape handling",
        "category": "formatting",
        "mode": "chat",
        "state": "complete",
        "slot": 0,
        "round": 0,
        "seed": {"text": "Keep a literal \\u0000 distinct from \x00"},
        "review": review,
    }
    try:
        with identity_scope(owner), TestClient(create_api(services)) as client:
            assert (
                client.post(
                    "/runs",
                    json={
                        "run_id": case["run_id"],
                        "config": {},
                        "token_cap": 1000,
                        "concurrency": 1,
                        "generated_tokens": 0,
                        "reserved_tokens": 0,
                        "state": "running",
                        "active_chats": 0,
                    },
                ).status_code
                == 200
            )
            assert client.post("/cases/batch", json={"cases": [case]}).status_code == 200
            detail = client.get("/cases/nul-case").json()
            assert detail["review"] == review
            assert detail["seed"] == case["seed"]
            matches = client.get(
                "/cases", params={"run_id": "nul-run", "breakage": "format_violation", "model": "small"}
            ).json()
            assert matches["total"] == 1
            assert matches["cases"][0]["review"] == review
            assert matches["cases"][0]["seed"] == case["seed"]
            summary = client.get("/summary", params={"run_id": "nul-run"}).json()
            assert summary["completed"] == 1
            assert summary["scores"][0]["score"] == 0.25
            assert summary["breakages"][0]["count"] == 1
            case["review"] = {"summary": "Updated review without a NUL."}
            assert client.post("/cases", json=case).status_code == 200
            assert client.get("/cases/nul-case").json()["review"] == case["review"]
    finally:
        engine.dispose()


def test_assigned_worker_saves_one_runs_results_while_owner_controls_queue_and_filter(database_url):
    engine = create_engine(database_url)
    with engine.begin() as connection:
        migrate(connection)
    services = Mock(spec=AppletServices)
    services.engine.return_value = engine
    owner = VerifiedIdentity(user_id="worker-test-owner", role="user")
    worker = VerifiedIdentity(user_id="worker@test-project.iam.gserviceaccount.com", role="user")
    run = {
        "run_id": "worker-test",
        "config": {},
        "token_cap": 1000,
        "concurrency": 2,
        "generated_tokens": 0,
        "reserved_tokens": 0,
        "state": "running",
        "active_chats": 0,
    }
    case = {
        "case_id": "worker-case",
        "run_id": run["run_id"],
        "title": "Result",
        "category": "coding",
        "mode": "agentic",
        "state": "generating",
        "slot": 0,
        "round": 0,
        "seed": {},
        "review": {},
    }
    trace = {
        "message_id": "worker-trace",
        "case_id": case["case_id"],
        "model": "small",
        "role": "assistant",
        "round": 0,
        "sequence": 0,
        "state": "complete",
        "payload": {"reasoning": "Thinking", "content": "Done"},
    }
    try:
        with identity_scope(owner), TestClient(create_api(services)) as client:
            assert client.post("/runs", json=run).status_code == 200
            assert client.post("/runs", json={**run, "run_id": "unrelated-test"}).status_code == 200
            assert client.post("/cases", json=case).status_code == 200
            assert client.post("/cases", json={**case, "case_id": "still-generating"}).status_code == 200
            assert client.post("/runs/worker-test/worker", json={"identity": worker.user_id}).status_code == 200
            with identity_scope(worker):
                assert client.post("/runs", json={**run, "generated_tokens": 12}).status_code == 200
                assert client.post("/runs", json={**run, "run_id": "new-worker-run"}).status_code == 403
                case["state"] = "complete"
                assert client.post("/cases", json=case).status_code == 200
                assert client.post("/traces", json=trace).status_code == 200
                assert client.post("/runs/worker-test/ideas/claim", json={"case_id": "claim"}).status_code == 200
                assert client.post("/runs/worker-test/control", json={"state": "stopped"}).status_code == 403
                assert client.post("/runs/worker-test/worker", json={"identity": worker.user_id}).status_code == 403
                assert (
                    client.post(
                        "/runs/worker-test/ideas", json={"text": "Topic", "kind": "topic", "chats": 1, "mode": "chat"}
                    ).status_code
                    == 403
                )
                assert client.post("/cases", json={**case, "run_id": "unrelated-test"}).status_code == 403
            assert client.get("/runs/worker-test").json()["owner"] == owner.user_id
            completed = client.get("/cases", params={"run_id": run["run_id"], "state": "complete"}).json()
            assert completed["total"] == 1
            assert completed["cases"][0]["case_id"] == case["case_id"]
            assert client.get(f"/cases/{case['case_id']}").json()["traces"][0]["payload"] == trace["payload"]
            streaming = {**trace, "message_id": "unfinished-trace", "sequence": 1, "state": "streaming"}
            assert client.post("/traces", json=streaming).status_code == 200
            recovery = client.get(f"/cases/{case['case_id']}", params={"trace_state": "streaming"}).json()
            assert [item["message_id"] for item in recovery["traces"]] == [streaming["message_id"]]
            assert recovery["traces"][0]["payload"] == streaming["payload"]
            assert len(client.get(f"/cases/{case['case_id']}").json()["traces"]) == 2
            assert client.post("/runs/worker-test/worker", json={"identity": None}).status_code == 200
            with identity_scope(worker):
                assert client.post("/traces", json=trace).status_code == 403
    finally:
        engine.dispose()


def test_idea_queue_claims_are_bounded_idempotent_and_link_to_saved_cases(database_url):
    engine = create_engine(database_url)
    with engine.begin() as connection:
        migrate(connection)
    services = Mock(spec=AppletServices)
    services.engine.return_value = engine
    owner = VerifiedIdentity(user_id="owner", role="user")
    try:
        with identity_scope(owner), TestClient(create_api(services)) as client:
            run: dict = {
                "run_id": "queue-test",
                "config": {"candidates": [{"key": "small", "revision": "abc"}]},
                "token_cap": 1000,
                "concurrency": 4,
                "generated_tokens": 0,
                "reserved_tokens": 0,
                "state": "running",
                "active_chats": 0,
            }
            assert client.post("/runs", json=run).status_code == 200
            run["config"]["coordinator"] = "/owned/replacement-serving-job"
            assert client.post("/runs", json=run).status_code == 200
            changed = {**run["config"], "candidates": [{"key": "small", "revision": "different"}]}
            assert client.post("/runs", json={**run, "config": changed}).status_code == 409
            assert client.get("/runs").json()[0]["config"] == run["config"]
            assert client.get("/runs/queue-test").json()["config"] == run["config"]
            idea = client.post(
                "/runs/queue-test/ideas",
                json={"text": "Follow JSON schemas", "kind": "topic", "chats": 4, "mode": "chat"},
            ).json()

            def claim(number):
                with identity_scope(owner):
                    return client.post("/runs/queue-test/ideas/claim", json={"case_id": f"case-{number}"}).json()

            with ThreadPoolExecutor(max_workers=8) as executor:
                claims = list(executor.map(claim, range(8)))
            assigned = [number for number, result in enumerate(claims) if result["idea"]]
            assert len(assigned) == 4
            for number in assigned:
                assert claim(number)["idea"]["idea_id"] == idea["idea_id"]
            row = client.get("/runs/queue-test/ideas").json()[0]
            assert row["assigned"] == 4 and row["completed"] == 0
            number = assigned[0]
            case = {
                "case_id": f"case-{number}",
                "run_id": "queue-test",
                "title": "Preparing",
                "category": "formatting",
                "mode": "agentic",
                "state": "generating",
                "slot": 0,
                "round": 0,
                "seed": {},
                "review": {},
            }
            assert client.post("/cases", json=case).status_code == 200
            case.update(state="complete", mode="chat", seed={"queued_idea": {"idea_id": idea["idea_id"]}})
            assert client.post("/cases", json=case).status_code == 200
            filtered = client.get("/cases", params={"run_id": "queue-test", "idea_id": idea["idea_id"]}).json()
            assert filtered["total"] == 1 and filtered["cases"][0]["mode"] == "chat"
            assert client.get("/runs/queue-test/ideas").json()[0]["completed"] == 1
            traces = [
                {
                    "message_id": f"message-{index}",
                    "case_id": case["case_id"],
                    "model": "small",
                    "role": "assistant",
                    "round": 0,
                    "sequence": index,
                    "state": "streaming",
                    "payload": {"reasoning": f"Thinking {index}", "content": ""},
                }
                for index in range(16)
            ]
            assert client.post("/traces/batch", json={"traces": traces}).status_code == 200
            case.update(state="reviewing")
            assert client.post("/cases/batch", json={"cases": [case]}).status_code == 200
            traces[0].update(state="complete", payload={"reasoning": "Thought", "content": "Verified"})
            assert client.post("/traces/batch", json={"traces": [traces[0]]}).status_code == 200
            detail = client.get(f"/cases/{case['case_id']}").json()
            assert detail["state"] == "reviewing"
            assert len(detail["traces"]) == 16
            assert detail["traces"][0]["payload"] == {"reasoning": "Thought", "content": "Verified"}
            cancelled = client.post(
                "/runs/queue-test/ideas", json={"text": "Unused idea", "kind": "topic", "chats": 3, "mode": "auto"}
            ).json()
            assert client.post(f"/runs/queue-test/ideas/{cancelled['idea_id']}/cancel").status_code == 200
            assert claim(100)["idea"] is None
            with identity_scope(VerifiedIdentity(user_id="other", role="user")):
                assert client.post("/runs/queue-test/ideas/claim", json={"case_id": "forbidden"}).status_code == 403
                assert client.post("/traces/batch", json={"traces": [traces[0]]}).status_code == 403
                assert client.post("/cases/batch", json={"cases": [case]}).status_code == 403
    finally:
        engine.dispose()


def test_summary_workload_filter_separates_scores_breakages_and_counts(database_url):
    engine = create_engine(database_url)
    with engine.begin() as connection:
        migrate(connection)
    services = Mock(spec=AppletServices)
    services.engine.return_value = engine
    owner = VerifiedIdentity(user_id="summary-owner", role="user")
    run = {
        "run_id": "workload-summary",
        "config": {},
        "token_cap": 1000,
        "concurrency": 2,
        "generated_tokens": 0,
        "reserved_tokens": 0,
        "state": "running",
        "active_chats": 0,
    }
    try:
        with identity_scope(owner), TestClient(create_api(services)) as client:
            assert client.post("/runs", json=run).status_code == 200
            assert client.post("/runs", json={**run, "run_id": "other-summary"}).status_code == 200
            for case_id, run_id, mode, category, score, breakage in [
                ("chat-summary", run["run_id"], "chat", "coding", 0.2, "incorrect_answer"),
                ("agentic-summary", run["run_id"], "agentic", "coding", 0.8, "invalid_tool_call"),
                ("agentic-reasoning", run["run_id"], "agentic", "reasoning", 0.6, "invalid_tool_call"),
                ("unrelated-summary", "other-summary", "chat", "coding", 1.0, "hallucination"),
            ]:
                case = {
                    "case_id": case_id,
                    "run_id": run_id,
                    "title": case_id,
                    "category": category,
                    "mode": mode,
                    "state": "complete",
                    "slot": 0,
                    "round": 0,
                    "seed": {},
                    "review": {
                        "assessments": [{"model": "small", "score": score}],
                        "breakages": [{"model": "small", "category": breakage, "severity": "high"}],
                        "verification": [{"model": "small", "score": 1.0 if mode == "chat" else 0.0}],
                        "verification_eligible": case_id != "agentic-reasoning",
                    },
                }
                assert client.post("/cases", json=case).status_code == 200
            for mode, count, expected_scores, expected_breakages in [
                (None, 3, {"coding": (0.5, 2), "reasoning": (0.6, 1)}, {"incorrect_answer": 1, "invalid_tool_call": 2}),
                ("chat", 1, {"coding": (0.2, 1)}, {"incorrect_answer": 1}),
                ("agentic", 2, {"coding": (0.8, 1), "reasoning": (0.6, 1)}, {"invalid_tool_call": 2}),
            ]:
                parameters = {"run_id": run["run_id"]}
                if mode is not None:
                    parameters["mode"] = mode
                response = client.get("/summary", params=parameters)
                assert response.status_code == 200
                summary = response.json()
                assert summary["cases"] == summary["completed"] == count
                assert {item["category"]: (item["score"], item["n"]) for item in summary["scores"]} == expected_scores
                expected_verified = {"coding": (0.5, 2) if mode is None else (1.0 if mode == "chat" else 0.0, 1)}
                assert {
                    item["category"]: (item["score"], item["n"]) for item in summary["verified_scores"]
                } == expected_verified
                assert {item["category"]: item["count"] for item in summary["breakages"]} == expected_breakages
                assert sum(item["cases"] for item in summary["categories"]) == count
                if mode is not None:
                    assert {item["mode"] for item in summary["categories"]} == {mode}
    finally:
        engine.dispose()
