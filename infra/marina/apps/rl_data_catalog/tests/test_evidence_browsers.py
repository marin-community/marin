# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen attempt and reviewer evidence stays bound to its task and original bytes."""

import hashlib
import io
import json
import socket
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from marina.client import MarinaRequestError
from rigging.filesystem.storage_path import StoragePath

from infra.marina.apps.rl_data_catalog.load_browse_evidence import catalog_session
from infra.marina.apps.rl_data_catalog.load_reviews import (
    matched_review_rows,
    read_batch,
    stage_content_reviews,
    task_request_evidence,
)
from infra.marina.apps.rl_data_catalog.load_rollouts import stage_attempt, stage_rollouts
from infra.marina.apps.rl_data_catalog.release_browse import browse_invariants
from infra.marina.apps.rl_data_catalog.review_mapping import original_batch_mapping
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference


def test_attempt_archive_preserves_transcript_tools_and_verifier_without_serving_config(tmp_path: Path) -> None:
    conversation = [{"role": "assistant", "tool_calls": [{"name": "terminal", "arguments": "python check.py"}]}]
    archive_path = tmp_path / "trial-artifacts.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, value in {
            "trial/agent/conversation.json": json.dumps(conversation),
            "trial/verifier/verdict.json": '{"reward":1}',
            "trial/config.json": "secret-environment-canary",
            "../outside": "path-traversal-canary",
        }.items():
            raw = value.encode()
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    receipt = tmp_path / "00" / "attempt.json"
    receipt.parent.mkdir()
    receipt.write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "status": "scored",
                "reward": 1.0,
                "reason": None,
                "archive": str(archive_path),
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "result": {
                    "id": "trial-id",
                    "started_at": "2026-11-17T10:00:00+00:00",
                    "finished_at": "2026-11-17T10:00:03+00:00",
                    "verifier_result": {"rewards": {"reward": 1}},
                    "config": {"env": "another-secret-canary"},
                },
            }
        )
    )
    ref = {"uri": str(receipt), "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest()}
    staged = stage_attempt(ref, {"task-a"}, "small")
    assert staged["summary"]["duration"] == 3
    assert staged["summary"]["score"] == 1
    assert staged["detail"]["conversation"] == conversation
    assert staged["detail"]["verifier"] == {"reward": 1}
    assert staged["detail"]["model_input"] is None
    assert all(
        secret not in json.dumps(staged)
        for secret in ("secret-environment-canary", "path-traversal-canary", "another-secret-canary")
    )
    with pytest.raises(ValueError, match="outside the frozen"):
        stage_attempt(ref, {"another-task"}, "small")
    archive_path.write_bytes(b"changed after sealing")
    with pytest.raises(ValueError, match=r"SHA|hash|digest|changed"):
        stage_attempt(ref, {"task-a"}, "small")


def test_raw_reviewer_view_keeps_only_the_tasks_requests_and_every_retry_output() -> None:
    batch = {"attempts": []}
    for attempt in range(2):
        batch["attempts"].append(
            {
                "number": attempt,
                "query_task_ids": {"request-a": ["task-a"], "request-b": ["task-b"]},
                "requests": {
                    "requests": [
                        {"custom_id": "request-a", "body": {"messages": [{"content": f"input-{attempt}"}]}},
                        {"custom_id": "request-b", "body": {"messages": [{"content": "unrelated-task-canary"}]}},
                    ],
                    "observations": [
                        {
                            "output": "\n".join(
                                json.dumps(row)
                                for row in [
                                    {"custom_id": "request-b", "response": {"body": "unrelated-output-canary"}},
                                    {"custom_id": "request-a", "response": {"body": f"original-output-{attempt}"}},
                                ]
                            )
                        }
                    ],
                },
            }
        )
    actual = task_request_evidence(batch)["task-a"]
    assert [row["request"]["body"]["messages"][0]["content"] for row in actual["input"]] == ["input-0", "input-1"]
    assert [row["response"]["response"]["body"] for row in actual["output"]] == [
        "original-output-0",
        "original-output-1",
    ]
    assert "unrelated" not in json.dumps(actual)


def test_private_review_and_rollout_staging_refuses_before_any_artifact_read(monkeypatch):
    def refused_read(*args, **kwargs):
        raise AssertionError("Private artifact was read")

    monkeypatch.setattr(StoragePath, "read_bytes", refused_read)
    source = {"id": "Fixture:private", "data_restriction": "private_always"}
    with pytest.raises(ValueError, match="Withheld reviews remain in their private store"):
        stage_content_reviews(ManifestReference(uri="s3://fixture/private.json", sha256="a" * 64), source, 1)
    with pytest.raises(ValueError, match="Restricted rollouts remain in their private store"):
        stage_rollouts({}, source, 1)


def test_consolidated_mapping_keeps_original_identity_and_marks_ambiguous_or_missing_rows(tmp_path: Path) -> None:
    rows = [
        {
            "task_id": task,
            "task_sha256": task * 8,
            "runtime_sha256": "runtime",
            "review": json.dumps(
                {"task_id": task, "status": "unavailable", "verdict": None, "detail": "Original failure"}
            ),
        }
        for task in ("original", "ambiguous", "missing", "wrong-input", "different-judgment")
    ]
    root = tmp_path / "campaign"
    for number, tasks in enumerate((["original", "ambiguous", "wrong-input", "different-judgment"], ["ambiguous"])):
        batch = root / "batches" / str(number) / "attempts" / "original.json"
        batch.parent.mkdir(parents=True)
        selected = [row for row in rows if row["task_id"] in tasks]
        document = {
            "identity": {
                "contracts": [{key: value for key, value in row.items() if key != "review"} for row in selected],
                "reviewer": {"model": "original-reviewer", "model_revision": "original-pin"},
            },
            "reviews": [json.loads(row["review"]) for row in selected],
            "attempts": [],
        }
        for contract in document["identity"]["contracts"]:
            if contract["task_id"] == "wrong-input":
                contract["runtime_sha256"] = "another-runtime"
        for observation in document["reviews"]:
            if observation["task_id"] == "different-judgment":
                observation["detail"] = "A later, different judgment must not substitute for the sealed row"
        batch.write_text(json.dumps(document))
        # A later pointer must never retarget the original stored observation.
        (batch.parent.parent / "latest.json").write_text(json.dumps({"path": "later.json", "sha256": "0" * 64}))
    chain = tmp_path / "chain.json"
    chain.write_text(
        json.dumps(
            {
                "identity": {},
                "origins": {},
                "sealed_batch_receipts": [str(root / "batches" / "1" / "attempts" / "original.json")],
                "repair_receipt": {
                    "kind": "batch_reviews",
                    "path": str(root / "batches" / "0" / "attempts" / "original.json"),
                    "sha256": (
                        hashlib.sha256((root / "batches" / "0" / "attempts" / "original.json").read_bytes()).hexdigest()
                    ),
                },
            }
        )
    )
    mapping = original_batch_mapping(rows, [str(chain)], 2)
    by_task = {row["task_id"]: row for row in mapping["entries"]}
    assert by_task["original"]["status"] == "mapped"
    assert by_task["original"]["reviewer"]["model_revision"] == "original-pin"
    assert by_task["original"]["hash_basis"] == "publisher_reference"
    assert by_task["original"]["receipt"]["uri"].endswith("original.json")
    assert by_task["ambiguous"]["status"] == "not_mapped"
    assert "ambiguous" in by_task["ambiguous"]["reason"]
    assert by_task["wrong-input"]["status"] == "mapped"
    assert by_task["wrong-input"]["runtime_sha256"] == "runtime"
    assert by_task["wrong-input"]["original_contract"]["runtime_sha256"] == "another-runtime"
    assert by_task["missing"]["status"] == by_task["different-judgment"]["status"] == "not_mapped"
    assert (mapping["mapped"], mapping["not_mapped"]) == (2, 3)


def test_original_review_receipt_does_not_follow_an_advanced_pointer(tmp_path: Path) -> None:
    batch = tmp_path / "batches" / "one"
    (batch / "attempts").mkdir(parents=True)
    original = batch / "attempts" / "first.json"
    original.write_text('{"original":true}')
    later = batch / "attempts" / "later.json"
    later.write_text('{"original":false}')
    (batch / "latest.json").write_text(
        json.dumps({"path": str(later), "sha256": hashlib.sha256(later.read_bytes()).hexdigest()})
    )
    document, provenance = read_batch(str(original))
    assert document == {"original": True}
    assert provenance["receipt"]["uri"] == str(original)
    assert provenance["receipt"]["sha256"] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert provenance["hash_basis"] == "staging_readback"
    (batch / "latest.json").write_text(json.dumps({"path": str(original), "sha256": "0" * 64}))
    with pytest.raises(ValueError, match="Publisher batch digest"):
        read_batch(str(original))


def test_release_rejects_missing_review_rows_and_unstaged_attempts() -> None:
    source = {
        "id": "Fixture:one",
        "quality": "SAMPLE",
        "difficulty": {
            "models": [{"measurement_id": "one", "attempts_receipt": {"uri": "sealed", "sha256": "a" * 64}}],
            "historical": [],
        },
    }
    panel = {"source_id": source["id"], "pipeline_key": "qc:one", "task_count": 100, "raw_count": 100, "receipt": {}}
    inventory = {"panels": [panel], "rollouts": [{"measurement_id": "one", "expected_count": 100, "attempt_count": 100}]}

    def fetch(path):
        return (
            inventory
            if path == "api/qc/inventory"
            else {"defects": [{"id": "ambiguity", "count": 2, "sources": [{"count": 2}]}]}
        )

    assert browse_invariants(fetch, [source])["qc_rows"] == 100
    panel["missing_chunks"] = 1
    with pytest.raises(ValueError, match="Original reviewer batch coverage"):
        browse_invariants(fetch, [source])
    panel["missing_chunks"] = 0
    panel["raw_count"] = 99
    with pytest.raises(ValueError, match="QC review coverage"):
        browse_invariants(fetch, [source])
    panel["raw_count"] = 100
    inventory["rollouts"] = []
    with pytest.raises(ValueError, match="Difficulty attempt coverage"):
        browse_invariants(fetch, [source])
    # Private sources need no content staging; their page renders a marker before requesting detail.
    assert browse_invariants(fetch, [{**source, "data_restriction": "private_always"}])["qc_sources"] == 0


def test_regional_loader_retries_a_lost_idempotent_write_reply() -> None:
    class LostReply(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            LostReply.calls += 1
            if LostReply.calls == 1:
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"row_count":0}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), LostReply)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        with catalog_session() as session:
            response = session.post(f"http://127.0.0.1:{server.server_port}/query", json={"sql": "idempotent insert"})
        assert response.json() == {"row_count": 0}
        assert LostReply.calls == 2
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize(("status", "size"), [(400, 2), (500, 2), (504, 1)])
def test_exact_readback_preserves_non_deadline_and_single_row_failures(status, size):
    calls = []

    def fail(sql, parameters):
        calls.append(parameters)
        raise MarinaRequestError(status, "original failure")

    with pytest.raises(MarinaRequestError) as error:
        matched_review_rows(fail, "SELECT exact rows", "load", [{"task_id": str(i)} for i in range(size)])
    assert error.value.status_code == status
    assert len(calls) == 1
