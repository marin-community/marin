# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import json
from dataclasses import asdict, replace

import httpx
import pytest
from rigging.filesystem.storage_path import StoragePath
from rigging.runtime_bundle import RuntimeBundle
from shellbox.backends.qemu.image import guest_code_id
from taskcompendium.environment import ArtifactKind, ShellVerifierSpec, VerifierArtifact
from taskcompendium.models import VerifierKind
from taskcompendium.parquet import write_tasks

from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.evaluation_journal import AttemptJournal, EvaluationJournal
from experiments.post_training.russell_rsi.rollout_eval import (
    DevelopmentEvaluationConfig,
    SupplementaryEvaluationConfig,
    evaluate_development,
    preserve_supplementary_submission,
    supplementary_evaluation_journal,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.token_preflight import PREFLIGHT_INSTRUCTION, preflight_task

QEMU_TEST_IMAGE = "unit@sha256:" + "0" * 64


@pytest.mark.parametrize("failure", ["transport", "invalid-json"])
def test_issued_http_attempt_cannot_be_repeated_after_failure(tmp_path, failure):
    requests = []

    def server(request):
        requests.append(request)
        issued = json.loads((tmp_path / "turns/000/issued.json").read_text())
        assert issued["request_sha256"] == compact_json_sha256(json.loads(request.content))
        assert (tmp_path / "reservation.json").exists()
        if failure == "transport":
            raise httpx.ReadError("Response was lost", request=request)
        return httpx.Response(200, content=b"invalid-json")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:

            async def operation():
                response = await attempt.post(client, "https://unit.test/v1/completions", {"prompt": [1]})
                return response.json()

            attempt = AttemptJournal(StoragePath(str(tmp_path)), {"slot": "parent/task-1"})
            with pytest.raises(httpx.ReadError if failure == "transport" else json.JSONDecodeError):
                await attempt.run(operation)
            with pytest.raises(RuntimeError, match="incomplete"):
                await AttemptJournal(StoragePath(str(tmp_path)), attempt.binding).run(operation)

    asyncio.run(run())
    assert len(requests) == 1
    assert not (tmp_path / "result.json").exists()
    response = tmp_path / "turns/000/response.json"
    assert response.exists() == (failure == "invalid-json")


@pytest.mark.parametrize("content", [b"", b"diff --git a/value b/value\n\x00\xff"])
def test_submission_bytes_survive_grading_failure(tmp_path, content):
    requests = []
    source = tmp_path / "collected.patch"
    source.write_bytes(content)
    artifact = VerifierArtifact(source="/tmp/model.patch", target="/tmp/model.patch", kind=ArtifactKind.FILE)
    attempt = AttemptJournal(StoragePath(str(tmp_path / "attempt")), {"slot": "authored/task-1"})

    def server(request):
        requests.append(request)
        return httpx.Response(200, json={"saved_response": True})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:

            async def operation():
                await attempt.post(client, "https://unit.test/v1/completions", {"prompt": [1]})
                await preserve_supplementary_submission(artifact, source)
                source.unlink()
                raise RuntimeError("Authored private grading loss")

            with pytest.raises(RuntimeError, match="grading loss"):
                await attempt.run(operation)
            with pytest.raises(RuntimeError, match="incomplete"):
                await AttemptJournal(attempt.directory, attempt.binding).run(operation)

    asyncio.run(run())
    saved = json.loads((tmp_path / "attempt/submission.json").read_text())
    preserved = base64.b64decode(saved["body_base64"], validate=True)
    assert preserved == content
    assert saved["sha256"] == hashlib.sha256(preserved).hexdigest()
    assert saved["binding"] == attempt.binding
    assert saved["artifact"] == artifact.model_dump(mode="json")
    assert (tmp_path / "attempt/turns/000/response.json").exists()
    assert not (tmp_path / "attempt/result.json").exists()
    assert len(requests) == 1


class TokenServer:
    def __init__(self, journal_root):
        self.journal_root = journal_root
        self.prompts = {}
        self.ids = {}
        self.completions = []

    def content_id(self, text):
        return self.ids.setdefault(text, 1000 + len(self.ids))

    def __call__(self, request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            tokens = []
            for message in body["messages"]:
                role = {"system": 1, "user": 2, "assistant": 3, "tool": 4}[message["role"]]
                text = "tool-call" if message.get("tool_calls") else message.get("content", "")
                tokens.extend([role, self.content_id(text)])
                if message["role"] == "assistant" and not body.get("continue_final_message"):
                    tokens.append(9)
            if body.get("add_generation_prompt"):
                tokens.append(3)
            self.prompts[tuple(tokens)] = body["messages"]
            return httpx.Response(200, json={"tokens": tokens})
        assert request.url.path == "/v1/completions"
        self.completions.append(body)
        issued = [json.loads(p.read_text()) for p in self.journal_root.rglob("issued.json")]
        assert any(row["request_sha256"] == compact_json_sha256(body) for row in issued)
        messages = self.prompts[tuple(body["prompt"])]
        tools = [message for message in messages if message["role"] == "tool"]
        if tools:
            text = next(value for value in ("48213", "73961", "23") if value in tools[-1]["content"])
            token = self.content_id(text)
        else:
            text = '<tool_call>{"name":"shell","arguments":{"command":"cat /workspace/preflight-value.txt"}}</tool_call>'
            token = self.content_id("tool-call")
        return httpx.Response(200, json={"choices": [{"text": text, "token_ids": [token, 9], "finish_reason": "stop"}]})


@pytest.fixture
def frozen_comparison(tmp_path):
    pytest.importorskip("skyrl_train.inference_engines.chat_continuation")
    tasks = [preflight_task(index, PREFLIGHT_INSTRUCTION, 23) for index in range(101, 105)]
    path = tmp_path / "tasks.parquet"
    write_tasks(str(path), tasks)
    runtime_dir = tmp_path / "runtime/unused"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "image.json").write_text(
        json.dumps(
            {
                "image_reference": QEMU_TEST_IMAGE,
                "manifest_digest": "sha256:" + "0" * 64,
                "guest_code_id": guest_code_id(),
            }
        )
    )
    runtime = RuntimeBundle(
        "/unused/runtime.json",
        "0" * 64,
        "/unused/runtime.tar.gz",
        "1" * 64,
        installation_parent=str(tmp_path / "runtime"),
    )
    manifest = {
        "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "runtime_bundle": asdict(runtime),
        "tasks": [{"task_sha256": digest(task.model_dump(mode="json"))} for task in tasks],
    }
    manifest_path = tmp_path / "panel-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = SupplementaryEvaluationConfig(
        DevelopmentEvaluationConfig(
            "/weights/parent",
            "parent",
            "frozen-four-task-panel",
            "tokenizer",
            "revision",
            str(path),
            str(tmp_path / "parent"),
            runtime,
            4,
            startup_attempts=3,
        ),
        str(tmp_path / "journal"),
        0,
        ("parent", "candidate"),
        str(manifest_path),
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    return config


def test_eight_matched_attempts_and_four_probes_resume_without_inference(tmp_path, frozen_comparison):
    config = frozen_comparison
    server = TokenServer(tmp_path / "journal")

    async def evaluate(config):
        journal = supplementary_evaluation_journal(config)
        await evaluate_development(
            config.evaluation,
            "https://unit.test/v1",
            "unit",
            {"source_image": QEMU_TEST_IMAGE, "directory_name": "unused"},
            journal=journal,
            http_transport=httpx.MockTransport(server),
        )

    candidate = replace(
        config,
        checkpoint_index=1,
        evaluation=replace(
            config.evaluation,
            model_uri="/weights/candidate",
            model_identity="candidate",
            output_path=str(tmp_path / "candidate"),
        ),
    )
    asyncio.run(evaluate(config))
    asyncio.run(evaluate(candidate))
    requests = len(server.completions)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    traces = {p: p.read_bytes() for p in tmp_path.rglob("traces.jsonl")}
    asyncio.run(evaluate(config))
    asyncio.run(evaluate(candidate))
    assert len(server.completions) == requests == 24
    assert all(p.read_bytes() == content for p, content in before.items())
    assert all(p.read_bytes() == content for p, content in traces.items())
    assert len(list((tmp_path / "journal").glob("*/task/*/*/result.json"))) == 8
    assert len(list((tmp_path / "journal").glob("*/preflight/*/result.json"))) == 4
    for label in ("parent", "candidate"):
        summary = json.loads((tmp_path / label / "failure_summary.json").read_text())
        assert summary["categories"] == {"passed": 4}
        assert len((tmp_path / label / "traces.jsonl").read_text().splitlines()) == 4


def test_interrupted_attempt_and_changed_candidate_stop_before_any_http(tmp_path, frozen_comparison):
    config = frozen_comparison
    server = TokenServer(tmp_path / "journal")
    journal = supplementary_evaluation_journal(config)
    asyncio.run(
        evaluate_development(
            config.evaluation,
            "https://unit.test/v1",
            "unit",
            {"source_image": QEMU_TEST_IMAGE, "directory_name": "unused"},
            journal=journal,
            http_transport=httpx.MockTransport(server),
        )
    )
    result = next((tmp_path / "journal/0/task").glob("*/*/result.json"))
    result.unlink()
    count = len(server.completions)
    with pytest.raises(RuntimeError, match="incomplete"):
        supplementary_evaluation_journal(config)
    changed = replace(config, model_identities=("parent", "different-candidate"))
    with pytest.raises(ValueError, match="Immutable record differs"):
        supplementary_evaluation_journal(changed)
    assert len(server.completions) == count
    assert list(result.parent.glob("turns/*/response.json"))


def test_one_failed_http_attempt_preserves_other_tasks_without_resampling(tmp_path, frozen_comparison):
    config = frozen_comparison
    server = TokenServer(tmp_path / "journal")
    failed = False

    def interrupted_server(request):
        nonlocal failed
        if request.url.path == "/v1/completions" and len(server.completions) == 4 and not failed:
            failed = True
            server.completions.append(json.loads(request.content))
            raise httpx.ReadError("Lost one task response", request=request)
        return server(request)

    async def evaluate():
        journal = supplementary_evaluation_journal(config)
        await evaluate_development(
            config.evaluation,
            "https://unit.test/v1",
            "unit",
            {"source_image": QEMU_TEST_IMAGE, "directory_name": "unused"},
            journal=journal,
            http_transport=httpx.MockTransport(interrupted_server),
        )

    asyncio.run(evaluate())
    count = len(server.completions)
    traces = (tmp_path / "parent/traces.jsonl").read_bytes()
    summary = json.loads((tmp_path / "parent/failure_summary.json").read_text())
    assert summary["categories"] == {"execution_model": 1, "passed": 3}
    assert len(list((tmp_path / "journal/0/task").glob("*/*/result.json"))) == 4
    asyncio.run(evaluate())
    assert len(server.completions) == count
    assert (tmp_path / "parent/traces.jsonl").read_bytes() == traces


@pytest.mark.parametrize("layout", ["multiple", "directory"])
def test_unsupported_submission_stops_before_http(tmp_path, frozen_comparison, layout):
    config = frozen_comparison
    task = preflight_task(101, PREFLIGHT_INSTRUCTION, 23)
    artifact = VerifierArtifact(source="/workspace/patch", target="/workspace/patch", kind=ArtifactKind.FILE)
    verifier = ShellVerifierSpec(
        argv=("true",),
        timeout=10,
        environment=task.environment.model_copy(update={"interaction": None}),
        artifacts=(
            (artifact, artifact)
            if layout == "multiple"
            else (artifact.model_copy(update={"kind": ArtifactKind.DIRECTORY}) if layout == "directory" else artifact,)
        ),
    )
    task = task.model_copy(
        update={
            "verifier": task.verifier.model_copy(
                update={
                    "kind": VerifierKind.SHELL,
                    "parameters_json": verifier.model_dump_json(),
                }
            ),
        }
    )
    write_tasks(config.evaluation.tasks_path, [task])
    server = TokenServer(tmp_path / "journal")

    async def evaluate():
        await evaluate_development(
            config.evaluation,
            "https://unit.test/v1",
            "unit",
            {},
            journal=EvaluationJournal(StoragePath(str(tmp_path / "journal")), {"attempts": {}}),
            http_transport=httpx.MockTransport(server),
        )

    with pytest.raises(ValueError, match="one collected file"):
        asyncio.run(evaluate())
    assert not server.prompts
    assert not server.completions
