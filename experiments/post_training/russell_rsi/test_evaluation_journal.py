# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pytest
from rigging.filesystem.storage_path import StoragePath
from rigging.runtime_bundle import RuntimeBundle
from rolloutengine.contracts import RolloutContractError
from shellbox.backends.qemu.image import guest_code_id
from taskcompendium.environment import (
    ArtifactKind,
    EnvironmentKind,
    EnvironmentSpec,
    ExitCodeReward,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.parquet import write_tasks

from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.evaluation_journal import AttemptJournal, EvaluationJournal
from experiments.post_training.russell_rsi.interrupted_calibration import retention_journal
from experiments.post_training.russell_rsi.rollout_eval import (
    DevelopmentEvaluationConfig,
    SupplementaryEvaluationConfig,
    calibration_evaluation_journal,
    evaluate_development,
    preserve_supplementary_submission,
    run_calibration_evaluation,
    run_development_evaluation,
    supplementary_evaluation_journal,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.token_preflight import PREFLIGHT_INSTRUCTION, PREFLIGHT_PROBES, preflight_task

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
                tokens.append(role)
                if text:
                    tokens.append(self.content_id(text))
                if message["role"] == "assistant" and not body.get("continue_final_message"):
                    tokens.append(9)
            if body.get("add_generation_prompt"):
                tokens.append(3)
            self.prompts[tuple(tokens)] = body["messages"]
            return httpx.Response(200, json={"tokens": tokens})
        assert request.url.path == "/v1/completions"
        self.completions.append(body)
        if self.journal_root is not None:
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


def test_retention_journal_resumes_without_http(tmp_path, frozen_comparison):
    tasks = [preflight_task(index, PREFLIGHT_INSTRUCTION, 23) for index in range(101, 104)]
    config = replace(frozen_comparison.evaluation, limit=3, startup_attempts=1)
    write_tasks(config.tasks_path, tasks)
    server = TokenServer(Path(config.output_path) / "journal")
    requests = []

    def respond(request):
        requests.append(request)
        return server(request)

    async def evaluate():
        journal = retention_journal(config)
        await evaluate_development(
            config,
            "https://unit.test/v1",
            "unit",
            {"source_image": QEMU_TEST_IMAGE, "directory_name": "unused"},
            journal=journal,
            http_transport=httpx.MockTransport(respond),
        )
        return journal

    journal = asyncio.run(evaluate())
    assert journal.complete()
    assert len(server.completions) == 2 * (len(tasks) + len(PREFLIGHT_PROBES))
    request_count = len(requests)
    saved = {path: path.read_bytes() for path in (Path(config.output_path) / "journal").rglob("*.json")}
    asyncio.run(evaluate())
    assert len(requests) == request_count
    assert all(path.read_bytes() == content for path, content in saved.items())


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


@pytest.mark.parametrize(
    "failure", ["syntax", "syntax-grade-failure", "transport", "tokenize", "no-response", "no-tokens"]
)
def test_initial_model_failure_grades_only_received_syntax_rejection_without_retry(tmp_path, frozen_comparison, failure):
    task = preflight_task(101, "Authored rejection fixture.", 23).model_copy(
        update={
            "verifier": VerifierSpec(
                kind=VerifierKind.SHELL,
                parameters_json=ShellVerifierSpec(
                    argv=("test", "-f", "/workspace/model-created"),
                    reward=ExitCodeReward(),
                    timeout=5,
                    environment=(
                        EnvironmentSpec(kind=EnvironmentKind.SHELLSIM) if failure == "syntax-grade-failure" else None
                    ),
                    artifacts=(
                        (VerifierArtifact(source="/workspace/absent", target="/tmp/submission", kind=ArtifactKind.FILE),)
                        if failure == "syntax-grade-failure"
                        else ()
                    ),
                ).model_dump_json(),
            )
        }
    )
    path = tmp_path / "authored.parquet"
    write_tasks(str(path), [task])
    config = replace(frozen_comparison.evaluation, tasks_path=str(path), limit=1)
    # Calibration does not require a journal. The trace must retain the raw rejected response itself.
    server = TokenServer(journal_root=None)
    rejected_requests = []
    raw_responses = []

    def endpoint(request):
        body = json.loads(request.content)
        if request.url.path == "/tokenize":
            authored = any("Authored rejection fixture." in message.get("content", "") for message in body["messages"])
            if authored and failure == "tokenize":
                raise httpx.ReadError("Authored tokenizer failure", request=request)
            return server(request)
        messages = server.prompts[tuple(body["prompt"])]
        if not any("Authored rejection fixture." in message.get("content", "") for message in messages):
            return server(request)
        rejected_requests.append(body)
        if failure == "transport":
            raise httpx.ReadError("Authored transport failure", request=request)
        payload = (
            {"choices": []}
            if failure == "no-response"
            else {
                "choices": [
                    {
                        "text": "<tool_call>{invalid</tool_call>",
                        "token_ids": [] if failure == "no-tokens" else [71, 72],
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        response = httpx.Response(200, json=payload)
        raw_responses.append(response.content)
        return response

    async def run():
        await evaluate_development(
            config,
            "https://unit.test/v1",
            "unit",
            {"source_image": QEMU_TEST_IMAGE, "directory_name": "unused"},
            http_transport=httpx.MockTransport(endpoint),
        )

    if failure == "no-tokens":
        with pytest.raises(ExceptionGroup) as caught:
            asyncio.run(run())
        assert isinstance(caught.value.exceptions[0], RolloutContractError)
        assert len(rejected_requests) == 1
        return
    asyncio.run(run())
    record = json.loads((tmp_path / "parent/traces.jsonl").read_text())
    assert record["interrupted_operation"] == ("grade" if failure == "syntax-grade-failure" else "model")
    assert record["steps"] == []
    assert record["prompt_token_ids"] == record["response_token_ids"] == record["loss_mask"] == []
    assert all(message["role"] != "assistant" for message in record["messages"])
    assert len(rejected_requests) == (0 if failure == "tokenize" else 1)
    if failure not in {"syntax", "syntax-grade-failure"}:
        assert record["grade"]["status"] == "unavailable"
        assert record["grade"]["reward"] is None
        assert record["failure"] is None
        return
    if failure == "syntax":
        assert record["grade"]["status"] == "graded"
        assert record["grade"]["reward"] == 0
        assert record["execution_error"]["type"] == "ModelResponseRejected"
    else:
        assert record["grade"]["status"] == "unavailable"
        assert record["grade"]["reward"] is None
    assert record["failure"]["exception_type"] == "ModelResponseRejected"
    assert record["failure"]["diagnostics"]["parse_error"] == {
        "type": "HermesSyntaxError",
        "message": "The model emitted invalid Hermes tool syntax",
    }
    evidence = record["failure"]["diagnostics"]["rejected_response"]
    assert base64.b64decode(evidence["response_body_base64"]) == raw_responses[0]
    assert evidence["response_sha256"] == hashlib.sha256(raw_responses[0]).hexdigest()
    assert evidence["request"] == rejected_requests[0]
    assert evidence["request_sha256"] == compact_json_sha256(rejected_requests[0])
    received = json.loads(base64.b64decode(evidence["response_body_base64"]))["choices"][0]
    assert received["token_ids"] == [71, 72]
    assert received["text"] == "<tool_call>{invalid</tool_call>"
    assert received["finish_reason"] == "stop"


@pytest.fixture
def calibration_config(frozen_comparison):
    config = frozen_comparison.evaluation
    tasks = [preflight_task(index, PREFLIGHT_INSTRUCTION, 23) for index in range(101, 133)]
    write_tasks(config.tasks_path, tasks)
    return replace(config, limit=32, samples_per_task=8, temperature=1.0)


def test_completed_journal_reconstructs_without_http_or_runtime(tmp_path, frozen_comparison, monkeypatch):
    config = frozen_comparison.evaluation
    server = TokenServer(tmp_path / "journal")
    journal = supplementary_evaluation_journal(frozen_comparison)
    asyncio.run(
        evaluate_development(
            config,
            "https://unit.test/v1",
            "unit",
            {"source_image": QEMU_TEST_IMAGE, "directory_name": "unused"},
            journal=journal,
            http_transport=httpx.MockTransport(server),
        )
    )
    assert len(server.completions) == 2 * (4 + len(PREFLIGHT_PROBES))
    assert len(list((tmp_path / "journal/0/task").glob("*/*/result.json"))) == 4
    response_urls = {json.loads(p.read_text())["url"] for p in (tmp_path / "journal").rglob("response.json")}
    assert response_urls == {"https://unit.test/tokenize", "https://unit.test/v1/completions"}
    traces = tmp_path / "parent/traces.jsonl"
    content = traces.read_bytes()
    traces.unlink()

    def unexpected_start(*args, **kwargs):
        pytest.fail("Saved calibration attempted runtime or model startup")

    monkeypatch.setattr("experiments.post_training.russell_rsi.rollout_eval.install_runtime_bundle", unexpected_start)
    monkeypatch.setattr("experiments.post_training.russell_rsi.rollout_eval.local_inference", unexpected_start)
    run_development_evaluation(config, journal=journal)
    assert traces.read_bytes() == content
    summary = json.loads((tmp_path / "parent/failure_summary.json").read_text())
    assert summary["categories"] == {"passed": 4}
    assert len(server.completions) == 2 * (4 + len(PREFLIGHT_PROBES))


def test_calibration_issued_incomplete_and_changed_binding_refuse_replay(tmp_path, calibration_config):
    config = calibration_config
    journal = calibration_evaluation_journal(config)
    assert len(journal.binding["attempts"]["task"]) == 256
    task_key, task_sha = next(iter(journal.binding["attempts"]["task"].items()))
    attempt = journal.attempt("task", task_key, task_sha)
    requests = []

    def interrupted(request):
        requests.append(request)
        raise httpx.ReadError("Lost response", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(interrupted)) as client:

            async def operation():
                await attempt.post(client, "https://unit.test/v1/completions", {"prompt": [1]})
                return {}

            with pytest.raises(httpx.ReadError):
                await attempt.run(operation)

    asyncio.run(run())
    assert (Path(str(attempt.directory)) / "turns/000/issued.json").exists()
    with pytest.raises(RuntimeError, match="incomplete"):
        run_calibration_evaluation(config)
    with pytest.raises(ValueError, match="Immutable record differs"):
        calibration_evaluation_journal(replace(config, model_identity="changed-model"))
    tasks = [preflight_task(index, PREFLIGHT_INSTRUCTION, 24) for index in range(101, 133)]
    write_tasks(config.tasks_path, tasks)
    with pytest.raises(ValueError, match="Immutable record differs"):
        calibration_evaluation_journal(config)
    assert len(requests) == 1
