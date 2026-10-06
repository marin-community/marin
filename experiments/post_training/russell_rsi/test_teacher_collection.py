# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest
from levanter.data.text.formats import ChatLmDatasetFormat
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tokenizers import load_tokenizer
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.contracts import ModelRequest, RolloutContractError
from rolloutengine.engine import ShellboxRolloutEngine
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineStartupError
from taskcompendium.environment import EnvironmentKind
from taskcompendium.grading_result import Outcome
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from experiments.post_training.glm import GLM_MODEL
from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.collection_recovery import (
    REASONING_MAPPING_VERSION,
    CollectionRecovery,
    StudentContextAmendment,
)
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_chat_collection import chat_teacher_evidence, run_teacher_chat
from experiments.post_training.russell_rsi.teacher_collection import (
    TeacherModelConfig,
    TeacherTask,
    TeacherTurnProvider,
    collect_teacher_rows,
    student_row,
)
from experiments.post_training.russell_rsi.token_preflight import PREFLIGHT_INSTRUCTION, preflight_task


def config():
    return TeacherModelConfig("frozen-trajectory", 4096, 0.0, "low")


def model_response(prompt, tokens, message, finish_reason="stop"):
    return {
        "prompt_token_ids": prompt,
        "choices": [{"token_ids": tokens, "message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(tokens)},
    }


@pytest.fixture
def student_tokenizer(tmp_path):
    source = Path(__file__).resolve().parents[3] / "lib/levanter/tests"
    destination = tmp_path / "tokenizer"
    destination.mkdir()
    return load_tokenizer(str(stage_gpt2_tokenizer(source, destination)))


@pytest.mark.parametrize("rewrite_prefix", [False, True])
def test_teacher_native_tools_keep_tokens_and_private_task_fields_off_wire(tmp_path, rewrite_prefix):
    requests = []

    async def send(request):
        index = len(requests)
        directory = tmp_path / "turns" / f"{index:03d}"
        body = json.loads(request.content)
        assert json.loads((directory / "request.json").read_text()) == body
        assert (directory / "issued.json").exists()
        assert (directory / "request-wire.bin").read_bytes() == request.content
        wire_record = json.loads((directory / "request-wire.json").read_text())
        assert wire_record["sha256"] == hashlib.sha256(request.content).hexdigest()
        assert request.headers["Content-Type"] == wire_record["content_type"] == "application/json"
        requests.append(body)
        if index == 0:
            assert "48213" not in request.content.decode()
            message = {
                "role": "assistant",
                "content": None,
                "reasoning": "Read the public workspace file.",
                "tool_calls": [
                    {
                        "id": "native-glm-call",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "cat /workspace/preflight-value.txt"}),
                        },
                    }
                ],
            }
            return httpx.Response(200, json=model_response([100], [101], message, "tool_calls"))
        prior = body["messages"][-2]
        preserves_reasoning = (
            prior.get("reasoning_content") == "Read the public workspace file." and "reasoning" not in prior
        )
        prompt = [100, 101, 102] if preserves_reasoning and not rewrite_prefix else [100, 999, 102]
        return httpx.Response(200, json=model_response(prompt, [103], {"role": "assistant", "content": "48213"}))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            provider = TeacherTurnProvider(
                client, lambda: "https://teacher.test/v1", config(), StoragePath(str(tmp_path))
            )
            engine = ShellboxRolloutEngine(
                provider,
                {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
                max_turns=4,
                command_timeout=10,
                convention=SubmissionConvention(id="teacher", answer_format=AnswerFormat.PLAIN),
            )
            return await engine.run(preflight_task(1, PREFLIGHT_INSTRUCTION, 48213))

    if rewrite_prefix:
        with pytest.raises(RolloutContractError, match="changed the served token prefix"):
            asyncio.run(run())
        return
    rollout = asyncio.run(run())
    assert rollout.steps[0].turn.message["reasoning"] == "Read the public workspace file."
    assert "reasoning_content" not in rollout.steps[0].turn.message
    assert rollout.grade.reward == 1
    assert rollout.prompt_token_ids == (100,)
    assert rollout.response_token_ids == (101, 102, 103)
    assert rollout.loss_mask == (1, 0, 1)
    assert requests[1]["messages"][-1]["tool_call_id"] == "native-glm-call"
    assert requests[1]["messages"][-2]["reasoning_content"] == "Read the public workspace file."
    saved = json.loads((tmp_path / "turns/001/model-turn.json").read_text())
    assert saved == json.loads(json.dumps(asdict(rollout.steps[-1].turn)))


@pytest.mark.parametrize("alias", ["original reasoning", None, "", "different reasoning", 123])
def test_teacher_dual_reasoning_preserves_prefix_or_rejects_before_issuance(tmp_path, alias):
    sent = []
    message = {
        "role": "assistant",
        "content": "",
        "reasoning": alias,
        "reasoning_content": "original reasoning",
    }

    async def send(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=model_response([100, 101], [102], {"role": "assistant", "content": "done"}))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            provider = TeacherTurnProvider(
                client, lambda: "https://teacher.test/v1", config(), StoragePath(str(tmp_path))
            )
            return await provider(ModelRequest((message,), {}, (100, 101), None))

    if alias in ("different reasoning", 123):
        with pytest.raises(RolloutContractError, match="reasoning fields"):
            asyncio.run(run())
        assert sent == []
        assert not (tmp_path / "turns/000/issued.json").exists()
    else:
        turn = asyncio.run(run())
        assert turn.prompt_token_ids == (100, 101)
        assert sent[0]["messages"][0]["reasoning_content"] == "original reasoning"
    assert message["reasoning"] == alias


@pytest.mark.parametrize("status,error", [(200, UnicodeDecodeError), (503, httpx.HTTPStatusError)])
def test_teacher_saves_raw_response_before_failure_and_never_reissues(tmp_path, status, error):
    sent = []
    raw = b"invalid JSON response\xff"

    async def send(request):
        sent.append(request.content)
        return httpx.Response(status, content=raw)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            for _ in range(2):
                provider = TeacherTurnProvider(
                    client, lambda: "https://teacher.test/v1", config(), StoragePath(str(tmp_path))
                )
                with pytest.raises(error):
                    await provider(ModelRequest(({"role": "user", "content": "public"},), {}, (), None))

    asyncio.run(run())
    record = json.loads((tmp_path / "turns/000/response.json").read_text())
    assert base64.b64decode(record["body_base64"]) == raw
    assert len(sent) == 1


def test_teacher_refuses_reissue_after_transport_failure(tmp_path):
    sent = []

    async def send(request):
        sent.append(request.content)
        raise httpx.ReadTimeout("response unavailable", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            request = ModelRequest(({"role": "user", "content": "public"},), {}, (), None)
            provider = TeacherTurnProvider(
                client, lambda: "https://teacher.test/v1", config(), StoragePath(str(tmp_path))
            )
            with pytest.raises(httpx.ReadTimeout):
                await provider(request)
            resumed = TeacherTurnProvider(
                client, lambda: "https://teacher.test/v1", config(), StoragePath(str(tmp_path))
            )
            with pytest.raises(RuntimeError, match="ambiguous"):
                await resumed(request)

    asyncio.run(run())
    assert len(sent) == 1
    assert (tmp_path / "turns/000/issued.json").exists()
    assert not (tmp_path / "turns/000/response.json").exists()


def test_teacher_preserves_raw_tokens_when_chat_changes_the_prefix(tmp_path):
    payload = model_response([100, 999, 102], [103], {"role": "assistant", "content": "answer"})

    async def send(request):
        return httpx.Response(200, json=payload)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            provider = TeacherTurnProvider(
                client, lambda: "https://teacher.test/v1", config(), StoragePath(str(tmp_path))
            )
            with pytest.raises(RolloutContractError, match="prefix"):
                await provider(ModelRequest(({"role": "user", "content": "public"},), {}, (100, 101), 1))

    asyncio.run(run())
    raw = json.loads((tmp_path / "turns/000/response.json").read_text())
    assert json.loads(base64.b64decode(raw["body_base64"])) == payload
    assert not (tmp_path / "turns/000/model-turn.json").exists()


def test_student_row_masks_observations_keeps_code_and_does_not_truncate(student_tokenizer):
    messages = [
        {"role": "system", "content": "SYSTEMMARKER"},
        {"role": "user", "content": "USERMARKER"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "REASONINGMARKER",
            "tool_calls": [
                {
                    "id": "call-shell",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": json.dumps({"command": "printf '<think>literal code</think>'"}),
                    },
                }
            ],
        },
        {"role": "tool", "name": "shell", "tool_call_id": "call-shell", "content": "TOOLMARKER"},
        {"role": "assistant", "content": "FINALMARKER " + "complete output " * 3000},
    ]
    row = student_row(messages, {}, student_tokenizer)
    rendered = student_tokenizer.decode(list(row.input_ids))
    targets = student_tokenizer.decode(
        [token for token, mask in zip(row.input_ids, row.assistant_mask, strict=True) if mask]
    )
    assert len(row.input_ids) > 4096
    assert "REASONINGMARKER" not in rendered
    assert "<think>literal code</think>" in targets
    assert "FINALMARKER" in targets
    assert all(marker not in targets for marker in ("SYSTEMMARKER", "USERMARKER", "TOOLMARKER"))
    assert messages[2]["reasoning_content"] == "REASONINGMARKER"


@pytest.mark.parametrize("reverse_properties", [False, True])
def test_student_row_matches_persisted_nested_tool_schema(tmp_path, student_tokenizer, reverse_properties):
    properties = {
        "zeta": {"description": "Second argument", "type": "string"},
        "alpha": {"type": "string", "description": "First argument"},
    }
    if reverse_properties:
        properties = dict(reversed(list(properties.items())))
    property_order = list(properties)
    options = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "inspect",
                    "description": "Inspect two values",
                    "parameters": {"type": "object", "properties": properties, "required": ["alpha", "zeta"]},
                },
            }
        ],
    }
    messages = [
        {"role": "user", "content": "Inspect the two values."},
        {"role": "assistant", "content": "Both values are present."},
    ]
    expected_example = {
        "messages": messages,
        "chat_template_kwargs": {**options, "enable_thinking": False},
    }
    expected_jsonl = json.dumps(expected_example, sort_keys=True) + "\n"
    row = student_row(messages, options, student_tokenizer)
    assert json.dumps(row.example, sort_keys=True) + "\n" == expected_jsonl
    path = StoragePath(str(tmp_path / "student-row.json"))
    write_once(path, asdict(row))
    saved = json.loads(path.read_text())
    processor = ChatLmDatasetFormat(
        chat_template=MARIN_CHAT_TEMPLATE, pack=False, mask_user_turns=True, slice_strategy="raise"
    ).build_preprocessor(student_tokenizer)
    for example in [saved["example"], json.loads(expected_jsonl)]:
        processed = processor([example])[0]
        assert list(map(int, processed["input_ids"])) == saved["input_ids"]
        assert list(map(int, processed["assistant_masks"])) == saved["assistant_mask"]
    original = processor([expected_example])[0]
    original_targets = [
        int(token) for token, mask in zip(original["input_ids"], original["assistant_masks"], strict=True) if mask
    ]
    saved_targets = [token for token, mask in zip(saved["input_ids"], saved["assistant_mask"], strict=True) if mask]
    assert saved_targets == original_targets
    assert list(options["tools"][0]["function"]["parameters"]["properties"]) == property_order


@pytest.mark.parametrize("contract_failure", [None, "empty_prompt", "missing_prompt", "missing_response"])
def test_collection_consumes_interrupted_slots_and_reuses_eight_complete_families(
    tmp_path, student_tokenizer, contract_failure
):
    selected = []
    tasks = {}
    for index in range(9):
        task = preflight_task(index + 10, PREFLIGHT_INSTRUCTION, 80000 + index)
        tasks[task.id] = task
        selected.append(TeacherTask(f"family-{index}", "boundaries", task.id, digest(task.model_dump(mode="json"))))
    directory = StoragePath(str(tmp_path / "collection"))
    for attempt in range(2):
        write_once(
            directory / "trajectories" / f"00-{attempt}" / "trajectory.json",
            {
                "task": asdict(selected[0]),
                "attempt": attempt,
            },
        )
    sent = []

    async def send(request):
        body = json.loads(request.content)
        sent.append(body)
        if contract_failure and not body["prompt_cache_key"].endswith("-preflight"):
            response = model_response([100], [101], {"role": "assistant", "content": "invalid"})
            if contract_failure == "empty_prompt":
                response["prompt_token_ids"] = []
            elif contract_failure == "missing_prompt":
                del response["prompt_token_ids"]
            else:
                del response["choices"][0]["token_ids"]
            return httpx.Response(200, json=response)
        if body["messages"][-1]["role"] == "tool":
            answer = json.loads(body["messages"][-1]["content"])["stdout"].strip()
            if body["prompt_cache_key"].endswith("-01-0"):
                answer = "0"
            return httpx.Response(
                200,
                json=model_response(
                    [100, 101, 102],
                    [103],
                    {
                        "role": "assistant",
                        "content": answer,
                    },
                ),
            )
        return httpx.Response(
            200,
            json=model_response(
                [100],
                [101],
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-shell",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": json.dumps({"command": "cat /workspace/preflight-value.txt"}),
                            },
                        }
                    ],
                },
                "tool_calls",
            ),
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            args = (
                tuple(selected),
                tasks,
                {"skills": [{"label": "boundaries"}]},
                student_tokenizer,
                "test-tokenizer-identity",
                client,
                lambda: "https://teacher.test/v1",
                config(),
                {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
                directory,
            )

            async def collect():
                records = {key: task.model_dump_json() for key, task in tasks.items()}
                return await collect_teacher_rows(args[0], records, *args[2:])

            admitted_task = tasks[selected[0].task_id]
            tasks[admitted_task.id] = admitted_task.model_copy(update={"tags": ("changed-task",)})
            with pytest.raises(ValueError, match="differs from the frozen admitted task"):
                await collect()
            assert sent == []
            assert not (tmp_path / "collection/plan.json").exists()
            tasks[admitted_task.id] = admitted_task
            if contract_failure:
                with pytest.raises(RolloutContractError, match="exact prompt and response token IDs"):
                    await collect()
                marker_path = tmp_path / "collection/contract-failure.json"
                marker = json.loads(marker_path.read_text())
                assert marker["task"] == asdict(selected[1])
                assert marker["slot"] == "01-0"
                assert marker["attempt"] == 0
                assert marker["exception_type"] == "RolloutContractError"
                assert marker["plan_sha256"] == compact_json_sha256(
                    json.loads((tmp_path / "collection/plan.json").read_text())
                )
                assert not (tmp_path / "collection/trajectories/01-0/rollout.json").exists()
                assert not (tmp_path / "collection/trajectories/01-0/qualification.json").exists()
                count = len(sent)
                saved_marker = marker_path.read_bytes()
                with pytest.raises(RolloutContractError, match=marker["exception_message"]):
                    await collect()
                assert len(sent) == count
                assert marker_path.read_bytes() == saved_marker
                return None
            result = await collect()
            assert len(sent) == 22  # Four preflight requests, eight successes, and one failed two-turn attempt.
            assert await collect() == result
            assert len(sent) == 22
            return result

    result = asyncio.run(run())
    if contract_failure:
        return
    assert result is not None
    assert result["status"] == "passed"
    assert [row["task"]["family"] for row in result["accepted"]] == [f"family-{index}" for index in range(1, 9)]
    assert len(result["attempts"]) == 11
    assert [row["status"] for row in result["attempts"][:2]] == ["interrupted_without_complete_rollout"] * 2
    failed = json.loads((tmp_path / "collection/trajectories/01-0/rollout.json").read_text())
    assert failed["grade"]["reward"] == 0
    assert result["attempts"][2]["status"] == "failed"
    for index in range(1, 9):
        attempt = 1 if index == 1 else 0
        row = json.loads((tmp_path / "collection/trajectories" / f"{index:02d}-{attempt}/student-row.json").read_text())
        assert len(row["input_ids"]) <= 4096
        assert sum(row["assistant_mask"]) > 0


@pytest.mark.parametrize("tamper", [None, "exhausted", "marker", "plan", "proof", "reservation", "preflight"])
def test_counted_recovery_retires_first_slot_and_reuses_exact_preflight(tmp_path, student_tokenizer, tamper):
    source = tmp_path / "predecessor"
    source.mkdir()
    tasks = {}
    selected = []
    for index in range(10):
        task = preflight_task(
            index + 10, PREFLIGHT_INSTRUCTION + (" public context " * 500 if index == 0 else ""), 80000 + index
        )
        tasks[task.id] = task
        selected.append(TeacherTask(f"family-{index}", "boundaries", task.id, digest(task.model_dump(mode="json"))))
    capabilities = {"skills": [{"label": "boundaries"}]}
    plan = {
        "selected": [asdict(entry) for entry in selected],
        "capabilities": capabilities,
        "model": asdict(config()),
        "student_tokenizer_identity": "test-tokenizer-identity",
        "student_template_sha256": hashlib.sha256(MARIN_CHAT_TEMPLATE.encode()).hexdigest(),
    }

    def save(name, value):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value.encode() if isinstance(value, str) else json.dumps(value, sort_keys=True).encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    info_hash = save(
        ".executor_info",
        {"name": "collection", "output_path": str(source), "config": {"version": "v1", "fingerprint": "abcd"}},
    )
    status_hash = save(".executor_status", "FAILED")
    plan_hash = save("plan.json", plan)
    marker = {
        "task": asdict(selected[0]),
        "attempt": 0,
        "slot": "00-0",
        "exception_type": "RolloutContractError",
        "exception_message": "Native GLM chat changed the served token prefix",
        "plan_sha256": compact_json_sha256(plan),
    }
    marker_hash = save("contract-failure.json", marker)
    reservation_hash = save("trajectories/00-0/trajectory.json", {"task": asdict(selected[0]), "attempt": 0})
    preflight_identity = f"{config().session_identity}-preflight"
    preflight_reservation_hash = save(
        "preflight/reservation.json", {**asdict(config()), "session_identity": preflight_identity}
    )
    request = {
        "model": GLM_MODEL,
        "prompt_cache_key": preflight_identity,
        "max_tokens": config().max_tokens,
        "temperature": config().temperature,
        "chat_template_kwargs": {"reasoning_effort": config().reasoning_effort},
        "return_token_ids": True,
    }
    preflight_hash = save(
        "preflight/token-preflight.json",
        {
            "status": "passed",
            "attempts": [
                {
                    "status": "passed",
                    "requests": [{"request": request}] * 2,
                    "rollout": {"grade": {"status": "graded", "reward": 1}},
                }
                for _ in range(2)
            ],
        },
    )
    artifacts = {}
    for name, filename, value in (
        ("source_request", "request.json", {"model": GLM_MODEL}),
        ("source_issued", "issued.json", {"prefix_token_ids": [100, 101]}),
        ("source_raw_response", "response.json", {"status_code": 200}),
    ):
        relative = f"trajectories/00-0/turns/003/{filename}"
        artifacts[name] = {"path": str(source / relative), "sha256": save(relative, value)}
    for name in (
        "original_tokenizer_request",
        "original_tokenizer_response",
        "mapped_tokenizer_request",
        "mapped_tokenizer_response",
    ):
        artifacts[name] = {"path": str(source / f"{name}.json"), "sha256": save(f"{name}.json", {"tokens": [100, 101]})}
    declaration = {
        "request_limit": 2,
        "maximum_total_output_tokens": 2,
        "http_retries": 0,
        "failed_teacher_slot": "00-0",
        "scientific_trajectory_budget_remaining": 19,
        "settings": {"model": GLM_MODEL, "max_tokens": 1, "temperature": 0, "return_token_ids": True},
    }
    artifacts["declaration"] = {
        "path": str(source / "diagnostic-declaration.json"),
        "sha256": save("diagnostic-declaration.json", declaration),
    }
    proof = {
        "status": "passed",
        "model": GLM_MODEL,
        "mapping_version": REASONING_MAPPING_VERSION,
        "proof_method": "bounded-real-relay-transport",
        "generation_requests": 2,
        "diagnostic_generation_requests": 2,
        "diagnostic_output_tokens": 2,
        "teacher_generation_requests": 0,
        "tokenizer_requests": 7,
        "tokenizer_successful_renders": 4,
        "http_retries": 0,
        "outputs_reused": False,
        "scientific_trajectory_budget_remaining": 19,
        "transport_diagnostic_declared_and_issued_before_send": True,
        "scientific_plan_unchanged": True,
        "scientific_trajectory_budget_unchanged": True,
        "failed_teacher_slot_still_consumed": True,
        "request_pair_difference_only_alias_mapping": True,
        "endpoint_identity": {"url": "https://teacher.test/v1/chat/completions"},
        "original_render_matches_saved_prompt": True,
        "mapped_render_preserves_expected_prefix": True,
        "reasoning_insertion_only": True,
        "expected_prefix_token_count": 2,
        "artifacts": artifacts,
        **{
            field: artifacts[name]["sha256"]
            for name, field in (
                ("source_request", "source_request_sha256"),
                ("source_issued", "source_issued_sha256"),
                ("source_raw_response", "source_raw_response_sha256"),
            )
        },
    }
    proof_hash = save("token-proof.json", proof)
    recovery = CollectionRecovery(
        str(source),
        "collection@v1:abcd",
        info_hash,
        status_hash,
        plan_hash,
        marker_hash,
        reservation_hash,
        preflight_identity,
        preflight_reservation_hash,
        preflight_hash,
        str(source / "token-proof.json"),
        proof_hash,
        REASONING_MAPPING_VERSION,
        "00-0",
    )
    amendment_record = {
        "protocol": "champion-rsi-teacher-sixteen-k-context-amendment-v1",
        "previous_context_tokens": 4096,
        "context_tokens": 16384,
        "predecessor_identity": recovery.predecessor_identity,
        "predecessor_plan_sha256": recovery.plan_sha256,
        "fatal_marker_sha256": recovery.fatal_marker_sha256,
        "plan_sha256": compact_json_sha256(plan),
        "consumed_slot": "00-0",
        "consumed_trajectories": 1,
        "remaining_trajectories": 19,
        "student_rows": 8,
        "sft_updates": 4,
        "assistant_only_loss": True,
        "full_untruncated_rows": True,
    }
    amendment = StudentContextAmendment(
        str(source / "context-amendment.json"),
        save("context-amendment.json", amendment_record),
        amendment_record["protocol"],
        4096,
        16384,
    )
    if tamper not in (None, "exhausted"):
        target = {
            "marker": "contract-failure.json",
            "plan": "plan.json",
            "proof": "token-proof.json",
            "reservation": "trajectories/00-0/trajectory.json",
            "preflight": "preflight/token-preflight.json",
        }[tamper]
        save(target, {"changed": True})
    sent = []

    async def send(request):
        body = json.loads(request.content)
        sent.append(body)
        assert not body["prompt_cache_key"].endswith("-preflight")
        if body["messages"][-1]["role"] == "tool":
            answer = "0" if tamper == "exhausted" else json.loads(body["messages"][-1]["content"])["stdout"].strip()
            return httpx.Response(
                200, json=model_response([100, 101, 102], [103], {"role": "assistant", "content": answer})
            )
        return httpx.Response(
            200,
            json=model_response(
                [100],
                [101],
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "shell",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": json.dumps({"command": "cat /workspace/preflight-value.txt"}),
                            },
                        }
                    ],
                },
                "tool_calls",
            ),
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            return await collect_teacher_rows(
                tuple(selected),
                {key: task.model_dump_json() for key, task in tasks.items()},
                capabilities,
                student_tokenizer,
                "test-tokenizer-identity",
                client,
                lambda: "https://teacher.test/v1",
                config(),
                {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
                StoragePath(str(tmp_path / "amended")),
                recovery,
                amendment,
            )

    if tamper not in (None, "exhausted"):
        with pytest.raises(ValueError):
            asyncio.run(run())
        assert sent == []
        return
    result = asyncio.run(run())
    if tamper == "exhausted":
        assert result["status"] == "insufficient_rows"
        assert len(result["attempts"]) == 20
        assert len(sent) == 38
        assert result["attempts"][0]["status"] == "contract_failure_predecessor"
        assert result["attempts"][-1]["task"] == asdict(selected[-1])
        assert result["attempts"][-1]["attempt"] == 1
        return
    assert result["status"] == "passed"
    assert result["attempts"][0]["status"] == "contract_failure_predecessor"
    assert result["accepted"][0]["task"] == asdict(selected[0])
    assert result["accepted"][0]["attempt"] == 1
    assert [row["task"]["family"] for row in result["accepted"]] == [f"family-{i}" for i in range(8)]
    assert sent[0]["prompt_cache_key"].endswith("-00-1")
    assert len(sent) == 16
    assert len(result["attempts"]) == 9
    assert not (tmp_path / "amended/trajectories/00-0").exists()
    lineage = json.loads((tmp_path / "amended/recovery-lineage.json").read_text())
    assert lineage["cumulative_attempt_limit"] == 20
    assert lineage["recovery"] == asdict(recovery)
    assert lineage["context_amendment"] == asdict(amendment)
    first_row = json.loads((tmp_path / "amended/trajectories/00-1/student-row.json").read_text())
    assert 4096 < len(first_row["input_ids"]) <= 16384
    assert any(first_row["assistant_mask"])


@pytest.mark.parametrize("chat_only", [False, True])
def test_chat_teacher_retains_changed_native_request_tokens_without_rl_stream(tmp_path, chat_only):
    requests = []
    responses = []

    async def send(request):
        index = len(requests)
        body = json.loads(request.content)
        requests.append(request.content)
        assert (tmp_path / "turns" / f"{index:03d}" / "request-wire.bin").read_bytes() == request.content
        if index == 0:
            message = {
                "role": "assistant",
                "content": None,
                "reasoning": "Read the file.",
                "tool_calls": [
                    {
                        "id": "read",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"command":"cat /workspace/preflight-value.txt"}'},
                    }
                ],
            }
            response = httpx.Response(200, json=model_response([100], [101], message, "tool_calls"))
        else:
            assert body["messages"][-1]["role"] == "tool"
            assert "48213" in body["messages"][-1]["content"]
            assert body["messages"][-2]["reasoning_content"] == "Read the file."
            response = httpx.Response(
                200, json=model_response([100, 999, 102], [103], {"role": "assistant", "content": "48213"})
            )
        responses.append(response.content)
        return response

    async def run():
        task = preflight_task(1, PREFLIGHT_INSTRUCTION, 48213)
        factories = {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()}
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            if chat_only:
                result = await run_teacher_chat(
                    task, StoragePath(str(tmp_path)), config(), client, lambda: "https://teacher.test/v1", factories
                )
                assert result.grade.reward == 1.0
                assert result.execution_error is None
                assert result.turns[0].response_token_ids == (101,)
                assert result.turns[1].prompt_token_ids == (100, 999, 102)
                evidence = chat_teacher_evidence(result)
                assert json.loads((tmp_path / "chat-result.json").read_text()) == json.loads(json.dumps(evidence))
                with pytest.raises(RuntimeError, match="reserved chat session"):
                    await run_teacher_chat(
                        task, StoragePath(str(tmp_path)), config(), client, lambda: "https://teacher.test/v1", factories
                    )
                return
            engine = ShellboxRolloutEngine(
                TeacherTurnProvider(client, lambda: "https://teacher.test/v1", config(), StoragePath(str(tmp_path))),
                factories,
                max_turns=16,
                command_timeout=120,
                convention=SubmissionConvention(id="russell-teacher", answer_format=AnswerFormat.PLAIN),
            )
            await engine.run(task)

    if chat_only:
        asyncio.run(run())
    else:
        with pytest.raises(RolloutContractError):
            asyncio.run(run())
    assert len(requests) == len(responses) == 2
    for index, raw in enumerate(responses):
        record = json.loads((tmp_path / "turns" / f"{index:03d}" / "response.json").read_text())
        assert base64.b64decode(record["body_base64"]) == raw


def test_chat_teacher_thinking_exhaustion_preserves_length_and_rejects_row(tmp_path):
    raw = json.dumps(
        model_response(
            [100], [101, 102], {"role": "assistant", "content": None, "reasoning": "Still thinking."}, "length"
        )
    ).encode()

    async def send(request):
        return httpx.Response(200, content=raw)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            return await run_teacher_chat(
                preflight_task(1, PREFLIGHT_INSTRUCTION, 48213),
                StoragePath(str(tmp_path)),
                config(),
                client,
                lambda: "https://teacher.test/v1",
                {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
            )

    result = asyncio.run(run())
    assert result.stop_reason == "length"
    assert result.grade.status == Outcome.EXTRACTION_ERROR
    assert result.grade.reward is None
    assert result.turns[0].response_token_ids == (101, 102)
    saved = json.loads((tmp_path / "turns/000/response.json").read_text())
    assert base64.b64decode(saved["body_base64"]) == raw


class StartupFailureThenShellSim:
    def __init__(self):
        self.failed = False

    async def create(self, spec):
        if not self.failed:
            self.failed = True
            raise MachineStartupError("Guest did not boot")
        return await ShellSimMachineFactory().create(spec)


def test_chat_teacher_startup_retry_preserves_later_http_failure(tmp_path):
    requests = []

    async def send(request):
        requests.append(request.content)
        return httpx.Response(503, content=b"endpoint unavailable")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            task = preflight_task(1, PREFLIGHT_INSTRUCTION, 48213)
            args = (
                task,
                StoragePath(str(tmp_path)),
                config(),
                client,
                lambda: "https://teacher.test/v1",
                {EnvironmentKind.SHELLSIM: StartupFailureThenShellSim()},
            )
            result = await run_teacher_chat(*args)
            with pytest.raises(RuntimeError, match="reserved chat session"):
                await run_teacher_chat(*args)
            return result

    result = asyncio.run(run())
    assert result.startup_attempt == 2
    assert result.interrupted_operation == "model"
    assert result.execution_error is not None
    assert result.execution_error["type"] == "HTTPStatusError"
    assert result.grade.reward is None
    assert len(requests) == 1
    assert (tmp_path / "startup-failure-1.json").exists()
    record = json.loads((tmp_path / "turns/000/response.json").read_text())
    assert record["status_code"] == 503
    assert base64.b64decode(record["body_base64"]) == b"endpoint unavailable"


def test_chat_teacher_agent_deadline_preserves_ambiguous_issuance(tmp_path):
    requests = []

    async def send(request):
        requests.append(request.content)
        await asyncio.Event().wait()
        raise AssertionError("Unreachable response")

    async def run():
        task = preflight_task(1, PREFLIGHT_INSTRUCTION, 48213).model_copy(update={"agent_timeout": 0.01})
        async with httpx.AsyncClient(transport=httpx.MockTransport(send)) as client:
            return await run_teacher_chat(
                task,
                StoragePath(str(tmp_path)),
                config(),
                client,
                lambda: "https://teacher.test/v1",
                {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
            )

    result = asyncio.run(run())
    assert result.execution_error is not None
    assert result.execution_error["type"] == "TimeoutError"
    assert result.interrupted_operation == "model"
    assert result.grade.reward is None
    assert len(requests) == 1
    assert (tmp_path / "turns/000/issued.json").exists()
    assert not (tmp_path / "turns/000/response.json").exists()
