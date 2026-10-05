# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from levanter.tokenizers import load_tokenizer
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.contracts import ModelRequest, RolloutContractError
from rolloutengine.engine import ShellboxRolloutEngine
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.sources import compact_json_sha256
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


def test_teacher_native_tools_keep_tokens_and_private_task_fields_off_wire(tmp_path):
    requests = []

    async def send(request):
        index = len(requests)
        directory = tmp_path / "turns" / f"{index:03d}"
        body = json.loads(request.content)
        assert json.loads((directory / "request.json").read_text()) == body
        assert (directory / "issued.json").exists()
        requests.append(body)
        if index == 0:
            assert "48213" not in request.content.decode()
            message = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Read the public workspace file.",
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
        return httpx.Response(
            200, json=model_response([100, 101, 102], [103], {"role": "assistant", "content": "48213"})
        )

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

    rollout = asyncio.run(run())
    assert rollout.grade.reward == 1
    assert rollout.prompt_token_ids == (100,)
    assert rollout.response_token_ids == (101, 102, 103)
    assert rollout.loss_mask == (1, 0, 1)
    assert requests[1]["messages"][-1]["tool_call_id"] == "native-glm-call"
    assert requests[1]["messages"][-2]["reasoning_content"] == "Read the public workspace file."
    saved = json.loads((tmp_path / "turns/001/model-turn.json").read_text())
    assert saved == json.loads(json.dumps(asdict(rollout.steps[-1].turn)))


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


def test_collection_consumes_interrupted_slots_and_reuses_eight_complete_families(tmp_path, student_tokenizer):
    selected = []
    tasks = {}
    for index in range(9):
        task = preflight_task(index + 10, PREFLIGHT_INSTRUCTION, 80000 + index)
        tasks[task.id] = task
        selected.append(
            TeacherTask(f"family-{index}", "boundaries", task.id, compact_json_sha256(task.model_dump(mode="json")))
        )
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
            result = await collect_teacher_rows(*args)
            assert len(sent) == 22  # Four preflight requests, eight successes, and one failed two-turn attempt.
            assert await collect_teacher_rows(*args) == result
            assert len(sent) == 22
            return result

    result = asyncio.run(run())
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
