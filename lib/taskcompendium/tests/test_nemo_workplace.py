# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Workplace row 0 import and mutable provider behavior."""

import asyncio
import hashlib
import json
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from urllib.request import urlopen

import pytest

from nemo_workplace.provider import SEED_SHA256, NemoWorkplaceProvider, _seed_digest, expected_state_json
from taskcompendium.grading import exact_answer
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.importers.nemo_workplace import (
    DATASET_REVISION,
    PROVIDER_GIT_REVISION,
    PROVIDER_REPOSITORY,
    ROW_SHA256,
    SOURCE_EXAMPLE_MAX_BYTES,
    SOURCE_EXAMPLE_SHA256,
    SOURCE_EXAMPLE_URL,
    import_row,
    select_row_zero,
)
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import AnswerType, ConversationInput, ConversationTrace, TaskSpec, TextMessage, VerifierKind
from taskcompendium.resources import ResourceVisibility
from taskcompendium.submission import PlainText, SubmissionConvention
from taskcompendium.verifier_registry import grade_answer


@pytest.fixture(scope="module")
def source_example() -> bytes:
    """Resolve the pinned upstream source in trusted test setup, before trials."""
    with urlopen(SOURCE_EXAMPLE_URL, timeout=30) as response:
        return response.read(SOURCE_EXAMPLE_MAX_BYTES + 1)


@pytest.fixture(scope="module")
def source_row(source_example: bytes) -> bytes:
    return select_row_zero(source_example)


@pytest.fixture(scope="module")
def trusted_provider_checkout(tmp_path_factory) -> Path:
    """Resolve the pinned source once, before any exported Harbor trial starts."""
    checkout = tmp_path_factory.mktemp("workplace-source") / "nemo_workplace"
    subprocess.run(["git", "clone", "--quiet", PROVIDER_REPOSITORY, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", PROVIDER_GIT_REVISION], check=True)
    return checkout


@contextmanager
def _serve_endpoint(endpoint: type[BaseHTTPRequestHandler]) -> Iterator[ThreadingHTTPServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _send_completion(handler: BaseHTTPRequestHandler, message: dict) -> None:
    body = json.dumps({"choices": [{"message": message}]}).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _provider() -> NemoWorkplaceProvider:
    return NemoWorkplaceProvider(seed_sha256=SEED_SHA256)


def _source(data: bytes) -> tuple[bytes, dict]:
    return data, json.loads(data)


class ProviderEnvironment:
    def __init__(self, provider: NemoWorkplaceProvider):
        self.provider = provider

    async def provider_state(self, name: str):
        assert name == "workplace"
        return self.provider.canonical_state()


async def _reward(specification: TaskSpec, convention: SubmissionConvention, provider: NemoWorkplaceProvider) -> float:
    conversation = ConversationTrace(
        events=(*specification.context.events, TextMessage(role="assistant", content="Done."))
    )
    result = await grade_answer(specification, convention, conversation, ProviderEnvironment(provider))
    assert result.reward is not None
    return result.reward


def test_workplace_import_pins_row_tool_surface_and_private_state(source_example: bytes, source_row: bytes):
    assert hashlib.sha256(source_example).hexdigest() == SOURCE_EXAMPLE_SHA256
    assert hashlib.sha256(source_row).hexdigest() == ROW_SHA256
    data, row = _source(source_row)
    specification, convention, binding = import_row(data)
    provider = binding.tool_providers["workplace"]
    assert _seed_digest() == provider.seed_sha256
    assert specification.source.revision == DATASET_REVISION
    assert specification.answer_type.value == convention.answer_format.value == "state"
    assert convention.provider == "workplace"
    assert specification.verifier.kind == VerifierKind.STRUCTURED_EXACT
    assert len(provider.tools) == len(row["responses_create_params"]["tools"]) == 27
    assert {resource.visibility for resource in specification.resources} == {ResourceVisibility.VERIFIER}
    visible = specification.context.model_dump_json()
    assert [(event.role, event.content) for event in specification.context.events] == [
        (message["role"], message["content"]) for message in row["responses_create_params"]["input"]
    ]
    assert all(resource.path not in visible for resource in specification.resources)
    assert "ground_truth" not in visible
    assert "source-row.json" in {resource.path for resource in specification.resources}
    private_provenance = next(
        resource for resource in specification.resources if resource.path == "source-provenance.json"
    )
    source_record = json.loads(private_provenance.content)
    assert source_record["source_file_sha256"] == SOURCE_EXAMPLE_SHA256
    assert (source_record["dataset_license"], source_record["dataset_owner"]) == (
        "CC-BY-4.0",
        "NVIDIA Corporation",
    )
    assert source_record["provider_git_revision"] == PROVIDER_GIT_REVISION
    assert source_record["provider_repository"] == PROVIDER_REPOSITORY


def test_workplace_import_rejects_unpinned_row_and_changed_tools(monkeypatch, source_example: bytes, source_row: bytes):
    data, row = _source(source_row)
    with pytest.raises(ValueError, match="pinned digest"):
        select_row_zero(source_example + b" ")
    with pytest.raises(ValueError, match="size limit"):
        select_row_zero(b" " * (SOURCE_EXAMPLE_MAX_BYTES + 1))
    with pytest.raises(ValueError, match="pinned raw digest"):
        import_row(data + b" ")
    row["responses_create_params"]["tools"][0]["name"] = "wrong_tool"
    changed = json.dumps(row).encode()
    monkeypatch.setattr("taskcompendium.importers.nemo_workplace.ROW_SHA256", hashlib.sha256(changed).hexdigest())
    with pytest.raises(ValueError, match="pinned provider"):
        import_row(changed)


async def test_workplace_success_wrong_and_noop_state(source_row: bytes):
    _, row = _source(source_row)
    specification, convention, _ = import_row(source_row)
    gold = row["ground_truth"]
    expected = json.loads(expected_state_json(gold))
    success, wrong, noop = (_provider() for _ in range(3))
    action = gold[0]
    await success.dispatch_action(action["name"], action["arguments"], "call-1")
    await wrong.dispatch_action(
        action["name"],
        '{"email_id":"00000057","body":"Thanks for the update - I will not follow up."}',
        "call-1",
    )
    await noop.dispatch_action(
        "email_get_email_information_by_id", '{"email_id":"00000057","field":"subject"}', "call-1"
    )
    assert success.canonical_state() == expected
    assert await _reward(specification, convention, success) == 1.0
    assert await _reward(specification, convention, wrong) == 0.0
    assert await _reward(specification, convention, noop) == 0.0


async def test_workplace_tool_error_recovers_and_retains_call_order(source_row: bytes):
    _, row = _source(source_row)
    specification, convention, _ = import_row(source_row)
    action = row["ground_truth"][0]
    provider = _provider()
    error = await provider.dispatch_action(action["name"], '{"email_id":"00000057","unknown":"x"}', "call-bad")
    success = await provider.dispatch_action(action["name"], action["arguments"], "call-good")
    assert [entry.call_id for entry in provider.trace] == ["call-bad", "call-good"]
    assert [entry.output for entry in provider.trace] == [error, success]
    assert await _reward(specification, convention, provider) == 1.0


async def test_workplace_concurrent_trials_start_from_fresh_seed(source_row: bytes):
    _, row = _source(source_row)
    specification, convention, _ = import_row(source_row)
    action = row["ground_truth"][0]
    first, second = _provider(), _provider()
    await asyncio.gather(
        first.dispatch_action(action["name"], action["arguments"], "first"),
        second.dispatch_action(
            "email_get_email_information_by_id", '{"email_id":"00000057","field":"subject"}', "second"
        ),
    )
    assert (await _reward(specification, convention, first), await _reward(specification, convention, second)) == (
        1.0,
        0.0,
    )


async def test_workplace_harbor_scripted_endpoint_recovers_after_tool_error(
    tmp_path, trusted_provider_checkout, source_row: bytes
):
    data, row = _source(source_row)
    specification, convention, binding = import_row(data)
    task_dir = lower_to_harbor(
        specification,
        convention,
        binding,
        tmp_path / "task",
        trusted_provider_sources={"workplace": trusted_provider_checkout},
    )
    gold = row["ground_truth"][0]
    calls = [
        ('{"email_id":"00000057","unknown":"x"}', "bad"),
        (gold["arguments"], "good"),
    ]
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if len(requests) <= len(calls):
                arguments, label = calls[len(requests) - 1]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{label}",
                            "type": "function",
                            "function": {"name": gold["name"], "arguments": arguments},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "Done."}
            _send_completion(self, message)

    with _serve_endpoint(Endpoint) as server:
        result = await run_trial(
            task_dir,
            binding,
            ChatLaunch(
                model="fixture",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                temperature=1.0,
                parallel_tool_calls=False,
                max_turns=4,
            ),
            tmp_path / "trials",
            "workplace",
        )
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert len(requests) == 3
    assert len(requests[0]["tools"]) == 27
    assert all(request["temperature"] == 1.0 and request["parallel_tool_calls"] is False for request in requests)
    assert requests[1]["messages"][-1]["tool_call_id"] == "call-bad"
    assert requests[2]["messages"][-1]["tool_call_id"] == "call-good"
    assert "ground_truth" not in json.dumps(requests)
    metadata = result.agent_result.metadata
    assert len(metadata["tool_definitions"]) == 27
    assert [action["call_id"] for action in metadata["tools"]] == ["call-bad", "call-good"]
    assert [action["observation"] for action in metadata["tools"]] == [
        requests[1]["messages"][-1]["content"],
        requests[2]["messages"][-1]["content"],
    ]
    assert [message["role"] for message in metadata["all_messages"]] == [
        "system",
        "user",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


async def test_workplace_chat_tools_can_answer_text_from_observation(
    tmp_path, trusted_provider_checkout, source_row: bytes
):
    imported, _, binding = import_row(source_row)
    subject = "Task Update on Develop prototype for report generation"
    specification = TaskSpec(
        id="workplace-subject-answer",
        context=ConversationInput(
            events=(TextMessage(role="user", content="Use the available tools to find the subject of email 00000057."),)
        ),
        verifier=exact_answer(subject),
        source=imported.source,
        environment_requirements=imported.environment_requirements,
        tool_providers=imported.tool_providers,
        answer_type=AnswerType.TEXT,
    )
    convention = PlainText(id="plain")
    task_dir = lower_to_harbor(
        specification,
        convention,
        binding,
        tmp_path / "task",
        trusted_provider_sources={"workplace": trusted_provider_checkout},
    )
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            if len(requests) == 1:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-subject",
                            "type": "function",
                            "function": {
                                "name": "email_get_email_information_by_id",
                                "arguments": '{"email_id":"00000057","field":"subject"}',
                            },
                        }
                    ],
                }
            else:
                observation = json.loads(payload["messages"][-1]["content"])
                message = {"role": "assistant", "content": observation["output"]["subject"]}
            _send_completion(self, message)

    with _serve_endpoint(Endpoint) as server:
        result = await run_trial(
            task_dir,
            binding,
            ChatLaunch(
                model="fixture",
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                temperature=1.0,
                parallel_tool_calls=False,
                max_turns=2,
            ),
            tmp_path / "trials",
            "subject",
        )
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert len(requests) == 2
    assert len(requests[0]["tools"]) == 27
    assert requests[1]["messages"][-1]["tool_call_id"] == "call-subject"
    assert result.agent_result.metadata["assistant_final"]["content"] == subject


async def test_workplace_harbor_trials_are_fresh_and_concurrent(tmp_path, trusted_provider_checkout, source_row: bytes):
    specification, convention, binding = import_row(source_row)
    task_dir = lower_to_harbor(
        specification,
        convention,
        binding,
        tmp_path / "task",
        trusted_provider_sources={"workplace": trusted_provider_checkout},
    )
    gold = json.loads(source_row)["ground_truth"][0]
    turns: dict[str, int] = {}
    lock = Lock()

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            model = payload["model"]
            with lock:
                turns[model] = turns.get(model, 0) + 1
                turn = turns[model]
            if turn == 1:
                if model.startswith("good"):
                    name, arguments = gold["name"], gold["arguments"]
                else:
                    name, arguments = "email_get_email_information_by_id", '{"email_id":"00000057","field":"subject"}'
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"{model}-call",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": "Done."}
            _send_completion(self, message)

    with _serve_endpoint(Endpoint) as server:

        async def trial(model: str):
            launch = ChatLaunch(
                model=model,
                api_base=f"http://127.0.0.1:{server.server_port}/v1",
                temperature=1.0,
                parallel_tool_calls=False,
                max_turns=3,
            )
            return await run_trial(task_dir, binding, launch, tmp_path / "trials", model)

        good, noop = await asyncio.gather(trial("good-1"), trial("noop"))
        good_again = await trial("good-2")

    assert all(result.exception_info is None for result in (good, noop, good_again))
    assert [result.verifier_result.rewards for result in (good, noop, good_again)] == [
        {"reward": 1.0},
        {"reward": 0.0},
        {"reward": 1.0},
    ]
