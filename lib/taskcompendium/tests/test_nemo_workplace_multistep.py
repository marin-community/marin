# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native Harbor coverage for the explicit derived Workplace workflow."""

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import msgspec
import pytest

from taskcompendium.execution import HarborExecutionConfig, HarborLaunchConfig
from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.nemo_workplace_multistep import DERIVED_ROW, build_multistep_sample
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import ContextRequirement, Outcome, ProviderStateVerifier
from taskcompendium.providers.nemo_workplace.provider import NemoWorkplaceEnvironment
from taskcompendium.providers.nemo_workplace.tools import get_tools
from taskcompendium.rendering import render_task

FIXTURES = Path(__file__).parent / "fixtures/nemo"


def _environment() -> NemoWorkplaceEnvironment:
    environment = object.__new__(NemoWorkplaceEnvironment)
    environment.tool_env = get_tools()
    environment.trace = []
    return environment


async def _dispatch_step(environment: NemoWorkplaceEnvironment, calls, step_index: int) -> None:
    for call_index, call in enumerate(calls, start=1):
        await environment.dispatch_action(call.name, call.arguments, f"step-{step_index}-call-{call_index}")


def _result(root: Path, trial_name: str, step_index: int) -> dict:
    return json.loads(
        (
            root / "trials" / trial_name / "steps" / f"step-{step_index}" / "verifier" / "taskcompendium-result.json"
        ).read_text()
    )


def _assistant_call(call, call_id: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
        ],
    }


@contextmanager
def _scripted_endpoint(steps):
    requests: list[dict] = []
    responses: list[dict] = []
    for step_index, calls in enumerate(steps, start=1):
        responses.extend(
            _assistant_call(call, f"step-{step_index}-call-{call_index}") for call_index, call in enumerate(calls, 1)
        )
        responses.append({"role": "assistant", "content": "Completed."})

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            message = responses[len(requests) - 1]
            body = json.dumps(
                {
                    "id": f"chatcmpl-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fixture",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_derived_workplace_uses_cumulative_provider_state_and_hides_private_data():
    sample = build_multistep_sample(FIXTURES)

    assert sample.specification.metadata.source.row == DERIVED_ROW
    assert sample.specification.success_policy.value == "mean"
    assert [step.context_requirement for step in sample.specification.steps] == [
        ContextRequirement.INSTRUCTION_AND_WORKSPACE,
        ContextRequirement.PRIOR_CONVERSATION,
        ContextRequirement.PRIOR_CONVERSATION,
    ]
    assert [len(step.verifier.parameters["ground_truth"]) for step in sample.specification.steps] == [2, 3, 5]
    derivation = next(resource for resource in sample.specification.resources if resource.path == "derivation.json")
    assert json.loads(derivation.content.data)["kind"] == "explicit_derived_workflow"

    public = msgspec.json.encode(render_task(sample.specification, sample.renderings)).decode()
    assert "ground_truth" not in public
    assert "source-row.json" not in public
    assert "derivation.json" not in public
    assert not any(term in public.lower() for term in ("verifier", "judge", "reward", "gold", "hidden test"))

    environment = _environment()
    for step_index, calls in enumerate(sample.all_good, start=1):
        await _dispatch_step(environment, calls, step_index)
        verifier = sample.specification.steps[step_index - 1].verifier
        assert isinstance(verifier, ProviderStateVerifier)
        result = await environment.grade_provider_state(verifier.adapter, verifier.parameters)
        assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
    assert [entry["call_id"] for entry in environment.trace] == [
        "step-1-call-1",
        "step-1-call-2",
        "step-2-call-1",
        "step-3-call-1",
        "step-3-call-2",
    ]


@pytest.mark.parametrize(
    ("attempt_name", "expected_rewards"),
    [
        pytest.param("all_good", [1.0, 1.0, 1.0], id="all-good"),
        pytest.param("later_wrong", [1.0, 0.0, 1.0], id="later-wrong"),
        pytest.param("missing_step", [1.0, 0.0, 1.0], id="missing-step"),
    ],
)
async def test_native_harbor_multistep_provider_trial_retains_state_and_conversation(
    tmp_path, monkeypatch, attempt_name, expected_rewards
):
    sample = build_multistep_sample(FIXTURES)
    attempt = getattr(sample, attempt_name)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    with _scripted_endpoint(attempt) as (endpoint, requests):
        task = lower_to_harbor(
            sample.specification,
            sample.renderings,
            sample.binding,
            tmp_path / "task",
            reference_execution=HarborExecutionConfig(sample.binding, HarborLaunchConfig("provider_chat")),
            model_name="fixture",
            agent_kwargs={"api_base": endpoint, "max_turns": 4},
        )
        execution = json.loads((task / "reference-execution.json").read_text())
        result = await run_trial(task, execution, tmp_path / "trials", attempt_name)

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": sum(expected_rewards) / len(expected_rewards)}
    assert [step.verifier_result.rewards["reward"] for step in result.step_results] == expected_rewards
    assert [_result(tmp_path, attempt_name, index)["reward"] for index in range(1, 4)] == expected_rewards
    assert execution["agent"]["kwargs"]["retain_conversation"] is True
    assert len(requests) == sum(len(calls) + 1 for calls in attempt)
    second_step_request = requests[len(attempt[0]) + 1]
    assert [message["role"] for message in second_step_request["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert second_step_request["messages"][0]["content"].startswith(sample.specification.steps[0].instructions)
    assert second_step_request["messages"][-1]["content"].startswith(sample.specification.steps[1].instructions)
    transcripts = [
        json.loads(
            (tmp_path / "trials" / attempt_name / "steps" / f"step-{index}" / "agent" / "transcript.json").read_text()
        )
        for index in range(1, 4)
    ]
    assert len(transcripts[2]) > len(transcripts[0])
    assert transcripts[2][-1]["content"] == "Completed."
