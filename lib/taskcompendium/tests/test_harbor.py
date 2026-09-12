# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Harbor trials without requiring target-model inference."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import msgspec
import pytest
from tasktrove_verify.spec import Mode

from taskcompendium.harbor.runner import run_trial
from taskcompendium.lowering import export_task
from taskcompendium.models import (
    AssistantFinal,
    Chat,
    ChatWithTools,
    ExecutionConfig,
    FileSubmission,
    JudgeConfig,
    JudgeModelPolicy,
    JudgeView,
    NoEnvironment,
    Protocol,
    PythonRuntime,
    ShellSimEnvironment,
    Source,
    TaskMetadata,
    TaskSpecification,
    VerifierSpec,
)


def _spec() -> TaskSpecification:
    return TaskSpecification(
        id="test/math",
        instructions="Compute three quarters as a fraction.",
        environment=NoEnvironment(),
        resources=(),
        verifier=VerifierSpec(Mode.MATH, {"expected": "3/4"}),
        verifier_runtime=PythonRuntime(),
        metadata=TaskMetadata(Source("test", "1", "0", "1")),
    )


def _task(root: Path, protocol: Protocol | None = None, environment=None) -> Path:
    return export_task(
        _spec(),
        protocol or Protocol("plain", Chat(), AssistantFinal()),
        ExecutionConfig("replay", environment or NoEnvironment()),
        root / "math",
    )


def _execution(task: Path, agent: dict) -> dict:
    config = json.loads((task / "execution.json").read_text())
    config["agent"] = {**config["agent"], **agent}
    return config


@pytest.mark.parametrize("response,reward", [("3/4", 1.0), ("4/3", 0.0)])
async def test_harbor_replay_semantic_grading(tmp_path, response, reward):
    task = _task(tmp_path)
    execution = _execution(
        task, {"import_path": "taskcompendium.harbor.agents:ReplayAgent", "kwargs": {"response": response}}
    )
    result = await run_trial(task, execution, tmp_path / "trials", "replay")
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": reward}
    outcome = json.loads((tmp_path / "trials/replay/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == "graded"


async def test_harbor_no_tool_rejects_execution_without_reward(tmp_path):
    task = _task(tmp_path)
    execution = _execution(
        task, {"import_path": "taskcompendium.harbor.agents:ReplayAgent", "kwargs": {"commands": ["echo 3/4"]}}
    )
    result = await run_trial(task, execution, tmp_path / "trials", "forbidden")
    assert result.verifier_result is None
    assert result.exception_info is not None


@pytest.mark.parametrize(
    "filename,field,value,error",
    [
        ("manifest.json", "harbor_revision", "wrong", "different Harbor revision"),
        ("specification.json", "instructions", "Changed instruction", "manifest hash"),
        ("protocol.json", "id", "changed", "protocol does not match"),
    ],
)
async def test_harbor_rejects_export_drift_before_starting_trial(tmp_path, filename, field, value, error):
    task = _task(tmp_path)
    document = json.loads((task / filename).read_text())
    document[field] = value
    (task / filename).write_text(json.dumps(document))
    execution = json.loads((task / "execution.json").read_text())
    with pytest.raises(ValueError, match=error):
        await run_trial(task, execution, tmp_path / "trials", "drift")
    assert not (tmp_path / "trials").exists()


@contextmanager
def _chat_endpoint(messages=None):
    requests = []
    messages = messages or [{"role": "assistant", "content": "3/4"}]

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            data = json.dumps(
                {
                    "model": "fixture-judge@1",
                    "system_fingerprint": "fixture-build",
                    "choices": [{"message": messages[len(requests) - 1]}],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, message, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_harbor_direct_chat_sends_only_agent_projection(tmp_path):
    task = _task(tmp_path)
    with _chat_endpoint() as (endpoint, requests):
        execution = _execution(
            task,
            {
                "import_path": "taskcompendium.harbor.agents:DirectChatAgent",
                "model_name": "fixture",
                "kwargs": {"api_base": endpoint},
            },
        )
        result = await run_trial(task, execution, tmp_path / "trials", "chat")
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert requests[0]["messages"] == [{"role": "user", "content": (task / "instruction.md").read_text()}]
    assert "tools" not in requests[0]


@pytest.mark.parametrize("answer,reward", [("3/4", 1.0), ("4/3", 0.0)])
async def test_harbor_shellsim_file_submission_grades_actual_file(tmp_path, bridge, answer, reward):
    protocol = Protocol("file", ChatWithTools(), FileSubmission("/app/answer.txt"))
    task = _task(tmp_path, protocol, ShellSimEnvironment())
    execution = _execution(
        task,
        {
            "import_path": "taskcompendium.harbor.agents:ReplayAgent",
            "kwargs": {"commands": [f"printf '%s' '{answer}' > answer.txt"]},
        },
    )
    execution["environment"] = {
        "import_path": "taskcompendium.harbor.environments:ShellSimEnvironment",
        "kwargs": {"bridge_path": str(Path(bridge).resolve())},
    }
    result = await run_trial(task, execution, tmp_path / "trials", "shellsim")
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": reward}


async def test_harbor_empty_submission_has_extraction_error_and_no_reward(tmp_path):
    task = _task(tmp_path)
    execution = _execution(task, {"import_path": "taskcompendium.harbor.agents:ReplayAgent"})
    result = await run_trial(task, execution, tmp_path / "trials", "empty")
    assert result.verifier_result is None
    assert result.exception_info is not None
    outcome = json.loads((tmp_path / "trials/empty/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == "extraction_error"
    assert outcome["reward"] is None


async def test_harbor_tool_chat_executes_shell_before_grading(tmp_path, bridge):
    protocol = Protocol("file", ChatWithTools(), FileSubmission("/app/answer.txt"))
    task = _task(tmp_path, protocol, ShellSimEnvironment())
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": json.dumps({"command": "printf '3/4' > answer.txt; cat answer.txt"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Done."},
    ]
    with _chat_endpoint(messages) as (endpoint, requests):
        execution = _execution(
            task,
            {
                "import_path": "taskcompendium.harbor.agents:ShellToolAgent",
                "model_name": "fixture",
                "kwargs": {"api_base": endpoint},
            },
        )
        execution["environment"] = {
            "import_path": "taskcompendium.harbor.environments:ShellSimEnvironment",
            "kwargs": {"bridge_path": str(Path(bridge).resolve())},
        }
        result = await run_trial(task, execution, tmp_path / "trials", "tool")
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": 1.0}
    observation = json.loads(requests[1]["messages"][-1]["content"])
    assert observation["stdout"] == "3/4"
    assert observation["return_code"] == 0


@pytest.mark.parametrize(
    "verdict,status,reward", [("Matches the rubric.\nSCORE: 1", "graded", 1.0), ("unknown", "infra_error", None)]
)
async def test_harbor_judge_transport_preserves_outcome_and_provenance(tmp_path, monkeypatch, verdict, status, reward):
    spec = _spec()
    monkeypatch.setenv("TASKCOMPENDIUM_TEST_JUDGE_KEY", "fixture")
    with _chat_endpoint([{"role": "assistant", "content": verdict}]) as (endpoint, requests):
        judge = JudgeConfig(JudgeModelPolicy("fixture", "small", "fixture", endpoint), JudgeView())
        spec = msgspec.structs.replace(
            spec,
            verifier=VerifierSpec(
                Mode.JUDGE,
                {"references": ["three quarters"], "exact_gate": False},
                judge=judge,
            ),
        )
        task = export_task(
            spec,
            Protocol("plain", Chat(), AssistantFinal()),
            ExecutionConfig("replay", NoEnvironment()),
            tmp_path / "judge",
        )
        execution = _execution(
            task,
            {
                "import_path": "taskcompendium.harbor.agents:ReplayAgent",
                "kwargs": {"response": "3/4"},
            },
        )
        execution["verifier"]["kwargs"] = {"judge_api_key_env": "TASKCOMPENDIUM_TEST_JUDGE_KEY"}
        result = await run_trial(task, execution, tmp_path / "trials", "judge")
    outcome = json.loads((tmp_path / "trials/judge/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == status
    assert outcome["reward"] == reward
    if reward is None:
        assert result.verifier_result is None
    else:
        assert result.verifier_result.rewards == {"reward": reward}
        assert outcome["detail"]["judgments"][0]["model"] == "fixture-judge@1"
        assert outcome["detail"]["judgments"][0]["revision"] == "fixture-build"
    assert len(requests) == 1
