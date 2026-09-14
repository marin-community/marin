# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native terminal agents against scripted HTTP responses, with real source grading."""

import json
import shlex
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from tasktrove_verify.spec import Mode

from taskcompendium.execution import (
    ChatWithTools,
    DockerEnvironment,
    HarborExecutionConfig,
    HarnessToolBinding,
    environment_for_requirements,
)
from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_answers import import_task
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import (
    AnswerRequirements,
    Capability,
    ContainerRuntime,
    Embedded,
    FileSubmission,
    FinalState,
    JsonPath,
    PlainText,
    Rejected,
    Rendering,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpecification,
    TaskTroveVerifier,
    WorkspaceState,
)

pytestmark = pytest.mark.docker

# Runs inside the disconnected agent container. Only the transport is scripted:
# the installed upstream CLI parses calls, executes bash, and decides termination.
MINI_ENDPOINT = r"""
import json, shlex, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
source = sys.argv[1]
requests = []
class Endpoint(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append(payload)
        Path('/logs/agent/fixture-requests.json').write_text(json.dumps(requests))
        command = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
        if len(requests) == 1:
            command = "printf '%s' " + shlex.quote(source) + " > /app/main.py"
        message = {"role": "assistant", "content": "Scripted action.", "tool_calls": [{
            "id": "call_" + str(len(requests)), "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }]}
        body = json.dumps({
            "id": "chatcmpl-" + str(len(requests)), "object": "chat.completion", "created": 1,
            "model": "gpt-4o", "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
HTTPServer(('127.0.0.1', 8765), Endpoint).serve_forever()
"""


def _spec(image, setup_commands=()):
    return TaskSpecification(
        id="native-agent/sum",
        requirements=TaskRequirements(
            (Capability.FILESYSTEM, Capability.SHELL, Capability.PROCESS),
            WorkspaceState(image, setup_commands=setup_commands),
        ),
        resources=(
            Resource("cases/input_1.txt", (ResourceRole.VERIFIER,), Embedded(b"2 3\n")),
            Resource("cases/output_1.txt", (ResourceRole.VERIFIER,), Embedded(b"5\n")),
            Resource("cases/input_2.txt", (ResourceRole.VERIFIER,), Embedded(b"-1 8\n")),
            Resource("cases/output_2.txt", (ResourceRole.VERIFIER,), Embedded(b"7\n")),
        ),
        metadata=TaskMetadata(Source("native-agent-fixture", "1", "0", "1")),
        steps=(
            StepSpecification(
                instructions="Write /app/main.py to read two integers from stdin and print their sum.",
                verifier=TaskTroveVerifier(Mode.STDIO, {"command": "python3 main.py"}, runtime=ContainerRuntime(image)),
                answer_requirements=AnswerRequirements("final_state"),
            ),
        ),
    )


@pytest.mark.parametrize(
    "source,reward,extractor",
    [
        ("print(sum(map(int,input().split())))", 1.0, None),
        ("print(99)", 0.0, None),
        ("C", 1.0, PlainText()),
        ("D", 0.0, PlainText()),
        ('{"answer":"C"}', 1.0, JsonPath()),
        ('{"answer":"D"}', 0.0, JsonPath()),
    ],
)
async def test_native_terminus_executes_terminal_commands(
    tmp_path, runtime_image, monkeypatch, source, reward, extractor
):
    requests = []
    output_path = "/app/main.py" if extractor is None else "/app/answer.txt"

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            commands = []
            if len(requests) == 1:
                commands = [{"keystrokes": f"printf '%s' {shlex.quote(source)} > {output_path}\n", "duration": 0.1}]
            message = {
                "role": "assistant",
                "content": json.dumps(
                    {"analysis": "Scripted action.", "plan": "Finish.", "commands": commands, "task_complete": True}
                ),
            }
            body = json.dumps(
                {
                    "id": f"chatcmpl-{len(requests)}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        spec = _spec(runtime_image)
        protocol = Rendering("code", FinalState(("main.py",)))
        if extractor is not None:
            archive = Path(__file__).parent / "fixtures/tasktrove/answers/mcq-row-1972.tar.gz"
            spec = import_task(read_archive(archive.read_bytes(), "1972", "qa-short-answer"))
            assert not isinstance(spec, Rejected)
            protocol = Rendering("mcq", FileSubmission(output_path, extractor))
        task = lower_to_harbor(
            spec,
            (protocol,),
            HarborExecutionConfig(
                "terminus-2",
                DockerEnvironment(runtime_image),
                timeout=45,
                interaction=ChatWithTools((HarnessToolBinding("terminus-2", "docker"),)),
            ),
            tmp_path / "task",
            model_name="openai/gpt-4o",
            agent_kwargs={
                "api_base": f"http://127.0.0.1:{server.server_port}/v1",
                "max_turns": 3,
                "record_terminal_session": False,
                "enable_summarize": False,
            },
        )
        execution = json.loads((task / "execution.json").read_text())
        result = await run_trial(task, execution, tmp_path / "trials", "terminus")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
    assert len(requests) == 2
    trajectory = json.loads((tmp_path / "trials/terminus/agent/trajectory.json").read_text())
    assert trajectory["agent"]["name"] == "terminus-2"
    assert any(step.get("tool_calls") for step in trajectory["steps"])


@pytest.mark.parametrize("source,reward", [("print(sum(map(int,input().split())))", 1.0), ("print(99)", 0.0)])
async def test_preinstalled_native_mini_swe_agent_executes_bash(tmp_path, runtime_image, monkeypatch, source, reward):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    start = f"python3 -c {shlex.quote(MINI_ENDPOINT)} {shlex.quote(source)} >/tmp/fixture.log 2>&1 &"
    ready = (
        "import time, urllib.request\nfor _ in range(100):\n try:\n"
        "  urllib.request.urlopen('http://127.0.0.1:8765'); break\n"
        " except OSError: time.sleep(.05)\nelse: raise RuntimeError('fixture endpoint failed to start')"
    )
    spec = _spec(runtime_image, (start + "\npython3 -c " + shlex.quote(ready),))
    task = lower_to_harbor(
        spec,
        (Rendering("code", FinalState(("main.py",))),),
        HarborExecutionConfig(
            "mini-swe-agent",
            environment_for_requirements(spec.requirements),
            timeout=45,
            interaction=ChatWithTools((HarnessToolBinding("mini-swe-agent", "docker"),)),
        ),
        tmp_path / "task",
        model_name="openai/gpt-4o",
        agent_env={"OPENAI_API_BASE": "http://127.0.0.1:8765/v1", "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    execution = json.loads((task / "execution.json").read_text())
    result = await run_trial(task, execution, tmp_path / "trials", "mini")
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": reward}
    logs = tmp_path / "trials/mini/agent"
    requests = json.loads((logs / "fixture-requests.json").read_text())
    assert len(requests) == 2
    assert requests[0]["tools"][0]["function"]["name"] == "bash"
    native = json.loads((logs / "mini-swe-agent.trajectory.json").read_text())
    assert native["info"]["mini_version"] == "2.4.6"
    assert native["info"]["exit_status"] == "Submitted"
    trajectory = json.loads((logs / "trajectory.json").read_text())
    assert any(step.get("tool_calls") for step in trajectory["steps"])
