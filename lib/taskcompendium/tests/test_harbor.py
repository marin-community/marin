# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise real Harbor trials without requiring target-model inference."""

import json
import shlex
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import msgspec
import pytest
from tasktrove_verify.spec import Mode

from taskcompendium.execution import (
    Chat,
    ChatWithTools,
    DockerEnvironment,
    HarborExecutionConfig,
    HarborLaunchConfig,
    HarborTaskBinding,
    HarnessToolBinding,
    NoEnvironment,
    ShellSimEnvironment,
    ShellToolBinding,
    environment_for_requirements,
)
from taskcompendium.harbor.runner import run_trial
from taskcompendium.importers.sequential import greeting_task, sentence_revision_task
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import (
    AssistantFinal,
    Capability,
    Embedded,
    FileSubmission,
    FinalState,
    JsonPath,
    JudgeConfig,
    JudgeModelPolicy,
    JudgeView,
    PlainText,
    Rendering,
    Resource,
    ResourceRole,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpecification,
    TaskSuccessPolicy,
    TaskTroveVerifier,
    WorkspaceState,
)

pytestmark = pytest.mark.harbor_conformance


def _reference_execution(binding: HarborTaskBinding, agent: str) -> HarborExecutionConfig:
    return HarborExecutionConfig(binding, HarborLaunchConfig(agent))


def _exported_execution(task: Path) -> dict:
    return json.loads((task / "reference-execution.json").read_text())


def _spec() -> TaskSpecification:
    return TaskSpecification(
        id="test/math",
        requirements=TaskRequirements(),
        resources=(),
        metadata=TaskMetadata(Source("test", "1", "0", "1")),
        steps=(
            StepSpecification(
                instructions="Compute three quarters as a fraction.",
                verifier=TaskTroveVerifier(Mode.MATH, {"expected": "3/4"}),
            ),
        ),
    )


def _task(root: Path, protocol: Rendering | None = None, environment=None) -> Path:
    selected_environment = environment or NoEnvironment()
    binding = HarborTaskBinding(
        selected_environment,
        (
            Chat()
            if isinstance(selected_environment, NoEnvironment)
            else ChatWithTools(
                (
                    (
                        HarnessToolBinding("terminal", "shellsim")
                        if isinstance(selected_environment, ShellSimEnvironment)
                        else HarnessToolBinding("terminal", "docker")
                    ),
                )
            )
        ),
    )
    return lower_to_harbor(
        _spec(),
        (protocol or Rendering("plain", AssistantFinal()),),
        binding,
        root / "math",
        reference_execution=HarborExecutionConfig(binding, HarborLaunchConfig("replay")),
    )


def _execution(task: Path, agent: dict) -> dict:
    config = _exported_execution(task)
    config["agent"] = {**config["agent"], **agent, "kwargs": {**config["agent"]["kwargs"], **agent.get("kwargs", {})}}
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
        ("specification.json", "id", "changed-task", "manifest hash"),
        ("renderings.json", "id", "changed", "rendering does not match"),
    ],
)
async def test_harbor_rejects_export_drift_before_starting_trial(tmp_path, filename, field, value, error):
    task = _task(tmp_path)
    document = json.loads((task / filename).read_text())
    target = document[0] if filename == "renderings.json" else document
    target[field] = value
    (task / filename).write_text(json.dumps(document))
    execution = _exported_execution(task)
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
                "kwargs": {"api_base": endpoint, "chat_template_kwargs": {"enable_thinking": False}},
            },
        )
        result = await run_trial(task, execution, tmp_path / "trials", "chat")
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert requests[0]["messages"] == [{"role": "user", "content": (task / "instruction.md").read_text()}]
    assert "tools" not in requests[0]
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("answer,reward", [("3/4", 1.0), ("4/3", 0.0)])
async def test_harbor_shellsim_file_submission_grades_actual_file(tmp_path, bridge, answer, reward):
    protocol = Rendering("file", FileSubmission("/app/answer.txt"))
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
    protocol = Rendering("file", FileSubmission("/app/answer.txt"))
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({"command": "printf '3/4' > answer.txt; cat answer.txt"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "Done."},
    ]
    with _chat_endpoint(messages) as (endpoint, requests):
        binding = HarborTaskBinding(ShellSimEnvironment(), ChatWithTools((ShellToolBinding("run_command", "shellsim"),)))
        task = lower_to_harbor(
            _spec(),
            (protocol,),
            binding,
            tmp_path / "tool",
            reference_execution=_reference_execution(binding, "tool_chat"),
            agent_kwargs={"api_base": endpoint},
            model_name="fixture",
            environment_kwargs={"bridge_path": str(Path(bridge).resolve())},
        )
        execution = _exported_execution(task)
        result = await run_trial(task, execution, tmp_path / "trials", "tool")
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": 1.0}
    observation = json.loads(requests[1]["messages"][-1]["content"])
    assert observation["stdout"] == "3/4"
    assert observation["return_code"] == 0

    definition = requests[0]["tools"][0]["function"]
    assert definition["name"] == "run_command"
    assert definition["parameters"]["required"] == ["command"]
    assert definition["parameters"]["properties"] == {"command": {"type": "string"}}


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
            steps=(
                msgspec.structs.replace(
                    spec.steps[0],
                    verifier=TaskTroveVerifier(
                        Mode.JUDGE,
                        {"references": ["three quarters"], "exact_gate": False},
                        judge=judge,
                    ),
                ),
            ),
        )
        binding = HarborTaskBinding(NoEnvironment(), Chat())
        task = lower_to_harbor(
            spec,
            (Rendering("plain", AssistantFinal()),),
            binding,
            tmp_path / "judge",
            reference_execution=_reference_execution(binding, "replay"),
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


@pytest.mark.parametrize("responses,expected", [(["3/4", "2"], 1.0), (["3/4", "7"], 0.5)])
async def test_multistep_chat_uses_each_semantic_verifier(tmp_path, responses, expected):
    base = _spec()
    second = msgspec.structs.replace(
        base.steps[0], instructions="Compute one plus one.", verifier=TaskTroveVerifier(Mode.MATH, {"expected": "2"})
    )
    spec = msgspec.structs.replace(base, steps=(*base.steps, second), success_policy=TaskSuccessPolicy.MEAN)
    binding = HarborTaskBinding(NoEnvironment(), Chat())
    task = lower_to_harbor(
        spec,
        (Rendering("plain", AssistantFinal()),) * 2,
        binding,
        tmp_path / "multi",
        reference_execution=_reference_execution(binding, "replay"),
        agent_kwargs={"steps": [{"response": r, "commands": []} for r in responses]},
    )
    result = await run_trial(task, _exported_execution(task), tmp_path / "trials", "multi")
    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": expected}
    assert [s.verifier_result.rewards["reward"] for s in result.step_results] == [1.0, 1.0 if expected == 1.0 else 0.0]


async def test_multistep_extraction_failure_has_no_aggregate_reward(tmp_path):
    base = _spec()
    spec = msgspec.structs.replace(base, steps=base.steps * 2, success_policy=TaskSuccessPolicy.MEAN)
    binding = HarborTaskBinding(NoEnvironment(), Chat())
    task = lower_to_harbor(
        spec,
        (Rendering("json", AssistantFinal(JsonPath())),) * 2,
        binding,
        tmp_path / "multi",
        reference_execution=_reference_execution(binding, "replay"),
        agent_kwargs={
            "steps": [{"response": "broken", "commands": []}, {"response": '{"answer":"3/4"}', "commands": []}]
        },
    )
    result = await run_trial(task, _exported_execution(task), tmp_path / "trials", "multi")
    assert result.verifier_result is None
    assert result.step_results[0].exception_info is not None
    diagnostic = json.loads((tmp_path / "trials/multi/steps/step-1/verifier/taskcompendium-result.json").read_text())
    assert diagnostic["status"] == "extraction_error"
    assert diagnostic["reward"] is None


@pytest.mark.parametrize("extractor", [PlainText(), JsonPath()])
@pytest.mark.parametrize(
    "revised,reward", [("Mira will meet Leo on Thursday.", 1.0), ("Mira will meet Leo on Friday.", 0.0)]
)
async def test_multistep_chat_preserves_required_visible_context(tmp_path, extractor, revised, reward):
    spec = sentence_revision_task()
    renderings = (Rendering("answer", AssistantFinal(extractor)),) * 2
    with pytest.raises(ValueError, match="prior conversation"):
        lower_to_harbor(spec, renderings, HarborTaskBinding(NoEnvironment(), Chat()), tmp_path / "invalid")
    answers = ["Mira will meet Leo on Tuesday.", revised]
    responses = [json.dumps({"answer": answer}) if isinstance(extractor, JsonPath) else answer for answer in answers]
    with _chat_endpoint([{"role": "assistant", "content": response} for response in responses]) as (endpoint, requests):
        binding = HarborTaskBinding(NoEnvironment(), Chat(), context="conversation")
        task = lower_to_harbor(
            spec,
            renderings,
            binding,
            tmp_path / "multi",
            reference_execution=_reference_execution(binding, "chat"),
            agent_kwargs={"api_base": endpoint},
            model_name="fixture",
        )
        result = await run_trial(task, _exported_execution(task), tmp_path / "trials", "multi")
    assert result.verifier_result.rewards == {"reward": reward}
    assert [step.verifier_result.rewards["reward"] for step in result.step_results] == [1.0, reward]
    assert [m["role"] for m in requests[0]["messages"]] == ["user"]
    assert "Thursday" not in requests[0]["messages"][0]["content"]
    assert [m["role"] for m in requests[1]["messages"]] == ["user", "assistant", "user"]
    assert requests[1]["messages"][1]["content"] == responses[0]
    assert "Thursday" in requests[1]["messages"][2]["content"]


async def test_multistep_shellsim_releases_inputs_at_their_step(tmp_path, bridge):
    base = _spec()
    first = msgspec.structs.replace(
        base.steps[0],
        instructions="Write ready to answer.txt.",
        verifier=TaskTroveVerifier(Mode.EXACT, {"expected": ["ready"]}),
    )
    second = msgspec.structs.replace(
        first,
        instructions="Copy input.txt to answer.txt.",
        verifier=TaskTroveVerifier(Mode.EXACT, {"expected": ["later"]}),
        resources=(Resource("input.txt", (ResourceRole.AGENT,), Embedded(b"later")),),
    )
    spec = msgspec.structs.replace(
        base,
        requirements=TaskRequirements((Capability.FILESYSTEM, Capability.SHELL), WorkspaceState()),
        steps=(first, second),
        success_policy=TaskSuccessPolicy.MEAN,
    )
    binding = HarborTaskBinding(ShellSimEnvironment(), ChatWithTools((HarnessToolBinding("terminal", "shellsim"),)))
    task = lower_to_harbor(
        spec,
        (Rendering("file", FileSubmission("/app/answer.txt")),) * 2,
        binding,
        tmp_path / "multi",
        reference_execution=_reference_execution(binding, "replay"),
        environment_kwargs={"bridge_path": bridge},
        agent_kwargs={
            "steps": [
                {
                    "commands": [
                        "if test -e input.txt; then echo leaked > answer.txt; else echo ready > answer.txt; fi"
                    ],
                    "response": "",
                },
                {"commands": ["cp input.txt answer.txt"], "response": ""},
            ]
        },
    )
    result = await run_trial(task, _exported_execution(task), tmp_path / "trials", "multi")
    assert result.exception_info is None
    assert [s.verifier_result.rewards["reward"] for s in result.step_results] == [1.0, 1.0]


@pytest.mark.docker
async def test_repository_followup_uses_workspace_with_fresh_conversation(tmp_path, runtime_image):
    spec = greeting_task(runtime_image)
    first = "def greet(name):\n    return f'Hello, {name}!'\n"
    extension = (
        "\noriginal = greet\ndef greet(name, uppercase=False):\n"
        "    value = original(name)\n    return value.upper() if uppercase else value\n"
    )
    commands = [
        f"printf '%s' {shlex.quote(first)} > greeting.py",
        f"cat greeting.py && printf '%s' {shlex.quote(extension)} >> greeting.py",
    ]
    messages = []
    for index, command in enumerate(commands):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {"name": "shell", "arguments": json.dumps({"command": command})},
                        }
                    ],
                },
                {"role": "assistant", "content": "Done."},
            ]
        )
    with _chat_endpoint(messages) as (endpoint, requests):
        environment = environment_for_requirements(spec.requirements)
        binding = HarborTaskBinding(
            environment,
            ChatWithTools(
                (ShellToolBinding("shell", "shellsim" if isinstance(environment, ShellSimEnvironment) else "docker"),)
            ),
            context="fresh",
        )
        task = lower_to_harbor(
            spec,
            (Rendering("workspace", FinalState(("greeting.py",))),) * 2,
            binding,
            tmp_path / "fresh-workspace",
            reference_execution=_reference_execution(binding, "tool_chat"),
            agent_kwargs={"api_base": endpoint},
            model_name="fixture",
        )
        result = await run_trial(task, _exported_execution(task), tmp_path / "trials", "fresh")
    assert result.exception_info is None
    assert [step.verifier_result.rewards["reward"] for step in result.step_results] == [1.0, 1.0]
    assert [m["role"] for m in requests[2]["messages"]] == ["user"]
    assert "uppercase" not in requests[0]["messages"][0]["content"]
    assert first in json.loads(requests[3]["messages"][2]["content"])["stdout"]


@pytest.mark.docker
async def test_chat_docker_environment_does_not_grant_model_tools(tmp_path, runtime_image):
    with _chat_endpoint() as (endpoint, requests):
        binding = HarborTaskBinding(DockerEnvironment(runtime_image), Chat())
        task = lower_to_harbor(
            _spec(),
            (Rendering("plain", AssistantFinal()),),
            binding,
            tmp_path / "chat-docker",
            reference_execution=_reference_execution(binding, "chat"),
            agent_kwargs={"api_base": endpoint},
            model_name="fixture",
        )
        result = await run_trial(task, _exported_execution(task), tmp_path / "trials", "chat")
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert "tools" not in requests[0]


@pytest.mark.docker
async def test_chat_replay_cannot_execute_commands_in_docker(tmp_path, runtime_image):
    binding = HarborTaskBinding(DockerEnvironment(runtime_image), Chat())
    task = lower_to_harbor(
        _spec(),
        (Rendering("plain", AssistantFinal()),),
        binding,
        tmp_path / "replay-docker",
        reference_execution=_reference_execution(binding, "replay"),
        agent_kwargs={"commands": ["echo unauthorized"], "response": "3/4"},
    )
    result = await run_trial(task, _exported_execution(task), tmp_path / "trials", "replay")
    assert result.exception_info is not None
    assert result.verifier_result is None
