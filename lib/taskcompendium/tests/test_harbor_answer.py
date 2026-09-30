# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A pinned answer task through Harbor's custom-verifier trial lifecycle."""

import json
import subprocess
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from threading import Thread

import pytest
from harbor.models.task.task import Task
from tasktrove_verify.spec import MathType

from taskcompendium.grading import exact_answer, numeric_answer
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.lowering import (
    HarborEnvironmentConfig,
    SelectionPolicy,
    compatible_lowerings,
    lower_to_harbor,
    read_specification,
    select_lowerings,
)
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    FinalTools,
    FunctionDefinition,
    Source,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.submission import (
    AnswerCall,
    AnswerFormat,
    GradingAttempt,
    JsonAnswer,
    PlainText,
    SubmissionConvention,
    submission_compatible,
)
from taskcompendium.verifier_registry import grade_answer
from taskcompendium.verifiers.mathematical import mathematical_answer

from .harbor_replay import run_replay_trial


def _answer_convention(answer_format: AnswerFormat) -> SubmissionConvention:
    if answer_format == AnswerFormat.PLAIN:
        return PlainText(id="plain")
    if answer_format == AnswerFormat.JSON:
        return JsonAnswer(id="json")
    if answer_format == AnswerFormat.ANSWER_CALL:
        return AnswerCall(id="answer_call")
    raise ValueError(f"Unsupported test answer format: {answer_format}")


def _answer_action(answer: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-answer",
                "type": "function",
                "function": {"name": "submit_answer", "arguments": json.dumps({"answer": answer})},
            }
        ],
    }


@dataclass
class ChatEndpoint:
    url: str
    authorizations: list[str | None]
    requests: list[dict]
    status: int
    body: bytes


@pytest.fixture
def chat_endpoint():
    endpoint = ChatEndpoint("", [], [], 200, b'{"choices":[{"message":{"role":"assistant","content":"12"}}]}')

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            endpoint.authorizations.append(self.headers.get("Authorization"))
            endpoint.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(endpoint.status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(endpoint.body)

        def log_message(self, format, *args):  # noqa: A002 - match BaseHTTPRequestHandler
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    endpoint.url = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def specification() -> TaskSpec:
    return TaskSpec(
        id="arithmetic-7-plus-5",
        context=ConversationInput(events=(TextMessage(role="user", content="What is 7 + 5?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer(12.0, tolerance_abs=0.0, tolerance_rel=0.0),
        source=Source(dataset="hand-authored", revision="2026-09-16", row="arithmetic-7-plus-5", importer_revision="1"),
    )


@pytest.mark.parametrize(
    "answer_format,response,reward,status",
    [
        (AnswerFormat.PLAIN, "12", 1.0, "graded"),
        (AnswerFormat.PLAIN, "12.0", 1.0, "graded"),
        (AnswerFormat.PLAIN, "13", 0.0, "graded"),
        (AnswerFormat.PLAIN, "not a number", 0.0, "graded"),
        (AnswerFormat.PLAIN, r"\boxed{12}", 0.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"12"}', 1.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"13"}', 0.0, "graded"),
        (AnswerFormat.JSON, '{"answer":"12"', 0.0, "submission_failure"),
    ],
)
async def test_direct_chat_harbor_trial_distinguishes_answer_outcomes(
    tmp_path, specification, answer_format, response, reward, status
):
    environment_config = HarborEnvironmentConfig()
    convention = _answer_convention(answer_format)
    task = lower_to_harbor(specification, convention, environment_config, tmp_path / "task")
    assert Task.is_valid_dir(task, disable_verification=True)
    assert "12" not in (task / "instruction.md").read_text()

    result = await run_replay_trial(task, {"role": "assistant", "content": response}, tmp_path / "trials", "run")

    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert outcome["status"] == status
    assert outcome["reward"] == reward
    if reward is None:
        assert result.verifier_result is None
    else:
        assert result.exception_info is None, result.exception_info
        assert result.verifier_result.rewards == {"reward": reward}


async def test_direct_chat_exact_comparison_uses_pinned_normalization(tmp_path, specification):
    specification = specification.model_copy(
        update={"verifier": exact_answer("Straße Park"), "answer_type": AnswerType.TEXT}
    )
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        environment_config,
        tmp_path / "task",
    )

    result = await run_replay_trial(task, {"role": "assistant", "content": "STRASSE   PARK"}, tmp_path / "trials", "run")

    assert result.verifier_result.rewards == {"reward": 1.0}


@pytest.mark.parametrize(
    "response,reward",
    [("12.05", 1.0), ("12.2", 0.0)],
)
async def test_numeric_answer_uses_explicit_tolerance(specification, response, reward):
    specification = specification.model_copy(
        update={"verifier": numeric_answer(12.0, tolerance_abs=0.1, tolerance_rel=0.0)}
    )
    convention = PlainText(id="plain")

    result = await grade_answer(
        specification,
        convention,
        GradingAttempt(
            conversation=ConversationTrace(
                events=(*specification.context.events, TextMessage(role="assistant", content=response))
            ),
            tool_providers={},
            workspace=object(),
        ),
    )

    assert (result.status, result.reward) == ("graded", reward)


async def test_direct_chat_rejects_invalid_private_metadata_before_launch(tmp_path, specification):
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        environment_config,
        tmp_path / "task",
    )
    (task / "submission_convention.json").write_text("{invalid")

    with pytest.raises(ValueError):
        await run_replay_trial(task, {"role": "assistant", "content": "12"}, tmp_path / "trials", "run")

    assert not (tmp_path / "trials").exists()


async def test_text_convention_rejects_tool_call_submission(tmp_path, specification):
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        environment_config,
        tmp_path / "task",
    )

    message = _answer_action("12")
    result = await run_replay_trial(task, message, tmp_path / "trials", "run")
    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert result.verifier_result.rewards == {"reward": 0.0}
    assert outcome["status"] == "submission_failure"
    trace = ConversationTrace.model_validate_json((tmp_path / "trials/run/agent/submission.json").read_text())
    assert trace.events[-1].calls[0].arguments == {"answer": "12"}


async def test_chat_records_incompatible_tool_call_for_convention_extraction(tmp_path, specification, chat_endpoint):
    message = _answer_action("12")
    chat_endpoint.body = json.dumps({"choices": [{"message": message}]}).encode()
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        environment_config,
        tmp_path / "task",
    )
    result = await run_trial(
        task,
        environment_config,
        ChatLaunch(model="model", api_base=chat_endpoint.url),
        tmp_path / "trials",
        "run",
    )
    outcome = json.loads((tmp_path / "trials/run/verifier/taskcompendium-result.json").read_text())
    assert result.verifier_result.rewards == {"reward": 0.0}
    assert outcome["status"] == "submission_failure"
    trace = ConversationTrace.model_validate_json((tmp_path / "trials/run/agent/submission.json").read_text())
    assert trace.events[-1].calls[0].arguments == {"answer": "12"}


@pytest.mark.parametrize(
    "answer_type,verifier,response",
    [
        (AnswerType.TEXT, exact_answer("12"), "12"),
        (AnswerType.NUMBER, mathematical_answer("13/2", MathType.SCALAR), "6.5"),
        (AnswerType.NUMBER, numeric_answer(12.0, tolerance_abs=0.0, tolerance_rel=0.0), "12.0"),
    ],
)
async def test_answer_call_grades_semantic_answers_through_harbor(
    tmp_path, specification, answer_type, verifier, response
):
    specification = specification.model_copy(update={"answer_type": answer_type, "verifier": verifier})
    convention = AnswerCall(id="answer-call")
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(specification, convention, environment_config, tmp_path / "task")

    correct = await run_replay_trial(task, _answer_action(response), tmp_path / "trials", "correct")
    wrong = await run_replay_trial(task, _answer_action("13"), tmp_path / "trials", "wrong")

    assert correct.verifier_result.rewards == {"reward": 1.0}
    assert wrong.verifier_result.rewards == {"reward": 0.0}
    trace = ConversationTrace.model_validate_json((tmp_path / "trials/correct/agent/submission.json").read_text())
    assert trace.events[-1].calls[0].arguments == {"answer": response}


async def test_answer_call_does_not_dispatch_and_requires_its_submission_function(tmp_path, specification, monkeypatch):
    convention = AnswerCall(id="answer-call")
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(specification, convention, environment_config, tmp_path / "task")
    requests = []

    def respond(request, **_kwargs):
        requests.append(json.loads(request.data))
        return BytesIO(json.dumps({"choices": [{"message": _answer_action("12")}]}).encode())

    monkeypatch.setattr("taskcompendium.harbor.adapter.urllib.request.urlopen", respond)
    result = await run_trial(
        task,
        environment_config,
        ChatLaunch(model="model", api_base="https://example.invalid"),
        tmp_path / "trials",
        "run",
    )

    assert result.verifier_result.rewards == {"reward": 1.0}
    assert len(requests) == 1
    assert requests[0]["tools"][0]["function"]["name"] == "submit_answer"
    assert requests[0]["tools"][0]["function"]["parameters"]["properties"]["answer"]["type"] == "string"
    assert requests[0]["tool_choice"] == "required"
    assert requests[0]["parallel_tool_calls"] is False
    assert (tmp_path / "trials/run/agent/submission.json").exists()

    invalid = _answer_action("12")
    invalid["tool_calls"][0]["function"]["name"] = "lookup"
    invalid_result = await run_replay_trial(task, invalid, tmp_path / "trials", "invalid")
    outcome = json.loads((tmp_path / "trials/invalid/verifier/taskcompendium-result.json").read_text())
    assert invalid_result.verifier_result.rewards == {"reward": 0.0}
    assert outcome["status"] == "submission_failure"


@pytest.mark.parametrize(
    "answer_format,tool_choice",
    [
        (AnswerFormat.PLAIN, None),
        (AnswerFormat.PLAIN, "auto"),
        (AnswerFormat.PLAIN, "none"),
        (AnswerFormat.JSON, "auto"),
        (AnswerFormat.ANSWER_CALL, None),
        (AnswerFormat.ANSWER_CALL, "auto"),
        (AnswerFormat.ANSWER_CALL, "required"),
    ],
)
async def test_answer_submission_preserves_advertised_tools_and_policy(
    tmp_path, specification, chat_endpoint, answer_format, tool_choice
):
    specification = specification.model_copy(
        update={
            "final_tools": FinalTools(
                functions=(FunctionDefinition(name="lookup", parameters={"type": "object"}),),
                tool_choice=tool_choice,
                parallel_tool_calls=True,
            )
        }
    )
    convention = _answer_convention(answer_format)
    task = lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    response = (
        _answer_action("12")
        if answer_format == AnswerFormat.ANSWER_CALL
        else {"role": "assistant", "content": '{"answer":"12"}' if answer_format == AnswerFormat.JSON else "12"}
    )
    chat_endpoint.body = json.dumps({"choices": [{"message": response}]}).encode()

    result = await run_trial(
        task,
        HarborEnvironmentConfig(),
        ChatLaunch(model="model", api_base=chat_endpoint.url),
        tmp_path / "trials",
        "run",
    )

    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards == {"reward": 1.0}
    request = chat_endpoint.requests[0]
    assert request["tools"][0] == {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
    assert [tool["function"]["name"] for tool in request["tools"]] == (
        ["lookup", "submit_answer"] if answer_format == AnswerFormat.ANSWER_CALL else ["lookup"]
    )
    assert request.get("tool_choice") == tool_choice
    assert request["parallel_tool_calls"] is True


@pytest.mark.parametrize(
    "answer_format,tool_choice,function_name",
    [
        (AnswerFormat.PLAIN, "required", "lookup"),
        (AnswerFormat.JSON, "required", "lookup"),
        (AnswerFormat.ANSWER_CALL, "none", "lookup"),
        (AnswerFormat.ANSWER_CALL, "auto", "submit_answer"),
    ],
)
def test_lowering_rejects_submission_policy_conflicts(
    tmp_path, specification, answer_format, tool_choice, function_name
):
    specification = specification.model_copy(
        update={
            "final_tools": FinalTools(
                functions=(FunctionDefinition(name=function_name, parameters={"type": "object"}),),
                tool_choice=tool_choice,
            )
        }
    )
    convention = _answer_convention(answer_format)

    assert compatible_lowerings(specification, (convention,), (HarborEnvironmentConfig(),)) == ()
    with pytest.raises(ValueError, match="incompatible"):
        lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    assert not (tmp_path / "task").exists()


def test_direct_chat_rejects_unsatisfied_requirements(tmp_path, specification):
    specification = specification.model_copy(
        update={"environment_requirements": EnvironmentRequirements(capabilities=("filesystem",))}
    )

    with pytest.raises(ValueError, match="cannot satisfy"):
        lower_to_harbor(
            specification,
            PlainText(id="plain"),
            HarborEnvironmentConfig(),
            tmp_path / "task",
        )


@pytest.mark.parametrize(
    "verifier,message",
    [
        (
            VerifierSpec(kind=VerifierKind.EXACT_ANSWER, parameters_json='{"expected": 12}'),
            "Invalid 'exact' verifier parameters",
        ),
        (
            VerifierSpec(kind=VerifierKind.EXACT_ANSWER, parameters_json='{"expected": "12", "extra": true}'),
            "Invalid 'exact' verifier parameters",
        ),
    ],
)
def test_lowering_rejects_invalid_verifier_before_writing(tmp_path, specification, verifier, message):
    specification = specification.model_copy(update={"verifier": verifier})

    with pytest.raises(ValueError, match=message):
        lower_to_harbor(
            specification,
            PlainText(id="plain"),
            HarborEnvironmentConfig(),
            tmp_path / "task",
        )
    assert not (tmp_path / "task").exists()


def test_exported_specification_resolves_verifier_in_fresh_process(tmp_path, specification):
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )
    script = (
        "import asyncio, json, sys; from pathlib import Path; "
        "from taskcompendium.verifier_registry import grade_answer; "
        "from taskcompendium.models import ConversationTrace, TextMessage; "
        "from taskcompendium.submission import GradingAttempt; "
        "from taskcompendium.lowering import read_submission_convention, read_specification; "
        "root = Path(sys.argv[1]); "
        "specification = read_specification(root / 'specification.json'); "
        "result = asyncio.run(grade_answer(specification, "
        "read_submission_convention(root / 'submission_convention.json'), "
        "GradingAttempt(ConversationTrace(events=(*specification.context.events, "
        "TextMessage(role='assistant', content='12'))), {}, object()))); "
        "print(json.dumps({'status': result.status, 'reward': result.reward}))"
    )

    completed = subprocess.run([sys.executable, "-c", script, str(task)], capture_output=True, text=True, check=True)

    assert json.loads(completed.stdout) == {"status": "graded", "reward": 1.0}


@pytest.mark.parametrize("schema_version", ["0.1", "0.11"])
def test_old_verifier_schema_is_rejected_on_read(tmp_path, specification, schema_version):
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )
    path = task / "specification.json"
    payload = json.loads(path.read_text())
    payload["schema_version"] = schema_version
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=f"Unsupported TaskSpec schema: {schema_version}"):
        read_specification(path)


def test_file_result_cannot_use_text_submission_convention(tmp_path, specification):
    specification = specification.model_copy(update={"answer_type": AnswerType.FILE})
    convention = PlainText(id="plain")

    assert submission_compatible(specification, convention).reasons == ("plain cannot carry file",)
    assert compatible_lowerings(specification, (convention,), (HarborEnvironmentConfig(),)) == ()
    with pytest.raises(ValueError, match="incompatible: plain cannot carry file"):
        lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / "task")
    assert not (tmp_path / "task").exists()


def test_selection_policies_use_compatible_conventions(specification):
    conventions = (
        PlainText(id="plain"),
        JsonAnswer(id="json"),
    )
    candidates = compatible_lowerings(specification, conventions, (HarborEnvironmentConfig(),))

    assert select_lowerings(candidates, SelectionPolicy.ALL) == candidates
    assert select_lowerings(candidates, SelectionPolicy.FIRST) == (candidates[0],)
    repeated = [select_lowerings(candidates, SelectionPolicy.SAMPLE, rng_key=42) for _ in range(10)]
    assert all(selection == repeated[0] for selection in repeated)
    sampled_ids = {
        select_lowerings(candidates, SelectionPolicy.SAMPLE, rng_key=key)[0].convention.id for key in range(16)
    }
    assert sampled_ids == {candidate.convention.id for candidate in candidates}


async def test_chat_trial_resolves_key_at_runtime_without_persisting_it(
    tmp_path, specification, chat_endpoint, monkeypatch
):
    secret = "taskcompendium-local-test-secret"
    monkeypatch.setenv("TASKCOMPENDIUM_TEST_API_KEY", secret)
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        environment_config,
        tmp_path / "task",
    )
    launch = ChatLaunch(model="fixture-model", api_base=chat_endpoint.url, api_key_env="TASKCOMPENDIUM_TEST_API_KEY")

    result = await run_trial(task, environment_config, launch, tmp_path / "trials", "run")

    assert result.verifier_result.rewards == {"reward": 1.0}
    assert chat_endpoint.authorizations == [f"Bearer {secret}"]
    artifacts = list((tmp_path / "trials/run").rglob("*.json"))
    assert any(path.name == "config.json" for path in artifacts)
    assert any(path.name == "result.json" for path in artifacts)
    assert all(secret not in path.read_text() for path in artifacts)


async def test_chat_trial_preserves_conversation_roles(tmp_path, specification, chat_endpoint):
    context = ConversationInput(
        events=(
            TextMessage(role="system", content="Answer arithmetic questions."),
            TextMessage(role="user", content="What is 2 + 2?"),
            TextMessage(role="assistant", content="4"),
            TextMessage(role="user", content="What is 7 + 5?"),
        )
    )
    specification = specification.model_copy(update={"context": context})
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        environment_config,
        tmp_path / "task",
    )

    result = await run_trial(
        task,
        environment_config,
        ChatLaunch(model="fixture-model", api_base=chat_endpoint.url),
        tmp_path / "trials",
        "run",
    )

    assert result.verifier_result.rewards == {"reward": 1.0}
    assert chat_endpoint.requests[0]["messages"] == [
        {"role": "system", "content": "Answer arithmetic questions."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "What is 7 + 5?"},
        {"role": "user", "content": "Give your answer as plain text."},
    ]


async def test_chat_http_error_preserves_server_diagnostic(tmp_path, specification, chat_endpoint):
    chat_endpoint.status = 400
    chat_endpoint.body = b'{"error":"model unavailable"}'
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        environment_config,
        tmp_path / "task",
    )
    launch = ChatLaunch(model="fixture-model", api_base=chat_endpoint.url)

    result = await run_trial(task, environment_config, launch, tmp_path / "trials", "run")

    assert result.exception_info is not None
    assert "model unavailable" in (tmp_path / "trials/run/result.json").read_text()
