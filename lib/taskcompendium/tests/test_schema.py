# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public schema loading and runtime boundaries preserve private contracts."""

import base64
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import ExactSpec, JudgeSpec, StripReasoningBlocks

from taskcompendium.grader import verifyit_package
from taskcompendium.grading import grade_answer, grade_remote_judge_answer
from taskcompendium.models import (
    SCHEMA_VERSION,
    AnswerType,
    CommandSemantics,
    ConversationInput,
    ConversationTrace,
    DockerBuildContext,
    EnvironmentRequirements,
    GraderMachine,
    GradingAttempt,
    NoGrader,
    PlainText,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    upgrade_task_payload,
)
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.submission import chat_request

IMAGE = "private/grader@sha256:" + "a" * 64
GRADING_ENVIRONMENT = EnvironmentRequirements(
    docker_image=IMAGE,
    command_semantics=CommandSemantics.LINUX_PROCESS,
    setup_commands=("pip install --no-index /tests/wheels/*.whl",),
    environment_variables={"CHECK_MODE": "strict"},
)
EXACT_DONE = verifyit_package(ExactSpec(expected=("done",))).grader
EXACT_DONE_IN_ENVIRONMENT = verifyit_package(
    ExactSpec(expected=("done",)), placement=GraderMachine(environment=GRADING_ENVIRONMENT)
).grader


@pytest.fixture
def specification():
    return TaskSpec(
        id="schema-example",
        context=ConversationInput(events=(TextMessage(role="user", content="Repair the project."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=EXACT_DONE,
        source=Source(dataset="org/project", revision="pinned-revision", row="0", importer_revision="1"),
    )


@pytest.mark.parametrize(
    "answer_type,output_paths,error",
    [
        (AnswerType.WORKSPACE_STATE, (), "requires a grader placement"),
        (AnswerType.FILE, (), "exactly one output path"),
        (AnswerType.FILE, ("/app/a.txt", "/app/b.txt"), "exactly one output path"),
    ],
)
def test_host_judge_rejects_answers_it_cannot_read_before_admission(specification, answer_type, output_paths, error):
    wire = specification.model_dump()
    wire.update(
        answer_type=answer_type,
        output_paths=output_paths,
        grader=verifyit_package(JudgeSpec(references=("done",))).grader,
    )
    with pytest.raises(ValidationError, match=error):
        TaskSpec.model_validate(wire)


@pytest.mark.parametrize(
    "update",
    [
        {"environment_requirements": EnvironmentRequirements(capabilities=("browser",))},
        {"environment_requirements": EnvironmentRequirements(command_semantics=CommandSemantics.SHELL_SIMULATOR)},
        {
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="org/image@sha256:" + "a" * 64
            )
        },
        {
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                docker_build=DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM mutable:latest\n"),)),
            )
        },
        {"environment_requirements": EnvironmentRequirements(working_directory="/app")},
        {"environment_requirements": EnvironmentRequirements(setup_commands=("initialize",))},
        {"environment_requirements": EnvironmentRequirements(environment_variables={"TASK_MODE": "repair"})},
        {
            "environment_requirements": EnvironmentRequirements.model_validate(
                {
                    "tool_providers": {
                        "company": {
                            "action_interface": "workplace:v1",
                            "initial_state": {"inbox": [], "company": "example"},
                        }
                    },
                }
            )
        },
    ]
    + [
        {"resources": {role: [{"path": "input.txt", "source": {"kind": "inline_file", "content_base64": "eA=="}}]}}
        for role in ("all", "worker", "oracle")
    ]
    + [
        {
            "resources": {
                "worker": [
                    {
                        "path": "project/input.txt",
                        "source": {"kind": "inline_file", "content_base64": "cHVibGljIGlucHV0"},
                        "mode": "0755",
                    }
                ]
            }
        },
        {"answer_type": AnswerType.FILE, "output_paths": ("/app/answer.txt",), "grader": EXACT_DONE_IN_ENVIRONMENT},
        {"answer_type": AnswerType.STATE},
        {"answer_type": AnswerType.WORKSPACE_STATE, "grader": EXACT_DONE_IN_ENVIRONMENT},
        {"grader": EXACT_DONE_IN_ENVIRONMENT},
    ],
)
def test_direct_chat_rejects_semantics_it_cannot_preserve_before_request(specification, update):
    record = TaskSpec.model_validate({**specification.model_dump(), **update})
    task = TaskSpec.model_validate_json(record.model_dump_json())
    with pytest.raises(NotImplementedError):
        chat_request(task)


@pytest.mark.parametrize("second_path", ["data", "data/input.txt"])
@pytest.mark.parametrize("role", ["worker", "oracle", "verifier"])
def test_shared_resource_destinations_cannot_overwrite_role_mounts(specification, second_path, role):
    wire = specification.model_dump()
    wire["resources"] = {
        "all": [{"path": "data", "source": {"kind": "inline_file", "content_base64": "c2hhcmVk"}}],
        role: [{"path": second_path, "source": {"kind": "inline_file", "content_base64": "cHJpdmF0ZQ=="}}],
    }
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(wire)


def test_private_role_mounts_reuse_paths_without_becoming_worker_visible(specification):
    wire = specification.model_dump(mode="json")
    wire["resources"] = {
        role: [
            {
                "path": "fixture.txt",
                "source": {"kind": "inline_file", "content_base64": base64.b64encode(content.encode()).decode("ascii")},
            }
        ]
        for role, content in (("worker", "public"), ("oracle", "gold"), ("verifier", "hidden test"))
    }
    resources = TaskSpec.model_validate_json(json.dumps(wire)).model_dump(mode="json")["resources"]
    assert resources["all"] == []
    assert base64.b64decode(resources["worker"][0]["source"]["content_base64"]) == b"public"
    assert base64.b64decode(resources["oracle"][0]["source"]["content_base64"]) == b"gold"
    assert base64.b64decode(resources["verifier"][0]["source"]["content_base64"]) == b"hidden test"


def test_build_context_roundtrip_keeps_bytes_metadata_and_role_boundaries(specification):
    dockerfile = inline_resource("Dockerfile", b"FROM mutable:latest\nCOPY payload.bin /input\n")
    payload = inline_resource("payload.bin", b"\x00\xff\x80").model_copy(update={"mode": "0755", "mtime_ns": 123456789})
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_build=DockerBuildContext(files=(dockerfile, payload))
    )
    wire = specification.model_dump()
    wire["environment_requirements"] = environment.model_dump()
    wire["grader"] = ScriptGrader(argv=("true",), placement=GraderMachine(environment=environment)).model_dump()
    task = TaskSpec.model_validate_json(json.dumps(wire))
    restored = task.model_dump(mode="json")
    for requirements in (restored["environment_requirements"], restored["grader"]["placement"]["environment"]):
        files = {resource["path"]: resource for resource in requirements["docker_build"]["files"]}
        assert (
            base64.b64decode(files["Dockerfile"]["source"]["content_base64"])
            == b"FROM mutable:latest\nCOPY payload.bin /input\n"
        )
        assert base64.b64decode(files["payload.bin"]["source"]["content_base64"]) == b"\x00\xff\x80"
        assert (files["payload.bin"]["mode"], files["payload.bin"]["mtime_ns"]) == ("0755", 123456789)
    assert task.resources == ResourceGroups()


@pytest.mark.parametrize("initial_state", [None, "company-snapshot", {"inbox": [], "counter": 3}])
def test_reader_keeps_literal_provider_state_but_direct_chat_cannot_request_it(specification, initial_state):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"]["tool_providers"] = {
        "company": {"action_interface": "workplace:v1", "initial_state": initial_state}
    }
    task = TaskSpec.model_validate_json(json.dumps(wire))
    assert task.environment_requirements.tool_providers["company"].initial_state == initial_state
    with pytest.raises(NotImplementedError):
        chat_request(task)


def test_reader_rejects_nested_nonfinite_provider_state(specification):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"]["tool_providers"] = {
        "company": {"action_interface": "workplace:v1", "initial_state": {"counters": [float("nan")]}}
    }
    with pytest.raises(ValidationError):
        TaskSpec.model_validate_json(json.dumps(wire))


GRADER_CONFIGURATION = {"checks": [{"tolerance": 0.125}], "label": "NaN"}


@pytest.mark.parametrize(
    "grader",
    [
        {
            "kind": "verifyit",
            "mode": "structured_exact",
            "parameters": {"expected": GRADER_CONFIGURATION},
            "placement": None,
        },
        {"kind": "none", "reason": "Source evaluator is unavailable", "contract": GRADER_CONFIGURATION},
    ],
)
@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e309"])
def test_grader_configuration_rejects_nested_nonfinite_json_numbers(specification, grader, number):
    # JsonValue alone accepts nonfinite numbers despite allow_inf_nan=False.
    wire = {**specification.model_dump(mode="json"), "answer_type": "json", "answer_format": {"kind": "json_value"}}
    text = json.dumps({**wire, "grader": grader})
    assert TaskSpec.model_validate_json(text).model_dump(mode="json")["grader"] == grader
    with pytest.raises(ValidationError):
        TaskSpec.model_validate_json(text.replace("0.125", number))


@pytest.mark.parametrize(
    "base_path,alias",
    [("foo", "foo."), ("foo", "foo "), ("inputs/answer", "inputs/answer:backup"), ("foo", "FOO"), ("a/b", "a\\b")],
)
def test_resource_groups_preserve_distinct_linux_files(tmp_path, specification, base_path, alias):
    wire = specification.model_dump(mode="json")
    wire["resources"] = {
        "all": [{"path": base_path, "source": {"kind": "inline_file", "content_base64": "cHVibGlj"}}],
        "worker": [{"path": alias, "source": {"kind": "inline_file", "content_base64": "b3ZlcndyaXRl"}}],
    }
    task = TaskSpec.model_validate(wire)
    for resource in task.resources.all + task.resources.worker:
        target = tmp_path / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(resource.source.content_base64))
    assert (tmp_path / base_path).read_bytes() == b"public"
    assert (tmp_path / alias).read_bytes() == b"overwrite"


@pytest.mark.parametrize("candidate,reward", [("done", 1.0), ("incorrect", 0.0)])
def test_in_process_grading_accepts_answers_acquired_in_a_worker_workspace(specification, candidate, reward):
    wire = specification.model_dump(mode="json")
    wire["environment_requirements"] = {"capabilities": ["shell", "filesystem"], "working_directory": "/app"}
    wire["resources"] = {
        "worker": [{"path": "project.txt", "source": {"kind": "inline_file", "content_base64": "d29ya2VyIGlucHV0"}}]
    }
    task = TaskSpec.model_validate(wire)
    conversation = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=candidate)))
    result = grade_answer(task, GradingAttempt(conversation))
    assert (result.status, result.reward) == ("graded", reward)


@pytest.mark.parametrize("payload", [b"UTF-8 text: \xe2\x98\x83\n", b"\x00\xff\x80\n"])
def test_inline_file_bytes_and_metadata_survive_json_reader(specification, payload):
    wire = specification.model_dump(mode="json")
    wire["resources"] = {
        "worker": [
            {
                "path": "input.dat",
                "source": {"kind": "inline_file", "content_base64": base64.b64encode(payload).decode("ascii")},
                "mode": "0500",
                "mtime_ns": 1_725_555_600_123_456_789,
            }
        ]
    }
    task = TaskSpec.model_validate(wire)
    restored = TaskSpec.model_validate_json(task.model_dump_json()).resources.worker[0]
    assert base64.b64decode(restored.source.content_base64) == payload
    assert restored.mode == "0500"
    assert restored.mtime_ns == 1_725_555_600_123_456_789


GRADER_MACHINE = {
    "kind": "grader_machine",
    "environment": {"docker_image": IMAGE, "command_semantics": "linux_process"},
}
SCRIPT = {"kind": "script", "argv": ["python3", "/tests/grade.py"], "placement": GRADER_MACHINE}
VERIFYIT_ENVIRONMENT = {"kind": "verifyit", "placement": GRADER_MACHINE}


@pytest.mark.parametrize(
    "update",
    [
        pytest.param(
            {"answer_type": "file", "output_paths": ["/tests/answer.txt"], "grader": SCRIPT | {"answer_path": None}},
            id="output-path-under-tests",
        ),
        pytest.param(
            {
                "answer_type": "file",
                "environment_requirements": {"capabilities": ["python3"]},
                "output_paths": ["/logs/verifier"],
                "grader": SCRIPT | {"answer_path": None},
            },
            id="output-directory-under-verifier-logs",
        ),
        pytest.param(
            {"answer_type": "file", "output_paths": ["/app/answer.txt"], "grader": SCRIPT},
            id="script-answer-path-for-file-answer",
        ),
        pytest.param({"grader": SCRIPT | {"answer_path": "/tests/answer.txt"}}, id="script-answer-path-under-tests"),
        pytest.param(
            {
                "grader": SCRIPT | {"conversation_path": "/tests/grade.py"},
                "resources": {
                    "verifier": [{"path": "grade.py", "source": {"kind": "inline_file", "content_base64": "eA=="}}]
                },
            },
            id="script-conversation-path-replaces-resource",
        ),
        pytest.param(
            {
                "grader": (
                    VERIFYIT_ENVIRONMENT
                    | {"mode": "exact", "parameters": {"expected": ["done"], "output": "/tests/answer.txt"}}
                )
            },
            id="verifyit-answer-file-under-tests",
        ),
        pytest.param(
            {
                "grader": {
                    "kind": "verifyit",
                    "mode": "exact",
                    "parameters": {"expected": ["done"], "output": "/logs/verifier/answer.txt"},
                    "placement": {"kind": "agent_machine"},
                }
            },
            id="agent-machine-answer-file-under-verifier-logs",
        ),
        pytest.param({"grader": {"kind": "script", "argv": ["true"]}}, id="script-without-placement"),
        pytest.param(
            {"grader": {**SCRIPT, "placement": {"kind": "grader_machine", "environment": {}}}},
            id="grader-machine-without-pinned-environment",
        ),
        pytest.param(
            {
                "grader": {
                    **SCRIPT,
                    "placement": {
                        **GRADER_MACHINE,
                        "environment": {
                            **GRADER_MACHINE["environment"],
                            "tool_providers": {"company": {"action_interface": "v1", "initial_state": None}},
                        },
                    },
                }
            },
            id="grader-machine-with-tool-providers",
        ),
        pytest.param(
            {"verifies_stopped_workspace": True},
            id="stopped-workspace-verdict-on-a-text-answer",
        ),
        pytest.param(
            {"grader": {**SCRIPT, "placement": {"kind": "agent_machine", "collect": [{"argv": ["true"]}]}}},
            id="agent-machine-collects-from-itself",
        ),
        pytest.param(
            {"grader": {**SCRIPT, "environment": GRADER_MACHINE["environment"]}},
            id="schema-0.30-grader-environment-in-current-schema",
        ),
    ],
)
def test_task_validation_rejects_invalid_grading_contracts(specification, update):
    wire = specification.model_dump(mode="json")
    with pytest.raises(ValidationError):
        TaskSpec.model_validate({**wire, **update})


def test_task_json_rejects_prior_schema_version(specification):
    payload = json.loads(specification.model_dump_json())
    payload["schema_version"] = "0.26"
    with pytest.raises(ValidationError):
        TaskSpec.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("judge", [JudgeSpec(), JudgeSpec(references=(" ",)), JudgeSpec(rubric="checklist")])
def test_invalid_script_judge_is_schema_feedback(judge):
    # Full S 20261108 d01.discrete.graph-analysis/06: InvalidTask escaped Pydantic and held the author.
    with pytest.raises(ValidationError, match="needs non-empty"):
        ScriptGrader(argv=("true",), placement=GraderMachine(environment=GRADING_ENVIRONMENT), judge=judge)


@pytest.mark.parametrize(
    "judge,match",
    [
        (JudgeSpec(model="fixture-judge", rubric=""), None),
        (JudgeSpec(model="fixture-judge", rubric="reference", references=("done",)), None),
        (JudgeSpec(model="fixture-judge", rubric="", request_timeout=0), "request timeout"),
        (JudgeSpec(model="fixture-judge", rubric="equivalence"), "unknown judge rubric"),
    ],
)
def test_script_judge_contract(judge, match):
    grader = dict(argv=("true",), placement=GraderMachine(environment=GRADING_ENVIRONMENT), judge=judge)
    if match is None:
        assert ScriptGrader(**grader).judge == judge
    else:
        with pytest.raises(ValidationError, match=match):
            ScriptGrader(**grader)


TASK_SPEC_0_30 = Path(__file__).parent / "fixtures" / "task_spec_0_30.json"
"""Tasks serialized by schema 0.30 models, one per grader shape."""


def test_schema_0_30_payloads_upgrade_to_grader_placements():
    payloads = {payload["id"]: json.dumps(payload) for payload in json.loads(TASK_SPEC_0_30.read_text())}
    tasks = {task_id: TaskSpec.model_validate_json(payload) for task_id, payload in payloads.items()}

    assert {task.schema_version for task in tasks.values()} == {SCHEMA_VERSION}
    in_process, verifyit_machine = tasks["in-process"].grader, tasks["verifyit-machine"].grader
    assert isinstance(in_process, VerifyitGrader) and in_process.placement is None
    assert isinstance(verifyit_machine, VerifyitGrader) and isinstance(verifyit_machine.placement, GraderMachine)
    script = tasks["script-collect"].grader
    original = json.loads(payloads["script-collect"])["grader"]
    assert isinstance(script, ScriptGrader) and script.placement.model_dump(mode="json") == {
        "kind": "grader_machine",
        "environment": original["environment"],
        # A 0.30 collect command has no deadline of its own; the grader's limit bounds it.
        "collect": [{**command, "timeout": None} for command in original["collect"]],
        "artifacts": original["artifacts"],
    }
    assert isinstance(tasks["no-grader"].grader, NoGrader)
    for task_id, payload in payloads.items():
        # The upgrade is deterministic and its result is a current-schema payload.
        assert TaskSpec.model_validate_json(payload) == tasks[task_id]
        assert TaskSpec.model_validate_json(tasks[task_id].model_dump_json()) == tasks[task_id]
        upgraded = upgrade_task_payload(json.loads(payload))
        assert upgrade_task_payload(upgraded) == upgraded


def test_schema_0_30_judge_candidate_flag_upgrades_to_steps_that_grade_the_same():
    payloads = {payload["id"]: payload for payload in json.loads(TASK_SPEC_0_30.read_text())}
    upgraded = TaskSpec.model_validate(payloads["judge-in-process"])
    current_judge = JudgeSpec(
        references=("Mars",),
        question="Which planet is red?",
        candidate_steps=(StripReasoningBlocks(),),
        transport_attempts=1,
    )
    current = upgraded.model_copy(update={"grader": verifyit_package(current_judge).grader})
    script = TaskSpec.model_validate(payloads["script-judge"]).grader

    assert isinstance(script, ScriptGrader) and script.judge == current_judge
    # Only the stripped answer passes the exact gate; the endpoint is never reached.
    connection = JudgeConnection("http://unused.invalid/v1", "secret")
    reply = TextMessage(role="assistant", content="<think>Venus?</think>Mars")
    attempt = GradingAttempt(ConversationTrace(events=(*upgraded.context.events, reply)))
    results = [grade_remote_judge_answer(task, attempt, connection) for task in (upgraded, current)]
    assert results[0] == results[1]
    assert (results[0].status, results[0].reward) == ("graded", 1.0)


def test_only_schema_0_30_upgrades():
    payload = json.loads(TASK_SPEC_0_30.read_text())[0]
    payload["schema_version"] = "0.29"
    with pytest.raises(ValidationError, match=r"Unsupported TaskSpec schema: 0\.29"):
        TaskSpec.model_validate(payload)


def test_grading_trace_keeps_an_empty_model_reply_that_a_source_conversation_rejects():
    """A model can reply with nothing mid-episode; the trace records it, while task context must not hold one."""
    events = (
        TextMessage(role="user", content="Reply in JSON."),
        TextMessage(role="assistant", content=""),
        TextMessage(role="user", content="Please send a JSON response again."),
    )

    trace = ConversationTrace(events=(*events, TextMessage(role="assistant", content='{"done": true}')))

    assert trace.events[1].content == ""
    with pytest.raises(ValidationError, match="nonempty content"):
        ConversationInput(events=events)
