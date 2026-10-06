# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field

import pytest
from rolloutengine.contracts import ModelRequest, ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from taskcompendium.chat import chat_conversation
from taskcompendium.environment import (
    ArtifactKind,
    DockerBuild,
    EnvironmentKind,
    ExitCodeReward,
    HealthcheckSpec,
    RegistryImage,
    RewardFileFormat,
    ShellVerifierSpec,
    StdoutReward,
    VerifierArtifact,
)
from taskcompendium.execution import StageExecution, TaskExecution
from taskcompendium.grading import grade_answer, structured_exact, verifier_descriptor
from taskcompendium.grading_contract import GradingAttempt
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    FunctionDefinition,
    Source,
    StageRewardStrategy,
    TaskSpec,
)
from taskcompendium.submission import FinalAction, JsonValueAnswer, PlainText
from verifyit.spec import ExactSpec, FunctionCall, McqSpec, NumericSpec, PredictedActionSpec

from taskforge.spec.draft import (
    Resources,
    Reward,
    assemble,
    environment,
    file,
    reward_file,
    shell_command,
    shell_verifier,
    stage,
    staged,
)

SOURCE = Source(dataset="taskforge-test", revision="r1", row="0", importer_revision="test")
PLAIN = PlainText(id="plain")
NO_EXECUTION = TaskExecution()
GRADE_ANSWER = 'if [ "$(cat /workspace/answer)" = 12 ]; then echo 1; else echo 0; fi'


@dataclass
class ReplayModel:
    messages: list[dict]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        index = len(self.requests) - 1
        prompt = (*request.prefix_token_ids, 90, 91) if request.prefix_token_ids else (10, 11)
        return ModelTurn(self.messages[index], prompt, (20 + index,), (-0.5,), "stop")


def shell_message(call_id: str, command: str) -> dict:
    arguments = json.dumps({"command": command})
    return {
        "role": "assistant",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "shell", "arguments": arguments}}],
    }


async def run_on_shellsim(task: TaskSpec, messages: list[dict], execution: TaskExecution = NO_EXECUTION):
    engine = ShellboxRolloutEngine(
        ReplayModel(messages).complete,
        {EnvironmentKind.SHELLSIM: ShellSimMachineFactory()},
        max_turns=6,
        command_timeout=5,
        cleanup_timeout=5,
        convention=PLAIN,
    )
    return await engine.run(TaskSpec.model_validate_json(task.model_dump_json()), execution=execution)


def answer_task(verifier) -> TaskSpec:
    return assemble(
        "answer",
        "What is six plus six?",
        AnswerType.TEXT,
        environment(EnvironmentKind.NULL),
        verifier_descriptor(verifier),
        SOURCE,
        execution=NO_EXECUTION,
        system="Answer briefly.",
    )


@pytest.mark.parametrize(
    "verifier,right,wrong",
    [
        (ExactSpec(expected=("twelve",)), "Twelve", "eleven"),
        (NumericSpec(expected="12", tolerance_abs=0.0, tolerance_rel=0.0), "12", "13"),
        (McqSpec(expected="B"), "B", "C"),
    ],
)
def test_answer_verifier_grades_after_json_round_trip(verifier, right, wrong):
    task = TaskSpec.model_validate_json(answer_task(verifier).model_dump_json())

    def reward(answer: str) -> float | None:
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": answer}]
        return grade_answer(task, PLAIN, GradingAttempt(chat_conversation(messages))).reward

    assert (reward(right), reward(wrong)) == (1.0, 0.0)
    assert task.environment_requirements.capabilities == ()


def test_predicted_action_verifier_grades_native_action_after_round_trip():
    lookup = FunctionDefinition(name="lookup", parameters={"type": "object", "properties": {"city": {"type": "string"}}})
    task = assemble(
        "action",
        "Look up the weather in Paris.",
        AnswerType.NATIVE_ACTION,
        environment(EnvironmentKind.NULL),
        verifier_descriptor(PredictedActionSpec(expected_calls=(FunctionCall("lookup", {"city": "Paris"}),))),
        SOURCE,
        execution=NO_EXECUTION,
        final_tools=(lookup,),
    )
    task = TaskSpec.model_validate_json(task.model_dump_json())

    def reward(city: str) -> float | None:
        call = {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"city": city})}}
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "tool_calls": [call]}]
        return grade_answer(task, FinalAction(id="action"), GradingAttempt(chat_conversation(messages))).reward

    assert (reward("Paris"), reward("Rome")) == (1.0, 0.0)


def test_structured_exact_verifier_grades_a_json_answer():
    task = assemble(
        "json",
        "Report the total as a JSON object.",
        AnswerType.JSON,
        environment(EnvironmentKind.NULL),
        structured_exact({"total": 12}),
        SOURCE,
        execution=NO_EXECUTION,
    )

    def reward(answer: str) -> float | None:
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": answer}]
        return grade_answer(task, JsonValueAnswer(id="json"), GradingAttempt(chat_conversation(messages))).reward

    assert (reward('{"total": 12}'), reward('{"total": 13}')) == (1.0, 0.0)


@pytest.mark.parametrize(
    "argv,reward",
    [
        (("sh", "/private/grade.sh"), StdoutReward()),
        (("sh", "-c", '[ "$(cat /workspace/answer)" = 12 ]'), ExitCodeReward()),
        (
            ("sh", "-c", 'mkdir -p /logs; echo "{\\"reward\\": $(sh /private/grade.sh)}" > /logs/reward.json'),
            reward_file("/logs/reward.json", RewardFileFormat.JSON),
        ),
    ],
)
@pytest.mark.parametrize("answer,expected", [("12", 1.0), ("13", 0.0)])
async def test_shell_verifier_grades_the_agent_workspace_on_shellsim(argv, reward: Reward, answer, expected):
    task = assemble(
        "file-answer",
        "Write six plus six to /workspace/answer.",
        AnswerType.FILE,
        environment(EnvironmentKind.SHELLSIM, files=(file("/workspace/README", "Write your answer here.\n"),)),
        shell_verifier(argv, reward, timeout=5, files=(file("/private/grade.sh", GRADE_ANSWER),)),
        SOURCE,
        execution=NO_EXECUTION,
    )

    result = await run_on_shellsim(
        task, [shell_message("c1", f"echo {answer} > /workspace/answer"), {"role": "assistant", "content": "Done."}]
    )

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, expected)
    assert "/private/grade.sh" not in json.dumps(result.messages)


async def test_staged_task_runs_stages_on_one_machine_and_stops_below_minimum():
    def check(path: str, value: str):
        return shell_verifier(("sh", "-c", f'[ "$(cat {path})" = {value} ]'), ExitCodeReward(), timeout=5)

    execution = TaskExecution(stages={"first": StageExecution(), "second": StageExecution()})
    task = assemble(
        "staged",
        "Write 12 to /workspace/a.",
        AnswerType.FILE,
        environment(EnvironmentKind.SHELLSIM),
        staged(StageRewardStrategy.MEAN),
        SOURCE,
        execution=execution,
        stages=(
            stage("first", check("/workspace/a", "12"), minimum_rewards={"reward": 1.0}),
            stage("second", check("/workspace/b", "24"), instruction="Now write 24 to /workspace/b."),
        ),
    )
    both = [
        shell_message("c1", "echo 12 > /workspace/a"),
        {"role": "assistant", "content": "Done."},
        shell_message("c2", "echo 24 > /workspace/b"),
        {"role": "assistant", "content": "Done."},
    ]
    passed = await run_on_shellsim(task, both, execution)
    failed = await run_on_shellsim(task, [shell_message("c1", "echo 11 > /workspace/a"), both[1]], execution)

    assert passed.grade.reward == 1.0
    assert [item["name"] for item in passed.grade.diagnostics["stages"]] == ["first", "second"]
    assert failed.grade.reward == 0.0
    assert [item["name"] for item in failed.grade.diagnostics["stages"]] == ["first"]


def test_docker_task_with_separate_grading_machine_round_trips():
    build = DockerBuild(files=(file("/Dockerfile", "FROM python:3.12-slim\nWORKDIR /app\n"),))
    grading = environment(
        EnvironmentKind.DOCKER,
        image=RegistryImage(reference="ghcr.io/example/grader@sha256:" + "0" * 64),
        workdir="/grade",
        resources=Resources(memory_mb=2048, cpus=2),
    )
    verifier = shell_verifier(
        ("python3", "/private/grade.py"),
        reward_file("/logs/reward.json", RewardFileFormat.JSON, key="score", pass_above=0.5),
        timeout=120,
        files=(file("/private/grade.py", "print(1)\n"),),
        user="root",
        grading_environment=grading,
        collect=(shell_command("cp -r /app /tmp/app", timeout=30),),
        artifacts=(VerifierArtifact(source="/app", target="/submission", kind=ArtifactKind.DIRECTORY),),
    )
    task = assemble(
        "docker",
        "Fix the bug in /app.",
        AnswerType.WORKSPACE_STATE,
        environment(
            EnvironmentKind.DOCKER,
            image=build,
            files=(file("/app/run.sh", "#!/bin/sh\npython3 main.py\n", mode=0o755),),
            setup=(shell_command("pip install pytest", timeout=300),),
            healthcheck=HealthcheckSpec(
                command=shell_command("true", timeout=5), interval=1, start_period=0, start_interval=1, retries=3
            ),
            workdir="/app",
            network=True,
            resources=Resources(memory_mb=4096, cpus=4, storage_mb=10240),
        ),
        verifier,
        SOURCE,
        execution=TaskExecution(attempt_timeout=3600, agent_user="agent"),
        metadata={"proposal": "abc"},
        tags=("taskforge",),
    )

    restored = TaskSpec.model_validate_json(task.model_dump_json())
    parameters = ShellVerifierSpec.model_validate_json(restored.verifier.parameters_json)

    assert restored == task
    assert restored.environment_requirements.capabilities == ("shell", "filesystem")
    assert (restored.environment.memory_mb, restored.environment.cpus, restored.environment.network) == (4096, 4, True)
    assert restored.environment.files[0].mode == 0o755
    assert restored.verifier.environment == grading
    assert restored.verifier.files[0].content == b"print(1)\n"
    assert parameters.artifacts[0].target == "/submission"


def test_assemble_rejects_private_grader_content_shipped_to_the_agent():
    copy = file("/workspace/notes.sh", GRADE_ANSWER)
    grader = shell_verifier(
        ("sh", "/private/grade.sh"), StdoutReward(), timeout=5, files=(file("/private/grade.sh", GRADE_ANSWER),)
    )
    with pytest.raises(ValueError, match="agent-visible"):
        assemble(
            "leak",
            "Write the answer.",
            AnswerType.FILE,
            environment(EnvironmentKind.SHELLSIM, files=(copy,)),
            grader,
            SOURCE,
            execution=NO_EXECUTION,
        )
    with pytest.raises(ValueError, match="agent-visible"):
        assemble(
            "staged-leak",
            "Write the answer.",
            AnswerType.FILE,
            environment(EnvironmentKind.SHELLSIM),
            staged(StageRewardStrategy.FINAL),
            SOURCE,
            execution=TaskExecution(stages={"only": StageExecution(workdir_files=(file("/notes.sh", GRADE_ANSWER),))}),
            stages=(stage("only", grader),),
        )


def test_stage_minimum_rewards_must_name_components_the_grader_reports():
    def staged_task(reward: Reward):
        gate = stage("first", shell_verifier(("true",), reward, timeout=5), minimum_rewards={"score": 1.0})
        return assemble(
            "gated",
            "Do it.",
            AnswerType.FILE,
            environment(EnvironmentKind.SHELLSIM),
            staged(StageRewardStrategy.FINAL),
            SOURCE,
            execution=TaskExecution(stages={"first": StageExecution()}),
            stages=(gate,),
        )

    for reward in (ExitCodeReward(), StdoutReward(), reward_file("/logs/score.txt", RewardFileFormat.NUMBER)):
        with pytest.raises(ValueError, match="reports only 'reward'"):
            staged_task(reward)
    gated = staged_task(reward_file("/logs/reward.json", RewardFileFormat.JSON))
    assert gated.stages[0].minimum_rewards == {"score": 1.0}


@pytest.mark.parametrize(
    "answer_type,env_kind,verifier,message",
    [
        (AnswerType.TEXT, EnvironmentKind.NULL, "shell", "executable task environment"),
        (AnswerType.FILE, EnvironmentKind.SHELLSIM, "exact", "requires a shell verifier"),
        (AnswerType.TEXT, EnvironmentKind.NULL, "action", "native_action"),
        (AnswerType.TEXT, EnvironmentKind.NULL, "structured", "json answer"),
        (AnswerType.JSON, EnvironmentKind.NULL, "numeric", "number or text answer"),
    ],
)
def test_assemble_rejects_graders_that_cannot_see_the_answer(answer_type, env_kind, verifier, message):
    verifiers = {
        "shell": shell_verifier(("true",), ExitCodeReward(), timeout=5),
        "exact": verifier_descriptor(ExactSpec(expected=("12",))),
        "action": verifier_descriptor(PredictedActionSpec(expected_calls=(FunctionCall("f", {}),))),
        "structured": structured_exact({"total": 12}),
        "numeric": verifier_descriptor(NumericSpec(expected="12", tolerance_abs=0.0, tolerance_rel=0.0)),
    }
    with pytest.raises(ValueError, match=message):
        assemble(
            "bad", "Do it.", answer_type, environment(env_kind), verifiers[verifier], SOURCE, execution=NO_EXECUTION
        )


@pytest.mark.parametrize("stages", [{}, {"first": StageExecution(), "extra": StageExecution()}])
def test_assemble_rejects_execution_settings_for_other_stages(stages):
    gate = stage("first", shell_verifier(("true",), ExitCodeReward(), timeout=5))
    with pytest.raises(ValueError, match="Execution stages"):
        assemble(
            "staged",
            "Do it.",
            AnswerType.FILE,
            environment(EnvironmentKind.SHELLSIM),
            staged(StageRewardStrategy.FINAL),
            SOURCE,
            execution=TaskExecution(stages=stages),
            stages=(gate,),
        )
