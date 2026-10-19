# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find the capabilities that the policy's failed runs lack, as catalog ideas for ``generate_tasks``.

``analyze_failed_runs`` gives the failed runs to one managed agent: a ``RolloutEngine`` attempt whose shell harness
serves two read-only tools, ``read_run`` and ``search_runs``. The agent answers with the capabilities whose absence
explains the failures, for example an answer format, the use of tools, a loop of repeated calls, or a careless
reading of the task. Each capability becomes a ``CapabilityIdea``, so ``task_workpieces`` and ``generate_tasks`` make
training tasks from it. The agent sees only what the policy saw and wrote, never a grader or its answer.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness, ToolCallResult, ToolContext
from rolloutengine.models import GradedRollout, Model, Rollout, RolloutLimits, TaskRolloutSpec, TraceMode
from taskcompendium.chat import chat_conversation
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    GradingAttempt,
    JsonValueAnswer,
    NoGrader,
    ProviderRequirement,
    Source,
    SubmissionFailure,
    TaskSpec,
    TextMessage,
)
from taskforge.content_hash import digest

from experiments.post_training.capability_driven_envs.catalog import CAPABILITY_KIND, CapabilityIdea
from experiments.post_training.capability_driven_envs.generate_tasks import FILE_SAFE_ID

SUBJECT_ID = "failed-runs"
SUBJECT_NAME = "Capabilities that failed runs lack"
ANALYSIS_REVISION = "1"
RUN_READER = "failed-runs/v1"
HARNESS = "failed-run-analyst"
ANSWER_FORMAT = JsonValueAnswer()
MAX_TASK_CHARS = 300
MAX_SNIPPET_CHARS = 200
MAX_TOOL_OUTPUT = 20_000
MAX_MATCHES = 50

READ_RUN = FunctionDefinition(
    name="read_run",
    description="Read the task's tools and the messages of one failed run, from message first_message on.",
    parameters={
        "type": "object",
        "properties": {"run_id": {"type": "string"}, "first_message": {"type": "integer", "minimum": 0}},
        "required": ["run_id"],
        "additionalProperties": False,
    },
)
SEARCH_RUNS = FunctionDefinition(
    name="search_runs",
    description="List the messages of all failed runs that contain a text, ignoring case.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
)


class SampleTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["entry", "representative"]
    instruction: str = Field(min_length=1)


class SamplingFacet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class MissingCapability(BaseModel):
    """A capability that the failed runs lack, with the catalog's capability fields and the runs that show it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="A file-safe ID, for example failed-runs.tool-use.stop-repeating-calls.")
    name: str = Field(min_length=1)
    outcome: str = Field(min_length=1, description="What an agent with the capability can do.")
    failure_pattern: str = Field(min_length=1, description="What the failed runs did, in general terms.")
    evidence: tuple[str, ...] = Field(min_length=1, description="The IDs of the failed runs that show the gap.")
    includes: tuple[str, ...] = Field(min_length=1)
    excludes: tuple[str, ...] = ()
    sample_tasks: tuple[SampleTask, ...] = Field(min_length=1)
    sampling_facets: tuple[SamplingFacet, ...] = ()

    @field_validator("id")
    @classmethod
    def file_safe(cls, value: str) -> str:
        if FILE_SAFE_ID.match(value) is None:
            raise ValueError(f"capability id {value!r} is not file-safe")
        return value


CAPABILITIES = TypeAdapter(list[MissingCapability])

SYSTEM_PROMPT = """\
You analyze the failed runs of a language-model agent to decide what the agent must learn next. A run is one \
attempt at a task: the task's messages, the agent's turns and tool calls, and the tool results.

Find the capabilities whose absence explains the failures. A capability can be any behavior that would have \
made a run succeed, for example: the answer format that the task asks for, the choice of tools and their \
arguments, a stop to loops that repeat a call or a mistake, a careful reading of the task and its constraints, \
a check of the work before the answer, or domain knowledge.

- Base each capability on runs that you read, and give their run IDs as evidence.
- Prefer a few general capabilities to many narrow ones. Merge failures that have the same cause.
- Write new sample tasks, each self-contained with one checkable answer. Do not copy the text, numbers, names, \
or answers of the failed tasks: the generated tasks train the agent, and the failed tasks evaluate it.

Read a run with read_run, and find a text across runs with search_runs. When you are done, answer without a \
tool call. The answer is one JSON array of capabilities that obeys this JSON schema:

{schema}
"""

USER_PROMPT = """\
{failed} of {total} runs failed. You can make at most {turns} model turns, so answer before the last one.

The failed runs, one JSON object for each:

{runs}
"""


@dataclass(frozen=True)
class EvaluatedRun:
    """One evaluated attempt: the task that it ran and its graded rollout."""

    task: TaskSpec
    result: GradedRollout


def policy_failures(runs: Sequence[EvaluatedRun], pass_threshold: float) -> list[EvaluatedRun]:
    """The runs that the policy failed: graded below ``pass_threshold``, or with no valid submission.

    An interrupted attempt, and an infrastructure, task, or grader failure, is not a failure of the policy, so it
    does not become a training target.
    """

    def failed(run: EvaluatedRun) -> bool:
        grade = run.result.grade
        if run.result.rollout.failure is not None:
            return False
        if grade.status == Outcome.SUBMISSION_FAILURE:
            return True
        return grade.status == Outcome.GRADED and grade.reward is not None and grade.reward < pass_threshold

    return [run for run in runs if failed(run)]


async def analyze_failed_runs(
    runs: Sequence[EvaluatedRun],
    model: Model,
    *,
    pass_threshold: float,
    limits: RolloutLimits,
    catalog_version: str,
) -> list[CapabilityIdea]:
    """The capabilities that the policy's failed runs lack, as ideas for ``generate_tasks``.

    ``model`` drives the analysis agent under ``limits``, for example a ``rolloutengine.chat.ChatModel``. With no
    policy failure, no agent runs. A failed analysis raises ``GradedRolloutInterrupted``, and an answer that is not a
    valid list of capabilities raises ``ValueError``.
    """
    failures = policy_failures(runs, pass_threshold)
    if not failures:
        return []
    # NOTE: One agent reads all the failed runs. A later version can divide the runs into shards with Zephyr and
    # analyze the shards in parallel, each with its own agent.
    by_id = {run.result.rollout.run_id: run for run in failures}
    reader = _RunReader(by_id)
    engine = RolloutEngine({}, {HARNESS: ShellHarness(command_timeout=None, tools={RUN_READER: reader})})
    spec = TaskRolloutSpec(
        task=_analysis_task(failures, len(runs), limits.max_turns, catalog_version),
        harness=HARNESS,
        task_machine=None,
        verifier_machine=None,
        limits=limits,
        trace=TraceMode.MESSAGES,
    )
    result = await engine.run_task(spec, model)
    capabilities = _answer(result.rollout)
    unknown = sorted({run_id for capability in capabilities for run_id in capability.evidence} - by_id.keys())
    if unknown:
        raise ValueError(f"The analysis cites runs that did not fail: {unknown}")
    ids = [capability.id for capability in capabilities]
    if len(set(ids)) != len(ids):
        raise ValueError(f"The analysis repeats capability IDs: {ids}")
    return [_idea(capability, catalog_version) for capability in capabilities]


def _analysis_task(failures: Sequence[EvaluatedRun], total: int, turns: int, catalog_version: str) -> TaskSpec:
    system = SYSTEM_PROMPT.format(schema=json.dumps(CAPABILITIES.json_schema()))
    runs = "\n".join(json.dumps(_summary(run)) for run in failures)
    user = USER_PROMPT.format(failed=len(failures), total=total, turns=turns, runs=runs)
    return TaskSpec(
        id=f"analyze-failed-runs-{catalog_version}",
        context=ConversationInput(
            events=(TextMessage(role="system", content=system), TextMessage(role="user", content=user))
        ),
        environment_requirements=EnvironmentRequirements(
            tool_providers={"runs": ProviderRequirement(action_interface=RUN_READER, initial_state=None)}
        ),
        answer_type=AnswerType.JSON,
        answer_format=ANSWER_FORMAT,
        grader=NoGrader(reason="An analysis of failed runs has no reference answer"),
        source=Source(dataset=SUBJECT_ID, revision=catalog_version, row="0", importer_revision=ANALYSIS_REVISION),
    )


def _summary(run: EvaluatedRun) -> dict[str, Any]:
    """The run's overview line: its grade, how it stopped, and its turns and tool calls."""
    rollout, grade = run.result.rollout, run.result.grade
    turns = [step.turn.message for conversation in rollout.conversations for step in conversation.steps]
    calls = [
        (call["function"]["name"], call["function"]["arguments"])
        for message in turns
        for call in message.get("tool_calls", ())
    ]
    events = run.task.context.events
    task = " ".join(event.content for event in events if isinstance(event, TextMessage) and event.role == "user")
    return {
        "run_id": rollout.run_id,
        "task_id": run.task.id,
        "tags": list(run.task.tags),
        "outcome": grade.status.value,
        "reward": grade.reward,
        "stop_reason": rollout.stop_reason,
        "turns": len(turns),
        "tool_calls": len(calls),
        "repeated_calls": sum(previous == call for previous, call in pairwise(calls)),
        "task": task[:MAX_TASK_CHARS],
    }


def _render(message: Mapping[str, Any]) -> str:
    calls = [f"{call['function']['name']}({call['function']['arguments']})" for call in message.get("tool_calls", ())]
    text = " ".join([str(message.get("content") or ""), *(f"<tool call> {call}" for call in calls)]).strip()
    return f"{message.get('role')}: {text}"


@dataclass(frozen=True)
class _RunReader:
    """The failed runs as two read-only tools for the analysis agent."""

    runs: Mapping[str, EvaluatedRun]

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        return (READ_RUN, SEARCH_RUNS)

    async def open(self, context: ToolContext) -> "_RunReader":
        return self

    async def close(self) -> None:
        pass

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        # A bad call gets an error message, so that the agent can correct it.
        try:
            if name == READ_RUN.name:
                return ToolCallResult(self._read(str(arguments["run_id"]), int(arguments.get("first_message", 0))))
            return ToolCallResult(self._search(str(arguments["text"])))
        except (KeyError, TypeError, ValueError) as error:
            return ToolCallResult(f"Tool error: {error!r}")

    def _messages(self, run: EvaluatedRun) -> list[str]:
        return [
            _render(message) for conversation in run.result.rollout.conversations for message in conversation.messages
        ]

    def _read(self, run_id: str, first: int) -> str:
        if run_id not in self.runs:
            return f"No failed run has the ID {run_id!r}."
        run = self.runs[run_id]
        messages = self._messages(run)
        if not 0 <= first < len(messages):
            raise ValueError(f"first_message must be from 0 to {len(messages) - 1}")
        task = run.task
        tools = [tool.model_dump(exclude_none=True) for tool in (*task.final_tools, *task.interaction_tools)]
        lines = [json.dumps({"tools": tools, "capabilities": list(task.environment_requirements.capabilities)})]
        size, end = len(lines[0]), first
        for index in range(first, len(messages)):
            line = f"[{index}] {messages[index]}"
            if index > first and size + len(line) > MAX_TOOL_OUTPUT:
                break
            lines.append(line[:MAX_TOOL_OUTPUT])
            size, end = size + len(line), index + 1
        lines.append(f"[Messages {first} to {end - 1} of {len(messages)}.]")
        return "\n".join(lines)

    def _search(self, text: str) -> str:
        matches = []
        for run_id, run in self.runs.items():
            for index, message in enumerate(self._messages(run)):
                at = message.lower().find(text.lower())
                if at >= 0:
                    start = max(0, at - MAX_SNIPPET_CHARS // 2)
                    matches.append(f"{run_id} [{index}] ...{message[start : start + MAX_SNIPPET_CHARS]}...")
        return "\n".join(matches[:MAX_MATCHES]) or "No matches."


def _answer(rollout: Rollout) -> list[MissingCapability]:
    messages = rollout.conversations[-1].messages
    attempt = GradingAttempt(chat_conversation(list(messages)), stop_reason=rollout.stop_reason)
    try:
        return CAPABILITIES.validate_python(ANSWER_FORMAT.extract(attempt).value)
    except (SubmissionFailure, ValidationError) as error:
        final = str(messages[-1].get("content"))[:500]
        stop = rollout.stop_reason
        raise ValueError(
            f"The analysis answer (stop reason {stop!r}) is not a list of capabilities: {final!r}"
        ) from error


def _idea(capability: MissingCapability, catalog_version: str) -> CapabilityIdea:
    record = {
        "kind": CAPABILITY_KIND,
        "parent_id": SUBJECT_ID,
        "prerequisites": [],
        **capability.model_dump(mode="json"),
    }
    return CapabilityIdea(
        capability_id=capability.id,
        subject_id=SUBJECT_ID,
        subject_name=SUBJECT_NAME,
        catalog_version=catalog_version,
        capability=record,
        prerequisite_edges=(),
        capability_hash=digest(record),
    )
