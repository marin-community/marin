# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find the capabilities that the policy's failed rollouts lack, as catalog ideas for ``generate_tasks``.

``analyze_failed_runs`` gives the failed rollouts to one managed agent as files: a ``RolloutEngine`` attempt whose
shell harness runs the agent's commands on a ShellSim machine that has the files. The agent answers with the general
skills whose absence explains the failures, for example an answer format, the use of tools, a stop to loops of
repeated calls, or a careful reading of the task. A capability describes the skill and not the failed tasks, so that
``generate_tasks`` writes the details of its training tasks. Each capability becomes a ``CapabilityIdea`` for
``task_workpieces`` and ``generate_tasks``. The files have only what the policy saw and wrote, never a grader or its
answer.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.shell import ShellHarness, task_prompt
from rolloutengine.models import (
    GradedRollout,
    MachineRuntimeSpec,
    Model,
    Rollout,
    RolloutLimits,
    TaskRolloutSpec,
    TraceMode,
)
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, NetworkPolicy
from taskcompendium.chat import chat_conversation
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    GradingAttempt,
    JsonValueAnswer,
    NoGrader,
    ResourceGroups,
    Source,
    SubmissionFailure,
    TaskSpec,
    TextMessage,
)
from taskcompendium.runtime.resources import inline_resource
from taskforge.content_hash import digest

from experiments.post_training.capability_driven_envs.catalog import CAPABILITY_KIND, CapabilityIdea
from experiments.post_training.capability_driven_envs.generate_tasks import FILE_SAFE_ID

SUBJECT_ID = "failed-runs"
SUBJECT_NAME = "Capabilities that failed runs lack"
ANALYSIS_REVISION = "1"
HARNESS = "failed-run-analyst"
ANSWER_FORMAT = JsonValueAnswer()
MAX_TASK_CHARS = 300
WORKSPACE = "rollouts"
"""The directory of the rollout files, below the machine's root. It is the agent's working directory."""
INDEX = "index.jsonl"
MACHINE = MachineRuntimeSpec(
    backend=Backend.SHELLSIM,
    network=NetworkPolicy.DENY,
    cpus=None,
    memory_mb=None,
    storage_mb=None,
    gpus=0,
    user=None,
    startup_timeout=None,
    cleanup_timeout=None,
)


class MissingCapability(BaseModel):
    """A general skill that the failed rollouts lack, with the catalog's capability fields and the evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="A file-safe ID, for example failed-runs.tool-use.stop-repeating-calls.")
    name: str = Field(min_length=1, description="The skill in a few words, with no detail of a failed task.")
    outcome: str = Field(min_length=1, description="What an agent with the skill can do, in any domain.")
    evidence: tuple[str, ...] = Field(min_length=1, description="The run IDs of the failed rollouts that show the gap.")
    includes: tuple[str, ...] = Field(min_length=1, description="The parts of the skill.")
    excludes: tuple[str, ...] = Field(default=(), description="Related skills that the capability does not cover.")

    @field_validator("id")
    @classmethod
    def file_safe(cls, value: str) -> str:
        if FILE_SAFE_ID.match(value) is None:
            raise ValueError(f"capability id {value!r} is not file-safe")
        return value


CAPABILITIES = TypeAdapter(list[MissingCapability])

SYSTEM_PROMPT = """\
You analyze the failed rollouts of a language-model agent to decide what the agent must learn next. A rollout is \
one attempt at a task: the task's messages, the agent's turns and tool calls, and the tool results.

Find the general capabilities whose absence explains the failures. A capability is a skill that transfers to \
many tasks and domains, for example: obey the answer format that a task asks for, select tools and their \
arguments, stop a loop that repeats a call or a mistake, read a task and its constraints carefully, do a check \
of the work before the answer, or reach an answer without unneeded steps.

- Use the failed rollouts only as evidence. Read them, and give their run IDs as the evidence of each capability.
- Write each capability as a catalog entry for the skill itself. Do not put the failed tasks, their domain, \
their data, or their mistakes in it: a different step writes new tasks that train the skill, and the failed \
tasks evaluate it.
- Prefer a few general capabilities to many narrow ones. Merge failures that come from the same missing skill.
- A rollout that stopped at its turn limit shows a gap only if it used its turns on repeated or unneeded steps. \
The agent does not see the limit unless its messages state it.

The workspace has the file {index}, with one JSON object for each failed rollout: its run ID, task, grade, \
stop reason, and counts of turns and tool calls. Each failed rollout also has the file <run_id>.txt. Its first line \
is a JSON object with the tools that the task gave. The policy's harness can also give a shell tool and the \
functions of the tool providers. After the first line, each message starts a new line with its number, for example \
[0]. Files can be long, so read parts of them, for example with grep, jq, head, or sed. When you are done, answer \
without a tool call. The answer is one JSON array of capabilities that obeys this JSON schema:

{schema}
"""

USER_PROMPT = """\
{failed} of {total} rollouts failed. You can make at most {turns} model turns, so answer before the last one.
"""


@dataclass(frozen=True)
class EvaluatedRollout:
    """One evaluated attempt: the task that it ran and its graded rollout."""

    task: TaskSpec
    result: GradedRollout


def policy_failures(rollouts: Sequence[EvaluatedRollout], pass_threshold: float) -> list[EvaluatedRollout]:
    """The rollouts that the policy failed: graded below ``pass_threshold``, or with no valid submission.

    An interrupted attempt, and an infrastructure, task, or grader failure, is not a failure of the policy, so it
    does not become a training target.
    """

    def failed(rollout: EvaluatedRollout) -> bool:
        grade = rollout.result.grade
        if rollout.result.rollout.failure is not None:
            return False
        if grade.status == Outcome.SUBMISSION_FAILURE:
            return True
        return grade.status == Outcome.GRADED and grade.reward is not None and grade.reward < pass_threshold

    return [rollout for rollout in rollouts if failed(rollout)]


async def analyze_failed_runs(
    rollouts: Sequence[EvaluatedRollout],
    model: Model,
    *,
    pass_threshold: float,
    limits: RolloutLimits,
    catalog_version: str,
) -> list[CapabilityIdea]:
    """The capabilities that the policy's failed rollouts lack, as ideas for ``generate_tasks``.

    ``model`` drives the analysis agent under ``limits``, for example a ``rolloutengine.chat.ChatModel``. With no
    policy failure, no agent runs. A failed analysis raises ``GradedRolloutInterrupted``, and an answer that is not a
    valid list of capabilities raises ``ValueError``.
    """
    failures = policy_failures(rollouts, pass_threshold)
    if not failures:
        return []
    # NOTE: One agent reads all the failed rollouts. A later version can divide the rollouts into shards with Zephyr,
    # analyze the shards in parallel, each with its own agent, and then merge their capabilities with one more agent.
    engine = RolloutEngine({Backend.SHELLSIM: ShellSimMachineFactory()}, {HARNESS: ShellHarness(command_timeout=None)})
    spec = TaskRolloutSpec(
        task=_analysis_task(failures, len(rollouts), limits.max_turns, catalog_version),
        harness=HARNESS,
        task_machine=MACHINE,
        verifier_machine=None,
        limits=limits,
        trace=TraceMode.MESSAGES,
    )
    result = await engine.run_task(spec, model)
    capabilities = _answer(result.rollout)
    run_ids = {failure.result.rollout.run_id for failure in failures}
    unknown = sorted({run_id for capability in capabilities for run_id in capability.evidence} - run_ids)
    if unknown:
        raise ValueError(f"The analysis cites rollouts that did not fail: {unknown}")
    ids = [capability.id for capability in capabilities]
    if len(set(ids)) != len(ids):
        raise ValueError(f"The analysis repeats capability IDs: {ids}")
    return [_idea(capability, catalog_version) for capability in capabilities]


def _analysis_task(failures: Sequence[EvaluatedRollout], total: int, turns: int, catalog_version: str) -> TaskSpec:
    system = SYSTEM_PROMPT.format(index=INDEX, schema=json.dumps(CAPABILITIES.json_schema()))
    user = USER_PROMPT.format(failed=len(failures), total=total, turns=turns)
    index = "".join(f"{json.dumps(_summary(failure))}\n" for failure in failures)
    files = (
        inline_resource(f"{WORKSPACE}/{INDEX}", index.encode()),
        *(
            inline_resource(f"{WORKSPACE}/{failure.result.rollout.run_id}.txt", _rollout_file(failure).encode())
            for failure in failures
        ),
    )
    return TaskSpec(
        id=f"analyze-failed-runs-{catalog_version}",
        context=ConversationInput(
            events=(TextMessage(role="system", content=system), TextMessage(role="user", content=user))
        ),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics=CommandSemantics.SHELL_SIMULATOR,
            working_directory=f"/{WORKSPACE}",
        ),
        answer_type=AnswerType.JSON,
        answer_format=ANSWER_FORMAT,
        grader=NoGrader(reason="An analysis of failed rollouts has no reference answer"),
        source=Source(dataset=SUBJECT_ID, revision=catalog_version, row="0", importer_revision=ANALYSIS_REVISION),
        resources=ResourceGroups(worker=files),
    )


def _summary(failure: EvaluatedRollout) -> dict[str, Any]:
    """The rollout's line in the index: its grade, how it stopped, and its turns and tool calls."""
    rollout, grade = failure.result.rollout, failure.result.grade
    turns = [step.turn.message for conversation in rollout.conversations for step in conversation.steps]
    calls = [_call_text(call) for message in turns for call in message.get("tool_calls") or ()]
    events = failure.task.context.events
    task = " ".join(event.content for event in events if isinstance(event, TextMessage) and event.role == "user")
    return {
        "run_id": rollout.run_id,
        "task_id": failure.task.id,
        "tags": list(failure.task.tags),
        "outcome": grade.status.value,
        "reward": grade.reward,
        "stop_reason": rollout.stop_reason,
        "turns": len(turns),
        "tool_calls": len(calls),
        "repeated_calls": sum(previous == call for previous, call in pairwise(calls)),
        "task": task[:MAX_TASK_CHARS],
    }


def _rollout_file(failure: EvaluatedRollout) -> str:
    """A line with the tools that the task gave, then each message from a new line that starts with its number."""
    task = failure.task
    requirements = task.environment_requirements
    tools = {
        "tools": list(task_prompt(task, None, ()).options.get("tools", ())),
        "tool_providers": sorted(provider.action_interface for provider in requirements.tool_providers.values()),
        "environment_capabilities": list(requirements.capabilities),
    }
    rollout = failure.result.rollout
    messages = [message for conversation in rollout.conversations for message in conversation.messages]
    return "".join([f"{json.dumps(tools)}\n", *(f"[{n}] {_render(message)}\n" for n, message in enumerate(messages))])


def _call_text(call: Any) -> str:
    """A tool call as ``name(arguments)``. The record keeps the policy's message, so a malformed call stays raw JSON."""
    function = call.get("function") if isinstance(call, Mapping) else None
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return f"{function['name']}({function.get('arguments', '')})"
    return f"malformed {json.dumps(call, default=str)}"


def _render(message: Mapping[str, Any]) -> str:
    calls = [_call_text(call) for call in message.get("tool_calls") or ()]
    text = " ".join([str(message.get("content") or ""), *(f"<tool call> {call}" for call in calls)]).strip()
    return f"{message.get('role')}: {text}"


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
