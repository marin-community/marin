# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find the capabilities that the policy's failed rollouts lack, as catalog ideas for ``generate_tasks``.

``analyze_failed_runs`` uses two kinds of structured model calls, map and reduce:

1. One call for each failed rollout names the general skills whose absence explains the failure.
2. One call merges the skills of all the rollouts into at most ``max_capabilities`` capabilities, with the IDs of the
   skills that each capability holds. The rollouts of those skills are the evidence of the capability.

A capability describes the skill and not the failed tasks, so that ``generate_tasks`` writes the details of its
training tasks. The calls see only what the policy saw and wrote, never a grader or its answer.

The merge reply lists the ID of each skill, so it grows with the rollouts, and GLM sometimes runs away to the output
limit in a long structured reply. Thus a merge is sampled again after an invalid reply. When the merge of all the
skills still fails, the skills merge in two levels: one merge for each chunk of rollouts, then one merge of the lists.
"""

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rolloutengine.harness.shell import task_prompt
from rolloutengine.models import GradedRollout
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskSpec, TextMessage
from taskforge.content_hash import digest
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError

from experiments.post_training.capability_driven_envs.catalog import CAPABILITY_KIND, CapabilityIdea
from experiments.post_training.capability_driven_envs.generate_tasks import FILE_SAFE_ID

SUBJECT_ID = "failed-runs"
SUBJECT_NAME = "Capabilities that failed runs lack"
MAX_TASK_CHARS = 300
SKILLS_PER_ROLLOUT = 3
CONCURRENCY = 32
MAX_FAILED_SHARE = 0.1
"""The largest share of the failed rollouts whose skill call can fail before the analysis fails."""
MERGE_ATTEMPTS = 3
CHUNK_ROLLOUTS = 100
"""The rollouts whose skills one first-level merge reads, when the merge of all the skills fails."""

logger = logging.getLogger(__name__)

Lines = dict[str, tuple[frozenset[str], dict[str, object]]]
"""For each line ID of a merge: the run IDs that the line stands for, and the fields that the merge call sees."""


class Skill(BaseModel):
    """A general skill that failed rollouts lack."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="The skill in a few words, with no detail of a failed task.")
    outcome: str = Field(min_length=1, description="What an agent with the skill can do, in any domain.")
    includes: tuple[str, ...] = Field(min_length=1, description="The parts of the skill.")
    excludes: tuple[str, ...] = Field(default=(), description="Related skills that the capability does not cover.")


class Capability(Skill):
    """A skill with a catalog ID."""

    id: str = Field(description="A file-safe ID, for example failed-runs.tool-use.stop-repeating-calls.")

    @field_validator("id")
    @classmethod
    def file_safe(cls, value: str) -> str:
        if FILE_SAFE_ID.match(value) is None:
            raise ValueError(f"capability id {value!r} is not file-safe")
        return value


class MissingCapability(Capability):
    """A general skill that the failed rollouts lack, with the catalog's capability fields and the evidence."""

    evidence: tuple[str, ...] = Field(min_length=1, description="The run IDs of the failed rollouts that show the gap.")


class RolloutSkills(BaseModel):
    """The general skills whose absence explains one failed rollout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    skills: tuple[Skill, ...] = Field(min_length=1, max_length=SKILLS_PER_ROLLOUT)


class Group(Capability):
    """One merged capability, with the IDs of the lines that it holds."""

    members: tuple[str, ...] = Field(min_length=1, description="The IDs of the lines that the capability holds.")

    @model_validator(mode="after")
    def includes_behaviors(self) -> "Group":
        # GLM sometimes copies the member IDs into includes, and then generate_tasks gets no behaviors to practice.
        if set(self.includes) & set(self.members):
            raise ValueError("includes must name behaviors that the capability covers, not the IDs of its members")
        return self


class Groups(BaseModel):
    """The merged capabilities."""

    capabilities: tuple[Group, ...] = Field(min_length=1)


CAPABILITY_RULES = """\
A capability is a skill that transfers to many tasks and domains, for example: obey the answer format that a task \
asks for, select tools and their arguments, stop a loop that repeats a call or a mistake, read a task and its \
constraints carefully, do a check of the work before the answer, or reach an answer without unneeded steps.

- Write each capability as a catalog entry for the skill itself. Do not put the failed tasks, their domain, \
their data, or their mistakes in it: a different step writes new tasks that train the skill, and the failed \
tasks evaluate it.
- A rollout that stopped at its turn limit shows a gap only if it used its turns on repeated or unneeded steps. \
The agent does not see the limit unless its messages state it."""

SKILLS_PROMPT = f"""\
You analyze one failed rollout of a language-model agent to decide what the agent must learn next. A rollout is one \
attempt at a task: the task's messages, the agent's turns and tool calls, and the tool results.

Find the general capabilities, at most {SKILLS_PER_ROLLOUT}, whose absence explains the failure. {CAPABILITY_RULES}

The first line of the rollout is a JSON object with its run ID, task, grade, stop reason, and counts of turns and \
tool calls. The second line is a JSON object with the tools that the task gave. The policy's harness can also give a \
shell tool and the functions of the tool providers. After it, each message starts a new line with its number, for \
example [0]."""

MERGE_SKILLS_PROMPT = f"""\
Each line is a skill that one failed rollout of a language-model agent lacks, with the ID of the skill. Merge the \
skills into at most {{count}} capabilities: merge the skills that are about the same capability, and keep the \
capabilities that explain the most rollouts. Give the IDs of the skills that each capability holds. {CAPABILITY_RULES}"""

MERGE_LISTS_PROMPT = f"""\
Several analyses each read a part of the failed rollouts of a language-model agent, and each gave capabilities with \
the number of rollouts that show them. Merge them into at most {{count}} capabilities: merge the capabilities that \
are about the same skill, and keep the skills that explain the most rollouts. Give the IDs of the capabilities that \
each merged capability holds. {CAPABILITY_RULES}"""


def policy_failures(
    rollouts: Sequence[tuple[TaskSpec, GradedRollout]], pass_threshold: float
) -> list[tuple[TaskSpec, GradedRollout]]:
    """The rollouts that the policy failed: graded below ``pass_threshold``, or with no valid submission.

    An interrupted attempt, and an infrastructure, task, or grader failure, is not a failure of the policy, so it
    does not become a training target.
    """

    def failed(graded: GradedRollout) -> bool:
        grade = graded.grade
        if graded.rollout.failure is not None:
            return False
        if grade.status == Outcome.SUBMISSION_FAILURE:
            return True
        return grade.status == Outcome.GRADED and grade.reward is not None and grade.reward < pass_threshold

    return [(task, graded) for task, graded in rollouts if failed(graded)]


async def analyze_failed_runs(
    rollouts: Sequence[tuple[TaskSpec, GradedRollout]],
    client: ModelClient,
    *,
    pass_threshold: float,
    max_capabilities: int,
    policy: LLMPolicy,
    merge_policy: LLMPolicy,
    catalog_version: str,
) -> list[CapabilityIdea]:
    """The capabilities that the policy's failed rollouts lack, as ideas for ``generate_tasks``.

    Each graded rollout comes with the task that it ran, because the record does not keep the tools that the task
    gave. ``policy`` samples the call for each rollout, and ``merge_policy`` samples the merges, whose replies are
    longer. The capabilities with more rollouts come first. With no policy failure, no model call occurs.

    A rollout whose reply stays invalid after the client's repairs gives no skills. When more than ``MAX_FAILED_SHARE``
    of the rollouts give none, the analysis raises ``ValueError``. A two-level merge whose reply stays invalid raises
    ``StructuredOutputError``.
    """
    failures = policy_failures(rollouts, pass_threshold)
    if not failures:
        return []
    skills = await _rollout_skills(client, failures, policy)
    try:
        groups = await _merged(client, MERGE_SKILLS_PROMPT, _skill_lines(skills), max_capabilities, merge_policy)
    except StructuredOutputError as error:
        logger.warning("The merge of %d skills failed, so they merge in two levels: %s", len(skills), error)
        groups = await _merged_in_two_levels(client, skills, max_capabilities, merge_policy)
    ideas, seen = [], set()
    for group, evidence in sorted(groups, key=lambda pair: -len(pair[1])):
        if group.id in seen or len(ideas) == max_capabilities:
            continue
        seen.add(group.id)
        record = MissingCapability(**group.model_dump(exclude={"members"}), evidence=tuple(sorted(evidence)))
        ideas.append(_idea(record, catalog_version))
    return ideas


async def _rollout_skills(
    client: ModelClient, failures: Sequence[tuple[TaskSpec, GradedRollout]], policy: LLMPolicy
) -> list[tuple[str, Skill]]:
    """The run ID and each skill that one call names for each failed rollout, in the order of the rollouts."""
    slots = asyncio.Semaphore(CONCURRENCY)

    async def named(task: TaskSpec, graded: GradedRollout) -> tuple[Skill, ...] | None:
        rollout = f"{json.dumps(_summary(task, graded))}\n{_rollout_file(task, graded)}"
        messages: list[Message] = [
            {"role": "system", "content": SKILLS_PROMPT},
            {"role": "user", "content": rollout},
        ]
        try:
            async with slots:
                return (await client.complete_structured(messages, RolloutSkills, policy)).value.skills
        except StructuredOutputError as error:
            # One rollout that the model cannot answer for, for example at the output limit, does not stop the others.
            logger.warning("No skills for rollout %s: %s", graded.rollout.run_id, error)
            return None

    found = await asyncio.gather(*(named(task, graded) for task, graded in failures))
    missing = sum(skills is None for skills in found)
    if missing > MAX_FAILED_SHARE * len(failures):
        raise ValueError(f"The skill call failed for {missing} of {len(failures)} failed rollouts")
    return [
        (graded.rollout.run_id, skill)
        for (_, graded), skills in zip(failures, found, strict=True)
        for skill in skills or ()
    ]


def _skill_lines(skills: Sequence[tuple[str, Skill]]) -> Lines:
    return {
        f"k{i}": (frozenset({run_id}), {"name": skill.name, "outcome": skill.outcome})
        for i, (run_id, skill) in enumerate(skills)
    }


async def _merged_in_two_levels(
    client: ModelClient, skills: Sequence[tuple[str, Skill]], count: int, policy: LLMPolicy
) -> list[tuple[Group, frozenset[str]]]:
    run_ids = list(dict.fromkeys(run_id for run_id, _ in skills))
    chunks = -(-len(run_ids) // CHUNK_ROLLOUTS)
    chunk_of = {run_id: i % chunks for i, run_id in enumerate(run_ids)}
    found = await asyncio.gather(
        *(
            _merged(client, MERGE_SKILLS_PROMPT, _skill_lines([s for s in skills if chunk_of[s[0]] == i]), count, policy)
            for i in range(chunks)
        )
    )
    lines: Lines = {
        f"c{i}.{j}": (evidence, {**group.model_dump(exclude={"id", "members"}), "rollouts": len(evidence)})
        for i, groups in enumerate(found)
        for j, (group, evidence) in enumerate(groups)
    }
    return await _merged(client, MERGE_LISTS_PROMPT, lines, count, policy)


async def _merged(
    client: ModelClient, prompt: str, lines: Lines, count: int, policy: LLMPolicy
) -> list[tuple[Group, frozenset[str]]]:
    """Each merged capability, with the run IDs of the lines that it holds."""
    listing = "\n".join(json.dumps({"id": key, **fields}) for key, (_, fields) in lines.items())
    messages: list[Message] = [
        {"role": "system", "content": prompt.format(count=count)},
        {"role": "user", "content": listing},
    ]
    for attempt in range(1, MERGE_ATTEMPTS):
        try:
            return _with_evidence((await client.complete_structured(messages, Groups, policy)).value, lines)
        except StructuredOutputError as error:
            logger.warning("Merge attempt %d of %d failed: %s", attempt, MERGE_ATTEMPTS, error)
    return _with_evidence((await client.complete_structured(messages, Groups, policy)).value, lines)


def _with_evidence(reply: Groups, lines: Lines) -> list[tuple[Group, frozenset[str]]]:
    merged = []
    for group in reply.capabilities:
        evidence = frozenset().union(*(lines[key][0] for key in group.members if key in lines))
        if evidence:
            merged.append((group, evidence))
    return merged


def _summary(task: TaskSpec, graded: GradedRollout) -> dict[str, Any]:
    """The rollout's first line: its grade, how it stopped, and its turns and tool calls."""
    rollout, grade = graded.rollout, graded.grade
    turns = [step.turn.message for conversation in rollout.conversations for step in conversation.steps]
    calls = [_call_text(call) for message in turns for call in message.get("tool_calls") or ()]
    events = task.context.events
    text = " ".join(event.content for event in events if isinstance(event, TextMessage) and event.role == "user")
    return {
        "run_id": rollout.run_id,
        "task_id": task.id,
        "tags": list(task.tags),
        "outcome": grade.status.value,
        "reward": grade.reward,
        "stop_reason": rollout.stop_reason,
        "turns": len(turns),
        "tool_calls": len(calls),
        "repeated_calls": sum(previous == call for previous, call in pairwise(calls)),
        "task": text[:MAX_TASK_CHARS],
    }


def _rollout_file(task: TaskSpec, graded: GradedRollout) -> str:
    """A line with the tools that the task gave, then each message from a new line that starts with its number."""
    requirements = task.environment_requirements
    tools = {
        "tools": list(task_prompt(task, None, ()).options.get("tools", ())),
        "tool_providers": sorted(provider.action_interface for provider in requirements.tool_providers.values()),
        "environment_capabilities": list(requirements.capabilities),
    }
    messages = [message for conversation in graded.rollout.conversations for message in conversation.messages]
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
