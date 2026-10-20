# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn failed policy rollouts into capability ideas for task generation."""

import asyncio
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from functools import partial
from itertools import chain, pairwise
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rolloutengine.models import GradedRollout
from taskcompendium.grading_result import Outcome
from taskforge.content_hash import digest
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError
from zephyr import counters
from zephyr.dataset import Dataset

from experiments.post_training.capability_driven_envs.catalog import CAPABILITY_KIND, CapabilityIdea
from experiments.post_training.capability_driven_envs.generate_tasks import FILE_SAFE_ID

SUBJECT_ID = "failed-runs"
SUBJECT_NAME = "Capabilities that failed runs lack"
MAX_TASK_CHARS = 300
IDEAS_PER_ROLLOUT = 3
MERGE_ATTEMPTS = 3
ROLLOUTS_WITHOUT_IDEAS = "failed_runs.rollouts_without_ideas"
"""The pipeline counter of the failed rollouts whose reply stayed invalid."""

logger = logging.getLogger(__name__)

MakeClient = Callable[[], AbstractAsyncContextManager[ModelClient]]
"""Makes a model client, for example ``functools.partial(GlmClient, source, provider)``.

Zephyr calls the functions of a pipeline on its workers, outside an event loop, so each model call makes its own
client there and closes it after the call.
"""


class Capability(BaseModel):
    """A general skill that failed rollouts lack, with a catalog ID."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, description="The skill in a few words, with no detail of a failed task.")
    outcome: str = Field(min_length=1, description="What an agent with the skill can do, in any domain.")
    includes: tuple[str, ...] = Field(min_length=1, description="The parts of the skill.")
    excludes: tuple[str, ...] = Field(default=(), description="Related skills that the capability does not cover.")
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


class RolloutIdeas(BaseModel):
    """The general skills whose absence explains one failed rollout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ideas: tuple[Capability, ...] = Field(min_length=1, max_length=IDEAS_PER_ROLLOUT)


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


@dataclass(frozen=True)
class MaybeIdea:
    """A capability idea with the failed rollouts that show its gap, before the merges end."""

    idea: Capability
    evidence: frozenset[str]
    """The run IDs. A merged idea holds the union of the evidence of its ideas, so a rollout counts once."""

    @property
    def num_rollouts(self) -> int:
        return len(self.evidence)


CAPABILITY_RULES = """\
A capability is a skill that transfers to many tasks and domains, for example: obey the answer format that a task \
asks for, select tools and their arguments, stop a loop that repeats a call or a mistake, read a task and its \
constraints carefully, do a check of the work before the answer, or reach an answer without unneeded steps.

- Write each capability as a catalog entry for the skill itself. Do not put the failed tasks, their domain, \
their data, or their mistakes in it: a different step writes new tasks that train the skill, and the failed \
tasks evaluate it.
- A rollout that stopped at its turn limit shows a gap only if it used its turns on repeated or unneeded steps. \
The agent does not see the limit unless its messages state it."""

EXTRACT_PROMPT = f"""\
You analyze one failed rollout of a language-model agent to decide what the agent must learn next. A rollout is one \
attempt at a task: the task's messages, the agent's turns and tool calls, and the tool results.

Find the general capabilities, at most {IDEAS_PER_ROLLOUT}, whose absence explains the failure. {CAPABILITY_RULES}

The first line of the rollout is a JSON object with its run ID, task messages, grade, stop reason, and counts of \
turns and tool calls. Each conversation starts with a JSON line containing its model options, including the tools \
it saw and its tool choice. Its messages follow, each starting with a number such as [0]."""

MERGE_PROMPT = f"""\
Each line is a capability that failed rollouts of a language-model agent lack, with the ID of the line and the \
number of rollouts that show the gap. Merge the capabilities into at most {{count}}: merge the capabilities that are \
about the same skill, and keep the capabilities that explain the most rollouts. Give the IDs of the lines that each \
merged capability holds. {CAPABILITY_RULES}"""


def analyze_failed_runs(
    rollouts: Dataset[GradedRollout],
    make_client: MakeClient,
    *,
    pass_threshold: float,
    max_new_ideas: int,
    num_shards: int,
    policy: LLMPolicy,
    merge_policy: LLMPolicy,
    catalog_version: str,
) -> Dataset[CapabilityIdea]:
    """The plan of a pipeline whose results are the capability ideas that the policy's failed rollouts lack.

    Each conversation record contains the model options and messages that the policy saw. ``policy`` samples the
    call for each rollout, and ``merge_policy`` samples the merges, whose replies are longer. Each of the
    ``num_shards`` groups gets one merge call. Choose it so that a group has at most about 1,000
    failed rollouts, because the merge prompt of a larger group can be too long for the context window. Run the plan
    outside an event loop, for example ``ctx.execute(plan).results``.
    """
    merge = partial(merge_ideas, make_client=make_client, max_new_ideas=max_new_ideas, policy=merge_policy)
    return (
        rollouts.filter(lambda graded: policy_failed(graded, pass_threshold))
        .flat_map(partial(yield_ideas, make_client=make_client, policy=policy))
        .reshard(num_shards)
        .reduce(merge, partial(merge_idea_lists, merge=merge))
        .flat_map(partial(to_ideas, catalog_version=catalog_version))
    )


def policy_failed(graded: GradedRollout, pass_threshold: float) -> bool:
    """A grade below ``pass_threshold``, or no valid submission.

    An interrupted attempt, and an infrastructure, task, or grader failure, is not a failure of the policy, so it
    does not become a training target.
    """
    grade = graded.grade
    if graded.rollout.failure is not None:
        return False
    if grade.status == Outcome.SUBMISSION_FAILURE:
        return True
    return grade.status == Outcome.GRADED and grade.reward is not None and grade.reward < pass_threshold


def yield_ideas(graded: GradedRollout, *, make_client: MakeClient, policy: LLMPolicy) -> Iterator[MaybeIdea]:
    """The ideas that one structured call finds in one failed rollout.

    A reply that stays invalid gives no ideas and adds 1 to the pipeline counter ``ROLLOUTS_WITHOUT_IDEAS``, which the
    execution result reports.
    """
    messages: list[Message] = [
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": f"{json.dumps(_summary(graded))}\n{_rollout_text(graded)}"},
    ]
    try:
        reply = _complete(make_client, messages, RolloutIdeas, policy, attempts=1)
    except StructuredOutputError as error:
        # One rollout that the model cannot answer for, for example at the output limit, does not stop the others.
        logger.warning("No ideas for rollout %s: %s", graded.rollout.run_id, error)
        counters.pipeline.update_counter(ROLLOUTS_WITHOUT_IDEAS, 1)
        return
    for idea in reply.ideas:
        yield MaybeIdea(idea, frozenset({graded.rollout.run_id}))


def merge_ideas(
    ideas: Iterable[MaybeIdea], *, make_client: MakeClient, max_new_ideas: int, policy: LLMPolicy
) -> list[MaybeIdea]:
    """At most ``max_new_ideas`` ideas that one structured call merges from ``ideas``, with more rollouts first.

    The merge reply lists the ID of each line, so it grows with the ideas, and GLM sometimes runs away to the output
    limit in a long structured reply. Thus a merge makes up to ``MERGE_ATTEMPTS`` calls until a reply is valid.
    """
    lines = {f"k{i}": candidate for i, candidate in enumerate(ideas)}
    if not lines:
        return []
    listing = "\n".join(
        json.dumps(
            {
                "id": key,
                "name": candidate.idea.name,
                "outcome": candidate.idea.outcome,
                "rollouts": candidate.num_rollouts,
            }
        )
        for key, candidate in lines.items()
    )
    messages: list[Message] = [
        {"role": "system", "content": MERGE_PROMPT.format(count=max_new_ideas)},
        {"role": "user", "content": listing},
    ]
    reply = _complete(make_client, messages, Groups, policy, attempts=MERGE_ATTEMPTS)
    merged: dict[str, MaybeIdea] = {}
    for group in reply.capabilities:
        evidence = frozenset().union(*(lines[key].evidence for key in group.members if key in lines))
        if evidence and group.id not in merged:
            merged[group.id] = MaybeIdea(Capability(**group.model_dump(exclude={"members"})), evidence)
    return sorted(merged.values(), key=lambda candidate: -candidate.num_rollouts)[:max_new_ideas]


def merge_idea_lists(
    lists: Iterable[list[MaybeIdea]], *, merge: Callable[[Iterable[MaybeIdea]], list[MaybeIdea]]
) -> list[MaybeIdea]:
    """The global reducer: one more ``merge`` of the lists of the groups. One list needs no second merge."""
    full = [ideas for ideas in lists if ideas]
    return full[0] if len(full) == 1 else merge(chain.from_iterable(full))


def to_ideas(ideas: list[MaybeIdea], *, catalog_version: str) -> list[CapabilityIdea]:
    records = (
        MissingCapability(**candidate.idea.model_dump(), evidence=tuple(sorted(candidate.evidence)))
        for candidate in ideas
    )
    return [_idea(record, catalog_version) for record in records]


def _complete[T: BaseModel](
    make_client: MakeClient, messages: list[Message], output_type: type[T], policy: LLMPolicy, *, attempts: int
) -> T:
    async def complete() -> T:
        async with make_client() as client:
            for attempt in range(1, attempts):
                try:
                    return (await client.complete_structured(messages, output_type, policy)).value
                except StructuredOutputError as error:
                    logger.warning("Attempt %d of %d failed: %s", attempt, attempts, error)
            return (await client.complete_structured(messages, output_type, policy)).value

    return asyncio.run(complete())


def _summary(graded: GradedRollout) -> dict[str, Any]:
    """The rollout's first line: its grade, how it stopped, and its turns and tool calls."""
    rollout, grade = graded.rollout, graded.grade
    turns = [step.turn.message for conversation in rollout.conversations for step in conversation.steps]
    calls = [_call_text(call) for message in turns for call in message.get("tool_calls") or ()]
    text = " ".join(
        str(message.get("content") or "")
        for conversation in rollout.conversations
        for message in conversation.messages
        if message.get("role") == "user"
    )
    return {
        "run_id": rollout.run_id,
        "outcome": grade.status.value,
        "reward": grade.reward,
        "stop_reason": rollout.stop_reason,
        "turns": len(turns),
        "tool_calls": len(calls),
        "repeated_calls": sum(previous == call for previous, call in pairwise(calls)),
        "task": text[:MAX_TASK_CHARS],
    }


def _rollout_text(graded: GradedRollout) -> str:
    """Render each conversation's model options and numbered messages for the analysis call."""
    lines = []
    message_index = 0
    for conversation_index, conversation in enumerate(graded.rollout.conversations):
        header = {"conversation": conversation_index, "options": conversation.options}
        lines.append(f"{json.dumps(header)}\n")
        for message in conversation.messages:
            lines.append(f"[{message_index}] {_render(message)}\n")
            message_index += 1
    return "".join(lines)


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
