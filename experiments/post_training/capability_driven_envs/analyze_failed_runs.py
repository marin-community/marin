# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find the capabilities that the policy's failed rollouts lack, as catalog ideas for ``generate_tasks``.

``analyze_failed_runs`` is one Zephyr map-reduce pipeline:

1. The source is the rollouts in batches, one shard for each batch. The rollouts of a batch need not be similar.
2. ``filter`` keeps the rollouts that the policy failed, with no model call.
3. ``flat_map(yield_ideas)``: one structured model call for each failed rollout names the general skills whose
   absence explains the failure.
4. ``reduce``: in each batch, one structured call (``merge_ideas``) merges the ideas, and one more call merges the
   lists of the batches. Each merged idea names the ideas that it holds, and code joins their evidence.
5. ``flat_map(to_ideas)`` makes a ``CapabilityIdea`` of each merged idea, the ideas with more rollouts first.

A capability describes the skill and not the failed tasks, so that ``generate_tasks`` writes the details of its
training tasks. The calls see only what the policy saw and wrote, never a grader or its answer.
"""

import asyncio
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from functools import partial
from itertools import batched, chain, pairwise
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rolloutengine.harness.shell import task_prompt
from rolloutengine.models import GradedRollout, RolloutOperation
from taskcompendium.grading_result import Outcome
from taskcompendium.models import TaskSpec, TextMessage
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
FAILURE_OPERATION = "operation"
"""The key of ``RolloutFailure.diagnostics`` that names the interrupted operation, as rolloutengine records it."""
POLICY_INTERRUPTIONS = frozenset({RolloutOperation.ATTEMPT, RolloutOperation.ADVANCE})
"""The interruptions that the policy can cause: a slow or looping policy runs past the attempt deadline, and a bad
tool call can stop the harness after a model turn."""

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
- A rollout that stopped at its turn limit or its time limit shows a gap only if it used its turns or its time on \
repeated or unneeded steps. The agent does not see the limits unless its messages state them."""

EXTRACT_PROMPT = f"""\
You analyze one failed rollout of a language-model agent to decide what the agent must learn next. A rollout is one \
attempt at a task: the task's messages, the agent's turns and tool calls, and the tool results.

Find the general capabilities, at most {IDEAS_PER_ROLLOUT}, whose absence explains the failure. {CAPABILITY_RULES}

The first line of the rollout is a JSON object with its run ID, task, grade, stop reason, the operation that \
interrupted it if any, and counts of turns and tool calls. The second line is a JSON object with the tools that the \
task gave. The policy's harness can also give a shell tool and the functions of the tool providers. After it, each \
message starts a new line with its number, for example [0]."""

MERGE_PROMPT = f"""\
Each line is a capability that failed rollouts of a language-model agent lack, with the ID of the line and the \
number of rollouts that show the gap. Merge the capabilities into at most {{count}}: merge the capabilities that are \
about the same skill, and keep the capabilities that explain the most rollouts. Give the IDs of the lines that each \
merged capability holds. {CAPABILITY_RULES}"""


def analyze_failed_runs(
    rollouts: Sequence[tuple[TaskSpec, GradedRollout]],
    make_client: MakeClient,
    *,
    pass_threshold: float,
    max_new_ideas: int,
    batch_size: int,
    policy: LLMPolicy,
    merge_policy: LLMPolicy,
    catalog_version: str,
) -> Dataset[CapabilityIdea]:
    """The plan of a pipeline whose results are the capability ideas that the policy's failed rollouts lack.

    Each graded rollout comes with the task that it ran, because the record does not keep the tools that the task
    gave. ``policy`` samples the call for each rollout, and ``merge_policy`` samples the merges, whose replies are
    longer. Each batch of ``batch_size`` rollouts is one shard: the calls of its rollouts run one after another on one
    worker, and its failed rollouts get one merge call. Batches run in parallel. Use at most about 1,000 rollouts in
    a batch, because the merge prompt of a larger batch can be too long for the context window. Run the plan outside
    an event loop, for example ``ctx.execute(plan).results``.
    """
    merge = partial(merge_ideas, make_client=make_client, max_new_ideas=max_new_ideas, policy=merge_policy)
    return (
        Dataset.from_list(list(batched(rollouts, batch_size)))
        .flat_map(lambda batch: batch)
        .filter(lambda pair: policy_failed(pair[1], pass_threshold))
        .flat_map(partial(yield_ideas, make_client=make_client, policy=policy))
        .reduce(merge, partial(merge_idea_lists, merge=merge))
        .flat_map(partial(to_ideas, catalog_version=catalog_version))
    )


def policy_failed(graded: GradedRollout, pass_threshold: float) -> bool:
    """A grade below ``pass_threshold``, no valid submission, or an interruption in ``POLICY_INTERRUPTIONS``.

    A task, grader, or infrastructure failure, and an interruption by the model server or the machine, shows no gap
    of the policy, so it does not become a training target.
    """
    failure = graded.rollout.failure
    if failure is not None:
        return failure.diagnostics.get(FAILURE_OPERATION) in POLICY_INTERRUPTIONS
    grade = graded.grade
    if grade.status == Outcome.SUBMISSION_FAILURE:
        return True
    return grade.status == Outcome.GRADED and grade.reward is not None and grade.reward < pass_threshold


def yield_ideas(
    rollout: tuple[TaskSpec, GradedRollout], *, make_client: MakeClient, policy: LLMPolicy
) -> Iterator[MaybeIdea]:
    """The ideas that one structured call finds in one failed rollout.

    A reply that stays invalid gives no ideas and adds 1 to the pipeline counter ``ROLLOUTS_WITHOUT_IDEAS``, which the
    execution result reports.
    """
    task, graded = rollout
    messages: list[Message] = [
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": f"{json.dumps(_summary(task, graded))}\n{_rollout_text(task, graded)}"},
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
    """The global reducer: one more ``merge`` of the lists of the batches. One list needs no second merge."""
    full = [ideas for ideas in lists if ideas]
    return full[0] if len(full) == 1 else merge(chain.from_iterable(full))


def to_ideas(ideas: list[MaybeIdea], *, catalog_version: str) -> list[CapabilityIdea]:
    """The catalog idea of each merged idea, with the run IDs of its evidence in the capability record."""
    return [_idea(candidate, catalog_version) for candidate in ideas]


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


def _summary(task: TaskSpec, graded: GradedRollout) -> dict[str, Any]:
    """The rollout's first line: its grade, how it stopped or what interrupted it, and its turns and tool calls."""
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
        "interrupted_during": rollout.failure.diagnostics.get(FAILURE_OPERATION) if rollout.failure else None,
        "turns": len(turns),
        "tool_calls": len(calls),
        "repeated_calls": sum(previous == call for previous, call in pairwise(calls)),
        "task": text[:MAX_TASK_CHARS],
    }


def _rollout_text(task: TaskSpec, graded: GradedRollout) -> str:
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


def _idea(candidate: MaybeIdea, catalog_version: str) -> CapabilityIdea:
    record = {
        "kind": CAPABILITY_KIND,
        "parent_id": SUBJECT_ID,
        "prerequisites": [],
        **candidate.idea.model_dump(mode="json"),
        "evidence": sorted(candidate.evidence),
    }
    return CapabilityIdea(
        capability_id=candidate.idea.id,
        subject_id=SUBJECT_ID,
        subject_name=SUBJECT_NAME,
        catalog_version=catalog_version,
        capability=record,
        prerequisite_edges=(),
        capability_hash=digest(record),
    )
