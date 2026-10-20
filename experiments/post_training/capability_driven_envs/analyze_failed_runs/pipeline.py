# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Zephyr job that turns a policy's failed rollouts into capability ideas for ``generate_tasks``.

``analyze_failed_runs`` is one map-reduce pipeline:

1. The source is the rollouts in shards of ``shard_size``. The rollouts of a shard need not be similar.
2. ``filter`` keeps the rollouts that the policy failed, with no model call.
3. ``flat_map(yield_ideas)``: one structured model call for each failed rollout names the general skills whose
   absence explains the failure.
4. ``reduce``: in each shard, one structured call (``merge_ideas``) merges the ideas, and one more call merges the
   lists of the shards. Each merged idea names the ideas that it holds, and code joins their evidence.
5. ``flat_map(to_ideas)`` makes a ``CapabilityIdea`` of each merged idea that cites enough tasks, the ideas with more
   rollouts first.

A capability describes the skill and not the failed tasks, so that ``generate_tasks`` writes the details of its
training tasks. The calls see only what the policy saw and wrote, never a grader or its answer.
"""

import asyncio
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager
from functools import partial
from itertools import batched, chain

from pydantic import BaseModel
from rolloutengine.models import GradedRollout
from taskcompendium.models import TaskSpec
from taskforge.content_hash import digest
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError
from zephyr import counters
from zephyr.dataset import Dataset

from experiments.post_training.capability_driven_envs.analyze_failed_runs.models import (
    Capability,
    Groups,
    MaybeIdea,
    RolloutIdeas,
)
from experiments.post_training.capability_driven_envs.analyze_failed_runs.prompts import EXTRACT_PROMPT, MERGE_PROMPT
from experiments.post_training.capability_driven_envs.analyze_failed_runs.rollouts import policy_failed, rollout_text
from experiments.post_training.capability_driven_envs.catalog import CAPABILITY_KIND, CapabilityIdea

SUBJECT_ID = "failed-runs"
SUBJECT_NAME = "Capabilities that failed runs lack"
MERGE_ATTEMPTS = 3
ROLLOUTS_WITHOUT_IDEAS = "failed_runs.rollouts_without_ideas"
"""The pipeline counter of the failed rollouts whose reply stayed invalid."""

logger = logging.getLogger(__name__)

MakeClient = Callable[[], AbstractAsyncContextManager[ModelClient]]
"""Makes a model client, for example ``functools.partial(GlmClient, source, provider)``.

Zephyr calls the functions of a pipeline on its workers, outside an event loop, so each model call makes its own
client there and closes it after the call.
"""


def analyze_failed_runs(
    rollouts: Sequence[tuple[TaskSpec, GradedRollout]],
    make_client: MakeClient,
    *,
    pass_threshold: float,
    max_new_ideas: int,
    shard_size: int,
    extract_ideas_llm_policy: LLMPolicy,
    merge_ideas_llm_policy: LLMPolicy,
    catalog_version: str,
    min_tasks_per_idea: int = 2,
) -> Dataset[CapabilityIdea]:
    """The plan of a pipeline whose results are the capability ideas that the policy's failed rollouts lack.

    Each graded rollout comes with the task that it ran, because the record does not keep the tools that the task
    gave. ``extract_ideas_llm_policy`` samples the call for each rollout, and ``merge_ideas_llm_policy`` samples the
    merges, whose replies are longer. Each shard has ``shard_size`` rollouts: the calls of its rollouts run one after
    another on one worker, and its failed rollouts get one merge call. Shards run in parallel. Use at most about 1,000
    rollouts in a shard, because the merge prompt of a larger shard can be too long for the context window. Run the
    plan outside an event loop, for example ``ctx.execute(plan).results``.

    An idea must cite failed rollouts of at least ``min_tasks_per_idea`` tasks, or of every failed task when there are
    fewer. Otherwise one defective grader, on a task with many samples, can make an idea by itself.
    """
    failed_tasks = {task.id for task, graded in rollouts if policy_failed(graded, pass_threshold)}
    min_tasks = min(min_tasks_per_idea, len(failed_tasks))
    merge = partial(merge_ideas, make_client=make_client, max_new_ideas=max_new_ideas, policy=merge_ideas_llm_policy)
    return (
        Dataset.from_list(list(batched(rollouts, shard_size)))
        .flat_map(lambda shard: shard)
        .filter(lambda pair: policy_failed(pair[1], pass_threshold))
        .flat_map(partial(yield_ideas, make_client=make_client, policy=extract_ideas_llm_policy))
        .reduce(merge, partial(merge_idea_lists, merge=merge))
        .flat_map(partial(to_ideas, catalog_version=catalog_version, min_tasks=min_tasks))
    )


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
        {"role": "user", "content": rollout_text(task, graded)},
    ]
    try:
        reply = _complete(make_client, messages, RolloutIdeas, policy, attempts=1)
    except StructuredOutputError as error:
        # One rollout that the model cannot answer for, for example at the output limit, does not stop the others.
        logger.warning("No ideas for rollout %s: %s", graded.rollout.run_id, error)
        counters.pipeline.update_counter(ROLLOUTS_WITHOUT_IDEAS, 1)
        return
    for idea in reply.ideas:
        yield MaybeIdea(idea, frozenset({graded.rollout.run_id}), frozenset({task.id}))


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
        members = [lines[key] for key in group.members if key in lines]
        if members and group.id not in merged:
            idea = Capability(**group.model_dump(exclude={"members"}))
            evidence = frozenset().union(*(member.evidence for member in members))
            merged[group.id] = MaybeIdea(idea, evidence, frozenset().union(*(member.tasks for member in members)))
    return sorted(merged.values(), key=lambda candidate: -candidate.num_rollouts)[:max_new_ideas]


def merge_idea_lists(
    lists: Iterable[list[MaybeIdea]], *, merge: Callable[[Iterable[MaybeIdea]], list[MaybeIdea]]
) -> list[MaybeIdea]:
    """The global reducer: one more ``merge`` of the lists of the shards. One list needs no second merge."""
    full = [ideas for ideas in lists if ideas]
    return full[0] if len(full) == 1 else merge(chain.from_iterable(full))


def to_ideas(ideas: list[MaybeIdea], *, catalog_version: str, min_tasks: int) -> list[CapabilityIdea]:
    """The catalog idea of each merged idea that cites rollouts of at least ``min_tasks`` tasks.

    The capability record holds the run IDs of the evidence.
    """
    return [_idea(candidate, catalog_version) for candidate in ideas if candidate.num_tasks >= min_tasks]


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
