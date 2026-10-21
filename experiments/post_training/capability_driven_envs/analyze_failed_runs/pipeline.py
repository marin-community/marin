# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn failed policy rollouts into capability ideas for task generation."""

import asyncio
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager
from functools import partial
from itertools import batched, chain

from pydantic import BaseModel
from rolloutengine.models import GradedRollout
from taskforge.content_hash import digest
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import StructuredOutputError
from zephyr import counters
from zephyr.dataset import Dataset

from experiments.post_training.capability_driven_envs.analyze_failed_runs.models import (
    Capability,
    MaybeIdea,
    RolloutIdeas,
    groups_of,
)
from experiments.post_training.capability_driven_envs.analyze_failed_runs.prompts import EXTRACT_PROMPT, MERGE_PROMPT
from experiments.post_training.capability_driven_envs.analyze_failed_runs.rollouts import policy_failed, rollout_text
from experiments.post_training.capability_driven_envs.catalog import CAPABILITY_KIND, CapabilityIdea

SUBJECT_ID = "failed-runs"
SUBJECT_NAME = "Capabilities that failed runs lack"
MERGE_ATTEMPTS = 3
INVALID_REPLIES = "failed_runs.invalid_replies"
"""The pipeline counter of the failed rollouts whose reply stayed invalid."""
DROPPED_ROLLOUTS = "failed_runs.dropped_rollouts"
"""The pipeline counter of the failed rollouts that no merged idea cites after a merge."""

logger = logging.getLogger(__name__)

MakeClient = Callable[[], AbstractAsyncContextManager[ModelClient]]
"""Factory for an async context manager that yields a model client."""


def analyze_failed_runs(
    rollouts: Sequence[GradedRollout],
    make_client: MakeClient,
    *,
    pass_threshold: float,
    max_new_ideas: int,
    shard_size: int,
    extract_ideas_llm_policy: LLMPolicy,
    merge_ideas_llm_policy: LLMPolicy,
    catalog_version: str,
    min_tasks_per_idea: int = 2,
    max_ideas_per_merge: int = 1000,
) -> Dataset[CapabilityIdea]:
    """Build a pipeline of capability ideas from the policy's failed rollouts.

    Recorded conversations supply the model options and messages seen by the policy. Each shard has ``shard_size``
    rollouts. Merge calls see at most ``max_ideas_per_merge`` ideas, with larger sets merged in rounds. Run the plan
    outside an event loop, for example ``ctx.execute(plan).results``.

    An idea must cite failed rollouts of at least ``min_tasks_per_idea`` tasks, or of every failed task when there are
    fewer. Otherwise one defective grader, on a task with many samples, can make an idea by itself. The merges rank the
    ideas of fewer tasks last, so that such an idea does not take the place of an idea that cites enough tasks.
    """
    if 2 * max_new_ideas > max_ideas_per_merge:
        raise ValueError(
            "max_ideas_per_merge must be at least 2 * max_new_ideas, so that each round of merges has fewer ideas"
        )
    failed_tasks = {graded.rollout.task_id for graded in rollouts if policy_failed(graded, pass_threshold)}
    min_tasks = min(min_tasks_per_idea, len(failed_tasks))
    merge = partial(
        merge_ideas,
        make_client=make_client,
        max_new_ideas=max_new_ideas,
        max_ideas_per_merge=max_ideas_per_merge,
        min_tasks=min_tasks,
        policy=merge_ideas_llm_policy,
    )
    return (
        Dataset.from_list(list(batched(rollouts, shard_size)))
        .flat_map(lambda shard: shard)
        .filter(lambda graded: policy_failed(graded, pass_threshold))
        .flat_map(partial(yield_ideas, make_client=make_client, policy=extract_ideas_llm_policy))
        .reduce(merge, partial(merge_idea_lists, merge=merge))
        .flat_map(partial(to_ideas, catalog_version=catalog_version, min_tasks=min_tasks))
    )


def yield_ideas(graded: GradedRollout, *, make_client: MakeClient, policy: LLMPolicy) -> Iterator[MaybeIdea]:
    """Yield ideas from one failed rollout.

    A reply that stays invalid also gives no ideas, and adds 1 to the pipeline counter ``INVALID_REPLIES``, which the
    execution result reports.
    """
    messages: list[Message] = [
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": rollout_text(graded)},
    ]
    try:
        reply = _complete(make_client, messages, RolloutIdeas, policy, attempts=1)
    except StructuredOutputError as error:
        # One rollout that the model cannot answer for, for example at the output limit, does not stop the others.
        logger.warning("No ideas for rollout %s: %s", graded.rollout.run_id, error)
        counters.pipeline.update_counter(INVALID_REPLIES, 1)
        return
    for idea in reply.ideas:
        yield MaybeIdea(idea, frozenset({graded.rollout.run_id}), frozenset({graded.rollout.task_id}))


def merge_ideas(
    ideas: Iterable[MaybeIdea],
    *,
    make_client: MakeClient,
    max_new_ideas: int,
    max_ideas_per_merge: int,
    min_tasks: int,
    policy: LLMPolicy,
) -> list[MaybeIdea]:
    """At most ``max_new_ideas`` ideas that structured calls merge from ``ideas``, in the order of ``_merged_batch``.

    One call merges at most ``max_ideas_per_merge`` ideas. Thus more ideas merge in rounds: each round merges batches
    of that size, and the next round merges the results. Ideas that cite enough tasks rank ahead of the rest, then
    ideas with more rollouts. The pipeline counter ``DROPPED_ROLLOUTS`` adds rollouts the merged ideas no longer cite.
    """
    merge = partial(
        _merged_batch, make_client=make_client, max_new_ideas=max_new_ideas, min_tasks=min_tasks, policy=policy
    )
    candidates = list(ideas)
    rollouts = frozenset().union(*(candidate.evidence for candidate in candidates))
    while len(candidates) > max_ideas_per_merge:
        candidates = list(chain.from_iterable(map(merge, batched(candidates, max_ideas_per_merge))))
    merged = merge(candidates)
    cited = frozenset().union(*(candidate.evidence for candidate in merged))
    counters.pipeline.update_counter(DROPPED_ROLLOUTS, len(rollouts - cited))
    return merged


def _merged_batch(
    ideas: Sequence[MaybeIdea], *, make_client: MakeClient, max_new_ideas: int, min_tasks: int, policy: LLMPolicy
) -> list[MaybeIdea]:
    """Return at most ``max_new_ideas`` ideas, ranking enough-task evidence before rollout count.

    Make at most ``MERGE_ATTEMPTS`` model calls when replies are invalid.
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
    reply = _complete(make_client, messages, groups_of(frozenset(lines)), policy, attempts=MERGE_ATTEMPTS)
    merged = [
        MaybeIdea(
            Capability(**group.model_dump(exclude={"members"})),
            frozenset().union(*(lines[key].evidence for key in group.members)),
            frozenset().union(*(lines[key].tasks for key in group.members)),
        )
        for group in reply.capabilities
    ]
    return sorted(merged, key=lambda candidate: (candidate.num_tasks < min_tasks, -candidate.num_rollouts))[
        :max_new_ideas
    ]


def merge_idea_lists(
    lists: Iterable[list[MaybeIdea]], *, merge: Callable[[Iterable[MaybeIdea]], list[MaybeIdea]]
) -> list[MaybeIdea]:
    """Merge nonempty groups into one list of ideas."""
    full = [ideas for ideas in lists if ideas]
    return full[0] if len(full) == 1 else merge(chain.from_iterable(full))


def to_ideas(ideas: list[MaybeIdea], *, catalog_version: str, min_tasks: int) -> list[CapabilityIdea]:
    """Keep merged ideas with evidence from at least ``min_tasks`` tasks."""
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
