# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controls of gate 8 that focus training on the policy's failures without synthesis.

Both follow the training groups, as the loop's analysis does, and feed tasks back at the synthesized share:

- ``replay_failures``: a seed task that the policy passes at most ``max_pass_rate`` of the time comes back again
  (the hard-seed control);
- ``retrieve_similar``: for each such failure, the unseen seed task most similar to it (BM25 over the task text) comes
  next (the retrieval control, a data-selection baseline).

If either matches the loop, the loop's gain comes from focusing on failures, not from Taskforge's new tasks.
"""

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import AsyncIterable, AsyncIterator, Sequence

from taskcompendium.models import TaskSpec

from experiments.post_training.rsi.feedback import SYNTHESIZED_TAG, capability_of
from experiments.post_training.rsi.rollouts import TaskRollouts
from experiments.post_training.rsi.seed import PoolRow, row_id

REPLAY_CAPABILITY = "capability:replay"
RETRIEVAL_CAPABILITY = "capability:retrieval"
TOKEN = re.compile(r"[a-z0-9]+")
BM25_K1 = 1.5
BM25_B = 0.75
MAX_QUERY_TOKENS = 64


def failed_seed_task(group: TaskRollouts, max_pass_rate: float) -> bool:
    rate = group.pass_rate
    return capability_of(group.task) is None and rate is not None and rate <= max_pass_rate


def tagged(task: TaskSpec, task_id: str, capability: str) -> TaskSpec:
    return task.model_copy(update={"id": task_id, "tags": (*task.tags, SYNTHESIZED_TAG, capability)})


async def replay_failures(groups: AsyncIterable[TaskRollouts], max_pass_rate: float) -> AsyncIterator[TaskSpec]:
    """Each failed seed task once more, under a new ID."""
    replays = 0
    async for group in groups:
        if failed_seed_task(group, max_pass_rate):
            replays += 1
            yield tagged(group.task, f"{group.task.id}.replay-{replays}", REPLAY_CAPABILITY)


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def task_text(task: TaskSpec) -> str:
    return " ".join(str(getattr(event, "content", "")) for event in task.context.events)


class Retriever:
    """BM25 over the user text of the seed rows. A row is returned at most once, and never after the seed stream
    claimed it, so that no task trains twice."""

    def __init__(self, rows: Sequence[PoolRow]):
        self._rows = rows
        self._ids = [row_id(row.payload) for row in rows]
        self._postings: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths = []
        for index, row in enumerate(rows):
            counts = Counter(tokens(_row_text(row.payload)))
            self._lengths.append(sum(counts.values()))
            for token, count in counts.items():
                self._postings[token].append((index, count))
        self._mean_length = sum(self._lengths) / max(len(self._lengths), 1)
        self._claimed: set[str] = set()

    def claim(self, task_id: str) -> bool:
        """Whether ``task_id`` was still unused; it is used from now on."""
        if task_id in self._claimed:
            return False
        self._claimed.add(task_id)
        return True

    def nearest(self, text: str) -> int | None:
        """The unused row most similar to ``text``, now used; ``None`` if no row shares a token."""
        query = Counter(tokens(text)).most_common(MAX_QUERY_TOKENS)
        scores: defaultdict[int, float] = defaultdict(float)
        for token, _ in query:
            postings = self._postings.get(token, [])
            if not postings:
                continue
            idf = math.log(1 + (len(self._rows) - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, count in postings:
                norm = count + BM25_K1 * (1 - BM25_B + BM25_B * self._lengths[index] / self._mean_length)
                scores[index] += idf * count * (BM25_K1 + 1) / norm
        for index in sorted(scores, key=scores.__getitem__, reverse=True):
            if self.claim(self._ids[index]):
                return index
        return None


def _row_text(payload: str) -> str:
    task = json.loads(payload)
    return " ".join(str(event.get("content", "")) for event in task["context"]["events"])


async def retrieve_similar(
    groups: AsyncIterable[TaskRollouts], retriever: Retriever, pool_task, max_pass_rate: float
) -> AsyncIterator[TaskSpec]:
    """For each failed seed task, the most similar unused seed task, under a new ID; ``pool_task(index)`` reads it."""
    retrieved = 0
    async for group in groups:
        if not failed_seed_task(group, max_pass_rate):
            continue
        index = retriever.nearest(task_text(group.task))
        if index is None:
            continue
        retrieved += 1
        task = pool_task(index).task
        yield tagged(task, f"{task.id}.retrieved-{retrieved}", RETRIEVAL_CAPABILITY)
