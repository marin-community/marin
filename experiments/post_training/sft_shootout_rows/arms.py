# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-arm attempt selection for the SFT shootout (task #211); rules from the experts-methods matrix.

Eligible tasks: pool v2 tasks that are not held out and have at least one passing GLM attempt that fits the context.

- plain (RS-1): the first passing attempt per eligible task, by sample index. One row per task; defines B.
- RS-multi: the same task slots; in each epoch every slot draws one of its task's passing attempts with a seeded
  generator, without replacement until the task's passes are used up, then the pool refills. One row per slot per
  epoch.
- relay: passing relay episodes with at least one GLM turn, taken in a seeded random order (whole episodes) until
  their supervised tokens reach B; v0 turns carry zero loss.
"""

import random
from dataclasses import dataclass, field

SEED = 20261016


@dataclass
class Attempt:
    task_id: str
    family: str
    trial_id: str
    sample: int
    retry: int
    supervised: int
    context: int
    authors: list[str] = field(default_factory=list)


def first_passing(attempts: list[Attempt]) -> dict[str, Attempt]:
    """Plain/RS-1: per task, the passing attempt with the lowest sample index (ties: lowest retry, trial id)."""
    chosen: dict[str, Attempt] = {}
    for attempt in sorted(attempts, key=lambda a: (a.task_id, a.sample, a.retry, a.trial_id)):
        chosen.setdefault(attempt.task_id, attempt)
    return chosen


def rs_multi_epochs(attempts: list[Attempt], tasks: list[str], epochs: int) -> list[dict[str, Attempt]]:
    """For each epoch, {task: attempt}; per-task draws are seeded and avoid repeats until a task's passes run out."""
    by_task: dict[str, list[Attempt]] = {}
    for attempt in sorted(attempts, key=lambda a: (a.task_id, a.sample, a.retry, a.trial_id)):
        by_task.setdefault(attempt.task_id, []).append(attempt)
    remaining = {task: [] for task in tasks}
    plan: list[dict[str, Attempt]] = []
    for epoch in range(epochs):
        rng = random.Random(f"{SEED}:rs-multi:{epoch}")
        selection = {}
        for task in tasks:
            if not remaining[task]:
                remaining[task] = list(by_task[task])
            selection[task] = remaining[task].pop(rng.randrange(len(remaining[task])))
        plan.append(selection)
    return plan


def relay_subsample(episodes: list[Attempt], budget: int) -> tuple[list[Attempt], int]:
    """Whole episodes in a seeded order until supervised tokens reach `budget`; returns (episodes, total)."""
    order = sorted(episodes, key=lambda a: a.trial_id)
    random.Random(f"{SEED}:relay").shuffle(order)
    chosen, total = [], 0
    for episode in order:
        if total >= budget:
            break
        chosen.append(episode)
        total += episode.supervised
    return chosen, total
