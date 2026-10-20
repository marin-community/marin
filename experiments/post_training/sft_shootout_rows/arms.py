# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-arm attempt selection for the SFT shootout; rules from the experts-methods matrix.

Eligible tasks: pool tasks that are not held out and have at least one passing GLM attempt that fits the context.

- plain (RS-1): the first passing attempt per eligible task, by sample index. One row per task; defines B.
- RS-multi: the same task slots; in each epoch every slot draws one of its task's passing attempts with a seeded
  generator, without replacement until the task's passes are used up, then the pool refills. One row per slot per
  epoch.
- relay: passing relay episodes with at least one GLM turn. Each family's episodes are taken in their own seeded random
  order (whole episodes) until their supervised tokens reach that family's share of plain's B; v0 turns carry zero
  loss.
"""

import random
from dataclasses import dataclass, field

from experiments.post_training.sweep_data.select import SEED


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
    # Trained tokens inside think blocks (reasoning), for the fewest-reasoning selection.
    reasoning: int = 0


def first_passing(attempts: list[Attempt]) -> dict[str, Attempt]:
    """Plain/RS-1: per task, the passing attempt with the lowest sample index (ties: lowest retry, trial id)."""
    chosen: dict[str, Attempt] = {}
    for attempt in sorted(attempts, key=lambda a: (a.task_id, a.sample, a.retry, a.trial_id)):
        chosen.setdefault(attempt.task_id, attempt)
    return chosen


def fewest_reasoning(attempts: list[Attempt]) -> dict[str, Attempt]:
    """Per task, the passing attempt with the fewest trained reasoning tokens (ties: sample, retry, trial id)."""
    chosen: dict[str, Attempt] = {}
    for attempt in sorted(attempts, key=lambda a: (a.task_id, a.reasoning, a.sample, a.retry, a.trial_id)):
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


def relay_subsample(episodes: list[Attempt], budgets: dict[str, int]) -> tuple[list[Attempt], dict[str, int]]:
    """Per family, whole episodes in a seeded order until supervised tokens reach the family's budget.

    Returns the chosen episodes (family by family) and each family's total; a total overshoots its budget by less
    than one episode. A family whose pool is below its budget takes every episode."""
    chosen, totals = [], {}
    for family, budget in sorted(budgets.items()):
        order = sorted((a for a in episodes if a.family == family), key=lambda a: a.trial_id)
        random.Random(f"{SEED}:relay:{family}").shuffle(order)
        totals[family] = 0
        for episode in order:
            if totals[family] >= budget:
                break
            chosen.append(episode)
            totals[family] += episode.supervised
    return chosen, totals


def match_plain_to_relay(
    chosen: dict[str, Attempt], plain_b: dict[str, int], relay_totals: dict[str, int]
) -> dict[str, Attempt]:
    """Fallback for a family whose relay pool ends below plain's B for it: keep plain's rows of that family in a seeded
    task order (whole rows) until their supervised tokens reach relay's total; other families keep every row."""
    kept = {task: a for task, a in chosen.items() if relay_totals.get(a.family, plain_b[a.family]) >= plain_b[a.family]}
    for family in sorted(f for f in plain_b if relay_totals.get(f, plain_b[f]) < plain_b[f]):
        order = sorted(task for task, a in chosen.items() if a.family == family)
        random.Random(f"{SEED}:plain-match:{family}").shuffle(order)
        total = 0
        for task in order:
            if total >= relay_totals[family]:
                break
            kept[task] = chosen[task]
            total += chosen[task].supervised
    return kept
