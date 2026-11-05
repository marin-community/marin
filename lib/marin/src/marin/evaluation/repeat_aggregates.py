# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project repeated evaluator scores onto independent benchmark items."""

import logging
import math

from marin.evaluation.lm_eval_samples import is_scratch_artifact
from marin.evaluation.records import EvalRunRecord, RepeatAggregate, TaskCoverage

ITEM_BERNOULLI_STDERR = "item_bernoulli_n_minus_1"
logger = logging.getLogger(__name__)


def _item_count(count: int | None, items: int, repeats: int, *, benchmark_size: bool = False) -> int | None:
    if count is None or count <= items or (benchmark_size and count != items * repeats):
        return count
    if count > (items - 1) * repeats:
        return items
    return count // repeats if count % repeats == 0 else None


def _item_coverage(source: TaskCoverage | None, items: int, repeats: int) -> TaskCoverage | None:
    if source is None:
        return TaskCoverage(n_scored=items)

    trials = items * repeats
    fully_scored_items = source.n_scored > (items - 1) * repeats
    if source.n_scored not in (0, items, trials) and not fully_scored_items:
        return None
    if source.n_scored == 0 and source.n_attempted not in (items, trials):
        return None
    attempted = _item_count(source.n_attempted, items, repeats)
    if source.n_attempted is not None and (attempted is None or attempted < items):
        return None
    # Trial-level unanswered counts cannot identify partly unanswered items. Every trial
    # unanswered does establish that every represented item lacked an extractable answer.
    if source.n_scored == items:
        unanswered = source.n_unanswered
    elif source.n_scored > 0 and source.n_unanswered == source.n_scored:
        unanswered = items
    else:
        unanswered = 0
    return TaskCoverage(
        n_benchmark=_item_count(source.n_benchmark, items, repeats, benchmark_size=True),
        n_attempted=attempted,
        n_scored=items,
        n_unanswered=unanswered,
    )


def normalize_repeat_record(record: EvalRunRecord) -> EvalRunRecord:
    """Materialize campaign-policy uncertainty and item coverage without altering the source record."""
    coverage = dict(record.coverage)
    aggregates = dict(record.repeat_aggregates)
    for task, metrics in record.metrics.items():
        if is_scratch_artifact(f"{task}/"):
            continue
        if task in aggregates:
            continue
        repeat_value = metrics.get("num_repeat")
        item_value = metrics.get("num_total")
        score = metrics.get("accuracy_avg")
        if repeat_value is None or item_value is None or score is None or repeat_value <= 1:
            continue
        if not float(repeat_value).is_integer() or not float(item_value).is_integer():
            logger.warning("Skipping repeated projection for %s/%s: nonintegral count", record.run_id, task)
            continue
        repeats, items = int(repeat_value), int(item_value)
        if items <= 1 or not 0 <= score <= 1:
            logger.warning("Skipping repeated projection for %s/%s: invalid score or item count", record.run_id, task)
            continue
        source = coverage.get(task)
        item_coverage = _item_coverage(source, items, repeats)
        if item_coverage is None:
            logger.warning("Skipping repeated projection for %s/%s: ambiguous partial coverage", record.run_id, task)
            continue
        coverage[task] = item_coverage
        aggregates[task] = RepeatAggregate(
            n_items=items,
            n_repeats=repeats,
            n_trials=items * repeats,
            n_scored_trials=source.n_scored if source is not None and source.n_scored > items else None,
            stderr=math.sqrt(score * (1 - score) / (items - 1)),
            stderr_basis=ITEM_BERNOULLI_STDERR,
            source_coverage=source,
        )
    if not aggregates:
        return record
    return record.model_copy(update={"coverage": coverage, "repeat_aggregates": aggregates})
