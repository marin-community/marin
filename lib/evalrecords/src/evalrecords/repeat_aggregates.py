# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project repeated evaluator scores onto independent benchmark items."""

import logging
import math

from evalrecords.lm_eval_samples import is_scratch_artifact
from evalrecords.records import EvalRunRecord, RepeatAggregate, RepeatCoverageBasis, TaskCoverage

ITEM_BERNOULLI_STDERR = "item_bernoulli_n_minus_1"
logger = logging.getLogger(__name__)


def _item_count(count: int | None, items: int, repeats: int) -> int | None:
    if count is None or count <= items:
        return count
    if count > (items - 1) * repeats:
        return items
    return count // repeats if count % repeats == 0 else None


def _item_coverage(
    source: TaskCoverage | None, items: int, repeats: int
) -> tuple[TaskCoverage | None, RepeatCoverageBasis]:
    if source is None:
        return TaskCoverage(n_scored=items), RepeatCoverageBasis.ITEMS

    trials = items * repeats
    if source.n_scored == 0:
        if source.n_attempted not in (items, trials):
            return None, RepeatCoverageBasis.SOURCE
        scored_items = items
        basis = RepeatCoverageBasis.ITEMS
    elif source.n_scored == items or ((items - 1) * repeats < source.n_scored <= trials):
        scored_items = items
        basis = RepeatCoverageBasis.ITEMS
    elif items < source.n_scored < trials:
        scored_items = math.ceil(source.n_scored / repeats)
        basis = RepeatCoverageBasis.ITEM_LOWER_BOUND
    else:
        return None, RepeatCoverageBasis.SOURCE
    attempted = _item_count(source.n_attempted, items, repeats)
    if source.n_attempted is not None and (attempted is None or attempted < scored_items):
        return None, RepeatCoverageBasis.SOURCE
    benchmark = source.n_benchmark
    trial_coverage = source.n_scored > items or (source.n_attempted is not None and source.n_attempted > items)
    # The trial-form producer multiplies the full benchmark size even when num_total is capped.
    if trial_coverage and benchmark is not None and benchmark >= trials and benchmark % repeats == 0:
        benchmark //= repeats
    # Trial-level unanswered counts cannot identify partly unanswered items. Every trial
    # unanswered does establish that every represented item lacked an extractable answer.
    if source.n_scored == items:
        unanswered = source.n_unanswered
    elif source.n_scored > 0 and source.n_unanswered == source.n_scored:
        unanswered = scored_items
    else:
        unanswered = 0
    return (
        TaskCoverage(
            n_benchmark=benchmark,
            n_attempted=attempted,
            n_scored=scored_items,
            n_unanswered=unanswered,
        ),
        basis,
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
        item_coverage, coverage_basis = _item_coverage(source, items, repeats)
        if item_coverage is None:
            logger.warning("Retaining source coverage for %s/%s: inconsistent repeated counts", record.run_id, task)
        else:
            coverage[task] = item_coverage
        aggregates[task] = RepeatAggregate(
            n_items=items,
            n_repeats=repeats,
            n_trials=items * repeats,
            n_scored_trials=source.n_scored if source is not None and source.n_scored > items else None,
            coverage_basis=coverage_basis,
            stderr=math.sqrt(score * (1 - score) / (items - 1)),
            stderr_basis=ITEM_BERNOULLI_STDERR,
            source_coverage=source,
        )
    if not aggregates:
        return record
    return record.model_copy(update={"coverage": coverage, "repeat_aggregates": aggregates})
