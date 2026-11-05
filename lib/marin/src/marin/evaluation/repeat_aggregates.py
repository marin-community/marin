# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project repeated evaluator scores onto independent benchmark items."""

import math

from marin.evaluation.lm_eval_samples import is_scratch_artifact
from marin.evaluation.records import EvalRunRecord, RepeatAggregate, TaskCoverage

ITEM_BERNOULLI_STDERR = "item_bernoulli_n_minus_1"


def _item_count(count: int | None, items: int, repeats: int, *, benchmark_size: bool = False) -> int | None:
    if count is None or count <= items or (benchmark_size and count != items * repeats):
        return count
    if count % repeats:
        raise ValueError(f"trial count {count} is not divisible by {repeats} repeats")
    return count // repeats


def _item_coverage(source: TaskCoverage | None, items: int, repeats: int) -> TaskCoverage:
    if source is None:
        return TaskCoverage(n_scored=items)

    trials = items * repeats
    if source.n_scored not in (0, items, trials):
        raise ValueError(f"repeated score covers {source.n_scored} samples, expected {items} items or {trials} trials")
    if source.errors and source.errors != {"ungraded": source.n_attempted}:
        raise ValueError(f"repeated score has unexplained coverage errors: {source.errors}")
    if source.n_scored == 0 and source.n_attempted not in (items, trials):
        raise ValueError("repeated score has no graded samples or matching aggregate count")
    return TaskCoverage(
        n_benchmark=_item_count(source.n_benchmark, items, repeats, benchmark_size=True),
        n_attempted=_item_count(source.n_attempted, items, repeats),
        n_scored=items,
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
            raise ValueError(f"{record.run_id}/{task}: nonintegral repeat or item count")
        repeats, items = int(repeat_value), int(item_value)
        if items <= 1 or not 0 <= score <= 1:
            raise ValueError(f"{record.run_id}/{task}: invalid repeated score or item count")
        source = coverage.get(task)
        coverage[task] = _item_coverage(source, items, repeats)
        aggregates[task] = RepeatAggregate(
            n_items=items,
            n_repeats=repeats,
            n_trials=items * repeats,
            stderr=math.sqrt(score * (1 - score) / (items - 1)),
            stderr_basis=ITEM_BERNOULLI_STDERR,
            source_coverage=source,
        )
    if not aggregates:
        return record
    return record.model_copy(update={"coverage": coverage, "repeat_aggregates": aggregates})
