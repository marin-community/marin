# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Primary-metric selection over lm-eval metric dicts.

lm-eval keys each metric ``"<name>,<filter>"`` (e.g. ``"acc,none"``, ``"exact_match,flexible-extract"``)
and pairs every score with a ``"<name>_stderr,<filter>"`` standard error. These helpers pick the one
headline metric for a task and its paired stderr, and the equivalent for the per-sample parquet columns
(which carry the base name only, e.g. ``acc``). Shared by the matrix/leaderboard views and the sample
browser so both rank metrics identically.
"""

from __future__ import annotations

# Headline metric for a task, matched on the base metric name (the ``,<filter>`` suffix stripped).
# The first base name present (ignoring ``*_stderr``) wins; otherwise the alphabetically-first metric.
# ``acc_norm`` outranks ``acc``: where lm-eval emits both (arc, hellaswag, piqa, openbookqa) the
# length-normalized score is the conventional headline.
PRIMARY_METRIC_PRIORITY = (
    "exact_match",
    # Evalchemy's chat-native benchmarks (MATH500) report ``accuracy``.
    "accuracy",
    "acc_norm",
    "acc",
    "pass@1",
)

# Tie-break among same-base metrics that differ only in lm-eval filter. ``flexible-extract`` outranks
# ``strict-match`` on gsm8k-style tasks: chat-templated models solve the problems but rarely emit the
# strict ``#### N`` answer format, so strict-match understates them (llama3.1-instruct: 0.20 strict
# vs 0.79 flexible on identical generations).
FILTER_PRIORITY = ("flexible-extract",)


def base_metric(name: str) -> str:
    """A metric key without lm-eval's ``,<filter>`` suffix (``exact_match,none`` -> ``exact_match``)."""
    return name.split(",", 1)[0]


def primary_metric(metrics: dict[str, float]) -> tuple[str, float] | None:
    """Pick the headline ``(key, value)`` for one task's metric dict, or None if empty.

    Standard-error metrics (base ends ``_stderr``) never headline; among the rest the
    ``PRIMARY_METRIC_PRIORITY`` order wins by base name, ``FILTER_PRIORITY`` breaks ties between
    same-base filters, and the alphabetically-first key is the final fallback at each step.
    """
    candidates = {name: value for name, value in metrics.items() if not base_metric(name).endswith("_stderr")}
    if not candidates:
        return None
    for preferred in PRIMARY_METRIC_PRIORITY:
        matches = {name: value for name, value in candidates.items() if base_metric(name) == preferred}
        if matches:
            for metric_filter in FILTER_PRIORITY:
                for name, value in matches.items():
                    if name.endswith(f",{metric_filter}"):
                        return name, value
            name = min(matches)
            return name, matches[name]
    name = min(candidates)
    return name, candidates[name]


def stderr_for(metrics: dict[str, float], metric_key: str) -> float | None:
    """The standard error paired with ``metric_key``: its ``<base>_stderr,<filter>`` value, or None.

    lm-eval names the stderr for ``acc,none`` as ``acc_stderr,none``; a filterless ``acc`` pairs with
    ``acc_stderr``.
    """
    base, _, flt = metric_key.partition(",")
    key = f"{base}_stderr,{flt}" if flt else f"{base}_stderr"
    value = metrics.get(key)
    return float(value) if value is not None else None


def primary_metric_column(columns: list[str]) -> str | None:
    """Pick the per-sample primary metric column from parquet metric column names, or None.

    Per-sample columns carry the base name only (``acc``, ``exact_match``), so priority matches on the
    name directly; ``*_stderr`` columns are excluded and the alphabetically-first remaining wins.
    """
    candidates = [column for column in columns if not column.endswith("_stderr")]
    if not candidates:
        return None
    for preferred in PRIMARY_METRIC_PRIORITY:
        if preferred in candidates:
            return preferred
    return min(candidates)
