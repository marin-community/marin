# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the lm-eval metric and extraction filter Marin presents as primary."""

from collections.abc import Iterable, Mapping

# Headline metric for a task, matched after stripping lm-eval's ``,<filter>`` suffix.
# ``acc_norm`` outranks ``acc`` where both exist; ``accuracy`` covers Evalchemy's chat-native tasks.
PRIMARY_METRIC_PRIORITY = ("exact_match", "accuracy", "acc_norm", "acc", "pass@1")

# lm-eval pairs ``acc`` with ``acc_stderr``. Evalchemy's repeated-sample tasks (aime24) report the mean
# over repeats as ``accuracy_avg`` and its standard error across repeats as ``accuracy_std_err``.
# Neither standard error is ever a headline score.
LM_EVAL_STDERR_SUFFIX = "_stderr"
REPEAT_MEAN_SUFFIX = "_avg"
REPEAT_STDERR_SUFFIX = "_std_err"
DISPERSION_SUFFIXES = (LM_EVAL_STDERR_SUFFIX, REPEAT_STDERR_SUFFIX)

# Chat models often solve gsm8k-style tasks without emitting the strict ``#### N`` format.
FILTER_PRIORITY = ("flexible-extract",)


def primary_filter(filters: Iterable[str]) -> str | None:
    """Pick the extraction filter Evaldash should show by default."""
    names = {name for name in filters if name}
    if not names:
        return None
    for preferred in FILTER_PRIORITY:
        if preferred in names:
            return preferred
    return min(names)


def base_metric(name: str) -> str:
    """Return a metric key without lm-eval's ``,<filter>`` suffix."""
    return name.split(",", 1)[0]


def declared_metric(metrics: Mapping[str, float], declared: str | None) -> tuple[str, float] | None:
    """Pick a declared base metric, applying the standard extraction-filter priority."""
    if declared is None:
        return primary_metric(metrics)
    candidates = {name: value for name, value in metrics.items() if base_metric(name) == declared}
    for metric_filter in FILTER_PRIORITY:
        filtered = sorted(name for name in candidates if name.endswith(f",{metric_filter}"))
        if filtered:
            name = filtered[0]
            return name, candidates[name]
    if not candidates:
        return None
    name = min(candidates)
    return name, candidates[name]


def primary_metric(metrics: Mapping[str, float]) -> tuple[str, float] | None:
    """Pick Marin's headline ``(key, value)`` from an evaluator metric mapping."""
    candidates = {name: value for name, value in metrics.items() if not base_metric(name).endswith(DISPERSION_SUFFIXES)}
    if not candidates:
        return None
    for preferred in PRIMARY_METRIC_PRIORITY:
        matches = {name: value for name, value in candidates.items() if base_metric(name) == preferred}
        if not matches:
            continue
        for metric_filter in FILTER_PRIORITY:
            for name, value in matches.items():
                if name.endswith(f",{metric_filter}"):
                    return name, value
        name = min(matches)
        return name, matches[name]
    name = min(candidates)
    return name, candidates[name]
