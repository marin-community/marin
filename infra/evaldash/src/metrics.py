# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-error lookup paired with the canonical primary-metric selection.

Primary-metric selection (``PRIMARY_METRIC_PRIORITY``, ``FILTER_PRIORITY``, ``primary_metric``) is
defined once, in ``marin.evaluation.samples`` (the per-sample export contract), and re-exported here
so the matrix/leaderboard views and the sample browser rank metrics identically. ``stderr_for`` is the
one piece specific to run-level metric dicts: finding the stderr paired with a metric key.
"""

from __future__ import annotations

from marin.evaluation.samples import primary_metric as primary_metric


def stderr_for(metrics: dict[str, float], metric_key: str) -> float | None:
    """The standard error paired with ``metric_key``: its ``<base>_stderr,<filter>`` value, or None.

    lm-eval names the stderr for ``acc,none`` as ``acc_stderr,none``; a filterless ``acc`` pairs with
    ``acc_stderr``.
    """
    base, _, flt = metric_key.partition(",")
    key = f"{base}_stderr,{flt}" if flt else f"{base}_stderr"
    value = metrics.get(key)
    return float(value) if value is not None else None
