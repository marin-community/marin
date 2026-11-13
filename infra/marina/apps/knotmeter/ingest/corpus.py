# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize immutable token structures during the explicit catalog load."""

import math
import statistics
from collections import Counter
from typing import Any, TypedDict


class HistogramBin(TypedDict):
    x0: float
    x1: float
    count: int


class DistributionStats(TypedDict):
    count: int
    min: float
    max: float
    mean: float
    median: float
    p90: float
    p95: float


class Distribution(TypedDict):
    bins: list[HistogramBin]
    stats: DistributionStats | None


def conversation_summary(structure: dict[str, Any]) -> dict[str, Any]:
    turns = [
        turn for node in structure["nodes"] for turn in (node["children"] if node["kind"] == "subagent" else [node])
    ]
    return structure["totals"] | {
        "requests": len(turns),
        "models": dict(Counter(turn.get("model", "unknown") for turn in turns)),
        "subagent_requests": sum(len(node["children"]) for node in structure["nodes"] if node["kind"] == "subagent"),
    }


def distribution(values: list[float]) -> Distribution:
    if not values:
        return {"bins": [], "stats": None}
    ordered = sorted(values)
    quantiles = statistics.quantiles(ordered, n=100, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 99
    lower, upper = ordered[0], ordered[-1]
    width = (upper - lower) / 24 if upper > lower else 1
    counts = Counter(min(23, math.floor((value - lower) / width)) for value in ordered)
    return {
        "bins": [
            {"x0": lower + i * width, "x1": lower + (i + 1) * width, "count": counts[i]}
            for i in range(24 if upper > lower else 1)
        ],
        "stats": {
            "count": len(ordered),
            "min": lower,
            "max": upper,
            "mean": statistics.mean(ordered),
            "median": statistics.median(ordered),
            "p90": quantiles[89],
            "p95": quantiles[94],
        },
    }


def dataset_summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    totals: Counter[str] = Counter()
    models: Counter[str] = Counter()
    samples: dict[str, list[float]] = {
        name: []
        for name in [
            "Input tokens per request",
            "Uncached input tokens per request",
            "Output tokens per request",
            "Cached input fraction per request",
            "Requests per conversation",
            "Subagent groups per conversation",
            "Subagent input tokens per request",
            "Subagent output tokens per request",
        ]
    }
    for row in rows:
        structure = row["structure"]
        summary = row["summary"]
        totals.update(
            {
                k: summary[k]
                for k in [
                    "in",
                    "out",
                    "cached",
                    "uncached",
                    "numTurns",
                    "numSubagentGroups",
                    "requests",
                    "subagent_requests",
                ]
            }
        )
        models.update(summary["models"])
        samples["Requests per conversation"].append(summary["requests"])
        samples["Subagent groups per conversation"].append(summary["numSubagentGroups"])
        for node in structure["nodes"]:
            turns = node["children"] if node["kind"] == "subagent" else [node]
            for turn in turns:
                samples["Input tokens per request"].append(turn["in"])
                samples["Uncached input tokens per request"].append(turn["uncached"])
                samples["Output tokens per request"].append(turn["out"])
                if turn["in"] > 0:
                    samples["Cached input fraction per request"].append(turn["cached"] / turn["in"])
                if node["kind"] == "subagent":
                    samples["Subagent input tokens per request"].append(turn["in"])
                    samples["Subagent output tokens per request"].append(turn["out"])
    return dict(totals) | {"models": dict(models), "conversations": len(rows)}, {
        name: distribution(values) for name, values in samples.items()
    }
