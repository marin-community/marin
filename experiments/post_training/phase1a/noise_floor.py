# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Three-seed Phase 0c noise estimates from the validated scoring summaries."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

SEEDS = (0, 1, 2)
STEPS = (10, 30, 50, 100)
COUNTS = {"P_X": 1000, "P_R": 1000, "P_chat": 1500, "P_think": 300}
FORWARD_POSITIONS = 1_702_261


@dataclass(frozen=True)
class RoutingProbe:
    base_model: str
    probe_identity: str
    tokens: int
    layers: tuple[int, ...]


@dataclass(frozen=True)
class SeedMetrics:
    method: str
    step: int
    seed: int
    metrics: dict[str, float]
    routing: dict[str, RoutingProbe] = field(default_factory=dict)


@dataclass(frozen=True)
class Spread:
    values: tuple[float, ...]
    mean: float
    sample_std: float
    minimum: float
    maximum: float
    range: float


def spread(values: tuple[float, ...]) -> Spread:
    return Spread(
        values, statistics.mean(values), statistics.stdev(values), min(values), max(values), max(values) - min(values)
    )


def seed_metrics(record: dict) -> SeedMetrics:
    """Validate probe coverage and extract token-weighted metrics and buckets."""
    summaries = record["summaries"]
    metrics = {}
    for name in (
        "kl_P_X",
        "reverse_P_X",
        "kl_P_R",
        "kl_P_chat",
        "kl_P_think",
        "flip_P_X",
        "flip_P_chat",
        "flip_P_think",
    ):
        summary = summaries[name]
        family = name.split("_", 1)[1]
        if summary["sequences"] != COUNTS[family]:
            raise ValueError(f"{name}: expected {COUNTS[family]} sequences, got {summary['sequences']}")
        if name in ("kl_P_X", "flip_P_X") and summary["positions"] != FORWARD_POSITIONS:
            raise ValueError(f"{name}: does not cover the frozen forward task-X positions")
        field = "flip_rate" if name.startswith("flip_") else "token_mean_kl"
        metric_name = f"token_{name}" if name.startswith("flip_") else name
        metrics[metric_name] = float(summary[field])
        buckets = summary["buckets"]
        if sum(b["positions"] for b in buckets.values()) != summary["positions"]:
            raise ValueError(f"{name}: buckets do not partition the scored positions")
        for bucket, result in buckets.items():
            metrics[f"{metric_name}/bucket/{bucket}"] = float(result[field])
        for quartile, result in summary.get("reference_margin_quartiles", {}).items():
            metrics[f"{metric_name}/margin/{quartile}"] = float(result[field])
    routing = {}
    if "routing" in record:
        reports = record["routing"]
        if set(reports) != set(COUNTS):
            raise ValueError("Routing evaluation must cover X, R, chat, and thinking")
        for family, report in reports.items():
            if not report["complete"] or report["sequences"] != COUNTS[family]:
                raise ValueError(f"{family}: incomplete routing evaluation")
            pairs = report["pairs"]
            if pairs != report["tokens"] * len(report["layers"]) or pairs <= 0:
                raise ValueError(f"{family}: routing denominator does not cover tokens and layers")
            routing[family] = RoutingProbe(
                report["base_model"], report["probe_identity"], report["tokens"], tuple(report["layers"])
            )
            name = f"routing_flip_{family}"
            metrics[name] = report["flips"] / pairs
            for category in ("by_layer", "base_margin_quartiles", "buckets", "position_buckets"):
                breakdown = report[category]
                if (
                    sum(row["pairs"] for row in breakdown.values()) != pairs
                    or sum(row["flips"] for row in breakdown.values()) != report["flips"]
                ):
                    raise ValueError(f"{family}: routing {category} does not partition token-layer pairs")
                for bucket, result in breakdown.items():
                    metrics[f"{name}/{category}/{bucket}"] = result["flips"] / result["pairs"]
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Noise estimates require finite metrics")
    if float(record["learning_rate"]) != 5e-7:
        raise ValueError("Noise-floor records must use the chosen common LR 5e-7")
    return SeedMetrics(record["method"], int(record["step"]), int(record["seed"]), metrics, routing)


def gate_inputs(records: list[SeedMetrics], forward_baseline: float, reverse_baseline: float) -> list[dict]:
    """Report each spread against forward and reverse mean single-task gains.

    The design does not specify whether 'noise' means standard deviation or
    range. Keep both comparisons explicit, without selecting one after seeing
    the results. Gains use the three-seed mean and retain their sign.
    """
    groups = defaultdict(list)
    for record in records:
        groups[(record.method, record.step)].append(record)
    if set(groups) != {(method, step) for method in ("m1", "m4") for step in STEPS}:
        raise ValueError("Expected both methods at every scored step")
    reference_routing = records[0].routing
    if any(record.routing != reference_routing for record in records):
        raise ValueError("Routing gate requires the same B, frozen probes, token coverage, and layers for every seed")
    output = []
    for (method, step), group in sorted(groups.items()):
        group.sort(key=lambda row: row.seed)
        if tuple(row.seed for row in group) != SEEDS:
            raise ValueError(f"{method} step {step}: expected exactly seeds 0, 1, 2")
        all_keys = set().union(*(row.metrics for row in group))
        keys = set.intersection(*(set(row.metrics) for row in group))
        unavailable = {
            key: {"missing_seeds": [row.seed for row in group if key not in row.metrics]}
            for key in sorted(all_keys - keys)
        }
        if any(not key.startswith("reverse_P_X/bucket/") for key in unavailable):
            raise ValueError(f"{method} step {step}: fixed-probe metrics or buckets differ across seeds")
        distributions = {key: spread(tuple(row.metrics[key] for row in group)) for key in sorted(keys)}
        gains = {
            "forward": forward_baseline - distributions["kl_P_X"].mean,
            "reverse": reverse_baseline - distributions["reverse_P_X"].mean,
        }
        progress = {
            direction: [1 - row.metrics[key] / baseline for row in group]
            for direction, key, baseline in (
                ("forward", "kl_P_X", forward_baseline),
                ("reverse", "reverse_P_X", reverse_baseline),
            )
        }
        comparisons = {}
        for key, distribution in distributions.items():
            comparisons[key] = {
                **asdict(distribution),
                "gate": {
                    direction: {
                        "threshold": gain / 3,
                        "sample_std_pass": distribution.sample_std < gain / 3,
                        "range_pass": distribution.range < gain / 3,
                    }
                    for direction, gain in gains.items()
                },
            }
        output.append(
            dict(
                method=method,
                step=step,
                seeds=SEEDS,
                gains=gains,
                progress=progress,
                metrics=comparisons,
                unavailable_metrics=unavailable,
                routing_flip_status="measured" if reference_routing else "not_measured",
                routing_reference={key: asdict(value) for key, value in reference_routing.items()},
                token_flip_definition="Next-token argmax disagreement; not selected-expert-set disagreement",
            )
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, required=True, help="Combined scoring summaries with explicit seed fields"
    )
    parser.add_argument("--forward-baseline", type=Path, required=True)
    parser.add_argument("--reverse-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baselines = [json.loads(path.read_text()) for path in (args.forward_baseline, args.reverse_baseline)]
    for baseline in baselines:
        if baseline["sequences"] != COUNTS["P_X"] or not 0 < baseline["token_mean_kl"] < math.inf:
            raise ValueError("Each baseline needs 1000 sequences and finite positive KL")
    if baselines[0]["positions"] != FORWARD_POSITIONS:
        raise ValueError("Forward baseline does not cover the frozen task-X positions")
    records = [seed_metrics(row) for row in json.loads(args.records.read_text())]
    result = gate_inputs(records, *(baseline["token_mean_kl"] for baseline in baselines))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
