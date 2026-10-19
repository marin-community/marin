"""Task-owned reductions of graded examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from eval.contracts.sample_results import REPEATED_SAMPLE_METRICS_ANNOTATION
from eval.passk import aggregate_pass_at_k


def outcome_report(outcomes: Sequence[Mapping[str, Any]], population: Sequence[str]) -> dict[str, Any]:
    """Collect native outcome partitions against the source's full population."""
    ids = {name: set() for name in outcomes[0] if name != "instance_id"}
    for row in outcomes:
        for name in ids:
            if row[name]:
                ids[name].add(row["instance_id"])
    ids["incomplete"] = set(population) - ids["submitted"]
    return {
        **{f"{name}_ids": sorted(values) for name, values in ids.items()},
        **{f"{name}_instances": len(values) for name, values in ids.items() if name != "incomplete"},
    }


def functional_pass_at_k(
    verdicts: Mapping[str, Sequence[Mapping[str, Any]]], ks: Sequence[int]
) -> dict[str, float]:
    """Reduce completion verdicts with the unbiased per-problem pass@k estimator."""
    if not verdicts or any(not rows for rows in verdicts.values()):
        raise ValueError("cannot aggregate missing functional correctness verdicts")
    return aggregate_pass_at_k(
        [len(rows) for rows in verdicts.values()],
        [sum(row["passed"] for row in rows) for rows in verdicts.values()],
        ks,
    )


def repeated_accuracy_results(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce independent accuracy trials, retaining their run statistics and standard error."""
    if not examples:
        raise ValueError("cannot aggregate an empty set of graded examples")
    repeats = {len(example[REPEATED_SAMPLE_METRICS_ANNOTATION]) for example in examples}
    if len(repeats) != 1:
        raise ValueError("graded examples have different repetition counts")
    num_repeat = repeats.pop()
    run_stats = [
        {
            "repetition": repeat + 1,
            "num_total": sum(repeat in example.get("valid_repeats", range(num_repeat)) for example in examples),
            "num_solved": sum(
                example[REPEATED_SAMPLE_METRICS_ANNOTATION][repeat]["accuracy"] for example in examples
                if repeat in example.get("valid_repeats", range(num_repeat))
            ),
        }
        for repeat in range(num_repeat)
    ]
    run_stats = [{**run, "accuracy": run["num_solved"] / run["num_total"]} for run in run_stats if run["num_total"]]
    if not run_stats:
        raise ValueError("cannot aggregate a task without a valid graded trial")
    accuracies = [run["accuracy"] for run in run_stats]
    return {
        "num_total": len(examples),
        "solved_avg": float(np.mean([run["num_solved"] for run in run_stats])),
        "run_stats": run_stats,
        "accuracy_avg": float(np.mean(accuracies)),
        "accuracy_std_err": float(np.std(accuracies) / np.sqrt(len(run_stats))),
        "num_repeat": num_repeat,
    }
