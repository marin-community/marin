# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge a quality-track candidate against its matched random control by capability group.

One held-out loss does not rank data selections the way downstream tasks do. A selection can
move the domain mix, and each loss suite rewards a different mix: Paloma rewards web text,
Uncheatable rewards code and arXiv. So the judge reads the ``downstream_eval`` outputs and
compares four groups against the matched random selection:

- E: English knowledge and commonsense (correct-answer BPB of the cloze tasks)
- C: code (gold-solution BPB of HumanEval and MBPP)
- M: math (gold-solution BPB of GSM8K and MATH-500)
- L: long-context use (loss on fixed targets at 3968 tokens of context), a guardrail only

A gain is the relative decrease against random. The weighted gain is the mean of E, C, and M.
A candidate is better only when the lower bound of the weighted gain is above zero and no group
can lose more than its harm limit. The bounds come from a bootstrap over training runs and
paired evaluation examples.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import click
import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath, prefix_join

CLOZE_TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "sciq",
    "winogrande",
    "lambada",
    "mmlu",
    "csqa",
    "openbookqa",
    "siqa",
)
GROUP_TASKS = {"E": CLOZE_TASKS, "C": ("humaneval", "mbpp"), "M": ("gsm8k", "math500")}
WEIGHTED_GROUPS = ("E", "C", "M")
# The largest relative loss in each group that a better candidate may have at the 95% lower bound.
HARM_LIMITS = {"E": 0.01, "C": 0.02, "M": 0.02, "L": 0.02}
LONG_CONTEXT_KEY = "fixed_target_context_3968_loss"
LONG_CONTEXT_MIN_DOCUMENTS = 5
BOOTSTRAP_SAMPLES = 1000


class Decision(StrEnum):
    BETTER = "better"
    WORSE = "worse"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class RunScores:
    """Per-example correct-answer BPB by task, in example-ID order, and the long-context loss."""

    gold_bpb: dict[str, np.ndarray]
    long_context_loss: float


@dataclass(frozen=True)
class Judgement:
    gains: dict[str, float]
    lower: dict[str, float]
    upper: dict[str, float]
    decision: Decision


def read_run_scores(downstream_dir: str) -> RunScores:
    """Read the records and metrics that ``downstream_eval`` wrote to ``downstream_dir``."""
    with StoragePath(prefix_join(downstream_dir, "downstream_records.parquet")).open("rb") as stream:
        records = pq.read_table(stream).to_pandas()
    with StoragePath(prefix_join(downstream_dir, "metrics.json")).open("r") as stream:
        text = json.load(stream)["text"]
    correct = records[records.choice == records.label].sort_values(["task", "example_id"])
    bpb = -correct.log_likelihood / (math.log(2) * correct["bytes"].clip(lower=1))
    gold_bpb = {task: bpb[correct.task == task].to_numpy() for task in correct.task.unique()}
    long_sets = [
        values[LONG_CONTEXT_KEY]
        for values in text.values()
        if LONG_CONTEXT_KEY in values and values["fixed_target_documents"] >= LONG_CONTEXT_MIN_DOCUMENTS
    ]
    return RunScores(gold_bpb, float(np.mean(long_sets)))


def _group_value(run: RunScores, tasks: Sequence[str], index: dict[str, np.ndarray] | None) -> float:
    per_task = [run.gold_bpb[task] if index is None else run.gold_bpb[task][index[task]] for task in tasks]
    return float(np.mean([values.mean() for values in per_task]))


def relative_gains(
    candidate: Sequence[RunScores],
    control: Sequence[RunScores],
    index: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Relative decrease of each group's mean score against the control; ``G`` is the weighted gain."""
    gains = {}
    for group, tasks in GROUP_TASKS.items():
        candidate_value = np.mean([_group_value(run, tasks, index) for run in candidate])
        control_value = np.mean([_group_value(run, tasks, index) for run in control])
        gains[group] = float((control_value - candidate_value) / control_value)
    candidate_long = np.mean([run.long_context_loss for run in candidate])
    control_long = np.mean([run.long_context_loss for run in control])
    gains["L"] = float((control_long - candidate_long) / control_long)
    gains["G"] = float(np.mean([gains[group] for group in WEIGHTED_GROUPS]))
    return gains


def judge(
    candidate: Sequence[RunScores],
    control: Sequence[RunScores],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> Judgement:
    """Compare matched runs with a bootstrap over runs and paired evaluation examples."""
    if len(candidate) < 2 or len(control) < 2:
        raise ValueError("the judge needs at least two candidate runs and two control runs")
    rng = np.random.default_rng(seed)
    example_counts = {task: len(values) for task, values in control[0].gold_bpb.items()}
    draws = []
    for _ in range(samples):
        index = {task: rng.integers(0, count, count) for task, count in example_counts.items()}
        candidate_draw = [candidate[i] for i in rng.integers(0, len(candidate), len(candidate))]
        control_draw = [control[i] for i in rng.integers(0, len(control), len(control))]
        draws.append(relative_gains(candidate_draw, control_draw, index))
    gains = relative_gains(candidate, control)
    lower = {key: float(np.percentile([draw[key] for draw in draws], 2.5)) for key in gains}
    upper = {key: float(np.percentile([draw[key] for draw in draws], 97.5)) for key in gains}
    if lower["G"] > 0 and all(lower[group] > -limit for group, limit in HARM_LIMITS.items()):
        decision = Decision.BETTER
    elif upper["G"] < 0 or any(upper[group] < -limit for group, limit in HARM_LIMITS.items()):
        decision = Decision.WORSE
    else:
        decision = Decision.INCONCLUSIVE
    return Judgement(gains, lower, upper, decision)


@click.command()
@click.option(
    "--candidate", "candidate_dirs", multiple=True, required=True, help="downstream_eval output of a candidate run."
)
@click.option("--control", "control_dirs", multiple=True, required=True, help="downstream_eval output of a random run.")
def main(candidate_dirs: tuple[str, ...], control_dirs: tuple[str, ...]) -> None:
    judgement = judge(
        [read_run_scores(path) for path in candidate_dirs], [read_run_scores(path) for path in control_dirs]
    )
    for key in (*GROUP_TASKS, "L", "G"):
        print(f"{key}: {judgement.gains[key]:+.4f} [{judgement.lower[key]:+.4f}, {judgement.upper[key]:+.4f}]")
    print(f"decision: {judgement.decision}")


if __name__ == "__main__":
    main()
