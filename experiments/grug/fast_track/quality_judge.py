# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Judge a quality-track candidate against its matched random control by capability group.

One held-out loss does not rank data selections the way downstream tasks do. A selection can
move the domain mix, and each loss suite rewards a different mix: Paloma rewards web text,
Uncheatable rewards code and arXiv. So the judge reads the ``downstream_eval`` outputs and
compares these groups against the matched random selection:

- K: English knowledge (correct-answer BPB of ARC, MMLU, SciQ, OpenBookQA)
- S: English commonsense and narrative (HellaSwag, PIQA, SIQA, Winogrande, CSQA, LAMBADA)
- C: code (gold-solution BPB of HumanEval and MBPP)
- M: math (gold-solution BPB of GSM8K and MATH-500)
- L: long-context use (loss on fixed targets at 3968 tokens of context), a guardrail only

Quality classifiers trade S for K, so the two English groups have separate harm limits. A gain is
the relative decrease against random. The weighted gain is the mean of English ((K + S) / 2), C, and M.
A candidate is better only when the lower bound of the weighted gain is above zero and no group
can lose more than its harm limit. The bounds are Welch t intervals. Their variance adds the seed
variance of the per-run gains of each arm and the variance of a bootstrap over paired evaluation
examples. With three runs per arm, a percentile bootstrap over runs covers only about 80% at 95%.
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
from scipy.stats import t as student_t

from experiments.grug.fast_track.downstream_eval import LONG_CONTEXT_LOSS_KEY, METRICS_FILE, RECORDS_FILE

GROUP_TASKS = {
    "K": ("arc_easy", "arc_challenge", "mmlu", "sciq", "openbookqa"),
    "S": ("hellaswag", "piqa", "siqa", "winogrande", "csqa", "lambada"),
    "C": ("humaneval", "mbpp"),
    "M": ("gsm8k", "math500"),
}
# The largest relative loss in each group that a better candidate may have at the 95% lower bound.
HARM_LIMITS = {"K": 0.01, "S": 0.01, "C": 0.02, "M": 0.02, "L": 0.02}
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
    example_ids: dict[str, tuple[str, ...]]
    long_context_loss: float
    version: str


@dataclass(frozen=True)
class Judgement:
    gains: dict[str, float]
    lower: dict[str, float]
    upper: dict[str, float]
    decision: Decision


def read_run_scores(downstream_dir: str) -> RunScores:
    """Read the records and metrics that ``downstream_eval`` wrote to ``downstream_dir``."""
    with StoragePath(prefix_join(downstream_dir, RECORDS_FILE)).open("rb") as stream:
        records = pq.read_table(stream).to_pandas()
    with StoragePath(prefix_join(downstream_dir, METRICS_FILE)).open("r") as stream:
        metrics = json.load(stream)
    correct = records[records.choice == records.label].sort_values(["task", "example_id"])
    bpb = -correct.log_likelihood / (math.log(2) * correct["bytes"].clip(lower=1))
    gold_bpb = {task: bpb[correct.task == task].to_numpy() for task in correct.task.unique()}
    example_ids = {task: tuple(correct.example_id[correct.task == task]) for task in gold_bpb}
    long_sets = [
        values[LONG_CONTEXT_LOSS_KEY]
        for values in metrics["text"].values()
        if LONG_CONTEXT_LOSS_KEY in values and values["fixed_target_documents"] >= LONG_CONTEXT_MIN_DOCUMENTS
    ]
    if not long_sets:
        raise ValueError(f"{downstream_dir} has no held-out set with long documents; score it without --skip-text")
    version = f"{metrics['tasks_version']}/{metrics['scorer_version']}"
    return RunScores(gold_bpb, example_ids, float(np.mean(long_sets)), version)


def _group_value(run: RunScores, tasks: Sequence[str], index: dict[str, np.ndarray] | None) -> float:
    per_task = [run.gold_bpb[task] if index is None else run.gold_bpb[task][index[task]] for task in tasks]
    return float(np.mean([values.mean() for values in per_task]))


def _run_values(run: RunScores) -> dict[str, float]:
    values = {group: _group_value(run, tasks, None) for group, tasks in GROUP_TASKS.items()}
    values["L"] = run.long_context_loss
    return values


def _weighted(gains: dict) -> float | np.ndarray:
    return np.mean([(gains["K"] + gains["S"]) / 2, gains["C"], gains["M"]], axis=0)


def _per_run_gains(
    runs: Sequence[RunScores], control_means: dict[str, float], scale: dict[str, float]
) -> dict[str, np.ndarray]:
    """Relative gain of each run against the control means, times ``scale``.

    The candidate and control means of these gains differ by ``relative_gains``.
    """
    values = [_run_values(run) for run in runs]
    gains = {
        key: np.array([scale[key] * (mean - value[key]) / mean for value in values])
        for key, mean in control_means.items()
    }
    gains["G"] = _weighted(gains)
    return gains


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
    gains["G"] = float(_weighted(gains))
    return gains


def judge(
    candidate: Sequence[RunScores],
    control: Sequence[RunScores],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> Judgement:
    """Compare matched runs with Welch t intervals over runs plus a bootstrap over paired evaluation examples."""
    if len(candidate) < 2 or len(control) < 2:
        raise ValueError("the judge needs at least two candidate runs and two control runs")
    runs = [*candidate, *control]
    if len({run.version for run in runs}) != 1 or any(run.example_ids != runs[0].example_ids for run in runs):
        raise ValueError("all runs must come from the same task copy and scorer version")
    rng = np.random.default_rng(seed)
    example_counts = {task: len(values) for task, values in control[0].gold_bpb.items()}
    example_draws = [
        relative_gains(
            candidate, control, {task: rng.integers(0, count, count) for task, count in example_counts.items()}
        )
        for _ in range(samples)
    ]
    gains = relative_gains(candidate, control)
    keys = (*GROUP_TASKS, "L")
    control_means = {key: float(np.mean([_run_values(run)[key] for run in control])) for key in keys}
    candidate_means = {key: float(np.mean([_run_values(run)[key] for run in candidate])) for key in keys}
    # The control mean is also the denominator of the gain, so its runs count with the factor
    # candidate mean / control mean (the delta method for a ratio).
    ratio = {key: candidate_means[key] / control_means[key] for key in keys}
    candidate_runs = _per_run_gains(candidate, control_means, dict.fromkeys(keys, 1.0))
    control_runs = _per_run_gains(control, control_means, ratio)
    lower, upper = {}, {}
    for key, gain in gains.items():
        candidate_var = candidate_runs[key].var(ddof=1) / len(candidate)
        control_var = control_runs[key].var(ddof=1) / len(control)
        seed_var = candidate_var + control_var
        if seed_var == 0:
            raise ValueError(f"the runs have no seed variance for {key}; compare runs with different seeds")
        # Welch degrees of freedom of the seed part; the example part has many examples, so it adds none.
        dof = seed_var**2 / (candidate_var**2 / (len(candidate) - 1) + control_var**2 / (len(control) - 1))
        example_var = np.var([draw[key] for draw in example_draws], ddof=1)
        half_width = float(student_t.ppf(0.975, dof) * math.sqrt(seed_var + example_var))
        lower[key], upper[key] = gain - half_width, gain + half_width
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
