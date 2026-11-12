# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Final eval metrics of Data Belay runs: answer BPB from their W&B summaries, choice loss from `data-belay score`."""

import dataclasses
import functools
import json
import re
import statistics
from collections.abc import Mapping
from typing import Any

import wandb
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.data_belay.judge import CHOICE_METRIC_PREFIX, GROUP_METRICS, group_values

# W&B project of the runs; the fast_track launcher logs there.
WANDB_PROJECT = "marin-community/marin_moe"
# Run ids start with this, apart from the runs that trained the Belay `vanilla` model.
RUN_PREFIX = "dbelay-main"
# Seeds of the shared control runs at each rung; every candidate at the rung is compared with all of them.
CONTROL_SEEDS = {"d512": (0, 1, 2, 3, 4, 5), "d768": (0, 1, 2), "d1024": (0, 1, 2)}
# Candidates whose text no eval measures, in scripts the English evals do not share: Japanese web text (ClimbLab) and
# Chinese, Thai and Arabic PDFs (FinePDFs), from the Datakit. Their cost against the control is what giving a
# candidate that share of the base mix costs on its own, and the judge measures harm against them. Keyed by the share
# in percent, as the candidate's label carries it (prepare.PreparedCache.candidate).
DILUTION_REFERENCES = {
    20: (
        "dk-climblab-ja-p20-cb41c3",
        "dk-finepdfs-cmn_hani-p20-860b9c",
        "dk-finepdfs-tha_thai-p20-cd56ab",
        "dk-finepdfs-arb_arab-p20-567b60",
    )
}
REFERENCE_SEEDS = (0, 1, 2)
# The share in a candidate's label: <name>-p<percent>-<cache hash>, or <name>-p<percent> for the first pilot caches.
LABEL_SHARE = re.compile(r"-p(\d+)(?:-[0-9a-f]{6})?$")
# Choice loss and accuracy of each run's final checkpoint, one JSON per run id, in this directory under the training
# cluster's MARIN_PREFIX. Outside the runs' 30-day TTL root, so the verdicts can be recomputed after the checkpoints
# are gone.
CHOICE_DIR = "data_belay/choice"


@dataclasses.dataclass(frozen=True)
class TaskScore:
    """One task's scores on a run's final checkpoint, from experiments.data_belay.accuracy."""

    acc: float
    acc_norm: float
    choice_bits: float
    questions: int


def control_run_id(rung: str, seed: int) -> str:
    return f"{RUN_PREFIX}-{rung}-control-s{seed}"


def treatment_run_id(rung: str, candidate: str, seed: int) -> str:
    return f"{RUN_PREFIX}-{rung}-{candidate}-s{seed}"


def choice_path(choice_root: str, run_id: str) -> str:
    return prefix_join(choice_root, f"{run_id}.json")


def scores_json(scores: Mapping[str, TaskScore]) -> str:
    return json.dumps({task: dataclasses.asdict(score) for task, score in scores.items()}, indent=2)


def parse_scores(text: str) -> dict[str, TaskScore]:
    return {task: TaskScore(**score) for task, score in json.loads(text).items()}


def run_exists(run_id: str) -> bool:
    return len(wandb.Api().runs(WANDB_PROJECT, filters={"name": run_id})) > 0


def run_scored(choice_root: str, run_id: str) -> bool:
    configure_coreweave_s3()
    return StoragePath(choice_path(choice_root, run_id)).exists()


def metrics_from(summary: Mapping[str, Any], scores: Mapping[str, TaskScore]) -> dict[str, float]:
    """Every metric in GROUP_METRICS, from a run's W&B summary and the scores that `data-belay score` wrote for it."""
    keys = [key for keys in GROUP_METRICS.values() for key in keys]
    return {
        key: (
            scores[key.removeprefix(CHOICE_METRIC_PREFIX)].choice_bits
            if key.startswith(CHOICE_METRIC_PREFIX)
            else float(summary[key])
        )
        for key in keys
    }


@functools.cache
def run_metrics(choice_root: str, run_id: str) -> dict[str, float]:
    """Every metric in GROUP_METRICS of a finished run; raises if the run did not finish or has no choice loss yet."""
    run = wandb.Api().run(f"{WANDB_PROJECT}/{run_id}")
    if run.state != "finished":
        raise ValueError(f"run {run_id} is {run.state}, not finished")
    path = choice_path(choice_root, run_id)
    if not run_scored(choice_root, run_id):
        raise ValueError(f"run {run_id} has no choice loss at {path}; `data-belay score` writes it")
    return metrics_from(run.summary, parse_scores(StoragePath(path).read_text()))


def arm_metrics(choice_root: str, rung: str, run_ids: list[str]) -> dict[str, list[float]]:
    """Per-group value of each run of one arm at ``rung``, in ``run_ids`` order."""
    controls = [run_metrics(choice_root, control_run_id(rung, seed)) for seed in CONTROL_SEEDS[rung]]
    control_means = {key: statistics.fmean(c[key] for c in controls) for key in controls[0]}
    per_run = [group_values(run_metrics(choice_root, run_id), control_means) for run_id in run_ids]
    return {group: [values[group] for values in per_run] for group in GROUP_METRICS}


def label_share(candidate: str) -> int | None:
    """The share in percent that a candidate's label carries, or None for a label without one."""
    match = LABEL_SHARE.search(candidate)
    return int(match.group(1)) if match else None


def dilution_references(candidate: str) -> tuple[str, ...]:
    """The dilution references that ``candidate`` is judged against: those at its share, leaving out itself."""
    return tuple(ref for ref in DILUTION_REFERENCES.get(label_share(candidate), ()) if ref != candidate)


def dilution_arm(choice_root: str, rung: str, candidate: str) -> dict[str, list[float]] | None:
    """Each dilution reference's mean group value at ``rung``, for a candidate at the references' share.

    One value per reference, not per seed: the references differ from each other by more than their seeds do, so
    pooling their seeds would narrow the interval. None when the candidate's share has no references. Raises when a
    reference is not scored at the rung, so that one ladder never mixes two harm baselines.
    """
    refs = dilution_references(candidate)
    if not refs:
        return None
    unscored = [
        ref for ref in refs if not all(run_scored(choice_root, treatment_run_id(rung, ref, s)) for s in REFERENCE_SEEDS)
    ]
    if unscored:
        raise ValueError(f"{candidate} at {rung}: dilution references not scored yet: {', '.join(unscored)}")
    arms = [arm_metrics(choice_root, rung, [treatment_run_id(rung, ref, s) for s in REFERENCE_SEEDS]) for ref in refs]
    return {group: [statistics.fmean(arm[group]) for arm in arms] for group in GROUP_METRICS}
