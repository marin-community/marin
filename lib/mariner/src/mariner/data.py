# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loading mariner-data, or one's own run tables, into the procedure's inputs.

A run table has one row per training run: its ``observation_id``, its mixture weights ``w__<bucket_id>`` (summing to
one), the fraction of each bucket's pool it drew from ``pf__<bucket_id>`` and its BPB on each task ``bpb__<task_id>``.
A swarm's run table also marks the runs a fit uses (``fit_inclusion``), labels the proportional mixture's fit run
``baseline_proportional`` (``run_label``) and gives the proportional repeats, which anchor the floors, the
``design_kind`` ``repeat``. A partition table lists each ``bucket_id`` with its ``pool_tokens``, and a tasks table each
``task_id`` with its ``objective`` and ``aggregate_weight``. A snapshot of the released dataset is read through its
``MANIFEST.json``, which gives each table's file and format.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from mariner.evaluate import HeldoutBank
from mariner.model import Anchor, Objective, Swarm
from mariner.optimize import MIXTURE_BLOCK_SIZE

DATASET_REPO = "marin-community/mariner-data"  # the published location; check at publication
SWARMS = ("qwen3_360m_1p6b", "llama_160m_1p2b", "llama_200m_6b")
MAIN_SWARM = "qwen3_360m_1p6b"
OBJECTIVES = ("uncheatable", "obe_easy")
PAPER_TARGET_BUDGET = 6_325_183_647_689  # D_tgt in tokens: the release's epochs per unit weight are D_tgt / P_i
CALIBRATION_RUN = "baseline_proportional"
REPEAT = "repeat"
JUNE_RUN_FLOPS = 3e18
JUNE_RUN_OBJECTIVE = "uncheatable"  # the one objective whose main-swarm anchors include the June run
TRAINED_MARINER = {"uncheatable": "mix_4c2c8b25f9e6d5ca", "obe_easy": "mix_dcbcb8cd4cfc72df"}

MANIFEST = "MANIFEST.json"
OBSERVATION_ID = "observation_id"
EXPOSURE_ID = "exposure_id"
WEIGHT_PREFIX = "w__"
POOL_FRACTION_PREFIX = "pf__"
OUTCOME_PREFIX = "bpb__"
WEIGHT_SUM_TOLERANCE = 1e-6
DOWNLOAD_PATTERNS = (
    MANIFEST,
    "README.md",
    "data/partition/*",
    "data/tasks/*",
    "data/swarm_*/*",
    "data/ladder/*",
    "data/heldout/*",
    "data/validation_3e18/*",
    "data/fits/*",
)


class TableFormat(StrEnum):
    CSV = "csv"
    PARQUET = "parquet"


@dataclasses.dataclass(frozen=True)
class Partition:
    """A partition's buckets with their pool sizes in tokens and the epochs one unit of weight materializes."""

    bucket_ids: tuple[str, ...]
    pool_tokens: np.ndarray
    epochs_per_unit_weight: np.ndarray  # D_tgt / P_i


# ---------------------------------------------------------------------------------------------------------------
# Run tables: partition, objectives, swarm, anchors and held-out bank
# ---------------------------------------------------------------------------------------------------------------


def read_csv(path: Path) -> pd.DataFrame:
    """``pd.read_csv`` with exact float parsing; pandas' default parser misreads some 17-digit values by one ulp."""
    return pd.read_csv(path, float_precision="round_trip")


def partition_from_table(table: pd.DataFrame, target_budget: float | None = None) -> Partition:
    """A partition table's buckets in table order.

    The epochs per unit weight are ``target_budget / pool_tokens`` for a target budget in tokens, else the table's
    ``epochs_per_unit_weight``.
    """
    pool_tokens = table["pool_tokens"].to_numpy(float)
    if target_budget is not None:
        epochs = target_budget / pool_tokens
    elif "epochs_per_unit_weight" in table.columns:
        epochs = table["epochs_per_unit_weight"].to_numpy(float)
    else:
        raise ValueError("the partition has no epochs_per_unit_weight column; give the target budget in tokens")
    return Partition(tuple(table["bucket_id"]), pool_tokens, epochs)


def read_partition(path: Path, target_budget: float | None = None) -> Partition:
    """The partition table (CSV) at ``path``; see ``partition_from_table``."""
    return partition_from_table(read_csv(path), target_budget)


def objective_from_tasks(tasks: pd.DataFrame, name: str) -> Objective:
    """The objective ``name`` of a tasks table: its tasks in table order, weighted by their ``aggregate_weight``."""
    rows = tasks[tasks["objective"] == name]
    if rows.empty:
        raise ValueError(f"no task belongs to objective {name}")
    return Objective(name, tuple(rows["task_id"]), rows["aggregate_weight"].to_numpy(float))


def require_columns(runs: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in runs.columns]
    if missing:
        raise ValueError(f"the run table has no column {', '.join(missing)}")


def bucket_values(runs: pd.DataFrame, prefix: str, bucket_ids: Sequence[str]) -> np.ndarray:
    columns = [f"{prefix}{bucket}" for bucket in bucket_ids]
    require_columns(runs, columns)
    return runs[columns].to_numpy(float)


def mixture_weights(runs: pd.DataFrame, bucket_ids: Sequence[str]) -> np.ndarray:
    """The runs' weights (runs x buckets); each run's must sum to one within 1e-6."""
    weights = bucket_values(runs, WEIGHT_PREFIX, bucket_ids)
    unnormalized = ~(np.abs(weights.sum(axis=1) - 1.0) <= WEIGHT_SUM_TOLERANCE)
    if unnormalized.any():
        raise ValueError(
            f"the weights of {unnormalized.sum()} of {len(runs)} runs do not sum to one within {WEIGHT_SUM_TOLERANCE:g}"
        )
    return weights


def pool_fractions(runs: pd.DataFrame, bucket_ids: Sequence[str]) -> np.ndarray:
    """The runs' pool fractions (runs x buckets); each must lie in (0, 1], since a zero gives infinite exposure."""
    fractions = bucket_values(runs, POOL_FRACTION_PREFIX, bucket_ids)
    outside = ~((fractions > 0.0) & (fractions <= 1.0))
    if outside.any():
        columns = [
            f"{POOL_FRACTION_PREFIX}{bucket}" for bucket, bad in zip(bucket_ids, outside.any(axis=0), strict=True) if bad
        ]
        raise ValueError(f"pool fractions must lie in (0, 1]: {', '.join(columns)}")
    return fractions


def measured_outcomes(runs: pd.DataFrame, task_ids: Sequence[str]) -> pd.DataFrame:
    """The runs' BPB on ``task_ids`` (runs x tasks, named by task id); every run must have every value."""
    columns = [f"{OUTCOME_PREFIX}{task}" for task in task_ids]
    require_columns(runs, columns)
    outcomes = runs[columns].astype(float)
    missing = outcomes.isna().sum()
    if missing.any():
        counts = ", ".join(f"{column} in {count} of {len(runs)} runs" for column, count in missing[missing > 0].items())
        raise ValueError(f"missing outcomes: {counts}")
    return outcomes.set_axis(list(task_ids), axis=1).reset_index(drop=True)


def swarm_from_runs(runs: pd.DataFrame, partition: Partition, task_ids: Sequence[str]) -> Swarm:
    """The fit rows (``fit_inclusion``) of a swarm's run table, in table order, over the partition's buckets.

    The calibration run is the fit row with ``run_label`` ``baseline_proportional``. Raises ``ValueError`` when no fit
    row has that label, a fit row lacks an outcome of ``task_ids``, a fit row's weights do not sum to one within 1e-6,
    a bucket's ``w__`` or ``pf__`` column is missing, or a pool fraction lies outside (0, 1].
    """
    fit = runs[runs["fit_inclusion"]].reset_index(drop=True)
    calibration = (fit["run_label"] == CALIBRATION_RUN).to_numpy()
    if not calibration.any():
        raise ValueError(f"no fit row has run_label {CALIBRATION_RUN}, the calibration run")
    return Swarm(
        runs=tuple(fit[OBSERVATION_ID]),
        buckets=partition.bucket_ids,
        weights=mixture_weights(fit, partition.bucket_ids),
        epochs_per_unit_weight=partition.epochs_per_unit_weight,
        pool_fractions=pool_fractions(fit, partition.bucket_ids),
        outcomes=measured_outcomes(fit, task_ids),
        calibration=calibration,
    )


def anchors_from_runs(runs: pd.DataFrame, task_ids: Sequence[str]) -> dict[str, Anchor]:
    """Each task's anchor over the proportional runs ``runs``: their mean and sample standard deviation (ddof 1)."""
    if len(runs) < 2:
        raise ValueError(f"the floors need at least two proportional runs for a standard deviation, got {len(runs)}")
    outcomes = measured_outcomes(runs, task_ids)
    return {task: Anchor(outcomes[task].mean(), outcomes[task].std(ddof=1)) for task in task_ids}


def heldout_bank(runs: pd.DataFrame, partition: Partition, objective: Objective) -> HeldoutBank:
    """Held-out runs as a bank of exposures, in order of first appearance.

    The runs that share an ``exposure_id`` (each run alone when the table has no such column) form one exposure, with
    its first run's weights and pool fractions. An exposure's measured value of a task is the mean over its runs that
    have one; NaN when none has.
    """
    runs = runs.reset_index(drop=True)
    weights = mixture_weights(runs, partition.bucket_ids)
    fractions = pool_fractions(runs, partition.bucket_ids)
    columns = [f"{OUTCOME_PREFIX}{task}" for task in objective.tasks]
    require_columns(runs, columns)
    keys = runs[EXPOSURE_ID if EXPOSURE_ID in runs.columns else OBSERVATION_ID]
    if keys.isna().any():
        raise ValueError(f"{keys.isna().sum()} of {len(runs)} runs have no {keys.name}")
    first = keys.drop_duplicates().index.to_numpy()
    grouped = runs[columns].astype(float).groupby(keys.to_numpy(), sort=False)
    return HeldoutBank(
        exposure_ids=tuple(keys.iloc[first]),
        weights=weights[first],
        pool_fractions=fractions[first],
        measured=grouped.mean().set_axis(list(objective.tasks), axis=1).reset_index(drop=True),
        runs=grouped.size().to_numpy(),
    )


# ---------------------------------------------------------------------------------------------------------------
# Snapshots of mariner-data
# ---------------------------------------------------------------------------------------------------------------


def read_table(root: Path, name: str) -> pd.DataFrame:
    """The table ``name`` of the snapshot at ``root``, read from the file and in the format its MANIFEST.json gives."""
    entries = [entry for entry in json.loads((root / MANIFEST).read_text())["tables"] if entry["table"] == name]
    if len(entries) != 1:
        raise ValueError(f"{MANIFEST} lists {len(entries)} tables named {name}")
    path = root / entries[0]["path"]
    if TableFormat(entries[0]["format"]) is TableFormat.PARQUET:
        return pd.read_parquet(path)
    return read_csv(path)


def release_partition(root: Path) -> Partition:
    return partition_from_table(read_table(root, "partition"))


def release_tasks(root: Path) -> pd.DataFrame:
    return read_table(root, "tasks")


def release_swarm(root: Path, swarm: str, objective: str) -> Swarm:
    """The fit rows of the released swarm ``swarm`` (table ``swarm_<swarm>``), with the outcomes of ``objective``."""
    task_ids = objective_from_tasks(release_tasks(root), objective).tasks
    return swarm_from_runs(read_table(root, f"swarm_{swarm}"), release_partition(root), task_ids)


def june_run(root: Path, swarm_table: pd.DataFrame) -> pd.DataFrame:
    """The June proportional run: the one ``ladder`` run of the proportional mixture at 3e18 FLOPs outside the swarm."""
    ladder = read_table(root, "ladder")
    june = ladder[
        (ladder["method"] == "proportional")
        & (ladder["compute_flops"] == JUNE_RUN_FLOPS)
        & ~ladder[OBSERVATION_ID].isin(swarm_table[OBSERVATION_ID])
    ]
    if len(june) != 1:
        raise ValueError(
            f"expected one proportional ladder run at {JUNE_RUN_FLOPS:g} FLOPs outside the swarm, got {len(june)}"
        )
    return june


def release_anchors(root: Path, swarm: str, objective: str) -> dict[str, Anchor]:
    """The anchors of a released swarm's fit, over its proportional repeats (``design_kind`` ``repeat``).

    The main swarm's Uncheatable anchors also include the June proportional run (``june_run``), as the paper's
    reference fit did.
    """
    table = read_table(root, f"swarm_{swarm}")
    runs = table[table["design_kind"] == REPEAT]
    if (swarm, objective) == (MAIN_SWARM, JUNE_RUN_OBJECTIVE):
        runs = pd.concat([runs, june_run(root, table)])
    return anchors_from_runs(runs, objective_from_tasks(release_tasks(root), objective).tasks)


def trained_counts(root: Path, mixture_id: str) -> np.ndarray:
    """The block counts of a trained mixture: its ``validation_3e18`` weights times 2,048, which must be integers."""
    table = read_table(root, "validation_3e18")
    rows = table[table["mixture_id"] == mixture_id]
    if rows.empty:
        raise ValueError(f"no validation_3e18 run trained {mixture_id}")
    blocks = bucket_values(rows, WEIGHT_PREFIX, release_partition(root).bucket_ids) * MIXTURE_BLOCK_SIZE
    counts = np.rint(blocks[0])
    if not (blocks == counts).all():
        raise ValueError(f"the runs of {mixture_id} do not share one mixture on the {MIXTURE_BLOCK_SIZE}-block grid")
    return counts.astype(np.int64)


def released_task_predictions(root: Path, table: str, objective: str) -> pd.DataFrame:
    """A released predictions table (``fits_swarm_predictions``, ``fits_heldout_predictions``) for ``objective``.

    Predicted BPB by observation id (rows) and task id (columns), both in the table's order.
    """
    frame = read_table(root, table)
    frame = frame[frame["objective"] == objective]
    wide = frame.pivot(index=OBSERVATION_ID, columns="task_id", values="predicted_bpb")
    return wide.loc[frame[OBSERVATION_ID].unique(), frame["task_id"].unique()]


def released_task_parameters(root: Path, swarm: str, objective: str) -> pd.DataFrame:
    """The released per-task parameters (``fits_task_parameters``) of one swarm and objective, indexed by task id."""
    frame = read_table(root, "fits_task_parameters")
    rows = frame[(frame["swarm"] == swarm) & (frame["objective"] == objective)]
    return rows.drop(columns=["swarm", "objective"]).set_index("task_id")


def release_heldout_bank(root: Path, objective: str) -> HeldoutBank:
    """The paper's held-out bank: the ``heldout`` runs that ``fits_heldout_predictions`` predicts for ``objective``."""
    predictions = read_table(root, "fits_heldout_predictions")
    population = predictions.loc[predictions["objective"] == objective, OBSERVATION_ID]
    heldout = read_table(root, "heldout")
    return heldout_bank(
        heldout[heldout[OBSERVATION_ID].isin(population)],
        release_partition(root),
        objective_from_tasks(release_tasks(root), objective),
    )


def download(local_dir: Path, repo_id: str = DATASET_REPO, revision: str | None = None) -> Path:
    """Download MANIFEST.json, the card and the tables this package reads into ``local_dir`` and return its path.

    The tables are partition, tasks, the three swarms, ladder, heldout, validation_3e18 and the fits; the StarCoder,
    domain-sweep, other ladder and run-set tables stay on the Hub.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415  (optional dependency: the download extra)

    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
            allow_patterns=list(DOWNLOAD_PATTERNS),
        )
    )
