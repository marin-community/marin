# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loading mariner-data, or one's own run tables, into the procedure's inputs.

A run table has one row per training run: its ``observation_id``, its mixture weights ``w__<bucket_id>`` (nonnegative,
summing to one), the fraction of each bucket's pool it drew from ``pf__<bucket_id>`` and its BPB on each task
``bpb__<task_id>`` (positive). A swarm's run table also marks the runs a fit uses (``fit_inclusion``, True or False),
labels the proportional mixture's fit run ``baseline_proportional`` (``run_label``) and gives the proportional repeats,
which anchor the floors and must train that run's mixture, the ``design_kind`` ``repeat``. A partition table lists
each ``bucket_id`` with its ``pool_tokens`` (positive), and a tasks table each ``task_id`` with its ``objective`` and
``aggregate_weight`` (positive). Identifier columns are read as text, and no row may leave a required one blank
(missing, empty or only whitespace). Observation ids and bucket ids are unique, and so are task ids within an
objective. A snapshot of the released dataset is read through its ``MANIFEST.json``, which gives each table's file and
format.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from mariner.model import WEIGHT_SUM_TOLERANCE, Anchor, HeldoutBank, Objective, Swarm, shown_values
from mariner.olmix import LogLinearLaw
from mariner.optimize import MIXTURE_BLOCK_SIZE

DATASET_REPO = "marin-community/mariner-data"  # the published location
SWARMS = ("qwen3_360m_1p6b", "llama_160m_1p2b", "llama_200m_6b")
MAIN_SWARM = "qwen3_360m_1p6b"


class ObjectiveName(StrEnum):
    """The paper's two objectives, by their names in the tasks table's ``objective`` column. A member equals and hashes
    as its name, so it can stand wherever the name does."""

    UNCHEATABLE = "uncheatable"
    OBE_EASY = "obe_easy"


OBJECTIVES = tuple(ObjectiveName)
PAPER_TARGET_BUDGET = 6_325_183_647_689  # D_tgt in tokens: the release's epochs per unit weight are D_tgt / P_i
JUNE_RUN_FLOPS = 3e18
JUNE_RUN_OBJECTIVE = ObjectiveName.UNCHEATABLE  # the one objective whose main-swarm anchors include the June run
TRAINED_MARINER: dict[str, str] = {
    ObjectiveName.UNCHEATABLE: "mix_4c2c8b25f9e6d5ca",
    ObjectiveName.OBE_EASY: "mix_dcbcb8cd4cfc72df",
}


class TrainedOlmix(NamedTuple):
    """A matched Olmix mixture that the paper trained, and the KL weight that proposed it under the epoch cap."""

    mixture_id: str
    kl: float


TRAINED_OLMIX: dict[str, TrainedOlmix] = {
    ObjectiveName.UNCHEATABLE: TrainedOlmix("mix_645a3bf43226a443", 0.05),
    ObjectiveName.OBE_EASY: TrainedOlmix("mix_40fb2116404045a4", 0.005),
}
OLMIX_CAP = 4.0

# The run-table schema, which mariner.design writes: columns, the prefixes of the per-bucket and per-task columns,
# the calibration run's label and the design kinds.
OBSERVATION_ID = "observation_id"
RUN_LABEL = "run_label"
DESIGN_KIND = "design_kind"
FIT_INCLUSION = "fit_inclusion"
EXPOSURE_ID = "exposure_id"
# The columns that name a trained mixture's runs in validation_3e18, heldout and ladder.
MIXTURE_ID = "mixture_id"
LAUNCH = "launch"
TRAINER_SEED = "trainer_seed"
PROPOSAL_OBJECTIVE = "proposal_objective"  # the objective a surrogate proposed the mixture for, "none" for a baseline
WEIGHT_PREFIX = "w__"
POOL_FRACTION_PREFIX = "pf__"
OUTCOME_PREFIX = "bpb__"
BASELINE_PREFIX = "baseline_"  # a baseline mixture's run label is baseline_<name>
PROPORTIONAL = "proportional"  # the baseline that calibrates the fit and that the repeats train
CALIBRATION_RUN = f"{BASELINE_PREFIX}{PROPORTIONAL}"
BASELINE = "baseline"
SAMPLED = "sampled"
DELETION = "deletion"
REPEAT = "repeat"
# The partition and tasks tables' columns.
BUCKET_ID = "bucket_id"
POOL_TOKENS = "pool_tokens"
EPOCHS_PER_UNIT_WEIGHT = "epochs_per_unit_weight"
TASK_ID = "task_id"
OBJECTIVE_COLUMN = "objective"
AGGREGATE_WEIGHT = "aggregate_weight"
SWARM_COLUMN = "swarm"  # the fits tables' column naming the released swarm of each fit

# Each table's required columns besides the w__, pf__ and bpb__ ones, and its name in error messages.
RUN_COLUMNS = (OBSERVATION_ID, RUN_LABEL, DESIGN_KIND, FIT_INCLUSION)
PARTITION_COLUMNS = (BUCKET_ID, POOL_TOKENS)
TASK_COLUMNS = (TASK_ID, OBJECTIVE_COLUMN, AGGREGATE_WEIGHT)
RUN_TABLE = "run"
# The snapshot's MANIFEST.json names these tables too.
PARTITION_TABLE = "partition"
TASKS_TABLE = "tasks"
HELDOUT_TABLE = "heldout"
VALIDATION_TABLE = "validation_3e18"
LADDER_TABLE = "ladder"
LADDER_ACCURACY_TABLE = "ladder_accuracy"
# The release's buckets by source, in the paper's order: the Common Crawl topics at two quality tiers, Dolma 3's other
# sources, and Dolmino's; and the partition table's pool names of the last two.
COMMON_CRAWL, DOLMA3, DOLMINO = "Common Crawl", "Dolma 3, other sources", "Dolmino"
SOURCES = (COMMON_CRAWL, DOLMA3, DOLMINO)
DOLMA3_POOL, DOLMINO_POOL = "dolma3", "dolmino"
# Read as text, so that bucket ids such as 00 and numeric observation ids keep their spelling.
IDENTIFIER_COLUMNS = (BUCKET_ID, TASK_ID, OBJECTIVE_COLUMN, OBSERVATION_ID, RUN_LABEL, DESIGN_KIND, EXPOSURE_ID)
MISSING_NOTE = "pandas reads empty cells and tokens such as NA, N/A, null, None and nan as missing"

MANIFEST = "MANIFEST.json"
FIGURES_INDEX = "figures_tables.csv"  # the paper's figures and tables with the configs each reads
# The released fits' per-task predictions at the swarm's runs and at the held-out runs.
SWARM_PREDICTIONS_TABLE = "fits_swarm_predictions"
HELDOUT_PREDICTIONS_TABLE = "fits_heldout_predictions"
PREDICTED_BPB = "predicted_bpb"  # the prediction tables' column of MARINER's predicted BPB
COEFFICIENT_PREFIX = "coef__"
# fits_reliability holds rows from three sources, named in its source_table column: each objective's tasks, with the
# objective's aggregate as the row without a task id, and OlmoBaseEval Easy's tasks with their subtasks collapsed.
RELIABILITY_TABLE = "fits_reliability"
RELIABILITY_SOURCES: dict[str, str] = {
    ObjectiveName.UNCHEATABLE: "snr_fit_uncheatable_delphi",
    ObjectiveName.OBE_EASY: "snr_fit_components_delphi",
}
RELIABILITY_COLLAPSED_SOURCE = "snr_fit_tasks_delphi"
# Floating rounding: the release's runs at one exposure differ by 4.2e-16 at most, and its proportional repeats
# differ from their calibration run by 3.1e-16 at most.
EXPOSURE_MIXTURE_TOLERANCE = 1e-12
HUGGINGFACE_HUB_MISSING = "downloading mariner-data needs huggingface_hub, which marin-mariner's download extra installs"
DOWNLOAD_PATTERNS = (
    MANIFEST,
    "README.md",
    FIGURES_INDEX,
    "data/partition/*",
    "data/tasks/*",
    "data/swarm_*/*",
    "data/ladder/*",
    "data/ladder_accuracy/*",
    "data/ladder_compute_equivalence/*",
    "data/heldout/*",
    "data/validation_3e18/*",
    "data/fits/*",
    "data/domain_sweeps/*",
    # The StarCoder tables that the figures read. The trajectories, the dense grid, the fit parameters and the repeat
    # noise stay on the Hub.
    "data/starcoder/starcoder_observations.*",
    "data/starcoder/starcoder_curve_membership.*",
    "data/starcoder/starcoder_curves.*",
    "data/starcoder/starcoder_diagonal.*",
    "data/starcoder/starcoder_epoch_matching.*",
    "data/starcoder/starcoder_epoch_matching_grid.*",
    "data/starcoder/starcoder_curricula.*",
    "data/starcoder/starcoder_curricula_matched_nd.*",
    "data/starcoder/starcoder_curricula_matched_nd_confirmation.*",
    "data/starcoder/starcoder_transition.*",
    "data/starcoder/starcoder_ramp_designs.*",
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
    """``pd.read_csv`` with exact float parsing, and with the identifier columns the table has read as text.

    Pandas' default parser misreads 16,973 values of the released tables this package reads, by up to 921 ulps at
    small magnitudes. As text, bucket ids such as ``00`` and numeric observation ids keep their spelling.
    """
    return pd.read_csv(path, float_precision="round_trip", dtype=dict.fromkeys(IDENTIFIER_COLUMNS, str))


def require_columns(table: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Raise ``ValueError`` naming the ``columns`` that ``table``, the table called ``name`` in messages, lacks."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"the {name} table has no column {', '.join(missing)}")


def blank_identifier_mask(identifiers: pd.Series) -> pd.Series:
    """The boolean mask of the ``identifiers`` that are missing, empty or only whitespace."""
    return identifiers.isna() | (identifiers.astype(str).str.strip() == "")


def require_identifiers(table: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Raise ``ValueError`` naming the identifier ``columns`` of ``table`` (called ``name``) that a row leaves blank.

    A blank identifier is missing, empty or only whitespace.
    """
    blanks = {column: int(blank_identifier_mask(table[column]).sum()) for column in columns}
    counts = ", ".join(f"{column} ({count} of {len(table)} rows)" for column, count in blanks.items() if count)
    if counts:
        raise ValueError(
            f"the {name} table has rows without {counts}; an identifier must not be empty or whitespace, and"
            f" {MISSING_NOTE}"
        )


def require_unique(ids: pd.Series, where: str) -> None:
    """Raise ``ValueError`` naming the values of the identifier column ``ids`` that repeat ``where``."""
    repeated = ids[ids.duplicated()].unique()
    if len(repeated):
        raise ValueError(f"{ids.name} must be unique {where}; repeated: {shown_values(repeated)}")


def require_positive(values: np.ndarray, ids: Sequence[str], name: str) -> None:
    """Raise ``ValueError`` naming the ``ids`` whose value of ``name`` in ``values`` is not positive and finite."""
    invalid = [
        identifier for identifier, value in zip(ids, values, strict=True) if not (np.isfinite(value) and value > 0)
    ]
    if invalid:
        raise ValueError(f"{name} must be positive and finite, which it is not for {shown_values(invalid)}")


def partition_from_table(table: pd.DataFrame, target_budget: float | None = None) -> Partition:
    """A partition table's buckets in table order.

    The epochs per unit weight are ``target_budget / pool_tokens`` for a target budget in tokens, else the table's
    ``epochs_per_unit_weight``. Every row needs a ``bucket_id`` that no other row has, and the pools, the target
    budget and the epochs per unit weight must be positive and finite.
    """
    require_columns(table, PARTITION_COLUMNS, PARTITION_TABLE)
    require_identifiers(table, (BUCKET_ID,), PARTITION_TABLE)
    require_unique(table[BUCKET_ID], f"in the {PARTITION_TABLE} table")
    bucket_ids = tuple(table[BUCKET_ID])
    pool_tokens = table[POOL_TOKENS].to_numpy(float)
    require_positive(pool_tokens, bucket_ids, POOL_TOKENS)
    if target_budget is not None:
        if not (np.isfinite(target_budget) and target_budget > 0):
            raise ValueError(f"the target budget must be a positive and finite number of tokens, got {target_budget:g}")
        epochs = target_budget / pool_tokens
    elif EPOCHS_PER_UNIT_WEIGHT in table.columns:
        epochs = table[EPOCHS_PER_UNIT_WEIGHT].to_numpy(float)
    else:
        raise ValueError(
            f"the {PARTITION_TABLE} table has no {EPOCHS_PER_UNIT_WEIGHT} column; give the target budget in tokens"
        )
    require_positive(epochs, bucket_ids, EPOCHS_PER_UNIT_WEIGHT)
    return Partition(bucket_ids, pool_tokens, epochs)


def read_partition(path: Path, target_budget: float | None = None) -> Partition:
    """The partition table (CSV) at ``path``; see ``partition_from_table``."""
    return partition_from_table(read_csv(path), target_budget)


def proportional_epochs(partition: Partition) -> float:
    """The epochs that proportional mixing gives every bucket at the paper's target budget: weighting the buckets by
    their pools, it covers the whole partition D_tgt / sum(P_i) times."""
    return float(PAPER_TARGET_BUDGET / partition.pool_tokens.sum())


def bucket_sources(partition: pd.DataFrame) -> pd.Series:
    """Each bucket of the release's partition table by its source, by bucket id: Common Crawl for the topics at a
    quality tier, else its pool's (``SOURCES``)."""
    require_columns(partition, [BUCKET_ID, "pool", "quality_tier"], PARTITION_TABLE)
    source = np.where(
        partition["quality_tier"].notna(),
        COMMON_CRAWL,
        np.where(partition["pool"] == DOLMINO_POOL, DOLMINO, DOLMA3),
    )
    unknown = sorted(set(partition.loc[~partition["pool"].isin([DOLMA3_POOL, DOLMINO_POOL]), "pool"]))
    if unknown:
        raise ValueError(f"the partition has buckets of the unknown pools {', '.join(map(str, unknown))}")
    return pd.Series(source, index=partition[BUCKET_ID])


def bucket_names(partition: pd.DataFrame) -> pd.Series:
    """Each bucket's name in the paper, by bucket id, with the release's partition table's LaTeX ampersands as plain
    text."""
    require_columns(partition, [BUCKET_ID, "paper_name"], PARTITION_TABLE)
    return partition.set_index(BUCKET_ID)["paper_name"].str.replace("\\&", "&", regex=False)


def objective_from_tasks(tasks: pd.DataFrame, name: str) -> Objective:
    """The objective ``name`` of a tasks table: its tasks in table order, weighted by their ``aggregate_weight``.

    Every row needs a ``task_id`` and an ``objective``, a task may appear once in an objective, and the objective's
    weights must be positive and finite.
    """
    require_columns(tasks, TASK_COLUMNS, TASKS_TABLE)
    require_identifiers(tasks, (TASK_ID, OBJECTIVE_COLUMN), TASKS_TABLE)
    rows = tasks[tasks[OBJECTIVE_COLUMN] == name]
    if rows.empty:
        raise ValueError(f"no task belongs to objective {name}")
    require_unique(rows[TASK_ID], f"within objective {name}")
    task_ids, weights = tuple(rows[TASK_ID]), rows[AGGREGATE_WEIGHT].to_numpy(float)
    require_positive(weights, task_ids, AGGREGATE_WEIGHT)
    return Objective(name, task_ids, weights)


def bucket_values(table: pd.DataFrame, prefix: str, bucket_ids: Sequence[str], name: str) -> np.ndarray:
    """The columns ``<prefix><bucket_id>`` of ``table``, called ``name`` in messages, as floats (rows x buckets)."""
    columns = [f"{prefix}{bucket}" for bucket in bucket_ids]
    require_columns(table, columns, name)
    return table[columns].to_numpy(float)


def flagged_columns(prefix: str, ids: Sequence[str], flagged: np.ndarray) -> str:
    """The columns ``<prefix><id>`` that hold a flagged value, at most ``SHOWN_VALUES`` of them (``shown_values``).

    ``flagged`` is (runs, ids), ids being buckets or tasks.
    """
    columns = [f"{prefix}{identifier}" for identifier, bad in zip(ids, flagged.any(axis=0), strict=True) if bad]
    return shown_values(columns)


def mixture_weights(table: pd.DataFrame, bucket_ids: Sequence[str], name: str) -> np.ndarray:
    """The weights (rows x buckets) of ``table``: finite and nonnegative, each row summing to one within 1e-6.

    ``name`` is the table's name in messages.
    """
    weights = bucket_values(table, WEIGHT_PREFIX, bucket_ids, name)
    invalid = ~(np.isfinite(weights) & (weights >= 0.0))
    if invalid.any():
        raise ValueError(
            f"weights must be finite and nonnegative: {flagged_columns(WEIGHT_PREFIX, bucket_ids, invalid)}"
        )
    unnormalized = ~(np.abs(weights.sum(axis=1) - 1.0) <= WEIGHT_SUM_TOLERANCE)
    if unnormalized.any():
        count = int(unnormalized.sum())
        noun = "row" if count == 1 else "rows"
        raise ValueError(
            f"the weights of {count} {noun} of the {name} table do not sum to one within {WEIGHT_SUM_TOLERANCE:g}"
        )
    return weights


def pool_fractions(table: pd.DataFrame, bucket_ids: Sequence[str], name: str) -> np.ndarray:
    """The pool fractions (rows x buckets) of ``table``; each must lie in (0, 1], since a zero gives infinite exposure.

    ``name`` is the table's name in messages.
    """
    fractions = bucket_values(table, POOL_FRACTION_PREFIX, bucket_ids, name)
    outside = ~((fractions > 0.0) & (fractions <= 1.0))
    if outside.any():
        raise ValueError(
            f"pool fractions must lie in (0, 1]: {flagged_columns(POOL_FRACTION_PREFIX, bucket_ids, outside)}"
        )
    return fractions


def measured_outcomes(runs: pd.DataFrame, task_ids: Sequence[str]) -> pd.DataFrame:
    """The runs' BPB on ``task_ids`` (runs x tasks, named by task id): every run's value, positive and finite."""
    columns = [f"{OUTCOME_PREFIX}{task}" for task in task_ids]
    require_columns(runs, columns, RUN_TABLE)
    outcomes = runs[columns].astype(float)
    missing = outcomes.isna().sum()
    if missing.any():
        counts = ", ".join(f"{column} in {count} of {len(runs)} runs" for column, count in missing[missing > 0].items())
        raise ValueError(f"missing outcomes: {counts}")
    values = outcomes.to_numpy()
    invalid = ~(np.isfinite(values) & (values > 0.0))
    if invalid.any():
        raise ValueError(f"outcomes must be positive and finite: {flagged_columns(OUTCOME_PREFIX, task_ids, invalid)}")
    return outcomes.set_axis(list(task_ids), axis=1).reset_index(drop=True)


def swarm_from_runs(runs: pd.DataFrame, partition: Partition, task_ids: Sequence[str]) -> Swarm:
    """The fit rows (``fit_inclusion``) of a swarm's run table, in table order, over the partition's buckets.

    The calibration run is the fit row with ``run_label`` ``baseline_proportional``. Raises ``ValueError`` when the
    table lacks ``observation_id``, ``run_label``, ``design_kind`` or ``fit_inclusion``, a row leaves one of the first
    three blank, two rows share an ``observation_id``, ``fit_inclusion`` holds anything but True and False (a missing
    value included), no fit row has that label, a fit row lacks an outcome of ``task_ids`` or has one that is not
    positive and finite, a fit row has a weight that is not finite and nonnegative or weights that do not sum to one
    within 1e-6, a bucket's ``w__`` or ``pf__`` column is missing, or a pool fraction lies outside (0, 1].
    """
    require_columns(runs, RUN_COLUMNS, RUN_TABLE)
    require_identifiers(runs, (OBSERVATION_ID, RUN_LABEL, DESIGN_KIND), RUN_TABLE)
    require_unique(runs[OBSERVATION_ID], f"in the {RUN_TABLE} table")
    inclusion = runs[FIT_INCLUSION]
    # is_bool_dtype holds for pandas' nullable boolean dtype, whose missing values the mask below would leave out.
    if not pd.api.types.is_bool_dtype(inclusion) or inclusion.isna().any():
        values = shown_values(inclusion.unique())
        raise ValueError(f"{FIT_INCLUSION} must be True or False, but it holds {inclusion.dtype} values: {values}")
    fit = runs[inclusion].reset_index(drop=True)
    calibration = (fit[RUN_LABEL] == CALIBRATION_RUN).to_numpy()
    if not calibration.any():
        raise ValueError(f"no fit row has run_label {CALIBRATION_RUN}, the calibration run")
    return Swarm(
        runs=tuple(fit[OBSERVATION_ID]),
        buckets=partition.bucket_ids,
        weights=mixture_weights(fit, partition.bucket_ids, RUN_TABLE),
        epochs_per_unit_weight=partition.epochs_per_unit_weight,
        pool_fractions=pool_fractions(fit, partition.bucket_ids, RUN_TABLE),
        outcomes=measured_outcomes(fit, task_ids),
        calibration=calibration,
    )


def anchors_from_runs(swarm: Swarm, runs: pd.DataFrame, task_ids: Sequence[str]) -> dict[str, Anchor]:
    """Each task's anchor over the proportional runs ``runs``: their mean and sample standard deviation (ddof 1).

    Every run must train the calibration mixture of ``swarm``: its weights and pool fractions must equal the
    calibration run's within 1e-12. Raises ``ValueError`` naming the runs that do not, or when there are fewer than
    two runs.
    """
    if len(runs) < 2:
        raise ValueError(f"the floors need at least two proportional runs for a standard deviation, got {len(runs)}")
    weights, fractions = mixture_weights(runs, swarm.buckets, RUN_TABLE), pool_fractions(runs, swarm.buckets, RUN_TABLE)
    mixtures = np.hstack([weights, fractions])
    calibration = np.hstack([swarm.weights, swarm.pool_fractions])[swarm.calibration]
    other = np.abs(mixtures[:, None, :] - calibration[None, :, :]).max(axis=(1, 2)) > EXPOSURE_MIXTURE_TOLERANCE
    if other.any():
        raise ValueError(
            f"the floors' proportional runs must train the mixture of {CALIBRATION_RUN}, but the weights or pool"
            f" fractions of {', '.join(map(str, runs[OBSERVATION_ID][other]))} differ from it by more than"
            f" {EXPOSURE_MIXTURE_TOLERANCE:g}"
        )
    outcomes = measured_outcomes(runs, task_ids)
    return {task: Anchor(outcomes[task].mean(), outcomes[task].std(ddof=1)) for task in task_ids}


def heldout_bank(runs: pd.DataFrame, bucket_ids: Sequence[str], objective: Objective) -> HeldoutBank:
    """Held-out runs as a bank of exposures over ``bucket_ids``, in order of first appearance.

    The runs that share an ``exposure_id`` (each run alone, by its ``observation_id``, when the table has no such
    column) form one exposure, and must share their weights and pool fractions within 1e-12. No run may leave that key
    blank (missing, empty or only whitespace). An exposure's measured value of a task is the mean over its runs that
    have one, NaN when none has. Its measured objective is the mean of its runs'
    ``bpb__<objective name>`` when the table has that column, which can record the objective for a run without
    per-task values; otherwise it is the task means weighted by the objective. A run may lack a value (NaN), but every
    value it has must be positive and finite, and the table must have a run.
    """
    if len(runs) == 0:
        raise ValueError(f"the {HELDOUT_TABLE} table has no runs")
    runs = runs.reset_index(drop=True)
    weights = mixture_weights(runs, bucket_ids, HELDOUT_TABLE)
    fractions = pool_fractions(runs, bucket_ids, HELDOUT_TABLE)
    columns = [f"{OUTCOME_PREFIX}{task}" for task in objective.tasks]
    require_columns(runs, columns, HELDOUT_TABLE)
    aggregate = f"{OUTCOME_PREFIX}{objective.name}"
    outcome_ids = (*objective.tasks, objective.name) if aggregate in runs.columns else objective.tasks
    values = runs[[f"{OUTCOME_PREFIX}{identifier}" for identifier in outcome_ids]].to_numpy(float)
    invalid = ~(np.isnan(values) | (np.isfinite(values) & (values > 0.0)))
    if invalid.any():
        raise ValueError(
            f"the {HELDOUT_TABLE} table's outcomes must be positive and finite where present:"
            f" {flagged_columns(OUTCOME_PREFIX, outcome_ids, invalid)}"
        )
    keys = runs[EXPOSURE_ID if EXPOSURE_ID in runs.columns else OBSERVATION_ID]
    blank = blank_identifier_mask(keys)
    if blank.any():
        raise ValueError(
            f"{blank.sum()} of {len(runs)} runs have no {keys.name}; an identifier must not be empty or whitespace"
        )
    exposure, _ = pd.factorize(keys)  # each run's exposure, numbered in order of first appearance
    first = keys.drop_duplicates().index.to_numpy()
    deviation = np.maximum(
        np.abs(weights - weights[first][exposure]).max(axis=1),
        np.abs(fractions - fractions[first][exposure]).max(axis=1),
    )
    differing = keys[deviation > EXPOSURE_MIXTURE_TOLERANCE].unique()
    if len(differing):
        raise ValueError(
            f"runs at one exposure differ in weights or pool fractions beyond {EXPOSURE_MIXTURE_TOLERANCE:g}:"
            f" {', '.join(map(str, differing))}"
        )
    grouped = runs[columns].astype(float).groupby(exposure, sort=False)
    measured = grouped.mean().set_axis(list(objective.tasks), axis=1).reset_index(drop=True)
    if aggregate in runs.columns:
        measured_objective = runs[aggregate].astype(float).groupby(exposure, sort=False).mean().to_numpy()
    else:
        measured_objective = measured.to_numpy() @ objective.weights
    return HeldoutBank(
        exposure_ids=tuple(map(str, keys.iloc[first])),
        buckets=tuple(bucket_ids),
        weights=weights[first],
        pool_fractions=fractions[first],
        measured=measured,
        measured_objective=measured_objective,
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
    return partition_from_table(read_table(root, PARTITION_TABLE))


def release_tasks(root: Path) -> pd.DataFrame:
    return read_table(root, TASKS_TABLE)


def release_runs(root: Path, swarm: str) -> pd.DataFrame:
    """The run table of the released swarm ``swarm`` (table ``swarm_<swarm>``)."""
    return read_table(root, f"swarm_{swarm}")


def release_swarm(root: Path, swarm: str, objective: str) -> Swarm:
    """The fit rows of the released swarm ``swarm``, with the outcomes of ``objective``."""
    task_ids = objective_from_tasks(release_tasks(root), objective).tasks
    return swarm_from_runs(release_runs(root, swarm), release_partition(root), task_ids)


def june_run(root: Path, swarm_table: pd.DataFrame) -> pd.DataFrame:
    """The June proportional run: the one ``ladder`` run of the proportional mixture at 3e18 FLOPs outside the swarm."""
    ladder = read_table(root, LADDER_TABLE)
    june = ladder[
        (ladder["method"] == PROPORTIONAL)
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
    table = release_runs(root, swarm)
    task_ids = objective_from_tasks(release_tasks(root), objective).tasks
    runs = table[table[DESIGN_KIND] == REPEAT]
    if (swarm, objective) == (MAIN_SWARM, JUNE_RUN_OBJECTIVE):
        runs = pd.concat([runs, june_run(root, table)])
    return anchors_from_runs(swarm_from_runs(table, release_partition(root), task_ids), runs, task_ids)


def trained_counts(root: Path, mixture_id: str) -> np.ndarray:
    """The block counts of a trained mixture: its ``validation_3e18`` weights times 2,048, which must be integers."""
    name = VALIDATION_TABLE
    table = read_table(root, name)
    rows = table[table[MIXTURE_ID] == mixture_id]
    if rows.empty:
        raise ValueError(f"no {name} run trained {mixture_id}")
    blocks = bucket_values(rows, WEIGHT_PREFIX, release_partition(root).bucket_ids, name) * MIXTURE_BLOCK_SIZE
    counts = np.rint(blocks[0])
    if not (blocks == counts).all():
        raise ValueError(f"the runs of {mixture_id} do not share one mixture on the {MIXTURE_BLOCK_SIZE}-block grid")
    return counts.astype(np.int64)


def released_task_predictions(root: Path, table: str, objective: str) -> pd.DataFrame:
    """A released predictions table (``fits_swarm_predictions``, ``fits_heldout_predictions``) for ``objective``.

    Predicted BPB by observation id (rows) and task id (columns), both in the table's order.
    """
    frame = read_table(root, table)
    require_columns(frame, [OBJECTIVE_COLUMN, OBSERVATION_ID, TASK_ID, PREDICTED_BPB], table)
    frame = frame[frame[OBJECTIVE_COLUMN] == objective]
    wide = frame.pivot(index=OBSERVATION_ID, columns=TASK_ID, values=PREDICTED_BPB)
    return wide.loc[frame[OBSERVATION_ID].unique(), frame[TASK_ID].unique()]


def reliability_rows(reliability: pd.DataFrame, source: str) -> pd.DataFrame:
    """The rows of a ``fits_reliability`` table from ``source``; raises ``ValueError`` naming it when it has none."""
    require_columns(reliability, ["source_table"], RELIABILITY_TABLE)
    rows = reliability[reliability["source_table"] == source]
    if rows.empty:
        raise ValueError(f"the {RELIABILITY_TABLE} table has no rows from {source}")
    return rows


def released_task_parameters(root: Path, swarm: str, objective: str) -> pd.DataFrame:
    """The released per-task parameters (``fits_task_parameters``) of one swarm and objective, indexed by task id."""
    frame = read_table(root, "fits_task_parameters")
    rows = frame[(frame[SWARM_COLUMN] == swarm) & (frame[OBJECTIVE_COLUMN] == objective)]
    return rows.drop(columns=[SWARM_COLUMN, OBJECTIVE_COLUMN]).set_index(TASK_ID)


def released_olmix_laws(root: Path, objective: str) -> tuple[LogLinearLaw, ...]:
    """The main swarm's released Olmix laws (``fits_olmix_task_laws``) for ``objective``, in the tasks table's order.

    Each law's coefficients follow the partition's bucket order.
    """
    name = "fits_olmix_task_laws"
    frame = read_table(root, name)
    require_columns(frame, [SWARM_COLUMN, OBJECTIVE_COLUMN, TASK_ID, "log_c", "huber_loss"], name)
    rows = frame[(frame[SWARM_COLUMN] == MAIN_SWARM) & (frame[OBJECTIVE_COLUMN] == objective)].set_index(TASK_ID)
    rows = rows.loc[list(objective_from_tasks(release_tasks(root), objective).tasks)]
    coefficients = bucket_values(rows, COEFFICIENT_PREFIX, release_partition(root).bucket_ids, name)
    return tuple(
        LogLinearLaw(float(log_c), task_coefficients, float(huber_loss))
        for log_c, task_coefficients, huber_loss in zip(rows["log_c"], coefficients, rows["huber_loss"], strict=True)
    )


def release_heldout_bank(root: Path, objective: str) -> HeldoutBank:
    """The paper's held-out bank: the ``heldout`` runs that ``fits_heldout_predictions`` predicts for ``objective``."""
    predictions = read_table(root, HELDOUT_PREDICTIONS_TABLE)
    require_columns(predictions, [OBJECTIVE_COLUMN, OBSERVATION_ID], HELDOUT_PREDICTIONS_TABLE)
    population = predictions.loc[predictions[OBJECTIVE_COLUMN] == objective, OBSERVATION_ID]
    heldout = read_table(root, HELDOUT_TABLE)
    require_columns(heldout, [OBSERVATION_ID], HELDOUT_TABLE)
    return heldout_bank(
        heldout[heldout[OBSERVATION_ID].isin(population)],
        release_partition(root).bucket_ids,
        objective_from_tasks(release_tasks(root), objective),
    )


def download(local_dir: Path, repo_id: str = DATASET_REPO, revision: str | None = None) -> Path:
    """Download MANIFEST.json, the card, figures_tables.csv and the tables this package reads into ``local_dir``.

    The tables are partition, tasks, the three swarms, ladder, heldout, validation_3e18 and the fits (Figure 20's
    fits_path_predictions among them), which the commands and figures read; ladder_accuracy, the domain sweeps and the
    StarCoder observations, curves, curve membership, diagonal, epoch-matching sweep and its design grid's realized
    shares (starcoder_epoch_matching_grid), two-phase gains (starcoder_curricula and its two matched_nd tables),
    mixture switches (starcoder_transition) and ramp designs, which the figures also read; and
    ladder_compute_equivalence, the baselines' released ladder laws, which the tests compare Figure 1's fits with. The
    StarCoder trajectories, dense grid, fit parameters and repeat noise, ladder_unimax_cap_sweep and run_sets stay on
    the Hub. Returns the directory's path.
    """
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415  (optional dependency: the download extra)
    except ImportError as error:
        raise ImportError(HUGGINGFACE_HUB_MISSING) from error

    return Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
            allow_patterns=list(DOWNLOAD_PATTERNS),
        )
    )
