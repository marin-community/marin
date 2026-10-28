# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of a MARINER fit: the paper's outer-fold cross-validation on a swarm, and a held-out bank's metrics.

The outer folds are mixture blocks, like the fit's inner folds: k-means clusters of the runs' square-root weights,
five per repeat, seeded FOLD_SEED + 100 r for repeat r, with the calibration runs pinned to training. The training
runs of outer fold f get three inner folds of their own the same way, seeded FOLD_SEED + 10,000 r + 100 f, on which
the fit selects every task's shape, ridge weight and floor multiplier before it predicts the fold's test runs. Every
fold uses the same anchors: they come from the proportional repeats, which are not among the swarm's runs.

The metrics score selection: the regret and rank of the mixture with the lowest prediction, the optimism of that
pick, and the RMSE and Spearman correlation of the predictions.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mariner.fit import FOLD_SEED, INNER_FOLDS, Folds, block_labels, fit_objective, folds_from_labels, pin_calibration
from mariner.model import Anchor, HeldoutBank, Objective, ObjectiveFit, Swarm, check_buckets

OUTER_FOLDS = 5
OUTER_REPEAT_SEED_STEP = 100
INNER_REPEAT_SEED_STEP = 10_000
INNER_FOLD_SEED_STEP = 100


@dataclasses.dataclass(frozen=True)
class Split:
    """One outer fold of one repeat, in row positions of the swarm.

    ``train`` (ascending) and ``test`` partition the swarm's rows. ``inner`` holds the inner folds of the training
    rows, which validate every training row but the calibration rows once.
    """

    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray
    inner: Folds


def outer_splits(swarm: Swarm, repeats: int) -> tuple[Split, ...]:
    """The paper's outer folds of ``swarm`` for ``repeats`` repeats, in repeat then fold order.

    The calibration runs train in every split, and no split tests or validates them. Raises ``ValueError`` when an
    outer fold would test no run, or an inner fold of a split validate no run, because its mixture block holds only
    calibration runs.
    """
    if repeats < 1:
        raise ValueError(f"cross-validation needs at least one repeat, got {repeats}")
    rows = np.arange(len(swarm.runs))
    splits = []
    for repeat in range(repeats):
        outer = pin_calibration(
            block_labels(swarm.weights, OUTER_FOLDS, FOLD_SEED + OUTER_REPEAT_SEED_STEP * repeat),
            np.flatnonzero(swarm.calibration),
        )
        for fold in range(OUTER_FOLDS):
            if not np.any(outer == fold):
                raise ValueError(
                    f"outer fold {fold} of repeat {repeat} tests no run: its mixture block holds only calibration runs"
                )
            train = rows[outer != fold]
            inner = pin_calibration(
                block_labels(
                    swarm.weights[train],
                    INNER_FOLDS,
                    FOLD_SEED + INNER_REPEAT_SEED_STEP * repeat + INNER_FOLD_SEED_STEP * fold,
                ),
                np.flatnonzero(swarm.calibration[train]),
            )
            for index in range(INNER_FOLDS):
                if not np.any(inner == index):
                    raise ValueError(
                        f"inner fold {index} of outer fold {fold} of repeat {repeat} validates no run: its mixture"
                        " block holds only calibration runs"
                    )
            splits.append(Split(repeat, fold, train, rows[outer == fold], folds_from_labels(train, inner, INNER_FOLDS)))
    return tuple(splits)


def cross_validate(
    swarm: Swarm,
    objective: Objective,
    anchors: dict[str, Anchor],
    repeats: int = 1,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Out-of-fold predictions of ``objective`` on the paper's outer folds (``outer_splits``).

    Each split refits every task on its training runs, with its own inner folds and the fixed ``anchors``, and
    predicts its test runs. The table has one row per tested run and repeat: ``repeat``, ``fold``,
    ``observation_id``, ``measured`` (the objective-weighted outcome) and ``predicted``.
    """
    measured = swarm.outcomes[list(objective.tasks)].to_numpy(float) @ objective.weights
    tables = []
    for split in outer_splits(swarm, repeats):
        if log is not None:
            log(f"repeat {split.repeat} fold {split.fold}: {len(split.train)} training runs, {len(split.test)} tested")
        # The training swarm's rows are the positions in split.train, which is ascending.
        inner = tuple(
            (np.searchsorted(split.train, train), np.searchsorted(split.train, validation))
            for train, validation in split.inner
        )
        fit = fit_objective(swarm.subset(split.train), objective, anchors, inner, log)
        tables.append(
            pd.DataFrame(
                {
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "observation_id": [swarm.runs[row] for row in split.test],
                    "measured": measured[split.test],
                    "predicted": fit.predict(swarm.weights[split.test], swarm.pool_fractions[split.test]),
                }
            )
        )
    return pd.concat(tables, ignore_index=True)


def selection_metrics(predicted: np.ndarray, measured: np.ndarray) -> dict[str, float | int]:
    """How well ``predicted`` selects among mixtures by their ``measured`` BPB, where lower is better.

    ``regret_at_1`` (its measured value minus the best), ``selected_rank`` (1 for the best) and ``optimism_at_pick``
    (its measured minus its predicted value) score the mixture with the lowest prediction; ``rmse`` and ``spearman``
    score every mixture.
    """
    pick = int(np.argmin(predicted))
    return {
        "rows": len(measured),
        "regret_at_1": float(measured[pick] - measured.min()),
        "selected_rank": int(np.sum(measured <= measured[pick])),
        "optimism_at_pick": float(measured[pick] - predicted[pick]),
        "rmse": float(np.sqrt(np.mean((predicted - measured) ** 2))),
        "spearman": float(spearmanr(predicted, measured).correlation),
    }


def heldout_metrics(fit: ObjectiveFit, bank: HeldoutBank) -> dict[str, float | int]:
    """``selection_metrics`` of the fit's predictions at the bank's exposures against their measured objective.

    Raises ``ValueError`` when the bank's buckets are not the fit's in the fit's order, it measures other tasks than
    the fit predicts, or an exposure has no measured objective.
    """
    check_buckets(fit, bank.buckets, "the held-out bank")
    if set(bank.measured.columns) != {task.task for task in fit.tasks}:
        raise ValueError(f"the bank measures other tasks than the fit of {fit.objective} predicts")
    unmeasured = [
        exposure for exposure, value in zip(bank.exposure_ids, bank.measured_objective, strict=True) if np.isnan(value)
    ]
    if unmeasured:
        raise ValueError(f"{len(unmeasured)} exposures have no measured {fit.objective}: {', '.join(unmeasured)}")
    return selection_metrics(fit.predict(bank.weights, bank.pool_fractions), bank.measured_objective)
