# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation of a fitted objective on a held-out bank of measured mixtures.

The metrics score selection: the regret and rank of the mixture with the lowest prediction, the optimism of that
pick, and the RMSE and Spearman correlation of the predictions over the bank.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclasses.dataclass(frozen=True)
class HeldoutBank:
    """Measured mixtures outside the swarm, one row per exposure; the runs at one exposure are averaged."""

    exposure_ids: tuple[str, ...]
    weights: np.ndarray  # (exposures, buckets)
    pool_fractions: np.ndarray  # (exposures, buckets)
    measured: pd.DataFrame  # (exposures, tasks) each task's mean over the exposure's runs; task names as columns
    runs: np.ndarray  # (exposures,) the number of runs at each exposure


def bank_metrics(predicted: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    pick = int(np.argmin(predicted))
    return {
        "rows": float(len(measured)),
        "regret_at_1": float(measured[pick] - measured.min()),
        "selected_rank": float(np.sum(measured <= measured[pick])),
        "optimism_at_pick": float(measured[pick] - predicted[pick]),
        "rmse": float(np.sqrt(np.mean((predicted - measured) ** 2))),
        "spearman": float(spearmanr(predicted, measured).correlation),
    }
