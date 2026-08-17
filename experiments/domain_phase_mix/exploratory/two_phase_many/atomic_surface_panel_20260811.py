# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The no-replay two-bucket atomic panel, and the spatial folds it is scored on (ATOM-001).

The simplest identifiable corner of the optimisation taxonomy: no forced replay, two buckets, one atomic
BPB objective at a time. Everything that makes the 39-bucket panel hard -- many buckets, many objectives,
and repetition -- is switched off here rather than fitted, so that a failure localises to the response
mechanism instead of being absorbed by a replay term.

The zero-replay claim is ASSERTED, not assumed: `assert_no_replay` fails loudly if any row reaches a full
pass over the StarCoder pool. On the current data the maximum total exposure is 0.0342 epochs, roughly
thirtyfold below activation, with no support wraps.

Folds are SPATIAL. The 125 coordinates form a dense grid in (phase-0 share, phase-1 share), so random-row
splits put near-neighbours on both sides of the split and measure interpolation rather than prediction.
Leaving out contiguous blocks of the mixture square asks the question we actually care about.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# The canonical explorer imports a sibling by bare module name, so its own directory has to be importable
# before it can be loaded. This is a property of that module, not a choice made here.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    plot_starcoder_wsd80_full_pool_atomic_surface_explorer_20260811 as explorer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_atomic_metrics as atomic,
)

REFERENCE = Path(__file__).resolve().parent / "reference_outputs"
COVERAGE = (
    REFERENCE
    / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811"
    / "coverage_observations.csv"
)
EXPLORER_DIR = REFERENCE / "starcoder_wsd80_full_pool_atomic_surface_explorer_20260811"
# StarCoder is bucket 0 and the complement (Nemotron) is bucket 1; two buckets, so one free share each phase.
BUCKETS = ("starcoder", "complement")


@dataclass(frozen=True)
class AtomicPanel:
    """One horizon's worth of full-pool observations, plus exact physical exposure."""

    frame: pd.DataFrame
    horizon: float

    @property
    def phase_0(self) -> np.ndarray:
        return self.frame["phase_0_starcoder"].to_numpy(float)

    @property
    def phase_1(self) -> np.ndarray:
        return self.frame["phase_1_starcoder"].to_numpy(float)

    @property
    def aggregate(self) -> np.ndarray:
        return self.frame["aggregate_starcoder"].to_numpy(float)

    @property
    def epochs_phase_0(self) -> np.ndarray:
        """Exact materialized StarCoder epochs during the stable phase."""
        return self.frame["starcoder_phase_0_epochs"].to_numpy(float)

    @property
    def epochs_phase_1(self) -> np.ndarray:
        return self.frame["starcoder_phase_1_epochs"].to_numpy(float)

    def target(self, key: str) -> np.ndarray:
        return self.frame[key].to_numpy(float)

    @property
    def complement_epochs_phase_0(self) -> np.ndarray:
        """Materialized epochs of the complement bucket during the stable phase.

        The panel records the complement pool's whole-run epoch capacity as `nemotron_max_total_epochs`.
        Splitting it by the schedule's phase fractions and scaling by the complement share gives the
        per-phase materialized exposure, exactly as the StarCoder columns already provide directly.
        The complement pool is about 26x larger than StarCoder in epoch terms, so its exposure is small
        and slowly varying -- which is the point: a readout of it is a different feature, not a rescaling.
        """
        capacity = self.frame["nemotron_max_total_epochs"].to_numpy(float)
        return explorer.PHASE_0_FRACTION * capacity * (1.0 - self.phase_0)

    @property
    def complement_epochs_phase_1(self) -> np.ndarray:
        capacity = self.frame["nemotron_max_total_epochs"].to_numpy(float)
        return (1.0 - explorer.PHASE_0_FRACTION) * capacity * (1.0 - self.phase_1)

    @property
    def tied(self) -> np.ndarray:
        return np.isclose(self.phase_0, self.phase_1)


def load_full_pool(refresh: bool = False) -> pd.DataFrame:
    observations = explorer.load_observations(COVERAGE, EXPLORER_DIR, refresh=refresh, workers=1)
    full = observations.loc[observations["support_id"] == "full"].reset_index(drop=True)
    assert_no_replay(full)
    return full


_PRELOADED: dict[str, pd.DataFrame] = {}


def seed_cache(frames: dict[str, pd.DataFrame]) -> None:
    """Install already-loaded per-support frames so `load_support` never touches disk.

    `load_observations` rewrites its metric cache on every call, so several worker processes calling it
    concurrently race on that file and some of them read it half-written. Parallel drivers load once in
    the parent and seed the workers through this, which removes the race and the repeated parse.
    """
    _PRELOADED.update(frames)


def load_all_supports(refresh: bool = False) -> dict[str, pd.DataFrame]:
    observations = explorer.load_observations(COVERAGE, EXPLORER_DIR, refresh=refresh, workers=1)
    return {
        support_id: group.reset_index(drop=True) for support_id, group in observations.groupby("support_id", sort=True)
    }


def load_support(support_id: str, refresh: bool = False) -> pd.DataFrame:
    """One replay condition. `full` is the zero-replay panel; the rest carry real repetition.

    The panel spans a repetition ladder that the no-replay work never used: maximum total StarCoder
    epochs run 0.034 at `full`, then 3.31, 6.62, 13.26, 26.53, 53.06 and 106.11 across m0125 to m400,
    with 288 to 492 of 500 rows exceeding one full pass. That range matters because the measured
    repetition-damage knee sits near 105 excess epochs, so m400 is the first condition that reaches it.

    It also matters because the zero-replay panel has NO positive two-phase gain to predict, so a model
    that cannot express phase structure scores well there regardless. Repetition is where the effect
    being modelled actually exists.
    """
    if support_id in _PRELOADED:
        frame = _PRELOADED[support_id]
    else:
        observations = explorer.load_observations(COVERAGE, EXPLORER_DIR, refresh=refresh, workers=1)
        frame = observations.loc[observations["support_id"] == support_id].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"unknown support_id {support_id!r}")
    if support_id == "full":
        assert_no_replay(frame)
    return frame


def repetition_summary(frame: pd.DataFrame) -> dict[str, float]:
    total = frame["starcoder_phase_0_epochs"].to_numpy(float) + frame["starcoder_phase_1_epochs"].to_numpy(float)
    return {
        "max_epochs": float(total.max()),
        "median_epochs": float(np.median(total)),
        "rows_repeated": float((total > 1.0).mean()),
    }


def assert_no_replay(frame: pd.DataFrame) -> None:
    """Fail loudly if this panel is not the zero-replay setting it is claimed to be."""
    total = frame["starcoder_phase_0_epochs"].to_numpy(float) + frame["starcoder_phase_1_epochs"].to_numpy(float)
    if total.max() >= 1.0:
        raise ValueError(f"full-pool rows reach {total.max():.4f} epochs; this is not a no-replay panel")
    if bool(frame["starcoder_support_wraps"].to_numpy().any()):
        raise ValueError("full-pool rows report support wraps")
    if not np.allclose(frame["starcoder_support_fraction"].to_numpy(float), 1.0):
        raise ValueError("full-pool rows do not all use the whole support")


def panels_by_horizon(frame: pd.DataFrame) -> list[AtomicPanel]:
    return [
        AtomicPanel(group.reset_index(drop=True), float(horizon))
        for horizon, group in frame.groupby("materialized_tokens_b", sort=True)
    ]


def spatial_folds(panel: AtomicPanel, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-region-out folds over contiguous blocks of the mixture square.

    Random-row splits are the wrong instrument on a dense grid: every held-out coordinate has an immediate
    neighbour in training, so the score measures interpolation. Blocks are formed by k-means on the
    (phase-0, phase-1) coordinates, which yields contiguous regions without assuming a rectangular layout.
    """
    points = np.column_stack([panel.phase_0, panel.phase_1])
    rng = np.random.default_rng(seed)
    centres = points[rng.choice(len(points), n_splits, replace=False)]
    for _ in range(50):
        assignment = np.argmin(((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)
        moved = np.array(
            [points[assignment == k].mean(axis=0) if (assignment == k).any() else centres[k] for k in range(n_splits)]
        )
        if np.allclose(moved, centres):
            break
        centres = moved
    return [(np.flatnonzero(assignment != k), np.flatnonzero(assignment == k)) for k in range(n_splits)]


def atomic_targets() -> tuple[str, ...]:
    return atomic.METRIC_KEYS
