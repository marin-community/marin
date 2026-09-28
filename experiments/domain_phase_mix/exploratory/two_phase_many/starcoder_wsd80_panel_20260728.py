# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 80/20 WSD StarCoder surface as a surrogate-fitting panel.

The existing StarCoder benchmark covers the original cosine schedule and the 50/50 boundary-aligned
WSD schedule. Neither is the 80/20 stable/decay schedule that the rest of the project standardised on,
and the 80/20 surface is the one panel where a two-phase policy is known to beat the entire one-phase
class. That makes it the sharpest available test of whether a surrogate can represent phase order at
all, so it is worth a loader of its own.

Two facts about this panel drive everything downstream.

The one-phase optimum sits at aggregate 0.30 and the two-phase optimum at aggregate 0.18, so a model
that gets the aggregate response right and the phase response wrong will still place its optimum in the
wrong place by twelve aggregate points. And any model in which the schedule enters only through a
per-domain reweighted cumulative exposure predicts exactly zero two-phase gain, because the tied policy
class already sweeps the whole effective-exposure simplex. Such a model cannot be wrong by a little
here; it is wrong by the entire two-phase advantage.

Epoch multipliers are carried over from the cosine panel, which shares the architecture, the token
budget and the data pools, and rescaled by the realised phase fractions. The 80/20 boundary is step
3040 of 3814.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
SURFACE_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714"
SURFACE_CSV = SURFACE_DIR / "wsd80_observed_metrics.csv"
FIBER_CSV = SURFACE_DIR / "wsd80_measured_fiber_observations.csv"

TARGET_COLUMN = "wsd80_bpb"
DOMAIN_NAMES = ("nemotron_full", "starcoder")
REFERENCE_SEED = 20260711
# Boundary-aligned 80/20 WSD: the decay window opens at step 3040 of 3814. Two phase fractions matter
# and they are not the same number. The design, the fibers and every prior analysis use the nominal
# 0.8/0.2 split to define aggregate and contrast coordinates, so fiber membership is exact only against
# that. Epoch multipliers describe physical token counts and use the realised step fractions.
PHASE_0_STEPS = 3040
TOTAL_STEPS = 3814
REALIZED_PHASE_0_FRACTION = PHASE_0_STEPS / TOTAL_STEPS
REALIZED_PHASE_1_FRACTION = 1.0 - REALIZED_PHASE_0_FRACTION
PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2
# Epochs per unit weight under the 50/50 cosine panel, which shares budget, pools and architecture.
# Nemotron is large enough that a full phase at weight one is under one epoch; StarCoder is not.
COSINE_EPOCHS_PER_UNIT_WEIGHT = {"nemotron_full": 0.5, "starcoder": 13.228934}
COSINE_PHASE_FRACTION = 0.5


@dataclass(frozen=True)
class Panel:
    """A fitting panel: one row per trained policy, weights shaped (n, 2, m)."""

    name: str
    frame: pd.DataFrame
    y: np.ndarray
    weights: np.ndarray
    c0: np.ndarray
    c1: np.ndarray
    domain_names: list[str]

    @property
    def phase_0(self) -> np.ndarray:
        return self.weights[:, 0, :]

    @property
    def phase_1(self) -> np.ndarray:
        return self.weights[:, 1, :]

    @property
    def aggregate(self) -> np.ndarray:
        return PHASE_0_FRACTION * self.phase_0 + PHASE_1_FRACTION * self.phase_1

    @property
    def contrast(self) -> np.ndarray:
        return self.phase_1 - self.phase_0

    @property
    def phase_tv(self) -> np.ndarray:
        return 0.5 * np.abs(self.contrast).sum(axis=1)

    @property
    def epochs(self) -> np.ndarray:
        """Epochs of each domain's pool consumed over the whole run."""
        return self.c0 * self.phase_0 + self.c1 * self.phase_1


def epoch_multipliers() -> tuple[np.ndarray, np.ndarray]:
    """Epochs per unit weight in each phase, rescaled from the cosine panel by phase length."""
    base = np.array([COSINE_EPOCHS_PER_UNIT_WEIGHT[name] for name in DOMAIN_NAMES], dtype=float)
    return (
        base * REALIZED_PHASE_0_FRACTION / COSINE_PHASE_FRACTION,
        base * REALIZED_PHASE_1_FRACTION / COSINE_PHASE_FRACTION,
    )


def _weights_from_starcoder_share(phase_0_share: np.ndarray, phase_1_share: np.ndarray) -> np.ndarray:
    """Two-bucket mixtures from the StarCoder share alone; the remainder is Nemotron."""
    phase_0 = np.column_stack([1.0 - phase_0_share, phase_0_share])
    phase_1 = np.column_stack([1.0 - phase_1_share, phase_1_share])
    return np.stack([phase_0, phase_1], axis=1)


def load_surface() -> Panel:
    """The 166 unique coordinates of the merged 80/20 WSD surface, at the reference seed."""
    frame = pd.read_csv(SURFACE_CSV).dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    frame = frame.drop_duplicates(subset=["phase_0_starcoder", "phase_1_starcoder"]).reset_index(drop=True)
    c0, c1 = epoch_multipliers()
    return Panel(
        name="wsd80_surface",
        frame=frame,
        y=frame[TARGET_COLUMN].to_numpy(dtype=float),
        weights=_weights_from_starcoder_share(
            frame["phase_0_starcoder"].to_numpy(dtype=float),
            frame["phase_1_starcoder"].to_numpy(dtype=float),
        ),
        c0=c0,
        c1=c1,
        domain_names=list(DOMAIN_NAMES),
    )


def load_fiber_replicates() -> pd.DataFrame:
    """Multi-seed observations on the two measured fixed-aggregate fibers.

    These share coordinates with the surface at the reference seed, so they are not extra fitting rows.
    They are the only estimate of training-seed noise on this panel and the only paired evidence about
    phase order, so they are loaded separately and used for scoring rather than for fitting.
    """
    frame = pd.read_csv(FIBER_CSV)
    frame["contrast"] = frame["phase_1_starcoder"] - frame["phase_0_starcoder"]
    frame["aggregate"] = PHASE_0_FRACTION * frame["phase_0_starcoder"] + PHASE_1_FRACTION * frame["phase_1_starcoder"]
    return frame


def training_seed_sigma(replicates: pd.DataFrame) -> float:
    """Pooled training-seed standard deviation over every replicated coordinate."""
    variances, weights = [], []
    for _key, block in replicates.groupby(["phase_0_starcoder", "phase_1_starcoder"]):
        if len(block) < 2:
            continue
        variances.append(block[TARGET_COLUMN].var(ddof=1))
        weights.append(len(block) - 1)
    assert variances, "no replicated coordinates found"
    return float(np.sqrt(np.average(variances, weights=weights)))


def tied_reference(panel: Panel) -> tuple[int, int]:
    """Row indices of the best one-phase policy and the best policy overall."""
    tied = np.flatnonzero(np.isclose(panel.contrast[:, 1], 0.0))
    assert len(tied) > 1, "expected a sampled tied diagonal"
    return int(tied[np.argmin(panel.y[tied])]), int(np.argmin(panel.y))
