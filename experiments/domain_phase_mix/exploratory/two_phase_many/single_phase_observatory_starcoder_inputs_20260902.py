# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Frozen StarCoder curve inputs for the single-phase Observatory benchmark.

Copied from ``fit_starcoder_all_tied_curves_canonical_dsp_20260902`` at commit 99bea291d7 so that later
edits to that script cannot change the benchmark's curve panels, curve order, input hashes, or cache keys.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_epoch_accounting as epoch_accounting,
)

SCRIPT_DIR = Path(__file__).resolve().parent
INVENTORY_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_single_phase_curve_inventory_20260902"
PRIMARY_TARGET = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
CURVE_INVENTORY_FILE = "curve_inventory.csv"
CURVE_MEMBERSHIPS_FILE = "curve_memberships.csv"
TARGET_OBSERVATIONS_FILE = "target_observations.csv"
INPUT_FILES = (CURVE_INVENTORY_FILE, CURVE_MEMBERSHIPS_FILE, TARGET_OBSERVATIONS_FILE)
EXPECTED_CURVES = 45
SUPPORT_EPOCH_MULTIPLIERS = {
    "m0125": 0.125,
    "m025": 0.25,
    "m050": 0.5,
    "m100": 1.0,
    "m200": 2.0,
    "m400": 4.0,
}


def load_inputs(inventory_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the 45 protocol-ready endpoint curves (sorted by family, id) and their primary-target points."""
    curves = pd.read_csv(inventory_dir / CURVE_INVENTORY_FILE)
    core = curves.loc[curves["protocol_group"].eq("core_endpoint")].copy()
    if len(core) != EXPECTED_CURVES or not core["primary_target_ready"].all():
        raise ValueError(f"Expected {EXPECTED_CURVES} protocol-ready endpoint curves")

    memberships = pd.read_csv(inventory_dir / CURVE_MEMBERSHIPS_FILE)
    targets = pd.read_csv(inventory_dir / TARGET_OBSERVATIONS_FILE)
    primary = targets.loc[targets["target_id"].eq(PRIMARY_TARGET), ["observation_id", "training_run_id", "bpb"]]
    points = memberships.merge(primary, on=["observation_id", "training_run_id"], validate="many_to_one")
    points = points.loc[points["curve_id"].isin(core["curve_id"])].copy()
    points = points.merge(
        core[["curve_id", "family", "primary_target_points"]],
        on="curve_id",
        validate="many_to_one",
    )
    if points.duplicated(["curve_id", "starcoder_weight"]).any():
        raise ValueError("A core curve has multiple primary observations at one mixture weight")
    counts = points.groupby("curve_id")["starcoder_weight"].nunique()
    expected = core.set_index("curve_id")["primary_target_points"].astype(int)
    if not counts.sort_index().equals(expected.sort_index()):
        raise ValueError("Primary observations do not match the frozen curve inventory")
    if not np.isfinite(points["bpb"]).all():
        raise ValueError("Primary curve data contain nonfinite BPB values")
    return core.sort_values(["family", "curve_id"]).reset_index(drop=True), points


def historical_starcoder_epoch_scale() -> float:
    return epoch_accounting.SIMULATED_EPOCH_TARGET_BUDGET / epoch_accounting.STARCODER_SOURCE_TOKENS


def starcoder_epoch_scale(metadata: pd.Series) -> float:
    """Return StarCoder materialized epochs at p=1 for one physical curve."""
    support_id = str(metadata["support_id"])
    if support_id == "full":
        return float(metadata["planned_materialized_tokens"]) / epoch_accounting.STARCODER_SOURCE_TOKENS
    if support_id in SUPPORT_EPOCH_MULTIPLIERS:
        return historical_starcoder_epoch_scale() * SUPPORT_EPOCH_MULTIPLIERS[support_id]
    if support_id in {"historical_simulated_support", "matched_nd_reference_support"}:
        return historical_starcoder_epoch_scale()
    raise ValueError(f"Unknown StarCoder support for epoch axis: {support_id}")
