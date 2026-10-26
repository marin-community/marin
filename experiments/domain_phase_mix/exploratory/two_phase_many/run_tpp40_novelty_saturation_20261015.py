# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""Registered amplitude-free native-feature diagnostics on policy metadata only."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE / "reference_outputs/tpp40_acquisition_retention_20261015/novelty_saturation"
RATES = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
POWERS = (0.3, 0.5, 0.7, 1.0)
QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
LOW_THRESHOLDS = (0.01, 0.1, 1.0)
HIGH_THRESHOLDS = (3.0, 5.0, 10.0)
INPUT_COLUMNS = {
    "policies.csv": (
        "order",
        "run_name",
        "source",
        "tied",
        "alpha_nominal",
        "alpha_realized",
        "simulated_epoch_target_budget",
    ),
    "policy_weights.csv": (
        "order",
        "bucket_index",
        "bucket",
        "phase0_weight",
        "phase1_weight",
        "exposure_epochs_nominal",
    ),
    "bucket_axis.csv": ("bucket_index", "bucket", "pool_tokens", "epochs_per_unit_weight"),
}
FEATURES = ("hazard_early", "hazard_total", "benefit_early", "benefit_total", "late_novelty", "positive_headroom")
OUTPUTS = (
    "coordinates.csv",
    "features.npz",
    "shape_support_summary.csv",
    "shape_bucket_summary.csv",
    "inventory_extrema.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime() -> dict[str, str]:
    return {"python": sys.version, **{key: importlib.metadata.version(key) for key in ("numpy", "pandas")}}


def native_grid(path: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read constant literals without importing native scientific code."""
    selected = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("SHAPE_RATES", "SHAPE_POWERS"):
                selected[name] = tuple(ast.literal_eval(node.value))
    return selected["SHAPE_RATES"], selected["SHAPE_POWERS"]


def read_metadata(directory: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(directory / name, usecols=list(columns), float_precision="round_trip")
        for name, columns in INPUT_COLUMNS.items()
    }


def coordinates(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Materialize the established exposure convention with physical phase duration."""
    policies, weights, axis = (frames[name] for name in INPUT_COLUMNS)
    policies = policies.sort_values("order")
    axis = axis.sort_values("bucket_index")
    if policies.order.duplicated().any() or axis.bucket_index.duplicated().any():
        raise ValueError("Duplicate policy or bucket identity")
    if weights.duplicated(["order", "bucket_index"]).any() or len(weights) != len(policies) * len(axis):
        raise ValueError("Policy-bucket table is not complete")
    result = (
        weights.merge(policies, on="order", validate="many_to_one")
        .merge(axis, on=["bucket_index", "bucket"], validate="many_to_one")
        .sort_values(["order", "bucket_index"])
        .reset_index(drop=True)
    )
    if len(result) != len(weights):
        raise ValueError("Policy/bucket joins dropped coordinates")
    if not result.tied.isin([True, False]).all():
        raise ValueError("Tied flags must be explicit booleans")
    for name in ("alpha_nominal", "alpha_realized", "simulated_epoch_target_budget"):
        if result[name].nunique() != 1:
            raise ValueError("Exposure metadata must be common across the complete panel")
    inventory = result.simulated_epoch_target_budget.to_numpy(float) / result.pool_tokens.to_numpy(float)
    w0, w1 = (result[f"phase{phase}_weight"].to_numpy(float) for phase in (0, 1))
    alpha = result.alpha_realized.to_numpy(float)
    nominal = result.alpha_nominal.to_numpy(float)
    if not np.isfinite(inventory).all() or np.any(inventory <= 0) or np.any((alpha <= 0) | (alpha >= 1)):
        raise ValueError("Invalid physical exposure inventory")
    if not np.isfinite(w0 + w1).all() or np.any(w0 < 0) or np.any(w1 < 0):
        raise ValueError("Invalid policy weights")
    expected = (nominal * w0 + (1 - nominal) * w1) * inventory
    if not np.allclose(expected, result.exposure_epochs_nominal, rtol=1e-13, atol=1e-13):
        raise ValueError("Nominal exposure metadata cannot be reproduced")
    sums = result.groupby("order")[["phase0_weight", "phase1_weight"]].sum().to_numpy()
    if not np.allclose(sums, 1.0, rtol=0, atol=1e-12):
        raise ValueError("Policy weights do not sum to one")
    result["early_epochs"] = alpha * w0 * inventory
    result["late_epochs"] = (1 - alpha) * w1 * inventory
    result["total_epochs"] = result.early_epochs + result.late_epochs
    result["support"] = np.select(
        [(w0 > 0) & (w1 > 0), (w0 > 0) & (w1 == 0), (w0 == 0) & (w1 > 0)],
        ["both_positive", "early_only", "late_only"],
        default="total_zero",
    )
    return result


def native_features(early: np.ndarray, late: np.ndarray, rate: float, power: float) -> dict[str, np.ndarray]:
    """Compute stable native benefits and remaining acquisition, without ODE state."""
    h0 = (rate * early) ** power
    dh = (rate * late) ** power
    positive = early > 0
    dh[positive] = h0[positive] * np.expm1(power * np.log1p(late[positive] / early[positive]))
    total = h0 + dh
    return dict(
        zip(
            FEATURES,
            (h0, total, -np.expm1(-h0), -np.expm1(-total), np.exp(-h0) * -np.expm1(-dh), np.exp(-total)),
            strict=True,
        )
    )


def summarize(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float | int]:
    result: dict[str, float | int] = {"n_coordinates": int(mask.sum())}
    for name, vector in values.items():
        quantiles = np.quantile(vector[mask], QUANTILES)
        result.update({f"{name}_q{q:g}": float(value) for q, value in zip(QUANTILES, quantiles, strict=True)})
    for name in ("hazard_early", "hazard_total"):
        result.update({f"{name}_le_{v:g}": int((values[name][mask] <= v).sum()) for v in LOW_THRESHOLDS})
        result.update({f"{name}_ge_{v:g}": int((values[name][mask] >= v).sum()) for v in HIGH_THRESHOLDS})
    return result


def diagnostic_tables(rows: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    early, late = rows.early_epochs.to_numpy(float), rows.late_epochs.to_numpy(float)
    stacked: dict[str, list[np.ndarray]] = {name: [] for name in FEATURES}
    support_rows, bucket_rows = [], []
    for rate in RATES:
        for power in POWERS:
            values = native_features(early, late, rate, power)
            for name in FEATURES:
                stacked[name].append(values[name])
            for role in ("all", "asymmetric", "tied"):
                mask = np.ones(len(rows), dtype=bool) if role == "all" else rows.tied.to_numpy(bool) == (role == "tied")
                prefix = {"rate": rate, "power": power, "policy_role": role}
                for support in ("all", "both_positive", "early_only", "late_only", "total_zero"):
                    selected = mask if support == "all" else mask & rows.support.eq(support).to_numpy()
                    if selected.any():
                        support_rows.append({**prefix, "support": support, **summarize(values, selected)})
                for bucket_index, bucket in rows[["bucket_index", "bucket"]].drop_duplicates().itertuples(index=False):
                    selected = mask & rows.bucket_index.eq(bucket_index).to_numpy()
                    if selected.any():
                        bucket_rows.append(
                            {
                                **prefix,
                                "bucket_index": int(bucket_index),
                                "bucket": bucket,
                                **summarize(values, selected),
                            }
                        )
    arrays: dict[str, np.ndarray] = {
        name: np.stack(value).reshape(len(RATES) * len(POWERS), rows.order.nunique(), rows.bucket_index.nunique())
        for name, value in stacked.items()
    }
    arrays["shape_rates"] = np.repeat(RATES, len(POWERS))
    arrays["shape_powers"] = np.tile(POWERS, len(RATES))
    return arrays, pd.DataFrame(support_rows), pd.DataFrame(bucket_rows)


def verify_registration(registration: dict) -> None:
    if registration["status"] != "registered" or registration["approval"] != "metadata_diagnostic_only":
        raise ValueError("Exact registration and root approval are required before evaluation")
    expected = {"policies": 280, "buckets": 39, "benefit_shapes": 28, "shape_policy_bucket_coordinates": 305760}
    if registration["counts"] != expected or registration["input_columns"] != {
        k: list(v) for k, v in INPUT_COLUMNS.items()
    }:
        raise ValueError("Registered scope differs from the reviewed complete-grid diagnostic")
    if registration["rates"] != list(RATES) or registration["powers"] != list(POWERS):
        raise ValueError("Benefit grid differs")
    if (
        registration["quantiles"] != list(QUANTILES)
        or registration["hazard_le"] != list(LOW_THRESHOLDS)
        or registration["hazard_ge"] != list(HIGH_THRESHOLDS)
    ):
        raise ValueError("Summary rules differ")
    if registration["runtime"] != runtime():
        raise ValueError("Runtime differs from registration")
    for name, expected_hash in registration["source_hashes"].items():
        if digest(Path(name)) != expected_hash:
            raise ValueError(f"Registered source changed: {name}")
    if str(Path(__file__).resolve()) not in registration["source_hashes"]:
        raise ValueError("Runner source is not pinned")
    if registration["native_source"] not in registration["source_hashes"]:
        raise ValueError("Native source is not pinned")
    if native_grid(Path(registration["native_source"])) != (RATES, POWERS):
        raise ValueError("Native benefit grid changed")
    source = Path(registration["input_directory"])
    if set(registration["input_hashes"]) != set(INPUT_COLUMNS):
        raise ValueError("Only the three whitelisted metadata inputs are admitted")
    for name, expected_hash in registration["input_hashes"].items():
        if digest(source / name) != expected_hash:
            raise ValueError(f"Metadata changed: {name}")


def run(registration_path: Path) -> dict:
    registration = json.loads(registration_path.read_text())
    verify_registration(registration)
    identity = digest(registration_path)
    output = Path(registration["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "completion.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["registration_sha256"] != identity or set(saved["outputs"]) != set(OUTPUTS):
            raise ValueError("Existing completion belongs to a different scope")
        for name, expected in saved["outputs"].items():
            if digest(output / name) != expected["sha256"]:
                raise ValueError("Completed output changed")
        return saved
    if any((output / name).exists() for name in OUTPUTS):
        raise ValueError("Partial output preserved; inspect before another attempt")
    rows = coordinates(read_metadata(Path(registration["input_directory"])))
    if (
        rows.order.nunique() != 280
        or rows.bucket_index.nunique() != 39
        or int(rows.groupby("order").tied.first().sum()) != 42
    ):
        raise ValueError("Complete panel counts differ")
    if rows.order.unique().tolist() != list(range(280)) or rows.bucket_index.unique().tolist() != list(range(39)):
        raise ValueError("Saved policy or bucket order differs")
    arrays, supports, buckets = diagnostic_tables(rows)
    inventory = rows[["bucket_index", "bucket", "pool_tokens"]].drop_duplicates()
    extrema = {
        name: inventory.loc[inventory.pool_tokens.eq(value)].to_dict("records")
        for name, value in (
            ("smallest_pool", inventory.pool_tokens.min()),
            ("largest_pool", inventory.pool_tokens.max()),
        )
    }
    rows.to_csv(output / "coordinates.csv", index=False)
    np.savez_compressed(output / "features.npz", **arrays)
    supports.to_csv(output / "shape_support_summary.csv", index=False)
    buckets.to_csv(output / "shape_bucket_summary.csv", index=False)
    (output / "inventory_extrema.json").write_text(json.dumps(extrema, indent=2, allow_nan=False) + "\n")
    report = {
        "completed": True,
        "scientific_gate": None,
        "scope": "amplitude-free metadata diagnostic only",
        "registration_sha256": identity,
        "counts": registration["counts"],
        "runtime": runtime(),
        "outputs": {
            name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size} for name in OUTPUTS
        },
    }
    completion.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=ROOT / "registration.json")
    args = parser.parse_args()
    print(json.dumps(run(args.registration), indent=2))


if __name__ == "__main__":
    main()
