# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate", "threadpoolctl"]
# ///
"""Apply a fixed interaction probe to reconstructed WSPU without refitting it.

Run this absolute script with PYTHONPATH pointing to the original benchmark's
reproduction_sources directory. The script writes and hashes predictions before
any label scoring; it never reads historical bank labels or launches jobs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import single_phase_observatory_models_20260902 as models
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
DEFAULT_FROZEN = REFERENCE / "delphi_offline_selection_20260906"
DEFAULT_OUTPUT = REFERENCE / "delphi_coupling_followup_20260906" / "incumbent_coupling"
MODEL_ID = "weibull_softplus_unscaled"
KAPPAS = ((0.0, "0"), (0.25, "0p25"), (0.5, "0p5"), (1.0, "1"))
FOLDS = (-1, 0, 1, 2, 3, 4)
PARITY_TOLERANCE = 1e-8


@dataclasses.dataclass(frozen=True)
class AdditiveHeads:
    inventory: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray
    rate: np.ndarray
    power: np.ndarray
    threshold: np.ndarray

    def bucket_effects(self, weights: np.ndarray) -> np.ndarray:
        exposure = np.atleast_2d(weights)[:, None, :] * self.inventory[None, None, :]
        benefit = -np.expm1(-((self.rate[None, :, None] * np.maximum(exposure, 0)) ** self.power[None, :, None]))
        harm = np.logaddexp(0.0, np.log1p(np.maximum(exposure, 0)) - self.threshold[None, :, None]) ** 2
        buckets = len(self.inventory)
        return -benefit * self.coefficients[None, :, :buckets] + harm * self.coefficients[None, :, buckets:]

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.intercept[None, :] + self.bucket_effects(weights).sum(axis=-1)


def coupled_values(anchor: np.ndarray, deltas: np.ndarray, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """Add fixed cross-bucket terms while preserving all one-bucket effects.

    Args:
        anchor: Positive original loss at the training-mean mixture, one per task.
        deltas: Bucket loss changes with shape (queries, tasks, buckets).
        kappa: Fixed interaction strength; zero is the additive limit.

    Returns:
        Atomic losses and counts of nonpositive bucket factors for each query/task.
    """
    if not np.isfinite(anchor).all() or (anchor <= 0).any():
        raise ValueError("All original WSPU anchor predictions must be positive and finite")
    if not np.isfinite(deltas).all() or kappa < 0:
        raise ValueError("Coupling requires finite bucket deltas and nonnegative kappa")
    if kappa == 0:
        return anchor[None, :] + deltas.sum(axis=-1), np.zeros(deltas.shape[:2], dtype=int)
    increments = kappa * deltas / anchor[None, :, None]
    factors = 1 + increments
    count = (factors <= 0).sum(axis=-1)
    positive = (factors > 0).all(axis=-1)
    zero = (factors == 0).any(axis=-1)
    product_minus_one = np.empty(deltas.shape[:2])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        product_minus_one[positive] = np.expm1(np.log1p(increments[positive]).sum(axis=-1))
        product_minus_one[zero] = -1.0
        signed = ~positive & ~zero
        values = factors[signed]
        signs = np.where(values < 0, -1.0, 1.0).prod(axis=-1)
        product_minus_one[signed] = signs * np.exp(np.log(np.abs(values)).sum(axis=-1)) - 1.0
        prediction = anchor[None, :] * (1 + product_minus_one / kappa)
    if not np.isfinite(prediction).all():
        raise ValueError("The fixed coupling probe produced a nonfinite prediction")
    return prediction, count


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze_protocol(frozen: Path, output: Path) -> tuple[dict, str]:
    data = benchmark.read_npz(frozen / "inputs" / "panel.npz")
    if data["weights"].shape != (280, 39):
        raise ValueError("The incumbent coupling probe requires the canonical 280-by-39 panel")
    paths = [frozen / "inputs" / "panel.npz"]
    paths.extend(frozen / "inputs" / f"{target}_bank_features.npz" for target in benchmark.TARGETS)
    paths.extend(
        frozen / "baseline_shards" / MODEL_ID / target / f"r0_f{fold}_c{component}.npz"
        for target in benchmark.TARGETS
        for fold in FOLDS
        for component in range(data[f"{target}_outcomes"].shape[1])
    )
    root = (frozen / "reproduction_sources").resolve()
    sources = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename or not str(filename).endswith(".py"):
            continue
        path = Path(filename).resolve()
        if path == Path(__file__).resolve() or "experiments" not in path.parts:
            continue
        if not path.is_relative_to(root):
            raise ValueError(f"Unpinned experiment import: {path}; use PYTHONPATH={root}")
        sources[str(path.relative_to(root))] = benchmark.sha256(path)
    protocol = {
        "version": 1,
        "timing": "Second-stage mechanistic probe added after the first fixed-basis coupling screen",
        "evidence_status": "retrospective development evidence; not a preregistered prospective confirmation",
        "scope": "canonical 280 final fit; original five outer folds; no shape, ridge, coefficient, or pooling changes",
        "read_set": {str(path.relative_to(frozen)): benchmark.sha256(path) for path in paths},
        "pinned_sources": sources,
        "script_sha256": benchmark.sha256(Path(__file__)),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "primary_kappa": 1.0,
        "sensitivity_kappas": [0.0, 0.25, 0.5],
        "tuning": "No kappa is selected using labels; the primary value and sensitivities are fixed here",
        "anchor": (
            "Each fit's training-mean mixture; A is that task's original WSPU prediction there and must be positive"
        ),
        "formula": "A * (1 + (product_b(1 + kappa * delta_b / A) - 1) / kappa); kappa=0 is original WSPU",
        "delta": "Original fitted additive bucket response at query minus its response at the anchor",
        "invariants": "Every original one-bucket response is unchanged; only cross-bucket terms are added",
        "numerics": (
            "Positive factors use log1p/expm1; other factors use signed log-absolute products with exact zero handling"
        ),
        "invalid_signs": (
            "Nonpositive factors and predictions are retained and counted; no clipping or positivity fallback"
        ),
        "reconstruction_parity_tolerance": PARITY_TOLERANCE,
        "scoring": (
            "This script writes and hashes all predictions before any bank-label join; it performs no label scoring"
        ),
    }
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("The frozen incumbent-coupling protocol changed; use a new output directory")
    else:
        atomic_json(path, protocol)
        (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    return protocol, digest


def reconstruct_heads(frozen: Path, output: Path, target: str, fold: int, digest: str) -> tuple[AdditiveHeads, dict]:
    path = output / "reconstructed_heads" / target / f"fold_{fold}.npz"
    if path.exists():
        saved = benchmark.read_npz(path)
        if str(saved["protocol_hash"]) != digest:
            raise ValueError(f"Stale incumbent reconstruction: {path}")
        heads = AdditiveHeads(
            *(saved[key] for key in ("inventory", "intercept", "coefficients", "rate", "power", "threshold"))
        )
        return heads, saved
    data = benchmark.read_npz(frozen / "inputs" / "panel.npz")
    bank = benchmark.read_npz(frozen / "inputs" / f"{target}_bank_features.npz")
    feature = benchmark.feature_set(data, benchmark.PANEL, data["weights"], data["exposures"])
    entry = registry.ENTRY_BY_ID[MODEL_ID]
    feature = registry.apply_transform(feature, entry)
    parts = [
        benchmark.read_npz(frozen / "baseline_shards" / MODEL_ID / target / f"r0_f{fold}_c{component}.npz")
        for component in range(data[f"{target}_outcomes"].shape[1])
    ]
    train, test = parts[0]["train"], parts[0]["test"]
    coefficients, intercepts, shapes = [], [], []
    for component, original in enumerate(parts):
        if not np.array_equal(original["train"], train) or not np.array_equal(original["test"], test):
            raise ValueError("Original WSPU components have inconsistent split rows")
        model = entry.build(dataclasses.replace(feature, component=str(data[f"{target}_components"][component])))
        if not isinstance(model, models.GridModel):
            raise ValueError("Original WSPU is not the expected fixed-design grid model")
        shape = json.loads(str(original["shape_json"]))
        design = model.design(feature, shape)
        expected_names = tuple(f"bucket_signal:{i}" for i in range(39)) + tuple(
            f"bucket_overexposure:{i}" for i in range(39)
        )
        if design.names != expected_names:
            raise ValueError("Original WSPU is not the expected 78-column additive design")
        head = models.fit_head(
            models.Design(design.values[train], design.ridge, design.names),
            data[f"{target}_outcomes"][train, component],
            float(original["ridge"]),
            model.head_for(shape),
        )
        coefficients.append(head.coefficients)
        intercepts.append(head.intercept)
        shapes.append(shape)
    heads = AdditiveHeads(
        data["inventory"],
        np.array(intercepts),
        np.stack(coefficients),
        np.array([shape["rate"] for shape in shapes]),
        np.array([shape["power"] for shape in shapes]),
        np.array([shape["threshold"] for shape in shapes]),
    )
    actual_train = heads.predict(data["weights"][train])
    actual_test = heads.predict(data["weights"][test])
    actual_bank = heads.predict(bank["weights"])
    train_error = np.max(np.abs(actual_train - np.column_stack([part["train_prediction"] for part in parts])), axis=0)
    test_error = np.max(np.abs(actual_test - np.column_stack([part["prediction"] for part in parts])), axis=0, initial=0)
    bank_error = np.max(np.abs(actual_bank - np.column_stack([part["bank_prediction"] for part in parts])), axis=0)
    if max(train_error.max(), test_error.max(), bank_error.max()) > PARITY_TOLERANCE:
        raise ValueError(f"Original WSPU reconstruction parity failed: {target}/{fold}")
    saved = {
        **dataclasses.asdict(heads),
        "train": train,
        "test": test,
        "component_names": data[f"{target}_components"],
        "train_max_absolute_error": train_error,
        "test_max_absolute_error": test_error,
        "bank_max_absolute_error": bank_error,
        "protocol_hash": digest,
    }
    harness.atomic_save(path, saved)
    return heads, saved


def predict_fold(frozen: Path, output: Path, target: str, fold: int, digest: str) -> None:
    data = benchmark.read_npz(frozen / "inputs" / "panel.npz")
    bank = benchmark.read_npz(frozen / "inputs" / f"{target}_bank_features.npz")
    heads, saved = reconstruct_heads(frozen, output, target, fold, digest)
    train, test = saved["train"], saved["test"]
    anchor_weights = data["weights"][train].mean(axis=0)
    anchor_effects = heads.bucket_effects(anchor_weights[None])[0]
    anchor_atomic = heads.intercept + anchor_effects.sum(axis=-1)
    if not np.isfinite(anchor_atomic).all() or (anchor_atomic <= 0).any():
        raise ValueError(f"Nonpositive original anchor prediction: {target}/{fold}")
    query = np.vstack([data["weights"][test], bank["weights"]])
    effects = heads.bucket_effects(query)
    deltas = effects - anchor_effects[None]
    original_atomic = heads.intercept[None] + effects.sum(axis=-1)
    for kappa, tag in KAPPAS:
        path = output / "prediction_shards" / target / f"fold_{fold}" / f"kappa_{tag}.npz"
        if path.exists():
            if str(benchmark.read_npz(path)["protocol_hash"]) != digest:
                raise ValueError(f"Stale coupling prediction shard: {path}")
            continue
        atomic, counts = coupled_values(anchor_atomic, deltas, kappa)
        if kappa == 0:
            atomic = original_atomic
        aggregate = atomic @ data[f"{target}_aggregation_weights"]
        harness.atomic_save(
            path,
            {
                "prediction": aggregate[: len(test)],
                "bank_prediction": aggregate[len(test) :],
                "atomic_prediction": atomic[: len(test)],
                "atomic_bank_prediction": atomic[len(test) :],
                "test": test,
                "train": train,
                "component_names": data[f"{target}_components"],
                "bank_coordinate_ids": bank["coordinate_id"],
                "anchor_weights": anchor_weights,
                "anchor_atomic_prediction": anchor_atomic,
                "nonpositive_factor_count": counts[: len(test)],
                "bank_nonpositive_factor_count": counts[len(test) :],
                "kappa": kappa,
                "protocol_hash": digest,
            },
        )


def collect_predictions(output: Path, digest: str) -> None:
    predictions, validity, parity = [], [], []
    for target in benchmark.TARGETS:
        for fold in FOLDS:
            original = benchmark.read_npz(output / "reconstructed_heads" / target / f"fold_{fold}.npz")
            for component, name in enumerate(original["component_names"]):
                parity.append(
                    {
                        "target": target,
                        "fold": fold,
                        "component": str(name),
                        **{
                            key: float(original[key][component])
                            for key in ("train_max_absolute_error", "test_max_absolute_error", "bank_max_absolute_error")
                        },
                    }
                )
            for kappa, tag in KAPPAS:
                method = f"wspu_coupling_kappa_{tag}"
                path = output / "prediction_shards" / target / f"fold_{fold}" / f"kappa_{tag}.npz"
                saved = benchmark.read_npz(path)
                if str(saved["protocol_hash"]) != digest:
                    raise ValueError(f"Stale coupling prediction shard: {path}")
                for row, value in zip(saved["test"], saved["prediction"], strict=True):
                    predictions.append(
                        {
                            "method": method,
                            "target": target,
                            "population": "panel_oof",
                            "repeat": 0,
                            "fold": fold,
                            "row_id": str(int(row)),
                            "prediction": float(value),
                        }
                    )
                if fold == -1:
                    for row, value in zip(saved["bank_coordinate_ids"], saved["bank_prediction"], strict=True):
                        predictions.append(
                            {
                                "method": method,
                                "target": target,
                                "population": "external_development",
                                "repeat": 0,
                                "fold": fold,
                                "row_id": str(row),
                                "prediction": float(value),
                            }
                        )
                validity.append(
                    {
                        "method": method,
                        "target": target,
                        "fold": fold,
                        "kappa": kappa,
                        "minimum_anchor_atomic_prediction": float(saved["anchor_atomic_prediction"].min()),
                        "nonpositive_factors": int(saved["nonpositive_factor_count"].sum()),
                        "bank_nonpositive_factors": int(saved["bank_nonpositive_factor_count"].sum()),
                        "nonpositive_atomic_predictions": int((saved["atomic_prediction"] <= 0).sum()),
                        "bank_nonpositive_atomic_predictions": int((saved["atomic_bank_prediction"] <= 0).sum()),
                        "minimum_bank_atomic_prediction": float(saved["atomic_bank_prediction"].min()),
                        "minimum_bank_macro_prediction": float(saved["bank_prediction"].min()),
                    }
                )
    table = pd.DataFrame(predictions)
    temporary = output / ".predictions.csv.tmp"
    table.to_csv(temporary, index=False)
    temporary.replace(output / "predictions.csv")
    original = table[table.method.eq("wspu_coupling_kappa_0")]
    original.to_csv(output / "original_wspu_means.csv", index=False)
    pd.DataFrame(validity).to_csv(output / "validity.csv", index=False)
    pd.DataFrame(parity).to_csv(output / "reconstruction_parity.csv", index=False)
    atomic_json(
        output / "prediction_hash.json",
        {
            "sha256": benchmark.sha256(output / "predictions.csv"),
            "original_wspu_means_sha256": benchmark.sha256(output / "original_wspu_means.csv"),
            "protocol_hash": digest,
            "created_before_truth_join": True,
            "rows": len(table),
            "shards": {
                str(path.relative_to(output)): benchmark.sha256(path)
                for path in sorted((output / "prediction_shards").rglob("*.npz"))
            },
            "scoring": "No historical bank labels read and no label scoring performed by this script",
        },
    )
    print(f"Frozen {len(table)} macro predictions and 48 atomic prediction shards before any label join", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("freeze", "predict"), default="predict")
    args = parser.parse_args()
    frozen, output = args.frozen_dir.resolve(), args.output_dir.resolve()
    _, digest = freeze_protocol(frozen, output)
    if args.stage == "freeze":
        return
    with threadpool_limits(limits=1):
        for target in benchmark.TARGETS:
            for fold in FOLDS:
                predict_fold(frozen, output, target, fold, digest)
        collect_predictions(output, digest)


if __name__ == "__main__":
    main()
