# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy", "pandas", "scikit-learn", "joblib", "tabulate"]
# ///
"""Offline Delphi selection benchmark with frozen inputs and resumable fits.

Run from the repository root with ``uv run --offline python -m
experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_delphi_selection_20260906``.
The prepare stage reads only local files. No stage submits training or eval jobs.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REFERENCE = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT = REFERENCE / "delphi_offline_selection_20260906"
HISTORICAL = REFERENCE / "single_phase_observatory_benchmark_20260902"
FROZEN_BANK = HISTORICAL / "heldout_round3_canonical" / "external_heldout_predictions.csv"
PANEL = "delphi_3e18_39bucket"
BASELINES = ("weibull_softplus_unscaled", "dsp_total_exposure", "olmix_loglinear_taskwise")
SEED = 20260902
TARGETS = ("uncheatable", "table9")
INTERVENTIONS = frozenset({"conditional_epoch_dose_response", "archive::delphi_baseline_mixtures_issue6607_20260623"})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(output: Path) -> None:
    """Freeze historical labels and current coordinates without reading newer bank outcomes."""
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"
    if inputs.exists():
        verify_inputs(output)
        return
    inputs.mkdir()
    panel = harness.canonical.load_panel(PANEL, harness.CANONICAL_INPUT_DIR)
    features = harness._tabular_features(panel)
    if len(panel.runs) != 280:
        raise ValueError("Canonical final-fit panel must contain exactly 280 training runs")
    payload = {
        "weights": features.weights,
        "exposures": features.exposures,
        "inventory": features.inventory,
        "early_fraction": features.early_fraction,
        "buckets": np.asarray(panel.buckets),
        "runs": np.asarray(panel.runs),
    }
    for group in panel.groups:
        payload[f"{group.name}_outcomes"] = group.outcomes
        payload[f"{group.name}_aggregate"] = group.outcomes @ group.aggregation_weights
        payload[f"{group.name}_components"] = np.asarray(group.components)
        payload[f"{group.name}_aggregation_weights"] = group.aggregation_weights
    harness.atomic_save(inputs / "panel.npz", payload)
    old = pd.read_csv(FROZEN_BANK)
    old = old[old.panel.eq(PANEL)].copy()
    old.to_csv(inputs / "historical_predictions.csv", index=False)
    weight_columns = [f"weight::{bucket}" for bucket in panel.buckets]
    coordinate_path = harness.HELDOUT_DIR / "heldout_coordinates.csv"
    coordinates = pd.read_csv(coordinate_path, usecols=["panel", "coordinate_id", *weight_columns])
    coordinates = coordinates[coordinates.panel.eq(PANEL)].set_index("coordinate_id")
    for target in TARGETS:
        bank = (
            old[old.model.eq(BASELINES[0]) & old.target.eq(target)].sort_values("coordinate_id").reset_index(drop=True)
        )
        weights = coordinates.loc[bank.coordinate_id, weight_columns].to_numpy(float)
        distance = np.abs(weights[:, None] - features.weights[None]).sum(axis=2).min(axis=1)
        if (distance < 1e-7).any():
            raise ValueError("External coordinates overlap the canonical panel")
        if bank.sources.str.contains("wspu.*scal|apriori|prospectively_frozen", case=False, regex=True).any():
            raise ValueError("Frozen bank contains a forbidden later source")
        bank["distance_tv"] = distance / 2
        bank.drop(columns=["model", "prediction", "prediction_hash"]).to_csv(
            inputs / f"{target}_bank_labels.csv", index=False
        )
        harness.atomic_save(
            inputs / f"{target}_bank_features.npz",
            {
                "weights": weights,
                "exposures": weights * features.inventory,
                "coordinate_id": bank.coordinate_id.to_numpy(str),
            },
        )
    rows = np.arange(280)
    manifest = []
    for repeat in range(5):
        outer = harness.olmix_benchmark.block_labels(features.weights, 5, SEED + 100 * repeat)
        for fold in range(5):
            train = rows[outer != fold]
            inner = harness.olmix_benchmark.block_labels(features.weights[train], 3, SEED + 10000 * repeat + 100 * fold)
            for row in rows:
                position = np.flatnonzero(train == row)
                manifest.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "row": row,
                        "role": "train" if len(position) else "test",
                        "inner_fold": int(inner[position[0]]) if len(position) else -1,
                    }
                )
    final_inner = harness.olmix_benchmark.block_labels(features.weights, 3, harness.HELDOUT_INNER_SEED)
    manifest.extend(
        {"repeat": 0, "fold": -1, "row": int(row), "role": "train", "inner_fold": int(final_inner[row])} for row in rows
    )
    pd.DataFrame(manifest).to_csv(inputs / "splits.csv", index=False)
    write_json(
        inputs / "provenance.json",
        {
            "canonical_inputs": panel.input_hashes,
            "historical_bank_path": str(FROZEN_BANK.relative_to(REPO_ROOT)),
            "historical_bank_sha256": sha256(FROZEN_BANK),
            "coordinate_features_path": str(coordinate_path.relative_to(REPO_ROOT)),
            "coordinate_features_sha256": sha256(coordinate_path),
            "scope": "canonical 280 final fit; frozen historical bank development only; no ladder inputs",
        },
    )
    write_json(output / "input_hashes.json", {path.name: sha256(path) for path in sorted(inputs.iterdir())})


def verify_inputs(output: Path) -> dict[str, str]:
    expected = json.loads((output / "input_hashes.json").read_text())
    for name, digest in expected.items():
        if sha256(output / "inputs" / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    return expected


def feature_set(data: dict, label: str, weights: np.ndarray, exposures: np.ndarray) -> models.Features:
    return models.Features(
        exposures,
        weights,
        data["inventory"],
        data["early_fraction"],
        models.families_from_buckets(data["buckets"]),
        label,
        tuple(data["buckets"]),
    )


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def partition(output: Path, fold: int, repeat: int = 0) -> harness.Split:
    table = pd.read_csv(output / "inputs" / "splits.csv")
    table = table[table.fold.eq(fold) & table.repeat.eq(repeat)]
    training = table[table.role.eq("train")]
    train = training.row.to_numpy(int)
    test = table[table.role.eq("test")].row.to_numpy(int)
    inner = tuple(
        (
            training.loc[training.inner_fold.ne(i), "row"].to_numpy(int),
            training.loc[training.inner_fold.eq(i), "row"].to_numpy(int),
        )
        for i in range(3)
    )
    if not len(train) or set(train) & set(test):
        raise ValueError("Invalid outer partition")
    return harness.Split(repeat, fold, train, test, inner)


def source_fingerprint(output: Path, extra: tuple[Path, ...] = ()) -> str:
    files = {Path(__file__).resolve(), *extra}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and str(filename).startswith(str(REPO_ROOT / "experiments")) and str(filename).endswith(".py"):
            files.add(Path(filename))
    record = {
        "inputs": verify_inputs(output),
        "sources": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in sorted(files)},
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    write_json(output / f"provenance_{digest[:16]}.json", record)
    return digest


def fit_baseline(
    output: Path, model_id: str, target: str, component: int, fold: int, repeat: int, fingerprint: str
) -> str:
    path = output / "baseline_shards" / model_id / target / f"r{repeat}_f{fold}_c{component}.npz"
    if path.exists() and str(read_npz(path)["fingerprint"]) == fingerprint:
        return "cached"
    data = read_npz(output / "inputs" / "panel.npz")
    bank = read_npz(output / "inputs" / f"{target}_bank_features.npz")
    feature = feature_set(data, PANEL, data["weights"], data["exposures"])
    query = feature_set(data, f"{PANEL}|frozen-bank", bank["weights"], bank["exposures"])
    split = partition(output, fold, repeat)
    response = data[f"{target}_outcomes"][:, component] if component >= 0 else data[f"{target}_aggregate"]
    name = str(data[f"{target}_components"][component]) if component >= 0 else "direct_macro"
    entry_id = BASELINES[0] if model_id == "wspu_direct_macro" else model_id
    entry = registry.ENTRY_BY_ID[entry_id]
    feature = dataclasses.replace(registry.apply_transform(feature, entry), component=name)
    model = entry.build(feature)
    task = harness.FitTask(entry_id, PANEL, target, max(0, component), name, repeat, max(0, fold))
    start = time.monotonic()
    fit = model.fit(feature, response, split.train, split.inner, harness._seed(task))
    prediction = model.predict(fit, feature, split.test) if len(split.test) else np.empty(0)
    bank_prediction = model.predict(fit, registry.apply_transform(query, entry), np.arange(len(bank["weights"])))
    if not np.isfinite(prediction).all() or not np.isfinite(bank_prediction).all():
        raise ValueError(f"Non-finite baseline: {model_id}/{target}/{component}/{fold}")
    harness.atomic_save(
        path,
        {
            "fingerprint": fingerprint,
            "prediction": prediction,
            "bank_prediction": bank_prediction,
            "test": split.test,
            "train": split.train,
            "train_prediction": model.predict(fit, feature, split.train),
            "shape_json": json.dumps(fit.shape, sort_keys=True),
            "ridge": fit.ridge,
            "diagnostics_json": json.dumps(fit.diagnostics, sort_keys=True),
            "elapsed": time.monotonic() - start,
        },
    )
    return "fitted"


def run_baselines(output: Path, workers: int, repeats: int) -> None:
    fingerprint = source_fingerprint(output)
    data = read_npz(output / "inputs" / "panel.npz")
    tasks = [
        (m, t, c, f, r)
        for m in (*BASELINES, "wspu_direct_macro")
        for t in TARGETS
        for c in (range(data[f"{t}_outcomes"].shape[1]) if m in BASELINES else [-1])
        for r in range(repeats)
        for f in ((-1, 0, 1, 2, 3, 4) if r == 0 else (0, 1, 2, 3, 4))
    ]
    with parallel_config(backend="loky", inner_max_num_threads=1):
        counts = Parallel(n_jobs=workers, verbose=10)(
            delayed(fit_baseline)(output, *task, fingerprint) for task in tasks
        )
    print(pd.Series(counts).value_counts().to_dict(), flush=True)


def selection_metrics(measured: np.ndarray, prediction: np.ndarray, order: np.ndarray | None = None) -> dict:
    if order is None:
        order = np.argsort(prediction, kind="stable")
    selected = int(order[0])
    best = float(measured.min())
    return {
        "rows": len(measured),
        "regret_at_1": float(measured[selected] - best),
        "best_of_5_regret": float(measured[order[:5]].min() - best),
        "best_of_10_regret": float(measured[order[:10]].min() - best),
        "selected_rank": float(stats.rankdata(measured, method="min")[selected]),
        "selected_row": selected,
        "selected_measured": float(measured[selected]),
        "optimism": float(measured[selected] - prediction[selected]),
        "rmse": float(np.sqrt(np.mean((prediction - measured) ** 2))),
        "spearman": float(stats.spearmanr(prediction, measured).statistic) if np.ptp(prediction) > 0 else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("prepare", "baselines"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1, choices=(1, 5))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output_dir)
    else:
        verify_inputs(args.output_dir)
        run_baselines(args.output_dir, args.workers, args.repeats)


if __name__ == "__main__":
    main()
