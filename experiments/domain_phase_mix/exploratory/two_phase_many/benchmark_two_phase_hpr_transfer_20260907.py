# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.7.2",
#   "cvxpy==1.7.5", "fsspec==2026.1.0", "gcsfs==2026.1.0", "plotly==6.5.1",
#   "tabulate==0.9.0", "threadpoolctl==3.6.0",
# ]
# ///
"""Evaluate the registered parameter-free aggregate replacement of frozen HPR."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import benchmark_two_phase_link_controls_20260907 as controls
import fit_two_phase_link_transfer_20260907 as previous
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "reference_outputs"
PREVIOUS = ROOT / "two_phase_link_transfer_20260907"
OUTPUT = ROOT / "two_phase_hpr_transfer_20260907"
MODELS = ("aggregate", "hpr", "hpr_aggregate_replacement")
NOISE = {"uncheatable": 0.00112710075, "table9": 0.00333003459}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(output: Path) -> None:
    """Preserve previous folds and verify exact contrast invariance before scoring."""
    module, panel, _ = previous.inputs(str(PREVIOUS))
    n = len(panel["runs"])
    all_rows = np.arange(n)
    pair_a, pair_t = panel["pair_asymmetric_rows"], panel["pair_tied_rows"]
    tied_weights = np.repeat(panel["aggregate"][:, None, :], 2, axis=1)
    metric_rows, prediction_rows, checks, bootstrap_rows = [], [], [], []
    sources = {Path(__file__), PREVIOUS / "inputs/panel.npz", PREVIOUS / "inputs/splits.npz"}
    sources.add(output / "PROTOCOL.md")
    gates = {}
    rng = np.random.default_rng(20260907)
    for objective in previous.OBJECTIVES:
        truth = panel[f"{objective}_aggregate"]
        oof = {name: np.full(n, np.nan) for name in MODELS}
        for context in previous.CONTEXTS:
            fold_name = "full" if context == "final" else context
            cell = PREVIOUS / "controls/hierarchical_phase_replay" / objective / fold_name
            complete = json.loads((cell / "complete.json").read_text())
            for name, digest in complete["sha256"].items():
                assert file_hash(cell / name) == digest, f"changed HPR output: {cell / name}"
                sources.add(cell / name)
            for name, digest in complete["protocol"]["source_sha256"].items():
                path = controls.REPO_ROOT / name
                assert file_hash(path) == digest, f"changed HPR source: {path}"
                sources.add(path)
            with (cell / "model.pkl").open("rb") as handle:
                fitted = pickle.load(handle)
            assert isinstance(fitted, controls.baseline.hierarchical_grp.Model)
            hpr = fitted.predict(panel["weights"])
            hpr_tied = fitted.predict(tied_weights)
            saved = np.load(cell / "prediction.npz", allow_pickle=False)["prediction"]
            hpr_parity = float(np.max(np.abs(saved - hpr)))
            assert hpr_parity < 1e-12
            aggregate = np.zeros(n)
            for index, weight in enumerate(panel[f"{objective}_aggregation_weights"]):
                path = PREVIOUS / "spines" / context / f"{objective}_c{index}.json"
                sources.add(path)
                spine = previous.load_spine(PREVIOUS, context, objective, index)
                values, _, _ = previous.basis_and_prediction(module, panel, spine)
                aggregate += weight * values
            transfer = aggregate + hpr - hpr_tied
            tied_parity = float(np.max(np.abs(transfer[panel["physical_tied"]] - aggregate[panel["physical_tied"]])))
            contrast_parity = float(np.max(np.abs((transfer[pair_a] - transfer[pair_t]) - (hpr[pair_a] - hpr[pair_t]))))
            assert tied_parity < 1e-12 and contrast_parity < 1e-12
            predictions = dict(zip(MODELS, (aggregate, hpr, transfer), strict=True))
            scored = ~panel["calibration_mask"] if context == "final" else panel["outer_fold"] == int(context[-1])
            for model, values in predictions.items():
                assert np.isfinite(values).all()
                if context != "final":
                    oof[model][scored] = values[scored]
                for row in all_rows:
                    prediction_rows.append(
                        {
                            "objective": objective,
                            "context": context,
                            "model": model,
                            "row": row,
                            "run": panel["runs"][row],
                            "group": panel["groups"][row],
                            "fold": panel["outer_fold"][row],
                            "tied": bool(panel["physical_tied"][row]),
                            "scored": bool(scored[row]),
                            "measured": truth[row],
                            "predicted": values[row],
                        }
                    )
                for population, subset in (
                    ("all", scored),
                    ("tied", scored & panel["physical_tied"]),
                    ("asymmetric", scored & ~panel["physical_tied"]),
                ):
                    if np.any(subset):
                        metric_rows.append(
                            {
                                "objective": objective,
                                "context": context,
                                "model": model,
                                "population": population,
                                **previous.metrics(truth[subset], values[subset], panel["physical_tied"][subset]),
                            }
                        )
            checks.append(
                {
                    "objective": objective,
                    "context": context,
                    "hpr_saved_prediction_error": hpr_parity,
                    "tied_parity_error": tied_parity,
                    "contrast_parity_error": contrast_parity,
                    "minimum_transfer_prediction": float(transfer.min()),
                    "negative_predictions": int(np.sum(transfer < 0)),
                }
            )
        mask = ~panel["calibration_mask"]
        pooled = {}
        for model, values in oof.items():
            for population, subset in (
                ("all", mask),
                ("tied", mask & panel["physical_tied"]),
                ("asymmetric", mask & ~panel["physical_tied"]),
            ):
                record = previous.metrics(truth[subset], values[subset], panel["physical_tied"][subset])
                metric_rows.append(
                    {"objective": objective, "context": "oof", "model": model, "population": population, **record}
                )
                pooled[(model, population)] = record
        for population, subset in (
            ("all", mask),
            ("tied", mask & panel["physical_tied"]),
            ("asymmetric", mask & ~panel["physical_tied"]),
        ):
            _, inverse = np.unique(panel["groups"][subset], return_inverse=True)
            count = np.bincount(inverse)
            samples = rng.integers(0, len(count), size=(3000, len(count)))
            denominator = count[samples].sum(axis=1)
            se = {model: np.bincount(inverse, weights=(oof[model][subset] - truth[subset]) ** 2) for model in MODELS}
            difference = np.sqrt(se[MODELS[2]][samples].sum(axis=1) / denominator) - np.sqrt(
                se["hpr"][samples].sum(axis=1) / denominator
            )
            low, high = np.quantile(difference, [0.025, 0.975])
            bootstrap_rows.append(
                {
                    "objective": objective,
                    "population": population,
                    "metric": "transfer_minus_hpr_rmse",
                    "difference": pooled[(MODELS[2], population)]["rmse"] - pooled[("hpr", population)]["rmse"],
                    "ci_low": low,
                    "ci_high": high,
                    "groups": len(count),
                    "draws": len(samples),
                }
            )
        a, h = pooled[(MODELS[2], "all")], pooled[("hpr", "all")]
        gates[objective] = {
            "endpoint_nonregression": a["rmse"] <= 1.05 * h["rmse"],
            "tied_or_asymmetric_improvement": any(
                pooled[(MODELS[2], pop)]["rmse"] < pooled[("hpr", pop)]["rmse"] for pop in ("tied", "asymmetric")
            ),
            "regret1_nonregression": a["regret1"] <= h["regret1"] + NOISE[objective],
            "regret5_nonregression": a["regret5"] <= h["regret5"] + NOISE[objective],
        }
    destination = output / "aggregate_replacement"
    destination.mkdir(exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(destination / "predictions.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(destination / "bootstrap.csv", index=False)
    (destination / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    (destination / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    manifest = {
        "source_sha256": {str(path): file_hash(path) for path in sorted(sources)},
        "output_sha256": {
            path.name: file_hash(path)
            for path in sorted(destination.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
        "scope": "Registered parameter-free replacement on previously inspected development folds; no new fitting.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        metrics[(metrics.context == "oof") & (metrics.population == "all")]
        .drop(columns="tie_policy")
        .to_string(index=False)
    )
    print(json.dumps(gates, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    evaluate(args.output)


if __name__ == "__main__":
    main()
