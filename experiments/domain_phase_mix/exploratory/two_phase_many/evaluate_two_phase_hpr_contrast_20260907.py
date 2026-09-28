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
"""Run registered LINK2-004 after its held-pair synthetic gate passed.

The original constrained fitter and completed synthetic inputs remain frozen.
This orchestration layer fits actual observed pair differences, combines the
result with matching saved aggregate spines, and reports matched development
fold diagnostics. It launches no training or evaluation jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE.parents[3]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[3]))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_two_phase_hpr_contrast_20260907 as contrast,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_two_phase_link_transfer_20260907 as previous,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_two_phase_link_spines_20260907 import (  # noqa: E402
    write_json_atomic,
)

PREVIOUS = HERE / "reference_outputs/two_phase_link_transfer_20260907"
OUTPUT = HERE / "reference_outputs/two_phase_hpr_transfer_20260907/contrast_fit"
MODELS = ("aggregate", "hpr", "hpr_contrast_refit")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_cell(argument: tuple[str, str, dict[str, Any], str, str]) -> dict[str, Any]:
    """Select and refit a head using only the registered context's training pairs."""
    root_string, output_string, identity, objective, context = argument
    root, output = Path(root_string), Path(output_string)
    destination = output / "cells" / objective / context
    complete_path = destination / "complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        assert complete["identity"] == identity, "changed LINK2-004 fit protocol"
        for name, digest in complete["output_sha256"].items():
            assert file_hash(destination / name) == digest
        return {"objective": objective, "context": context, "cached": True}
    module, panel, splits = previous.inputs(str(root))
    dataset, _, _ = contrast.read_policy_inputs(root)
    train = previous.training_rows(panel, splits, context)
    assert panel["calibration_mask"][train].sum() == 2
    asymmetric, tied = previous.selected_pairs(panel, train)
    assert not panel["calibration_mask"][asymmetric].any()
    assert not panel["calibration_mask"][tied].any()
    response = panel[f"{objective}_aggregate"]
    deltas = response[asymmetric] - response[tied]
    inner = []
    for index in range(3):
        train_rows = splits[f"{context}_inner{index}_train"]
        test_rows = splits[f"{context}_inner{index}_test"]
        assert panel["calibration_mask"][train_rows].sum() == 2
        assert not panel["calibration_mask"][test_rows].any()
        inner_train_a, inner_train_t = previous.selected_pairs(panel, train_rows)
        inner_test_a, inner_test_t = previous.selected_pairs(panel, test_rows)
        assert np.isin(inner_train_t, tied).all() and np.isin(inner_test_t, tied).all()
        inner.append(
            (np.flatnonzero(np.isin(asymmetric, inner_train_a)), np.flatnonzero(np.isin(asymmetric, inner_test_a)))
        )
    configs = contrast.candidate_configs()
    selected, head, sweep = contrast.select_contrast_config(
        dataset, panel["weights"][asymmetric], deltas, panel["groups"][asymmetric], tuple(inner), configs
    )
    prediction_config = configs[0] if selected is None else selected
    design = contrast.contrast_design(dataset, prediction_config, panel["weights"])
    delta = head.predict(design)
    assert np.max(np.abs(delta[panel["physical_tied"]])) < 1e-12
    aggregate = np.zeros(len(response))
    source_paths = []
    for index, weight in enumerate(panel[f"{objective}_aggregation_weights"]):
        path = root / "spines" / context / f"{objective}_c{index}.json"
        source_paths.append(path)
        spine = previous.load_spine(root, context, objective, index)
        base, _, _ = previous.basis_and_prediction(module, panel, spine)
        aggregate += weight * base
    old_context = "full" if context == "final" else context
    hpr_directory = root / "controls/hierarchical_phase_replay" / objective / old_context
    old_complete = json.loads((hpr_directory / "complete.json").read_text())
    for name, digest in old_complete["sha256"].items():
        assert file_hash(hpr_directory / name) == digest
    with np.load(hpr_directory / "prediction.npz", allow_pickle=False) as archive:
        hpr_prediction = archive["prediction"].copy()
        assert np.max(np.abs(response - archive["observed"])) < 1e-12
        assert np.array_equal(train, archive["train"])
    source_paths.append(hpr_directory / "prediction.npz")
    full_prediction = aggregate + delta
    assert np.isfinite(full_prediction).all()
    assert np.max(np.abs(full_prediction[panel["physical_tied"]] - aggregate[panel["physical_tied"]])) < 1e-12
    scored = ~panel["calibration_mask"] if context == "final" else panel["outer_fold"] == int(context[-1])
    table = pd.DataFrame(
        {
            "objective": objective,
            "context": context,
            "row": np.arange(len(response)),
            "run": panel["runs"],
            "group": panel["groups"],
            "fold": panel["outer_fold"],
            "tied": panel["physical_tied"],
            "scored": scored,
            "measured": response,
            "aggregate": aggregate,
            "hpr": hpr_prediction,
            "hpr_contrast_refit": full_prediction,
            "delta": delta,
        }
    )
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "predictions.csv", index=False)
    write_json_atomic(destination / "selection_sweep.json", sweep)
    fit = {
        "objective": objective,
        "context": context,
        "selected_config": None if selected is None else asdict(selected),
        "prediction_config": asdict(prediction_config),
        "selected_inner_mse": min(float(row["paired_mse"]) for row in sweep),
        "n_train_pairs": len(asymmetric),
        "train_rows": train.tolist(),
        "train_asymmetric_rows": asymmetric.tolist(),
        "train_tied_rows": tied.tolist(),
        "inner_pairs": [
            {
                "train_asymmetric_rows": asymmetric[left].tolist(),
                "train_tied_rows": tied[left].tolist(),
                "test_asymmetric_rows": asymmetric[right].tolist(),
                "test_tied_rows": tied[right].tolist(),
            }
            for left, right in inner
        ],
        "coefficients": head.coefficients.tolist(),
        "feature_names": head.feature_names,
        "training_null_columns": [
            name for name, active in zip(head.feature_names, head.active_columns, strict=True) if not active
        ],
        "ridge_multipliers": head.ridge_multipliers.tolist(),
        "objective_value": head.objective,
        "kkt_violation": head.kkt_violation,
        "n_positive_coefficients": int(np.sum(head.coefficients > 1e-12)),
        "minimum_prediction": float(full_prediction.min()),
        "maximum_abs_phase_correction": float(np.max(np.abs(delta))),
        "tied_max_abs_phase_correction": float(np.max(np.abs(delta[panel["physical_tied"]]))),
    }
    write_json_atomic(destination / "fit.json", fit)
    complete = {
        "identity": identity,
        "source_sha256": {str(path): file_hash(path) for path in source_paths},
        "output_sha256": {
            name: file_hash(destination / name) for name in ("fit.json", "selection_sweep.json", "predictions.csv")
        },
    }
    write_json_atomic(complete_path, complete)
    return {
        "objective": objective,
        "context": context,
        "n_train_pairs": len(asymmetric),
        "selected_config": fit["selected_config"],
        "kkt_violation": head.kkt_violation,
    }


def phase_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    """Measure held-pair prediction and the binary asymmetric-versus-tied decision."""
    prediction = prediction.copy()
    prediction[np.abs(prediction) < 1e-12] = 0.0
    regret = np.where(prediction < 0, np.maximum(truth, 0), np.maximum(-truth, 0))
    return {
        "n": len(truth),
        "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "bias_predicted_minus_measured": float(np.mean(prediction - truth)),
        "phase_gain_bias": float(np.mean(truth - prediction)),
        "spearman": previous.safe_rank(truth, prediction),
        "sign_accuracy": float(np.mean(np.sign(prediction) == np.sign(truth))),
        "mean_binary_decision_regret": float(np.mean(regret)),
        "asymmetric_pick_fraction": float(np.mean(prediction < 0)),
    }


def summarize(root: Path, output: Path) -> None:
    """Assemble frozen OOF predictions and report matched controls and full fits separately."""
    _, panel, _ = previous.inputs(str(root))
    frames = [
        pd.read_csv(output / "cells" / objective / context / "predictions.csv")
        for objective in previous.OBJECTIVES
        for context in previous.CONTEXTS
    ]
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions.to_csv(output / "all_predictions.csv", index=False)
    oof = all_predictions.loc[all_predictions.context.ne("final") & all_predictions.scored].copy()
    assert oof.groupby("objective").size().eq(518).all()
    assert not oof.duplicated(["objective", "row"]).any()
    oof.to_csv(output / "oof_predictions.csv", index=False)
    levels, pairs, pair_predictions = [], [], []
    for objective in previous.OBJECTIVES:
        for context in (*previous.CONTEXTS, "oof"):
            if context == "oof":
                frame = oof.loc[oof.objective.eq(objective)].sort_values("row")
            else:
                frame = all_predictions.loc[
                    all_predictions.objective.eq(objective)
                    & all_predictions.context.eq(context)
                    & all_predictions.scored
                ].sort_values("row")
            rows = frame.row.to_numpy(dtype=int)
            assert np.array_equal(frame.run.to_numpy(), panel["runs"][rows])
            for model in MODELS:
                for population, mask in (
                    ("all", np.ones(len(frame), dtype=bool)),
                    ("tied", frame.tied.to_numpy()),
                    ("asymmetric", ~frame.tied.to_numpy()),
                ):
                    if mask.any():
                        record = previous.metrics(
                            frame.measured.to_numpy()[mask], frame[model].to_numpy()[mask], frame.tied.to_numpy()[mask]
                        )
                        levels.append(
                            {
                                "objective": objective,
                                "context": context,
                                "model": model,
                                "population": population,
                                **record,
                            }
                        )
            indexed = frame.set_index("row")
            a, t = previous.selected_pairs(panel, rows)
            if not len(a):
                continue
            truth = indexed.loc[a, "measured"].to_numpy() - indexed.loc[t, "measured"].to_numpy()
            for model in MODELS:
                prediction = indexed.loc[a, model].to_numpy() - indexed.loc[t, model].to_numpy()
                pairs.append(
                    {"objective": objective, "context": context, "model": model, **phase_metrics(truth, prediction)}
                )
                if context == "oof":
                    for index in range(len(a)):
                        pair_predictions.append(
                            {
                                "objective": objective,
                                "model": model,
                                "asymmetric_row": a[index],
                                "tied_row": t[index],
                                "group": panel["groups"][a[index]],
                                "fold": panel["outer_fold"][a[index]],
                                "measured_delta": truth[index],
                                "predicted_delta": prediction[index],
                            }
                        )
    pd.DataFrame(levels).to_csv(output / "endpoint_metrics.csv", index=False)
    pd.DataFrame(pairs).to_csv(output / "pair_metrics.csv", index=False)
    pd.DataFrame(pair_predictions).to_csv(output / "oof_pair_predictions.csv", index=False)
    manifest = {
        str(path.relative_to(output)): file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_json_atomic(
        output / "manifest.json",
        {"output_sha256": manifest, "scope": "Registered development-fold comparison; no prospective confirmation."},
    )
    print(pd.DataFrame(pairs).query("context == 'oof'").to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PREVIOUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    synthetic = args.output.parent / "contrast_synthetic"
    results = pd.read_csv(synthetic / "summary.csv", keep_default_na=False)
    for objective, tolerance in (("uncheatable", 0.0013), ("table9", 0.0039)):
        selected = results.loc[results.objective.eq(objective) & results["mode"].eq("selected_shape")]
        signal = selected.loc[selected.scenario.ne("null")]
        assert (signal.rmse.to_numpy(dtype=float) < signal.truth_rms.to_numpy(dtype=float)).sum() >= 2
        assert selected.loc[selected.scenario.eq("null"), "positive_gain_rms"].item() <= tolerance
    source_paths = [
        Path(__file__),
        Path(contrast.__file__),
        Path(previous.__file__),
        Path(contrast.hpr.__file__),
        Path(contrast.hpr.family_grp.__file__),
        Path(contrast.observatory.__file__),
        args.root / "inputs/panel.npz",
        args.root / "inputs/splits.npz",
        args.root / "inputs/single_phase.py",
        synthetic / "protocol.json",
        synthetic / "summary.csv",
    ]
    identity = {
        "source_sha256": {str(path): file_hash(path) for path in source_paths},
        "config_grid": [asdict(config) for config in contrast.candidate_configs()],
        "exact_zero_head_included": True,
        "protocol": (
            "LINK2-004: no-intercept nonnegative HPR feature-difference head with source hierarchy ridge; "
            "observed pair inner SSE."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == identity
    write_json_atomic(protocol_path, identity)
    arguments = [
        (str(args.root), str(args.output), identity, objective, context)
        for objective in previous.OBJECTIVES
        for context in previous.CONTEXTS
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit_cell, argument) for argument in arguments]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    summarize(args.root, args.output)


if __name__ == "__main__":
    main()
