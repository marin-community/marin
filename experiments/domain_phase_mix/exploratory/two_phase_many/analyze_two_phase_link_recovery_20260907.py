# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "scipy==1.17.0", "pandas==2.2.2"]
# ///
"""Objective-level synthetic phase recovery under measured cross-task noise.

Truth uses the 58 final tied-only spines. One scalar per objective/direction
sets the aggregated true contrast RMS, preserving the task aggregation. Noise
resamples entire centered 58-task repeat vectors independently across runs.
Both known and re-estimated spines use an unshrunk temporal estimator. The
re-estimated spine holds shape, ridge, and floor fixed and refits its intercept
and nonnegative amplitudes on generated noisy tied training outcomes only.
This is an optimistic conditional sensitivity check, not endpoint validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
if str(HERE.parents[3]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[3]))

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_two_phase_link_spines_20260907 import (  # noqa: E402
    THREAD_VARIABLES,
    write_json_atomic,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.fit_two_phase_link_transfer_20260907 import (  # noqa: E402
    basis_and_prediction,
    inputs,
    load_spine,
    selected_pairs,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.two_phase_link_residual_20260907 import (  # noqa: E402
    LOG_CLIP,
    fit_bpb_contrasts,
    predict_bpb_delta,
)

DEFAULT_OUTPUT = HERE / "reference_outputs/two_phase_link_transfer_20260907"
OBJECTIVES = ("uncheatable", "table9")
SIGNAL_RMS = (0.0013, 0.0039)
DIRECTIONS = np.array([(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -1.0)])
SCENARIOS = ("benefit", "harm", "aligned", "opposed", "null")
BASES = ("known", "refitted")
SEED = 2026090700


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_signal_error(scale: float, q: np.ndarray, projected: np.ndarray, weights: np.ndarray, target: float) -> float:
    """Return the aggregated true contrast RMS minus its prescribed target."""
    delta = q * np.expm1(np.clip(scale * projected, -LOG_CLIP, LOG_CLIP))
    return float(np.sqrt(np.mean((delta @ weights) ** 2)) - target)


def truth_arrays(output: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Generate task truths with one amplitude scale for each objective/direction."""
    module, panel, _ = inputs(str(output))
    with np.load(output / "noise/noise.npz", allow_pickle=False) as noise:
        assert np.array_equal(noise["components"], panel["anchor_components"])
        weights = noise["objective_weights"].copy()
        centered = noise["centered_outcomes"].copy()
    assert weights.shape == (2, 58) and centered.shape == (11, 58)
    assert np.max(np.abs(centered.mean(axis=0))) < 1e-12
    components = [
        (objective, index) for objective in OBJECTIVES for index in range(len(panel[f"{objective}_components"]))
    ]
    bases, deficits, designs = [], [], []
    for objective, component in components:
        base, deficit, design = basis_and_prediction(module, panel, load_spine(output, "final", objective, component))
        bases.append(base)
        deficits.append(deficit)
        designs.append(design)
    base = np.stack(bases, axis=1)
    q = np.stack(deficits, axis=1)
    columns = np.stack(designs, axis=1)
    pair_rows = panel["pair_asymmetric_rows"]
    norms = np.linalg.norm(q[pair_rows, :, None] * columns[pair_rows], axis=0)
    signals = np.zeros((len(SCENARIOS), len(base), 58))
    theta = np.zeros((len(SCENARIOS), 58, 2))
    records = []
    for objective_id, objective in enumerate(OBJECTIVES):
        selected = weights[objective_id] > 0
        for direction_id, raw in enumerate(DIRECTIONS):
            direction = np.divide(raw, norms[selected], out=np.zeros_like(norms[selected]), where=norms[selected] > 0)
            projected = np.einsum("ntj,tj->nt", columns[:, selected], direction)

            arguments = (
                q[pair_rows][:, selected],
                projected[pair_rows],
                weights[objective_id, selected],
                SIGNAL_RMS[objective_id],
            )
            upper = 1.0
            while macro_signal_error(upper, *arguments) < 0.0 and upper < 1e6:
                upper *= 2
            scale = brentq(macro_signal_error, 0.0, upper, args=arguments, xtol=1e-14)
            correction = scale * projected
            assert np.max(np.abs(correction)) < LOG_CLIP, "Synthetic truth reached the numerical clip"
            theta[direction_id, selected] = scale * direction
            signals[direction_id][:, selected] = q[:, selected] * np.expm1(correction)
            macro = signals[direction_id, pair_rows] @ weights[objective_id]
            actual_rms = float(np.sqrt(np.mean(macro**2)))
            assert abs(actual_rms - SIGNAL_RMS[objective_id]) < 1e-10
            records.append(
                {
                    "objective": objective,
                    "scenario": SCENARIOS[direction_id],
                    "common_scale": scale,
                    "true_macro_rms": actual_rms,
                    "max_abs_log_correction": float(np.max(np.abs(correction))),
                    "task_theta": theta[direction_id, selected].tolist(),
                }
            )
    return {
        "base": base,
        "q": q,
        "columns": columns,
        "signals": signals,
        "theta": theta,
        "weights": weights,
        "centered_noise": centered,
        "column_norms": norms,
    }, records


def run_draw(argument: tuple[str, int, dict[str, str]]) -> dict[str, Any]:
    """Fit both conditional estimators on one shared task-correlated noise draw."""
    output_string, draw, identity = argument
    output = Path(output_string)
    directory = output / "objective_recovery"
    target = directory / f"draw{draw}.npz"
    metadata_path = directory / f"draw{draw}.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity or metadata["output_sha256"] != file_hash(target):
            raise ValueError(f"recovery cache changed: {metadata_path}")
        return {"draw": draw, "cached": True}
    module, panel, splits = inputs(output_string)
    with np.load(directory / "truths.npz", allow_pickle=False) as data:
        truth = {key: data[key] for key in data.files}
    pair_a, pair_t = panel["pair_asymmetric_rows"], panel["pair_tied_rows"]
    positions = np.full(len(panel["runs"]), -1, dtype=int)
    positions[pair_a] = np.arange(len(pair_a))
    random = np.random.default_rng(SEED + draw)
    sampled = random.integers(len(truth["centered_noise"]), size=len(panel["runs"]))
    noise = truth["centered_noise"][sampled]
    predictions = np.full((len(BASES), len(SCENARIOS), len(pair_a), 58), np.nan)
    fitted_theta = np.zeros((len(BASES), len(SCENARIOS), 3, 58, 2))
    floor_clamp_count = 0
    max_abs_test_correction = 0.0
    start = time.monotonic()
    components = [
        (objective, index) for objective in OBJECTIVES for index in range(len(panel[f"{objective}_components"]))
    ]
    for fold in range(3):
        train_rows = splits[f"outer{fold}_train"]
        train_tied = train_rows[panel["physical_tied"][train_rows]]
        train_a, train_t = selected_pairs(panel, train_rows)
        test_a, _ = selected_pairs(panel, splits[f"outer{fold}_test"])
        if not len(test_a):
            continue
        assert len(train_a)
        for global_component, (objective, component) in enumerate(components):
            spine = load_spine(output, "final", objective, component)
            matrix = module.design_matrix(panel["epochs"], spine.shape)
            noisy_deficit = (
                truth["base"][train_tied, global_component] + noise[train_tied, global_component] - spine.head.floor
            )
            floor_clamp_count += int(np.sum(noisy_deficit <= module.DEFICIT_FLOOR))
            intercept, coefficients = module.nonnegative_solve(
                matrix[train_tied], np.log(np.maximum(noisy_deficit, module.DEFICIT_FLOOR)), spine.ridge
            )
            refitted = replace(spine, head=module.Head(intercept, coefficients, spine.head.floor))
            _, refit_q, refit_columns = basis_and_prediction(module, panel, refitted)
            basis_parameters = (
                (truth["q"][:, global_component], truth["columns"][:, global_component]),
                (refit_q, refit_columns),
            )
            noise_delta = noise[train_a, global_component] - noise[train_t, global_component]
            base_delta = truth["base"][train_a, global_component] - truth["base"][train_t, global_component]
            for basis_id, (q, columns) in enumerate(basis_parameters):
                for scenario in range(len(SCENARIOS)):
                    observed = truth["signals"][scenario, train_a, global_component] + base_delta + noise_delta
                    fit = fit_bpb_contrasts(q[train_a], columns[train_a], observed, 0.0)
                    if not fit.success:
                        raise RuntimeError(
                            f"synthetic temporal fit failed: draw={draw}, fold={fold}, task={global_component}"
                        )
                    predictions[basis_id, scenario, positions[test_a], global_component] = predict_bpb_delta(
                        q[test_a], columns[test_a], fit.theta
                    )
                    fitted_theta[basis_id, scenario, fold, global_component] = fit.theta
                    max_abs_test_correction = max(
                        max_abs_test_correction, float(np.max(np.abs(columns[test_a] @ fit.theta)))
                    )
    assert np.isfinite(predictions).all(), "Every exact pair must receive a held-fold prediction"
    macro_prediction = np.einsum("bsnt,ot->bsno", predictions, truth["weights"])
    macro_truth = np.einsum("snt,ot->sno", truth["signals"][:, pair_a], truth["weights"])
    with target.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(
            stream,
            macro_prediction=macro_prediction,
            macro_truth=macro_truth,
            noise_sample_rows=sampled,
            fitted_theta=fitted_theta,
            pair_asymmetric_rows=pair_a,
            pair_tied_rows=pair_t,
            pair_groups=panel["groups"][pair_a],
            outer_fold=panel["outer_fold"][pair_a],
        )
    target.with_suffix(".tmp").replace(target)
    metadata = {
        "draw": draw,
        "identity": identity,
        "output_sha256": file_hash(target),
        "elapsed_seconds": time.monotonic() - start,
        "floor_clamp_count": floor_clamp_count,
        "max_abs_test_log_correction": max_abs_test_correction,
        "seed": SEED + draw,
    }
    write_json_atomic(metadata_path, metadata)
    return {key: value for key, value in metadata.items() if key != "identity"}


def summarize(output: Path, draws: int) -> list[dict[str, Any]]:
    """Persist pair predictions and objective-level errors for conditional inference."""
    directory = output / "objective_recovery"
    records = []
    for draw in range(draws):
        with np.load(directory / f"draw{draw}.npz", allow_pickle=False) as data:
            for objective_id, objective in enumerate(OBJECTIVES):
                for basis_id, basis in enumerate(BASES):
                    for scenario_id, scenario in enumerate(SCENARIOS):
                        truth = data["macro_truth"][scenario_id, :, objective_id]
                        prediction = data["macro_prediction"][basis_id, scenario_id, :, objective_id]
                        records.extend(
                            {
                                "objective": objective,
                                "basis": basis,
                                "scenario": scenario,
                                "draw": draw,
                                "pair": pair,
                                "group": str(data["pair_groups"][pair]),
                                "outer_fold": int(data["outer_fold"][pair]),
                                "true_delta": float(truth[pair]),
                                "predicted_delta": float(prediction[pair]),
                            }
                            for pair in range(len(truth))
                        )
    frame = pd.DataFrame(records)
    frame.to_csv(directory / "pair_predictions.csv", index=False)
    summaries = []
    for (objective, basis, scenario), rows in frame.groupby(["objective", "basis", "scenario"], sort=False):
        truth = rows["true_delta"].to_numpy()
        prediction = rows["predicted_delta"].to_numpy()
        rms = float(np.sqrt(np.mean(truth**2)))
        rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
        summaries.append(
            {
                "objective": objective,
                "basis": basis,
                "scenario": scenario,
                "pairs_per_draw": int(rows["pair"].nunique()),
                "draws": int(rows["draw"].nunique()),
                "true_rms": rms,
                "rmse": rmse,
                "rmse_signal_ratio": rmse / rms if rms > 0 else None,
                "rmse_reference_signal_ratio": rmse / SIGNAL_RMS[OBJECTIVES.index(objective)],
                "bias": float(np.mean(prediction - truth)),
                "predicted_gain_rms": float(np.sqrt(np.mean(np.minimum(prediction, 0.0) ** 2))),
                "predicted_max_gain": float(max(0.0, -prediction.min())),
                "mean_binary_regret": float(np.mean(np.where(prediction < 0.0, truth, 0.0) - np.minimum(truth, 0.0))),
            }
        )
    pd.DataFrame(summaries).to_csv(directory / "summary.csv", index=False)
    write_json_atomic(
        directory / "summary.json",
        {
            "results": summaries,
            "scope": (
                "Conditional on final tied-only fitted shapes, ridge and floors. Empirical noise resamples the eleven "
                "centered proportional variable-subset run vectors; uniform resampling has their ddof=0 covariance. "
                "Known-basis and refitted-basis arms share all generated endpoints. Phase penalty is zero, "
                "an optimistic unshrunk diagnostic. No actual asymmetric endpoint is used. Noise draws are "
                "a bounded sensitivity screen, not prospective evidence or a hard promotion gate."
            ),
        },
    )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.draws < 1 or args.workers < 1:
        raise ValueError("draws and workers must be positive")
    output = args.output.resolve()
    directory = output / "objective_recovery"
    paths = [
        Path(__file__),
        HERE / "two_phase_link_residual_20260907.py",
        HERE / "fit_two_phase_link_transfer_20260907.py",
        HERE / "fit_two_phase_link_spines_20260907.py",
        output / "inputs/panel.npz",
        output / "inputs/splits.npz",
        output / "inputs/single_phase.py",
        output / "noise/noise.npz",
        output / "noise/manifest.json",
        *sorted((output / "spines/final").glob("*_c*.json")),
    ]
    identity = {str(path): file_hash(path) for path in paths}
    noise_manifest = json.loads((output / "noise/manifest.json").read_text())
    assert identity[str(output / "noise/noise.npz")] == noise_manifest["output_sha256"]["noise.npz"]
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != identity or manifest["truths_sha256"] != file_hash(directory / "truths.npz"):
            raise ValueError("recovery identity changed; preserve existing output and choose a new output root")
    else:
        arrays, parameters = truth_arrays(output)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "truths.tmp").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        (directory / "truths.tmp").replace(directory / "truths.npz")
        write_json_atomic(directory / "signal_parameters.json", {"signals": parameters})
        write_json_atomic(
            manifest_path,
            {"identity": identity, "truths_sha256": file_hash(directory / "truths.npz"), "scenarios": SCENARIOS},
        )
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_draw, (str(output), draw, identity)) for draw in range(args.draws)]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    print(json.dumps(summarize(output, args.draws), indent=2), flush=True)


if __name__ == "__main__":
    main()
