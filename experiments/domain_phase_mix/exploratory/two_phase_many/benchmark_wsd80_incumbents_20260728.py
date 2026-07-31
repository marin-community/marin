# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""How the incumbent surrogates do on the 80/20 WSD StarCoder surface, and where they fail.

Fit quality alone would not settle anything here. A model can carry a good cross-validated error on a
dense surface while being structurally unable to represent the one feature that matters, because the
surface is dominated by the aggregate response and the phase effect is a hundredth of its range. So
this benchmark reports accuracy and then asks three mechanistic questions of every model.

Where does it put the optimum? The measured optimum is a two-phase policy at aggregate 0.18, twelve
aggregate points below the best tied policy. A model that collapses phase order will place its optimum
on or near the tied diagonal.

What phase gain does it predict at each aggregate? The measured phase gain closes to zero exactly at
the one-phase optimum and grows as the aggregate falls. Reproducing that profile is the real test.

And can it reproduce the two fibers? Those are the only paired, seeded, fixed-aggregate evidence on the
panel, so predicted-versus-measured on them is the cleanest read of the phase channel.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    aggregate_conditioned_phase_control_model_20260729 as aggregate_control,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as geometry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as retained,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_state_dynamics_model_20260729 as state_dynamics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search import (  # noqa: E402
    plot_separate_heads_starcoder_u_shape_fit as separate_heads,
)

DEFAULT_OUTPUT_DIR = wsd80.REFERENCE_OUTPUTS / "wsd80_incumbent_benchmark_20260728"
# Sentinel for the new model, which has its own fit entry point rather than a geometry FitConfig.
RETAINED_POWER_LAW = "retained_power_law"
RETAINED_STATE_DYNAMICS = "retained_state_dynamics"
AGGREGATE_CONDITIONED_PHASE_CONTROL = "aggregate_conditioned_phase_control"
MODEL_CONFIGS = (
    ("effective_exposure", geometry.FitConfig("effective_exposure", False)),
    (
        "effective_exposure_geometry",
        geometry.FitConfig("effective_exposure_geometry", True, "effective_exposure", (0, 1)),
    ),
    ("canonical", geometry.FitConfig("canonical", False, "canonical")),
    ("separate_heads", None),
    ("retained_power_law", RETAINED_POWER_LAW),
    ("retained_state_dynamics", RETAINED_STATE_DYNAMICS),
    ("aggregate_conditioned_phase_control", AGGREGATE_CONDITIONED_PHASE_CONTROL),
)
CV_SPLITS = 5
CV_SEEDS = (0, 1, 2)
OPTIMUM_GRID = 201
# Matches the hyperparameters the existing cosine/WSD StarCoder benchmark uses, so the incumbent
# numbers here are comparable to the ones already on record for those panels.
FIT_KWARGS = {"linear_reg": 0.01, "maxiter": 16, "coarse_top_k": 2}
SHAPE_SELECTION_SEED = 100
# Both fold protocols are reported. Random folds are what the incumbents were developed under and what
# every number previously on record used; blocked folds hold out whole regions of the mixture square
# and are the honest test of prediction on a new policy. Neither is quietly preferred.
FOLD_BUILDER = None
# Every fixed-aggregate fiber the panel measures. The phase gain g(a) is V-shaped across them with its
# minimum exactly at the one-phase optimum, and the best contrast changes sign there: code late below
# a = 0.30, code early above it. Reproducing that profile, and especially the sign flip, is a far
# stronger test of a surrogate's phase channel than matching two points was.
FIBER_AGGREGATES = (0.18, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
# Measured best two-phase policy, and its advantage over the best measured tied policy.
TRUE_OPTIMUM = (0.100, 0.500)
TRUE_TWO_PHASE_GAIN = 0.009594


def random_folds(weights: np.ndarray, indices: np.ndarray, n_splits: int, seed: int):
    """Plain shuffled K-fold, the protocol the incumbent models were developed under."""
    return [
        (indices[train], indices[test])
        for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(indices)
    ]


def mixture_blocked_folds(weights: np.ndarray, indices: np.ndarray, n_splits: int, seed: int):
    """Folds that hold out contiguous regions of mixture space rather than scattered points.

    This surface is densely sampled, so a randomly held-out coordinate almost always has a near
    neighbour left in training. Out-of-fold error under random folds therefore measures interpolation
    between adjacent mixtures, not prediction of new ones, and it rewards capacity that will not
    survive on a genuinely unseen policy. Blocking removes that, at the cost of a harder and more
    honest number.
    """
    coordinates = np.column_stack([weights[indices, 0, :], weights[indices, 1, :]])
    blocks = KMeans(n_clusters=n_splits, n_init=10, random_state=seed).fit_predict(coordinates)
    folds = []
    for block in np.unique(blocks):
        held = indices[blocks == block]
        if len(held) in (0, len(indices)):
            continue
        folds.append((np.setdiff1d(indices, held), held))
    assert len(folds) >= 2, f"mixture blocking produced {len(folds)} usable folds"
    return folds


def fit_predictor(data: pooled.Dataset, indices: np.ndarray, config) -> Callable[[np.ndarray], np.ndarray]:
    """Fit one incumbent and return a plain weights-to-prediction callable.

    The DSP family and the separate-heads family expose different prediction signatures, so both are
    wrapped here rather than special-cased at every call site.
    """
    if config in (RETAINED_POWER_LAW, RETAINED_STATE_DYNAMICS, AGGREGATE_CONDITIONED_PHASE_CONTROL):
        # Everything below is restricted to `indices`. Shape and ridge are selected on inner folds of
        # the training rows, and the final head is refitted on those same training rows only. An
        # earlier version selected correctly but refitted on the whole panel, which let each held-out
        # target influence the coefficients used to predict it.
        weights, target = data.weights[indices], data.y[indices]
        positions = np.arange(len(indices))
        lookup = {row: position for position, row in enumerate(indices)}
        inner = FOLD_BUILDER(data.weights, indices, CV_SPLITS, SHAPE_SELECTION_SEED)
        folds = tuple(
            (
                np.isin(positions, [lookup[row] for row in train]),
                np.isin(positions, [lookup[row] for row in test]),
            )
            for train, test in inner
        )
        model_module = {
            RETAINED_POWER_LAW: retained,
            RETAINED_STATE_DYNAMICS: state_dynamics,
            AGGREGATE_CONDITIONED_PHASE_CONTROL: aggregate_control,
        }[config]
        fit_model = model_module.fit_two_stage if config == AGGREGATE_CONDITIONED_PHASE_CONTROL else model_module.fit
        model = fit_model(
            weights,
            target,
            model_module.Geometry(c0=data.c0, c1=data.c1, phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION),
            folds=folds,
        )
        return model.predict
    if config is None:
        model = separate_heads.fit_separate_heads(geometry.packet(data, indices))
        return model.predict
    alpha0, alpha1 = geometry.phase_fractions(data)
    model = geometry.fit_model(data, indices, config, **FIT_KWARGS)
    return lambda weights: geometry.predict(model, weights, alpha0, alpha1)


def as_pooled_dataset(panel: wsd80.Panel) -> pooled.Dataset:
    return pooled.Dataset(
        name=panel.name,
        frame=panel.frame,
        y=panel.y,
        weights=panel.weights,
        c0=panel.c0,
        c1=panel.c1,
        domain_names=panel.domain_names,
    )


def grid_weights(phase_0_share: np.ndarray, phase_1_share: np.ndarray) -> np.ndarray:
    phase_0 = np.column_stack([1.0 - phase_0_share, phase_0_share])
    phase_1 = np.column_stack([1.0 - phase_1_share, phase_1_share])
    return np.stack([phase_0, phase_1], axis=1)


def predicted_optimum(predict, resolution: int) -> dict[str, float]:
    """Argmin of the fitted response over the whole feasible square, not just the sampled points."""
    axis = np.linspace(0.0, 1.0, resolution)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    weights = grid_weights(grid_0.ravel(), grid_1.ravel())
    prediction = predict(weights)
    best = int(np.argmin(prediction))
    phase_0, phase_1 = float(grid_0.ravel()[best]), float(grid_1.ravel()[best])
    return {
        "phase_0": phase_0,
        "phase_1": phase_1,
        "aggregate": wsd80.PHASE_0_FRACTION * phase_0 + wsd80.PHASE_1_FRACTION * phase_1,
        "contrast": phase_1 - phase_0,
        "prediction": float(prediction[best]),
    }


def predicted_phase_gain(predict, aggregate: float, resolution: int) -> dict[str, float]:
    """Best gain the model believes phase ordering buys at a fixed aggregate."""
    low = max(-aggregate / wsd80.PHASE_0_FRACTION, (aggregate - 1.0) / wsd80.PHASE_1_FRACTION)
    high = min(aggregate / wsd80.PHASE_1_FRACTION, (1.0 - aggregate) / wsd80.PHASE_0_FRACTION)
    contrast = np.linspace(low, high, resolution)
    phase_0 = aggregate - wsd80.PHASE_1_FRACTION * contrast
    phase_1 = aggregate + wsd80.PHASE_0_FRACTION * contrast
    prediction = predict(grid_weights(phase_0, phase_1))
    tied = predict(grid_weights(np.array([aggregate]), np.array([aggregate])))[0]
    best = int(np.argmin(prediction))
    return {
        "aggregate": aggregate,
        "tied_prediction": float(tied),
        "best_contrast": float(contrast[best]),
        "phase_gain": float(tied - prediction[best]),
    }


def two_phase_advantage(predict, resolution: int) -> dict[str, float]:
    """How much better than its own best tied policy the model thinks a two-phase policy can be.

    This is the property the panel exists to test and the one no error metric reports. A model whose
    response depends on the two phases only through an additive phase-weighted dose predicts *exactly*
    zero here, whatever its RMSE: the tied class already attains every reachable dose, so nothing is
    left for a schedule to win. The measurement on this panel is +0.009594 BPB, so a zero prediction is
    not a small error, it is a structural inability to represent the surface's defining feature.
    """
    # The tied search must be at least as fine as the two-dimensional one, or a model whose true
    # advantage is zero still shows a small positive: the square realises effective-dose values that
    # fall between the tied grid's points. At equal axis resolution the additive-dose incumbents
    # reported about 1.2e-6 instead of their structural zero. Refining the tied axis by the grid count
    # makes the comparison symmetric in what each side can reach.
    axis = np.linspace(0.0, 1.0, resolution)
    tied_axis = np.linspace(0.0, 1.0, resolution * resolution)
    tied = predict(grid_weights(tied_axis, tied_axis))
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    everywhere = predict(grid_weights(grid_0.ravel(), grid_1.ravel()))
    best = int(np.argmin(everywhere))
    return {
        "predicted_tied_bpb": float(tied.min()),
        "predicted_best_bpb": float(everywhere[best]),
        "predicted_two_phase_gain": float(tied.min() - everywhere[best]),
        "optimum_distance": float(
            np.hypot(grid_0.ravel()[best] - TRUE_OPTIMUM[0], grid_1.ravel()[best] - TRUE_OPTIMUM[1])
        ),
    }


def tied_diagonal_fit(predict, panel: wsd80.Panel, resolution: int) -> dict[str, float]:
    """How well the model reproduces the one-phase response, which is the easy half of the problem.

    If a surrogate cannot place the best constant mixture it has no chance at the two-phase optimum,
    and the failure is in the aggregate channel rather than the phase channel. Reported separately for
    that reason.
    """
    tied = np.flatnonzero(np.isclose(panel.contrast[:, 1], 0.0, atol=1e-9))
    residual = predict(panel.weights[tied]) - panel.y[tied]
    share = np.linspace(0.0, 1.0, resolution)
    curve = predict(grid_weights(share, share))
    return {
        "tied_rmse": float(np.sqrt(np.mean(residual**2))),
        "tied_rows": len(tied),
        "predicted_tied_optimum": float(share[int(np.argmin(curve))]),
        "measured_tied_optimum": float(panel.phase_0[tied[int(np.argmin(panel.y[tied]))], 1]),
    }


def measured_phase_gain(panel: wsd80.Panel, aggregate: float) -> dict[str, float]:
    on_fiber = np.isclose(panel.aggregate[:, 1], aggregate, atol=1e-6)
    assert on_fiber.sum() > 3, f"aggregate {aggregate} has only {on_fiber.sum()} coordinates; fiber match failed"
    tied = on_fiber & np.isclose(panel.contrast[:, 1], 0.0, atol=1e-9)
    assert tied.any(), f"no tied policy measured at aggregate {aggregate}"
    rows = np.flatnonzero(on_fiber)
    best = rows[int(np.argmin(panel.y[rows]))]
    tied_value = float(panel.y[np.flatnonzero(tied)[0]])
    return {
        "aggregate": aggregate,
        "tied_bpb": tied_value,
        "best_contrast": float(panel.contrast[best, 1]),
        "phase_gain": tied_value - float(panel.y[best]),
        "coordinates": int(on_fiber.sum()),
    }


def cross_validated(data: pooled.Dataset, name: str, config, seeds: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        out_of_fold = np.full(len(data.y), np.nan)
        for train, test in FOLD_BUILDER(data.weights, np.arange(len(data.y)), CV_SPLITS, seed):
            predict = fit_predictor(data, train, config)
            out_of_fold[test] = predict(data.weights[test])
        residual = out_of_fold - data.y
        rows.append(
            {
                "model": name,
                "seed": seed,
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "mae": float(np.mean(np.abs(residual))),
                # The surface is smooth and the scatter around it is measurement noise with occasional
                # far outliers, so squared error is set by the outliers rather than by how well the
                # response is recovered. The median absolute residual says how well a typical policy is
                # predicted, and the two can move in opposite directions.
                "median_absolute": float(np.median(np.abs(residual))),
                "spearman": float(spearmanr(out_of_fold, data.y).statistic),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid", type=int, default=OPTIMUM_GRID)
    parser.add_argument("--folds", choices=("random", "blocked"), default="blocked")
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Fit once and audit the raw response before paying for repeated outer cross-validation.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[name for name, _config in MODEL_CONFIGS],
        help="Run only the named models. By default every registered model is evaluated.",
    )
    args = parser.parse_args()
    global FOLD_BUILDER
    FOLD_BUILDER = random_folds if args.folds == "random" else mixture_blocked_folds
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = wsd80.load_surface()
    replicates = wsd80.load_fiber_replicates()
    sigma = wsd80.training_seed_sigma(replicates)
    data = as_pooled_dataset(panel)
    best_tied, best_overall = wsd80.tied_reference(panel)

    print("=" * 104)
    print(
        f"80/20 WSD StarCoder surface: {len(panel.y)} coordinates, training-seed sigma {sigma:.6f} BPB, "
        f"{args.folds} folds"
    )
    print("=" * 104)
    print(
        f"  measured best one-phase   p = {panel.phase_0[best_tied, 1]:.3f}"
        f"                    {panel.y[best_tied]:.6f}"
    )
    print(
        f"  measured best two-phase   p0 = {panel.phase_0[best_overall, 1]:.3f},"
        f" p1 = {panel.phase_1[best_overall, 1]:.3f}   {panel.y[best_overall]:.6f}"
        f"   aggregate {panel.aggregate[best_overall, 1]:.4f}"
    )
    print(
        f"  two-phase advantage       {panel.y[best_tied] - panel.y[best_overall]:+.6f} BPB"
        f"  ({(panel.y[best_tied] - panel.y[best_overall]) / sigma:.1f} sigma)\n"
    )

    truth = [measured_phase_gain(panel, aggregate) for aggregate in FIBER_AGGREGATES]
    print("  measured phase gain by aggregate:")
    for row in truth:
        print(
            f"    a = {row['aggregate']:.2f}  tied {row['tied_bpb']:.6f}"
            f"  best contrast {row['best_contrast']:+.4f}  gain {row['phase_gain']:+.6f}"
            f"  ({row['coordinates']} coordinates)"
        )
    print()

    selected_configs = MODEL_CONFIGS
    if args.models:
        selected = set(args.models)
        selected_configs = tuple((name, config) for name, config in MODEL_CONFIGS if name in selected)

    accuracy, diagnostics = [], []
    for name, config in selected_configs:
        if args.diagnostics_only:
            scores = pd.DataFrame(
                [
                    {
                        "model": name,
                        "seed": -1,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "median_absolute": np.nan,
                        "spearman": np.nan,
                    }
                ]
            )
        else:
            scores = cross_validated(data, name, config, CV_SEEDS)
        accuracy.append(scores)
        predict = fit_predictor(data, np.arange(len(panel.y)), config)
        print(f"  {name}: full-panel fit complete", flush=True)
        optimum = predicted_optimum(predict, args.grid)
        gains = [predicted_phase_gain(predict, aggregate, args.grid) for aggregate in FIBER_AGGREGATES]
        tied_fit = tied_diagonal_fit(predict, panel, args.grid)
        advantage = two_phase_advantage(predict, args.grid)
        diagnostics.append(
            {
                "model": name,
                "rmse": scores["rmse"].mean(),
                "rmse_sigma": scores["rmse"].mean() / sigma,
                "median_absolute_sigma": scores["median_absolute"].mean() / sigma,
                "spearman": scores["spearman"].mean(),
                "optimum_phase_0": optimum["phase_0"],
                "optimum_phase_1": optimum["phase_1"],
                "optimum_aggregate": optimum["aggregate"],
                "optimum_contrast": optimum["contrast"],
                **tied_fit,
                **advantage,
                **{f"phase_gain_at_{row['aggregate']:.2f}": row["phase_gain"] for row in gains},
                **{f"best_contrast_at_{row['aggregate']:.2f}": row["best_contrast"] for row in gains},
            }
        )

    table = pd.DataFrame(diagnostics)
    print("=" * 104)
    print("ACCURACY  (5-fold, 3 seeds; rmse in training-seed sigma)")
    print("=" * 104)
    print(
        table[["model", "rmse_sigma", "median_absolute_sigma", "spearman"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    print("\n" + "=" * 104)
    print("WHERE EACH MODEL PUTS THE OPTIMUM  (truth: p0 = 0.100, p1 = 0.500, aggregate 0.180)")
    print("=" * 104)
    print(
        table[["model", "optimum_phase_0", "optimum_phase_1", "optimum_aggregate", "optimum_contrast"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"
        )
    )

    print("\n" + "=" * 104)
    print("CAN THE MODEL REPRESENT A TWO-PHASE ADVANTAGE AT ALL?")
    print("=" * 104)
    print(
        f"  measured: best tied 0.945062, best two-phase 0.935468 at {TRUE_OPTIMUM}, "
        f"advantage {TRUE_TWO_PHASE_GAIN:+.6f} BPB"
    )
    print("  a model that is a function of one additive phase-weighted dose must predict exactly 0.000000")
    columns = ["model", "predicted_two_phase_gain", "optimum_phase_0", "optimum_phase_1", "optimum_distance"]
    print(table[columns].to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    print("\n" + "=" * 104)
    print("THE EASY HALF: DOES IT FIT THE ONE-PHASE RESPONSE?  (measured tied optimum p = 0.300)")
    print("=" * 104)
    tied_columns = ["model", "tied_rmse", "predicted_tied_optimum"]
    tied_table = table[tied_columns].copy()
    tied_table["tied_rmse_sigma"] = table["tied_rmse"] / sigma
    print(tied_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n" + "=" * 104)
    print("PHASE GAIN PROFILE g(a), PREDICTED AGAINST MEASURED")
    print("=" * 104)
    measured = {row["aggregate"]: row for row in truth}
    header = "".join(f"{a:>10.2f}" for a in FIBER_AGGREGATES)
    print(f"{'aggregate':<30}{header}")
    print(f"{'measured g(a)':<30}" + "".join(f"{measured[a]['phase_gain']:>10.4f}" for a in FIBER_AGGREGATES))
    print(
        f"{'measured best contrast':<30}" + "".join(f"{measured[a]['best_contrast']:>10.3f}" for a in FIBER_AGGREGATES)
    )
    print("-" * (30 + 10 * len(FIBER_AGGREGATES)))
    for _, row in table.iterrows():
        print(f"{row['model']:<30}" + "".join(f"{row[f'phase_gain_at_{a:.2f}']:>10.4f}" for a in FIBER_AGGREGATES))
        print(
            f"{'  its best contrast':<30}"
            + "".join(f"{row[f'best_contrast_at_{a:.2f}']:>10.3f}" for a in FIBER_AGGREGATES)
        )
    print("\nsign agreement of the best contrast against measurement, per model:")
    for _, row in table.iterrows():
        agree = sum(
            np.sign(row[f"best_contrast_at_{a:.2f}"]) == np.sign(measured[a]["best_contrast"])
            for a in FIBER_AGGREGATES
            if abs(measured[a]["best_contrast"]) > 1e-9
        )
        total = sum(1 for a in FIBER_AGGREGATES if abs(measured[a]["best_contrast"]) > 1e-9)
        print(f"  {row['model']:<32} {agree}/{total}")

    pd.concat(accuracy).to_csv(args.output_dir / f"cv_scores_{args.folds}.csv", index=False)
    table.to_csv(args.output_dir / f"diagnostics_{args.folds}.csv", index=False)
    pd.DataFrame(truth).to_csv(args.output_dir / "measured_phase_gain.csv", index=False)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
