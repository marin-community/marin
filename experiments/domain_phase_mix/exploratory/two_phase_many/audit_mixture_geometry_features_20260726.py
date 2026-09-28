# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Do global mixture-geometry features fix the surrogate's optimum drift?

The hypothesis under test is that the existing designs price exposure only
bucket-by-bucket and family-by-family, so nothing in them is sensitive to
*breadth* or to *peak epoch load* as properties of the whole mixture, and their
constrained optima therefore drift to the wrong concentration.

Three arms.

``measure``
    What the best-observed policies actually look like in geometry space, on the
    60M and 300M panels only. Reported before any model is fitted, because the
    stated characterization of that region is itself a claim to be checked.

``score``
    Five augmentations of ``hierarchical_phase_replay``, each adding global
    geometry columns and nothing else, scored by ``dual_objective_harness`` on
    both targets. Every augmentation strictly nests the incumbent: nonnegative
    least squares can zero the new columns, so an in-sample loss is impossible
    and any out-of-fold loss is a variance cost.

``optimum``
    Where the KL-regularized optimum path lands in the same geometry space, for
    the incumbent and each augmentation, at 300M. This is the mechanism check: a
    feature that improves the metrics without moving the optimum is not fixing
    the drift, and one that moves the optimum without improving the metrics is
    just a prior.

``bootstrap``
    Paired resampling of the fit panel stratified by proposal series, with the
    censored evaluation set held fixed and draws shared across variants, so the
    reported quantity is a per-draw difference rather than two marginal
    intervals.

The 3e18 panels are never loaded and the sealed ``targeted_pairwise`` panel is
never read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dual_objective_harness_20260726 import (  # noqa: E402
    CENSOR_FRACTIONS,
    Benchmark,
    Fitted,
    build_benchmark,
    fit_metrics,
    fit_on,
    out_of_fold_predictions,
    score_candidate,
    select_by,
    summarize,
)
from proposal_metrics_20260726 import phase_decision_skill  # noqa: E402
from swarm39_harness_20260725 import (  # noqa: E402
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    Design,
    Model,
    Panel,
    load_scale,
)
from swarm39_models_20260725 import (  # noqa: E402
    EPSILON,
    _state_shapes,
    build_hierarchical_phase_replay,
    reference_mixtures,
)

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "mixture_geometry_features_20260726"
TARGETS = (UNCHEATABLE, TABLE9)
CRITERION = "rmse"

# Deviation targets for the two-sided geometry columns. Effective buckets are
# counted out of 39; the token-proportional policy scores 14.8. The epoch targets
# are in log space and bracket both the value claimed for the best-observed region
# (12 epochs, log 2.48) and the value measured here.
BREADTH_TARGETS = (16.0, 20.0, 24.0)
LOG_PEAK_TARGETS = (2.5, 3.7, 4.3)
SOFT_PEAK_BETA = 0.25

# Constrained optimum path, same penalty as the deployment-regularization audit:
# alpha KL(p0||proportional) + (1 - alpha) KL(p1||proportional).
LAMBDA_GRID = (0.0, 0.03, 0.1, 0.3, 1.0)
RESTARTS = 4
OPTIMIZER_SEED = 20260726

BOOTSTRAP_DRAWS = 120
BOOTSTRAP_SEED = 20260726
BEST_DECILE = 0.10
WORKERS = 12


# ---------------------------------------------------------------------------
# Global geometry features
# ---------------------------------------------------------------------------


def _herfindahl(panel: Panel) -> np.ndarray:
    """Aggregate concentration; 1/this is the effective bucket count."""
    return (panel.aggregate**2).sum(axis=1, keepdims=True)


def _effective_buckets(panel: Panel) -> np.ndarray:
    return 1.0 / np.maximum(_herfindahl(panel), EPSILON)


def _entropy(panel: Panel) -> np.ndarray:
    safe = np.clip(panel.aggregate, EPSILON, None)
    return -(safe * np.log(safe)).sum(axis=1, keepdims=True)


def _kl_to_proportional(panel: Panel) -> np.ndarray:
    safe = np.clip(panel.aggregate, EPSILON, None)
    return (safe * np.log(safe / panel.proportional[None, :])).sum(axis=1, keepdims=True)


def _peak_epochs(panel: Panel) -> np.ndarray:
    return panel.epochs.max(axis=1, keepdims=True)


def _log_peak_epochs(panel: Panel) -> np.ndarray:
    return np.log1p(_peak_epochs(panel))


def _soft_peak_epochs(panel: Panel, beta: float = SOFT_PEAK_BETA) -> np.ndarray:
    """Log-sum-exp of per-bucket epochs; a smooth upper envelope of the peak."""
    epochs = panel.epochs
    shift = epochs.max(axis=1, keepdims=True)
    return shift + np.log(np.exp(beta * (epochs - shift)).sum(axis=1, keepdims=True)) / beta


def _contrast_participation(panel: Panel) -> np.ndarray:
    """(sum|d|)^2 / sum(d^2): how many buckets the phase contrast spreads over."""
    absolute = np.abs(panel.contrast)
    numerator = absolute.sum(axis=1, keepdims=True) ** 2
    denominator = (absolute**2).sum(axis=1, keepdims=True)
    return np.where(denominator > EPSILON, numerator / np.maximum(denominator, EPSILON), 0.0)


def _two_sided(values: np.ndarray, target: float) -> tuple[np.ndarray, np.ndarray]:
    """Squared shortfall and squared excess relative to a target, both nonnegative."""
    return np.maximum(target - values, 0.0) ** 2, np.maximum(values - target, 0.0) ** 2


def _augment(base: Design, blocks: list[np.ndarray], names: list[str]) -> Design:
    return Design(matrix=np.hstack([base.matrix, *blocks]), names=tuple([*base.names, *names]))


def build_hpr_breadth(panel: Panel, shape: dict) -> Design:
    """Incumbent plus one global concentration harm column."""
    return _augment(build_hierarchical_phase_replay(panel, shape), [_herfindahl(panel)], ["aggregate_herfindahl"])


def build_hpr_peak(panel: Panel, shape: dict) -> Design:
    """Incumbent plus global peak epoch load, hard and soft."""
    base = build_hierarchical_phase_replay(panel, shape)
    return _augment(
        base,
        [_log_peak_epochs(panel), _soft_peak_epochs(panel)],
        ["log_peak_epochs", "softmax_epoch_load"],
    )


def build_hpr_geometry(panel: Panel, shape: dict) -> Design:
    """Incumbent plus breadth, peak load, divergence, and contrast breadth.

    Divergence and contrast breadth enter twice with opposite signs, because the
    two targets disagree about which direction helps and nonnegative least squares
    cannot flip a coefficient.
    """
    base = build_hierarchical_phase_replay(panel, shape)
    kl = _kl_to_proportional(panel)
    participation = _contrast_participation(panel)
    return _augment(
        base,
        [
            _herfindahl(panel),
            _log_peak_epochs(panel),
            _soft_peak_epochs(panel),
            -_entropy(panel),
            kl,
            -kl,
            participation,
            -participation,
        ],
        [
            "aggregate_herfindahl",
            "log_peak_epochs",
            "softmax_epoch_load",
            "entropy_benefit",
            "kl_proportional_harm",
            "kl_proportional_benefit",
            "contrast_participation_harm",
            "contrast_participation_benefit",
        ],
    )


def build_hpr_geometry_target(panel: Panel, shape: dict) -> Design:
    """Incumbent plus two-sided deviation from a target breadth and peak load.

    The two targets are shape hyperparameters chosen by the same out-of-fold
    criterion as every other hyperparameter, so nothing about the censored rows
    reaches them. This is the only form in which a preferred operating point can
    be expressed without a coefficient sign flip.
    """
    base = build_hierarchical_phase_replay(panel, shape)
    breadth_short, breadth_over = _two_sided(_effective_buckets(panel), float(shape["breadth_target"]))
    peak_short, peak_over = _two_sided(_log_peak_epochs(panel), float(shape["log_peak_target"]))
    return _augment(
        base,
        [breadth_short, breadth_over, peak_short, peak_over],
        ["breadth_shortfall", "breadth_excess", "peak_shortfall", "peak_excess"],
    )


def build_hpr_kl_references(panel: Panel, shape: dict) -> Design:
    """Incumbent plus KL from the aggregate to four reference mixtures.

    Uniform is a poor concentration reference when bucket corpora differ by orders
    of magnitude, so the same block the crs_plus geometry extension uses is
    transplanted here: proportional, two unimax caps, and uniform.
    """
    base = build_hierarchical_phase_replay(panel, shape)
    references = reference_mixtures(panel)
    safe = np.clip(panel.aggregate, EPSILON, None)
    columns = [
        (safe * np.log(safe / np.clip(reference, EPSILON, None)[None, :])).sum(axis=1, keepdims=True)
        for reference in references.values()
    ]
    return _augment(base, columns, [f"kl_to:{name}" for name in references])


def build_hpr_peak_shortfall(panel: Panel, shape: dict) -> Design:
    """Incumbent plus the shortfall half of the two-sided peak-load column only.

    Splitting the two-sided column is what separates the two mechanisms inside
    ``hpr_geometry_target``. The excess half prices peak epoch load above the
    target and is a genuine geometry term; the shortfall half correlates 0.97 with
    the ``domain_deletion`` series indicator, because every deletion policy sits at
    about one epoch on its largest bucket, so it acts as a series-level level
    correction rather than as mixture geometry.
    """
    base = build_hierarchical_phase_replay(panel, shape)
    shortfall, _ = _two_sided(_log_peak_epochs(panel), float(shape["log_peak_target"]))
    return _augment(base, [shortfall], ["peak_shortfall"])


def geometry_target_shapes() -> Iterator[dict]:
    for shape in _state_shapes(True):
        for breadth in BREADTH_TARGETS:
            for log_peak in LOG_PEAK_TARGETS:
                yield {**shape, "breadth_target": breadth, "log_peak_target": log_peak}


def peak_target_shapes() -> Iterator[dict]:
    for shape in _state_shapes(True):
        for log_peak in LOG_PEAK_TARGETS:
            yield {**shape, "log_peak_target": log_peak}


def incumbent() -> Model:
    return Model("hierarchical_phase_replay", build_hierarchical_phase_replay, lambda: _state_shapes(True))


_WORKER: dict[str, object] = {}


def _worker_state() -> tuple[Benchmark, dict[str, Model]]:
    """Per-process benchmark and model catalogue.

    Models carry closures and cannot be pickled, so worker tasks are addressed by
    model name and the catalogue is rebuilt once per process.
    """
    if "benchmark" not in _WORKER:
        _WORKER["benchmark"] = build_benchmark()
        _WORKER["catalogue"] = {model.name: model for model in [incumbent(), *augmentations()]}
    return _WORKER["benchmark"], _WORKER["catalogue"]  # type: ignore[return-value]


def augmentations() -> list[Model]:
    return [
        Model("hpr_breadth", build_hpr_breadth, lambda: _state_shapes(True)),
        Model("hpr_peak", build_hpr_peak, lambda: _state_shapes(True)),
        Model("hpr_geometry", build_hpr_geometry, lambda: _state_shapes(True)),
        Model("hpr_geometry_target", build_hpr_geometry_target, geometry_target_shapes),
        Model("hpr_kl_references", build_hpr_kl_references, lambda: _state_shapes(True)),
        Model("hpr_peak_shortfall", build_hpr_peak_shortfall, peak_target_shapes),
    ]


# ---------------------------------------------------------------------------
# Arm 1: what the best-observed region looks like
# ---------------------------------------------------------------------------


def geometry_frame(panel: Panel) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "effective_buckets": _effective_buckets(panel)[:, 0],
            "max_simulated_epochs": _peak_epochs(panel)[:, 0],
            "softmax_epoch_load": _soft_peak_epochs(panel)[:, 0],
            "kl_to_proportional": _kl_to_proportional(panel)[:, 0],
            "aggregate_entropy": _entropy(panel)[:, 0],
            "contrast_participation": _contrast_participation(panel)[:, 0],
            "phase_tv": panel.phase_tv,
            "buckets_over_one_epoch": (panel.epochs > 1.0).sum(axis=1).astype(float),
        }
    )


def measure(output: Path) -> None:
    rows = []
    for scale in ("60m", "300m"):
        fit, heldout = load_scale(scale)
        for split, panel in (("fit", fit), ("heldout", heldout)):
            frame = geometry_frame(panel)
            for target in TARGETS:
                observed = panel.targets[target]
                available = np.isfinite(observed)
                if not available.any():
                    continue
                count = max(1, round(BEST_DECILE * available.sum()))
                order = np.argsort(np.where(available, observed, np.inf))
                best = np.zeros(len(observed), dtype=bool)
                best[order[:count]] = True
                for group, mask in (("best_decile", best), ("rest", available & ~best), ("all", available)):
                    for column in frame.columns:
                        values = frame[column].to_numpy()[mask]
                        rows.append(
                            {
                                "scale": scale,
                                "split": split,
                                "target": target,
                                "group": group,
                                "n": int(mask.sum()),
                                "feature": column,
                                "p10": float(np.quantile(values, 0.10)),
                                "median": float(np.median(values)),
                                "p90": float(np.quantile(values, 0.90)),
                                "mean": float(values.mean()),
                            }
                        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "best_region_geometry.csv", index=False)
    pd.set_option("display.width", 220)
    for target in TARGETS:
        block = frame[(frame["scale"] == "300m") & (frame["split"] == "fit") & (frame["target"] == target)]
        print(f"\n=== 300M fit panel, {target}: best decile versus rest ===")
        pivot = block.pivot_table(index="feature", columns="group", values=["p10", "median", "p90"])
        print(pivot.to_string())


# ---------------------------------------------------------------------------
# Arm 2: dual-objective scoring
# ---------------------------------------------------------------------------


def score(benchmark: Benchmark, output: Path, models: list[Model]) -> pd.DataFrame:
    rows = []
    for model in models:
        for target in TARGETS:
            result = score_candidate(benchmark, model, target, CRITERION)
            rows.append(
                {
                    "model": model.name,
                    "target": target,
                    "n_columns": model.build(benchmark.fit_300m, result["shape_300m"]).matrix.shape[1],
                    "n_shapes": result["selection"]["evaluated"],
                    "l2_300m": result["l2_300m"],
                    **summarize(result),
                    "cens20_bias": result["censored"]["0.20"]["bias"],
                    "cens20_spearman": result["censored"]["0.20"]["spearman"],
                    "phase_delta_correlation": result["phase"]["delta_correlation"],
                }
            )
            print(f"  scored {model.name} / {target}")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "dual_objective_scores.csv", index=False)
    return frame


# ---------------------------------------------------------------------------
# Arm 3: constrained optimum path at 300M
# ---------------------------------------------------------------------------


def _simplex(vector: np.ndarray) -> np.ndarray:
    weights = np.exp(vector - vector.max())
    return weights / weights.sum()


def _single_row(reference: Panel, phase0: np.ndarray, phase1: np.ndarray) -> Panel:
    return Panel(
        scale=reference.scale,
        split="candidate",
        alpha=reference.alpha,
        buckets=reference.buckets,
        c0=reference.c0,
        c1=reference.c1,
        family_index=reference.family_index,
        family_names=reference.family_names,
        phase0=phase0.reshape(1, -1),
        phase1=phase1.reshape(1, -1),
        targets={t: np.array([np.nan]) for t in TARGETS},
        series=np.array(["candidate"]),
        policy_class=np.array(["two_phase"]),
        group=np.array(["candidate"]),
        row_id=np.array(["candidate"]),
    )


def _kl_penalty(phase0: np.ndarray, phase1: np.ndarray, prior: np.ndarray, alpha: float) -> float:
    def kl(weights: np.ndarray) -> float:
        safe = np.clip(weights, EPSILON, None)
        return float((safe * np.log(safe / prior)).sum())

    return alpha * kl(phase0) + (1.0 - alpha) * kl(phase1)


def _optimum_task(task: tuple[str, str]) -> list[dict]:
    """One model-target optimum path, run inside a worker process."""
    name, target = task
    benchmark, catalogue = _worker_state()
    model = catalogue[name]
    panel = benchmark.fit_300m
    prior = panel.proportional
    n = len(panel.buckets)
    log_prior = np.log(prior)
    rng = np.random.default_rng([OPTIMIZER_SEED, TARGETS.index(target)])
    shape, l2, _ = select_by(panel, model, target, CRITERION)
    fitted = fit_on(panel, model, target, shape, l2)
    rows = []
    for lam in LAMBDA_GRID:

        def objective(z: np.ndarray, fitted: Fitted = fitted, lam: float = lam) -> float:
            phase0, phase1 = _simplex(z[:n]), _simplex(z[n:])
            candidate = _single_row(panel, phase0, phase1)
            return float(fitted.predict(candidate)[0]) + lam * _kl_penalty(phase0, phase1, prior, panel.alpha)

        best = None
        for restart in range(RESTARTS):
            start = np.concatenate([log_prior, log_prior]) if restart == 0 else rng.normal(0.0, 1.5, 2 * n)
            result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 400})
            if best is None or result.fun < best.fun:
                best = result
        assert best is not None
        phase0, phase1 = _simplex(best.x[:n]), _simplex(best.x[n:])
        candidate = _single_row(panel, phase0, phase1)
        rows.append(
            {
                "model": name,
                "target": target,
                "lambda": lam,
                "predicted_bpb": float(fitted.predict(candidate)[0]),
                "effective_buckets": float(_effective_buckets(candidate)[0, 0]),
                "max_simulated_epochs": float(_peak_epochs(candidate)[0, 0]),
                "kl_to_proportional": float(_kl_to_proportional(candidate)[0, 0]),
                "phase_tv": float(candidate.phase_tv[0]),
                "max_bucket_weight": float(candidate.aggregate.max()),
                "aggregate_tv_to_nearest_fit_row": float(
                    0.5 * np.abs(panel.aggregate - candidate.aggregate).sum(axis=1).min()
                ),
            }
        )
    return rows


def optimum_path(output: Path, names: list[str]) -> pd.DataFrame:
    tasks = [(name, target) for name in names for target in TARGETS]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for result in pool.map(_optimum_task, tasks):
            rows.extend(result)
            print(f"  optimum path done for {result[0]['model']} / {result[0]['target']}")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "optimum_path.csv", index=False)
    return frame


# ---------------------------------------------------------------------------
# Arm 4: paired bootstrap over fit-panel rows
# ---------------------------------------------------------------------------


def _take(panel: Panel, index: np.ndarray) -> Panel:
    return Panel(
        scale=panel.scale,
        split=panel.split,
        alpha=panel.alpha,
        buckets=panel.buckets,
        c0=panel.c0,
        c1=panel.c1,
        family_index=panel.family_index,
        family_names=panel.family_names,
        phase0=panel.phase0[index],
        phase1=panel.phase1[index],
        targets={k: v[index] for k, v in panel.targets.items()},
        series=panel.series[index],
        policy_class=panel.policy_class[index],
        group=panel.group[index],
        row_id=panel.row_id[index],
    )


def _stratified_draw(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample row positions with replacement inside each proposal series."""
    parts = []
    for value in np.unique(series):
        members = np.flatnonzero(series == value)
        parts.append(rng.choice(members, size=len(members), replace=True))
    return np.sort(np.concatenate(parts))


def _censored_mask(panel: Panel, target: str, fraction: float) -> np.ndarray:
    observed = panel.targets[target]
    available = np.isfinite(observed)
    count = max(1, int(fraction * available.sum()))
    order = np.argsort(np.where(available, observed, np.inf))
    mask = np.zeros(len(observed), dtype=bool)
    mask[order[:count]] = True
    return mask


def _bootstrap_draw(task: tuple[str, int, tuple[str, ...]]) -> list[dict]:
    """One shared resampling draw scored by every variant, inside a worker process."""
    target, draw, names = task
    benchmark, catalogue = _worker_state()
    panel_300m, panel_60m = benchmark.fit_300m, benchmark.fit_60m
    censored = _censored_mask(panel_300m, target, CENSOR_FRACTIONS[0])
    train_pool = np.flatnonzero(np.isfinite(panel_300m.targets[target]) & ~censored)
    held = panel_300m.subset(censored)
    truth = held.targets[target]

    rng = np.random.default_rng([BOOTSTRAP_SEED, TARGETS.index(target), draw])
    index_300m = train_pool[_stratified_draw(panel_300m.series[train_pool], rng)]
    usable_60m = np.flatnonzero(np.isfinite(panel_60m.targets[target]))
    index_60m = usable_60m[_stratified_draw(panel_60m.series[usable_60m], rng)]
    resampled_300m = _take(panel_300m, index_300m)
    resampled_60m = _take(panel_60m, index_60m)

    ranks = lambda values: np.argsort(np.argsort(values))  # noqa: E731
    rows = []
    for name in names:
        model = catalogue[name]
        shape, l2, _ = select_by(resampled_300m, model, target, CRITERION)
        oof = out_of_fold_predictions(resampled_300m, model, target, shape, l2)
        fit_quality = fit_metrics(resampled_300m.targets[target], oof)
        fitted = fit_on(resampled_300m, model, target, shape, l2)
        prediction = fitted.predict(held)
        residual = prediction - truth
        shape_60m, l2_60m, _ = select_by(resampled_60m, model, target, CRITERION)
        fitted_60m = fit_on(resampled_60m, model, target, shape_60m, l2_60m)
        delta = fitted_60m.predict(benchmark.paired_300m.two_phase_panel) - fitted_60m.predict(
            benchmark.paired_300m.tied_panel
        )
        skill = phase_decision_skill(delta, benchmark.paired_300m.observed_delta[target])
        rows.append(
            {
                "target": target,
                "draw": draw,
                "model": name,
                "oof_rmse": fit_quality["rmse"],
                "oof_spearman": fit_quality["spearman"],
                "cens_rmse": float(np.sqrt(np.mean(residual**2))),
                "cens_bias": float(np.mean(residual)),
                "cens_spearman": float(np.corrcoef(ranks(prediction), ranks(truth))[0, 1]),
                "phase_skill": skill["phase_skill_score"],
            }
        )
    return rows


def bootstrap(output: Path, names: list[str], draws: int) -> pd.DataFrame:
    """Shared-draw paired bootstrap; the censored evaluation rows never move."""
    tasks = [(target, draw, tuple(names)) for target in TARGETS for draw in range(draws)]
    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for result in pool.map(_bootstrap_draw, tasks):
            rows.extend(result)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(tasks)} draws complete")
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "bootstrap_draws.csv", index=False)
    return frame


def paired_differences(frame: pd.DataFrame, baseline: str) -> pd.DataFrame:
    metrics = ["oof_rmse", "oof_spearman", "cens_rmse", "cens_bias", "cens_spearman", "phase_skill"]
    # Lower is better for these; the rest are higher-is-better.
    lower_better = {"oof_rmse", "cens_rmse"}
    rows = []
    for target, block in frame.groupby("target"):
        wide = block.pivot(index="draw", columns="model", values=metrics)
        for model in sorted(set(block["model"]) - {baseline}):
            for metric in metrics:
                difference = (wide[(metric, model)] - wide[(metric, baseline)]).to_numpy()
                difference = difference[np.isfinite(difference)]
                if metric == "cens_bias":
                    better = np.abs(wide[(metric, model)].to_numpy()) < np.abs(wide[(metric, baseline)].to_numpy())
                elif metric in lower_better:
                    better = difference < 0
                else:
                    better = difference > 0
                rows.append(
                    {
                        "target": target,
                        "model": model,
                        "metric": metric,
                        "n_draws": len(difference),
                        "mean_difference": float(np.mean(difference)),
                        "ci_low": float(np.quantile(difference, 0.025)),
                        "ci_high": float(np.quantile(difference, 0.975)),
                        "fraction_better": float(np.mean(better[np.isfinite(better.astype(float))])),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--arms", nargs="*", default=["measure", "score", "optimum"])
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-models", nargs="*", default=None)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)

    if "measure" in args.arms:
        measure(output)

    if not ({"score", "optimum", "bootstrap"} & set(args.arms)):
        return

    benchmark = build_benchmark()
    print(json.dumps(benchmark.metadata, sort_keys=True))
    catalogue = {model.name: model for model in [incumbent(), *augmentations()]}

    if "score" in args.arms:
        frame = score(benchmark, output, list(catalogue.values()))
        for target in TARGETS:
            print(f"\n=== dual-objective scores, {target} ===")
            block = frame[frame["target"] == target].drop(columns=["target"])
            print(block.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    if "optimum" in args.arms:
        frame = optimum_path(output, list(catalogue))
        for target in TARGETS:
            print(f"\n=== constrained optimum path at 300M, {target} ===")
            block = frame[frame["target"] == target].drop(columns=["target"])
            print(block.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if "bootstrap" in args.arms:
        names = args.bootstrap_models or list(catalogue)
        frame = bootstrap(output, names, args.draws)
        differences = paired_differences(frame, incumbent().name)
        differences.to_csv(output / "bootstrap_paired_differences.csv", index=False)
        for target in TARGETS:
            print(f"\n=== paired bootstrap differences versus incumbent, {target} ===")
            block = differences[differences["target"] == target].drop(columns=["target"])
            print(block.to_string(index=False, float_format=lambda v: f"{v:+.6f}"))


if __name__ == "__main__":
    main()
