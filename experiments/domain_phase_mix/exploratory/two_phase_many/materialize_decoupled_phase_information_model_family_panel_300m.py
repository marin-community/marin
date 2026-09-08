# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "kaleido==0.2.1",
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
"""Materialize fixed-aggregate phase-information paths across surrogate families."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_decoupled_phase_information_constraints_300m as decoupled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_table9_phase_split_dsp_300m as phase_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as coverage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_best_phase_model_validation_panel_300m as geometry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_original_style_matched_sepheads_ablation_300m as matched,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "reference_outputs" / "decoupled_phase_information_model_family_panel_20260712"
)
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_decoupled_phase_information_model_family_20260712/mixtures"
)
CONTROLLED_RESULTS = (
    Path(__file__).resolve().parent
    / "reference_outputs"
    / "delphi_phase_model_validation_results_20260711"
    / "observed_3e18_results.csv"
)
CONTROLLED_MIXTURES = (
    Path(__file__).resolve().parent / "reference_outputs" / "separate_heads_frontier_tied_panel_20260710" / "mixtures"
)
PHASE_INFORMATION_BUDGETS = (0.0, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2)
DSP_LINEAR_REG = 0.01
DSP_MAXITER = 40
DSP_COARSE_TOP_K = 3
GEOMETRY_MAXITER = 16
GEOMETRY_COARSE_TOP_K = 2
FEASIBILITY_TOLERANCE = 1e-7
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Anchor:
    tag: str
    objective: str
    label: str
    source_candidate: str


@dataclass(frozen=True)
class Predictor:
    family: str
    objective: str
    predict: Callable[[np.ndarray], float]
    fit_rmse: float
    fit_spearman: float


@dataclass(frozen=True)
class SolveResult:
    weights: np.ndarray
    prediction: float
    successful_starts: int


ANCHORS = (
    Anchor("unch05", "uncheatable", "Uncheatable aggregate KL coefficient 0.05", "origstyle_sep_unch_1p_kl0p05"),
    Anchor("t9s05", "table9", "Table-9 stable aggregate KL coefficient 0.05", "origstyle_sep_t9_1p_kl0p05"),
    Anchor("t9b075", "table9", "Table-9 observed-best aggregate KL coefficient 0.075", "origstyle_sep_t9_1p_kl0p075"),
)
FAMILY_TAGS = {
    "separate_heads": "sep",
    "canonical": "can",
    "effective_exposure": "eff",
    "effective_exposure_geometry": "geo",
}


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def epsilon_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def candidate_name(anchor: Anchor, family: str, phase_information_budget: float) -> str:
    if phase_information_budget == 0.0:
        return f"dphase_{anchor.tag}_tied"
    return f"dphase_{anchor.tag}_{FAMILY_TAGS[family]}_e{epsilon_tag(phase_information_budget)}"


def generic_optimize_fixed_aggregate(
    predict: Callable[[np.ndarray], float],
    aggregate: np.ndarray,
    phase_information_budget: float,
) -> SolveResult:
    alpha0, alpha1 = matched.PHASE_FRACTIONS
    tied = np.stack([aggregate, aggregate])
    if phase_information_budget == 0.0:
        return SolveResult(tied, predict(tied), 1)

    lower = -aggregate / alpha1 + 1e-10
    upper = aggregate / alpha0 - 1e-10

    def weights_from_delta(delta: np.ndarray) -> np.ndarray:
        return decoupled.fixed_aggregate.weights_from_delta(aggregate, delta, alpha0, alpha1)

    def phase_information(delta: np.ndarray) -> float:
        weights = weights_from_delta(delta)
        return decoupled.fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1)

    rng = np.random.default_rng(0)
    starts = [np.zeros_like(aggregate)]
    starts.extend(
        decoupled.feasible_start(
            decoupled.fixed_aggregate.random_start(aggregate, lower, upper, rng),
            aggregate,
            lower,
            upper,
            phase_information_budget,
        )
        for _index in range(8)
    )
    constraints = [
        {"type": "eq", "fun": lambda delta: float(np.sum(delta))},
        {"type": "ineq", "fun": lambda delta: phase_information_budget - phase_information(delta)},
    ]
    bounds = list(zip(lower, upper, strict=True))
    best_prediction = np.inf
    best_weights: np.ndarray | None = None
    successful_starts = 0
    for start in starts:
        result = minimize(
            lambda delta: predict(weights_from_delta(np.asarray(delta, dtype=float))),
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-11},
        )
        if result.success:
            successful_starts += 1
        weights = weights_from_delta(np.asarray(result.x, dtype=float))
        information = decoupled.fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1)
        prediction = predict(weights)
        if (
            np.isfinite(prediction)
            and information <= phase_information_budget + FEASIBILITY_TOLERANCE
            and float(weights.min()) >= -FEASIBILITY_TOLERANCE
            and prediction < best_prediction
        ):
            best_prediction = prediction
            best_weights = weights
    if best_weights is None:
        raise RuntimeError(f"No feasible solve at epsilon_phase={phase_information_budget:g}")
    return SolveResult(best_weights, best_prediction, successful_starts)


def predictor_metrics(predict: Callable[[np.ndarray], float], dataset: matched.pooled.Dataset) -> tuple[float, float]:
    prediction = np.asarray([predict(weights) for weights in dataset.weights], dtype=float)
    rmse = float(np.sqrt(np.mean((prediction - dataset.y) ** 2)))
    rank = float(spearmanr(dataset.y, prediction).statistic)
    return rmse, rank


def fit_predictors(
    datasets: dict[str, matched.pooled.Dataset],
    separate_models: dict[str, matched.SeparateHeadsModel],
) -> dict[str, dict[str, Predictor]]:
    output: dict[str, dict[str, Predictor]] = {}
    for objective, dataset in datasets.items():
        packet = coverage.packet(dataset, np.arange(dataset.n))
        alpha0, alpha1 = coverage.phase_fractions(dataset)

        separate_model = separate_models[objective]

        def separate_predict(
            weights: np.ndarray,
            *,
            model: matched.SeparateHeadsModel = separate_model,
            fit_dataset: matched.pooled.Dataset = dataset,
        ) -> float:
            return float(matched.predict_model(model, fit_dataset, weights[None, :, :])[0])

        canonical_model, _canonical_tuning = phase_dsp.fit_variant_with_l2(
            packet,
            "canonical",
            DSP_LINEAR_REG,
            maxiter=DSP_MAXITER,
            coarse_top_k=DSP_COARSE_TOP_K,
            basin_hopping_iters=0,
        )
        effective_model, _effective_tuning = phase_dsp.fit_variant_with_l2(
            packet,
            "effective_exposure",
            DSP_LINEAR_REG,
            maxiter=DSP_MAXITER,
            coarse_top_k=DSP_COARSE_TOP_K,
            basin_hopping_iters=0,
        )
        geometry_model = coverage.fit_model(
            dataset,
            np.arange(dataset.n),
            geometry.MODEL_CONFIG,
            linear_reg=coverage.dataset_linear_reg(dataset),
            maxiter=GEOMETRY_MAXITER,
            coarse_top_k=GEOMETRY_COARSE_TOP_K,
        )

        def canonical_predict(weights: np.ndarray, *, model: dsp.FittedDSPModel = canonical_model) -> float:
            return float(dsp.predict(model, weights[None, :, :])[0])

        def effective_predict(weights: np.ndarray, *, model: dsp.FittedDSPModel = effective_model) -> float:
            return float(dsp.predict(model, weights[None, :, :])[0])

        def geometry_predict(
            weights: np.ndarray,
            *,
            model: coverage.CoverageModel = geometry_model,
            phase0_fraction: float = alpha0,
            phase1_fraction: float = alpha1,
        ) -> float:
            return float(coverage.predict(model, weights[None, :, :], phase0_fraction, phase1_fraction)[0])

        family_functions = {
            "separate_heads": separate_predict,
            "canonical": canonical_predict,
            "effective_exposure": effective_predict,
            "effective_exposure_geometry": geometry_predict,
        }
        output[objective] = {}
        for family, function in family_functions.items():
            rmse, rank = predictor_metrics(function, dataset)
            output[objective][family] = Predictor(family, objective, function, rmse, rank)
    return output


def controlled_phase_gain_rows(
    predictors: dict[str, dict[str, Predictor]],
    datasets: dict[str, matched.pooled.Dataset],
) -> pd.DataFrame:
    observed = pd.read_csv(CONTROLLED_RESULTS)
    rows: list[dict[str, object]] = []
    for objective, tag in (("uncheatable", "unch"), ("table9", "t9")):
        dataset = datasets[objective]
        two_weights = decoupled.weights_from_frame(CONTROLLED_MIXTURES / f"sepfront_{tag}_2p.csv", dataset.domain_names)
        tied_weights = decoupled.weights_from_frame(
            CONTROLLED_MIXTURES / f"sepfront_{tag}_tied.csv",
            dataset.domain_names,
        )
        target_column = "uncheatable_bpb" if objective == "uncheatable" else "table9_macro_bpb"
        two_rows = observed.loc[observed["base"].str.startswith(f"sepfront_{tag}_2p")].set_index("data_seed")
        tied_rows = observed.loc[observed["base"].str.startswith(f"sepfront_{tag}_tied")].set_index("data_seed")
        common_seeds = sorted(set(two_rows.index).intersection(tied_rows.index))
        observed_gains = tied_rows.loc[common_seeds, target_column] - two_rows.loc[common_seeds, target_column]
        for predictor in predictors[objective].values():
            predicted_gain = predictor.predict(tied_weights) - predictor.predict(two_weights)
            rows.append(
                {
                    "objective": objective,
                    "family": predictor.family,
                    "fit_rmse": predictor.fit_rmse,
                    "fit_spearman": predictor.fit_spearman,
                    "observed_gain_mean": float(observed_gains.mean()),
                    "observed_gain_sd": float(observed_gains.std(ddof=1)),
                    "observed_gain_min": float(observed_gains.min()),
                    "observed_gain_max": float(observed_gains.max()),
                    "predicted_gain": predicted_gain,
                    "gain_error": predicted_gain - float(observed_gains.mean()),
                    "correct_sign": bool(predicted_gain > 0.0),
                }
            )
    return pd.DataFrame(rows)


def write_candidate(
    output_dir: Path,
    gcs_output_dir: str,
    candidate: str,
    frame: pd.DataFrame,
    *,
    upload: bool,
) -> None:
    path = output_dir / "mixtures" / f"{candidate}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if upload:
        with fsspec.open(f"{gcs_output_dir.rstrip('/')}/{candidate}.csv", "wt") as handle:
            frame.to_csv(handle, index=False)


def materialize_candidates(
    output_dir: Path,
    gcs_output_dir: str,
    phase_information_budgets: tuple[float, ...],
    requested_families: tuple[str, ...],
    datasets: dict[str, matched.pooled.Dataset],
    predictors: dict[str, dict[str, Predictor]],
    natural: dict[str, np.ndarray],
    token_counts: dict[str, np.ndarray],
    target_budgets: dict[str, int],
    *,
    upload: bool,
) -> pd.DataFrame:
    unknown = sorted(set(requested_families).difference(FAMILY_TAGS))
    if unknown:
        raise ValueError(f"Unknown model families: {unknown}")
    rows: list[dict[str, object]] = []
    emitted: set[str] = set()
    for anchor in ANCHORS:
        dataset = datasets[anchor.objective]
        anchor_weights = decoupled.weights_from_frame(
            decoupled.PANEL_DIR / "mixtures" / f"{anchor.source_candidate}.csv",
            dataset.domain_names,
        )
        if not np.allclose(anchor_weights[0], anchor_weights[1], atol=1e-10):
            raise ValueError(f"Anchor {anchor.source_candidate} is not tied")
        aggregate = anchor_weights[0]
        for family in requested_families:
            predictor = predictors[anchor.objective][family]
            tied_prediction = predictor.predict(np.stack([aggregate, aggregate]))
            for phase_information_budget in phase_information_budgets:
                candidate = candidate_name(anchor, family, phase_information_budget)
                if candidate in emitted:
                    continue
                result = generic_optimize_fixed_aggregate(
                    predictor.predict,
                    aggregate,
                    phase_information_budget,
                )
                emitted.add(candidate)
                geometry_values = decoupled.policy_geometry(result.weights, natural[anchor.objective])
                aggregate_check = (
                    matched.PHASE_FRACTIONS[0] * result.weights[0] + matched.PHASE_FRACTIONS[1] * result.weights[1]
                )
                max_aggregate_error = float(np.max(np.abs(aggregate_check - aggregate)))
                epochs = matched.olmix.simulated_epochs(
                    result.weights,
                    token_counts[anchor.objective],
                    target_budget=target_budgets[anchor.objective],
                )
                nearest_tv, nearest_bpb = decoupled.fixed_aggregate.nearest_observed_tv(dataset, result.weights)
                frame = matched.per_component.mixture_frame(
                    domains=dataset.domain_names,
                    natural=natural[anchor.objective],
                    weights=result.weights,
                    token_counts=token_counts[anchor.objective],
                    target_budget=target_budgets[anchor.objective],
                )
                write_candidate(
                    output_dir,
                    gcs_output_dir,
                    candidate,
                    frame,
                    upload=upload,
                )
                rows.append(
                    {
                        "candidate": candidate,
                        "objective": anchor.objective,
                        "anchor_tag": anchor.tag,
                        "anchor_label": anchor.label,
                        "anchor_candidate": anchor.source_candidate,
                        "family": "control" if phase_information_budget == 0.0 else family,
                        "phase_information_budget": phase_information_budget,
                        "phase_information": geometry_values["phase_information"],
                        "phase_tv": geometry_values["phase_tv"],
                        "aggregate_kl": geometry_values["aggregate_kl"],
                        "aggregate_tv": geometry_values["aggregate_tv"],
                        "predicted_bpb": result.prediction,
                        "tied_prediction": tied_prediction,
                        "predicted_gain_vs_tied": tied_prediction - result.prediction,
                        "max_weight": float(result.weights.max()),
                        "max_simulated_epoch": float(epochs.max()),
                        "q95_simulated_epoch": float(np.quantile(epochs, 0.95)),
                        "nearest_observed_tv": nearest_tv,
                        "nearest_observed_bpb": nearest_bpb,
                        "successful_starts": result.successful_starts,
                        "max_aggregate_error": max_aggregate_error,
                    }
                )
    return pd.DataFrame(rows).sort_values(["objective", "anchor_tag", "family", "phase_information_budget"])


def write_report(manifest: pd.DataFrame, retrodiction: pd.DataFrame, output_dir: Path) -> None:
    summary = (
        manifest.loc[manifest["phase_information_budget"].gt(0)]
        .groupby(["objective", "anchor_tag", "family"], as_index=False)
        .agg(
            max_predicted_gain=("predicted_gain_vs_tied", "max"),
            max_phase_information=("phase_information", "max"),
            max_phase_tv=("phase_tv", "max"),
            max_weight=("max_weight", "max"),
            max_nearest_observed_tv=("nearest_observed_tv", "max"),
            successful_starts_min=("successful_starts", "min"),
        )
    )
    lines = [
        "# Decoupled phase-information model-family panel",
        "",
        "All model families use identical fixed aggregate anchors. Only phase ordering changes with "
        "the explicit phase-information budget.",
        "",
        "## Controlled prior-pair retrodiction",
        "",
        retrodiction.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Candidate-path summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation boundary",
        "",
        "The controlled retrodiction pair has three matched data seeds and is the strongest local "
        "transfer check. Candidate paths remain continuous surrogate optima; only 3e18 training can "
        "establish a new frontier.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def write_plots(manifest: pd.DataFrame, output_dir: Path) -> None:
    nonzero = manifest.loc[manifest["phase_information_budget"].gt(0)].copy()
    for metric in ("predicted_gain_vs_tied", "phase_tv", "max_weight"):
        figure = px.line(
            nonzero,
            x="phase_information_budget",
            y=metric,
            color="family",
            facet_col="anchor_tag",
            markers=True,
            color_discrete_sequence=px.colors.sample_colorscale("RdYlGn_r", [0.05, 0.35, 0.65, 0.95]),
            title=f"Decoupled phase-information family comparison: {metric}",
        )
        figure.write_html(
            output_dir / f"{metric}.html",
            include_plotlyjs=True,
            config=PLOT_CONFIG,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument(
        "--phase-information-budgets",
        default=",".join(str(value) for value in PHASE_INFORMATION_BUDGETS),
    )
    parser.add_argument("--families", default=",".join(FAMILY_TAGS))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    budgets = parse_float_tuple(args.phase_information_budgets)
    families = parse_str_tuple(args.families)
    datasets, separate_models, natural, token_counts, target_budgets = decoupled.load_context()
    predictors = fit_predictors(datasets, separate_models)
    retrodiction = controlled_phase_gain_rows(predictors, datasets)
    manifest = materialize_candidates(
        args.output_dir,
        args.gcs_output_dir,
        budgets,
        families,
        datasets,
        predictors,
        natural,
        token_counts,
        target_budgets,
        upload=args.upload,
    )
    if manifest["candidate"].duplicated().any():
        raise AssertionError("Candidate names must be unique")
    if float(manifest["max_aggregate_error"].max()) > 1e-9:
        raise AssertionError("Fixed aggregate changed")
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    retrodiction.to_csv(args.output_dir / "controlled_pair_retrodiction.csv", index=False)
    (args.output_dir / "panel_config.json").write_text(
        json.dumps(
            {
                "phase_information_budgets": budgets,
                "families": families,
                "gcs_output_dir": args.gcs_output_dir,
                "candidate_count": len(manifest),
                "upload": args.upload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_report(manifest, retrodiction, args.output_dir)
    write_plots(manifest, args.output_dir)
    print(retrodiction.to_string(index=False))
    print(manifest.to_string(index=False))
    print(f"Wrote {len(manifest)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
