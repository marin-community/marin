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
"""Materialize raw and decoupled-regularized family-GRP optima at 300M.

The two compared two-phase models share retained exposure, bucket and family
power responses, and family replay penalties. They differ only in whether late
learning is represented by one scalar multiplier or independent early/late
nonnegative heads. Their honest single-phase ablation is therefore shared: one
aggregate response head with no forgetting or late multiplier.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_decoupled_phase_information_constraints_300m as phase_information,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_grp_domain_saturation_phase_heads_20260714 as phase_heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_grp_saturation_hierarchy_20260714 as hierarchy,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_per_component_dsp_kl_sweep_300m as per_component,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_decoupled_phase_information_model_family_panel_300m as decoupled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_original_style_matched_sepheads_ablation_300m as matched,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_two_phase_canonical_bowl_candidates_300m as objective_data,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/bucket_family_power_heads_validation_panel_20260714"
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_bucket_family_power_heads_validation_20260714/mixtures"
)
OBJECTIVES = ("uncheatable", "table9")
OBJECTIVE_ABBREVIATIONS = {"uncheatable": "unch", "table9": "t9"}
TWO_PHASE_MODELS = {
    "eta": phase_heads.VARIANT_BY_NAME["power_eta"],
    "separate_heads": phase_heads.VARIANT_BY_NAME["power_separate_heads"],
}
ONE_PHASE_VARIANT = phase_heads.VARIANT_BY_NAME["power_eta"]
NUM_SHAPES = 16
INNER_SPLITS = 5
INNER_SEED = 3142
AGGREGATE_KL_VALUES = (0.0, 0.05, 0.1)
REGULARIZED_AGGREGATE_KL_VALUES = (0.05,)
PHASE_INFORMATION_BUDGETS = (0.005, 0.02, 0.05)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FittedPowerModel:
    dataset: family_grp.Dataset
    variant: phase_heads.Variant
    selection: phase_heads.SharedSelection
    head: family_grp.FittedHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design, _names, _layout = phase_heads.build_design(
            candidate,
            self.variant,
            self.selection.shape,
            None,
        )
        return self.head.predict_design(design)


@dataclass(frozen=True)
class OptimizerDataset:
    """Minimal observed panel needed to seed the shared mixture optimizer."""

    weights: np.ndarray
    y: np.ndarray

    @property
    def m(self) -> int:
        return self.weights.shape[2]


def optimizer_dataset(dataset: family_grp.Dataset) -> OptimizerDataset:
    return OptimizerDataset(weights=dataset.weights, y=dataset.target)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def dataset_id(objective: str) -> phase_heads.DatasetId:
    if objective == "uncheatable":
        return phase_heads.DatasetId.THREE_HUNDRED_M_UNCHEATABLE
    if objective == "table9":
        return phase_heads.DatasetId.THREE_HUNDRED_M_TABLE9
    raise ValueError(f"Unknown objective {objective!r}")


def family_dataset(raw: pooled.Dataset) -> family_grp.Dataset:
    family_names, family_members = hierarchy.family_partition(raw)
    return family_grp.Dataset(
        frame=raw.frame,
        target=np.asarray(raw.y, dtype=float),
        weights=np.asarray(raw.weights, dtype=float),
        c0=np.asarray(raw.c0, dtype=float),
        c1=np.asarray(raw.c1, dtype=float),
        domains=tuple(raw.domain_names),
        family_names=family_names,
        family_members=family_members,
        quality=np.full(raw.m, -1, dtype=int),
    )


def load_datasets(objective: str) -> dict[str, family_grp.Dataset]:
    two_phase_raw = pooled.load_300m_dataset(objective)
    one_phase_raw = observatory.load_300m_single_phase_dataset(objective, two_phase_raw)
    if two_phase_raw.n != 280 or one_phase_raw.n != 280:
        raise ValueError(f"Expected 280 rows per policy, got 1p={one_phase_raw.n}, 2p={two_phase_raw.n}")
    return {"1p": family_dataset(one_phase_raw), "2p": family_dataset(two_phase_raw)}


def candidate_shapes(variant: phase_heads.Variant, policy: str) -> tuple[Any, ...]:
    shapes = phase_heads.candidate_shapes(variant, NUM_SHAPES)
    if policy == "1p":
        shapes = tuple(replace(shape, late_multiplier=1.0, forgetting_rate=0.0) for shape in shapes)
    return tuple(dict.fromkeys(shapes))


def fit_model(
    dataset: family_grp.Dataset,
    objective: str,
    variant: phase_heads.Variant,
    policy: str,
) -> FittedPowerModel:
    indices = np.arange(dataset.n)
    selection = phase_heads.select_shared_hyperparameters(
        dataset,
        dataset_id(objective),
        variant,
        candidate_shapes(variant, policy),
        indices,
        INNER_SEED,
        INNER_SPLITS,
    )
    design, names, _layout = phase_heads.build_design(dataset, variant, selection.shape, None)
    head = family_grp.fit_head(design, dataset.target, indices, selection.l2, names)
    return FittedPowerModel(dataset, variant, selection, head)


def scalar_predictor(model: FittedPowerModel):
    def predict(weights: np.ndarray) -> float:
        return float(model.predict(np.asarray(weights, dtype=float)[None, :, :])[0])

    return predict


def entropy_effective_count(weights: np.ndarray) -> float:
    positive = np.clip(np.asarray(weights, dtype=float), 1e-16, 1.0)
    return float(np.exp(-np.sum(positive * np.log(positive))))


def candidate_record(
    *,
    candidate: str,
    objective: str,
    model_name: str,
    policy: str,
    model: FittedPowerModel,
    weights: np.ndarray,
    natural: np.ndarray,
    token_counts: np.ndarray,
    target_budget: int,
    aggregate_kl: float | None,
    phase_information_budget: float | None,
    successful_starts: int,
) -> dict[str, object]:
    alpha0, alpha1 = matched.PHASE_FRACTIONS
    aggregate = alpha0 * weights[0] + alpha1 * weights[1]
    epochs = matched.olmix.simulated_epochs(weights, token_counts, target_budget=target_budget)
    prediction = scalar_predictor(model)(weights)
    return {
        "candidate": candidate,
        "objective": objective,
        "model": model_name,
        "policy": policy,
        "selected_l2": model.selection.l2,
        "shape_exponent": model.selection.shape.exponent,
        "shape_late_multiplier": model.selection.shape.late_multiplier,
        "shape_forgetting_rate": model.selection.shape.forgetting_rate,
        "shape_penalty_threshold": model.selection.shape.penalty_threshold,
        "aggregate_kl_coefficient": aggregate_kl,
        "phase_information_budget": phase_information_budget,
        "predicted_bpb": prediction,
        "aggregate_kl_to_proportional": matched.weighted_kl(np.stack([aggregate, aggregate]), natural),
        "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - natural).sum()),
        "phase_information_kl": phase_information.fixed_aggregate.phase_order_kl(
            weights,
            aggregate,
            alpha0,
            alpha1,
        ),
        "phase_tv": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
        "phase_0_effective_bucket_count": entropy_effective_count(weights[0]),
        "phase_1_effective_bucket_count": entropy_effective_count(weights[1]),
        "aggregate_effective_bucket_count": entropy_effective_count(aggregate),
        "near_zero_weight_count": int(np.sum(weights < 1e-5)),
        "max_weight": float(weights.max()),
        "max_simulated_epoch": float(epochs.max()),
        "q95_simulated_epoch": float(np.quantile(epochs, 0.95)),
        "optimizer_successful_starts": successful_starts,
    }


def write_candidate(
    output_dir: Path,
    gcs_output_dir: str,
    candidate: str,
    frame: pd.DataFrame,
    *,
    upload: bool,
) -> None:
    mixture_dir = output_dir / "mixtures"
    mixture_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(mixture_dir / f"{candidate}.csv", index=False)
    if upload:
        with fsspec.open(f"{gcs_output_dir.rstrip('/')}/{candidate}.csv", "wt") as handle:
            frame.to_csv(handle, index=False)


def render_diagnostics(manifest: pd.DataFrame, output_dir: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: prediction versus maximum epochs",
            "Table-9: prediction versus maximum epochs",
            "Uncheatable: phase versus aggregate divergence",
            "Table-9: phase versus aggregate divergence",
        ),
    )
    colors = {"shared_1p": "#6F8190", "eta": "#E36F2C", "separate_heads": "#238443"}
    for column, objective in enumerate(OBJECTIVES, start=1):
        selected = manifest.loc[manifest["objective"].eq(objective)]
        for model_name, group in selected.groupby("model", sort=False):
            customdata = np.column_stack([group["candidate"], group["policy"], group["predicted_bpb"]])
            figure.add_trace(
                go.Scatter(
                    x=group["max_simulated_epoch"],
                    y=group["predicted_bpb"],
                    mode="markers+text",
                    text=group["candidate"],
                    textposition="top center",
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=column == 1,
                    marker={"color": colors[model_name], "size": 10},
                    customdata=customdata,
                    hovertemplate=(
                        "%{customdata[0]}<br>policy=%{customdata[1]}<br>prediction=%{customdata[2]:.6f}"
                        "<br>max epochs=%{x:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=group["aggregate_kl_to_proportional"],
                    y=group["phase_information_kl"],
                    mode="markers+text",
                    text=group["candidate"],
                    textposition="top center",
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=False,
                    marker={"color": colors[model_name], "size": 10},
                    customdata=customdata,
                    hovertemplate=(
                        "%{customdata[0]}<br>aggregate KL=%{x:.5f}<br>phase information=%{y:.5f}" "<extra></extra>"
                    ),
                ),
                row=2,
                col=column,
            )
    figure.update_xaxes(type="log", title_text="maximum simulated epochs", row=1)
    figure.update_yaxes(title_text="predicted BPB", row=1)
    figure.update_xaxes(title_text="aggregate KL to proportional", row=2)
    figure.update_yaxes(title_text="phase-information KL", row=2)
    figure.update_layout(
        title="Bucket-resolved family GRP deployment candidates",
        template="plotly_white",
        width=1500,
        height=1050,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08},
        margin={"l": 80, "r": 30, "t": 100, "b": 100},
    )
    figure.write_html(output_dir / "candidate_diagnostics.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(manifest: pd.DataFrame, output_dir: Path) -> None:
    raw = manifest.loc[manifest["candidate"].str.endswith("raw") | manifest["candidate"].str.endswith("akl0")].copy()
    compact = raw[
        [
            "objective",
            "model",
            "policy",
            "predicted_bpb",
            "max_weight",
            "max_simulated_epoch",
            "near_zero_weight_count",
            "aggregate_kl_to_proportional",
            "phase_information_kl",
        ]
    ]
    lines = [
        "# Bucket-resolved family GRP deployment panel",
        "",
        "## Question",
        "",
        (
            "Compare the bucket-resolved family power model with a scalar late multiplier against the same "
            "retained-exposure, bucket-response, family-response, and replay-penalty model with independent "
            "early/late response heads. The single-phase ablation is shared because both variants collapse to "
            "one aggregate response head when phase weights are tied."
        ),
        "",
        "## Fit and selection",
        "",
        "- Each policy/objective fit uses its 280-row 300M panel and all 39 top-level buckets.",
        "- Shape and ridge hyperparameters are selected by five-fold panel-stratified CV, then refit on all rows.",
        "- No validation outcome is used in fitting, hyperparameter selection, or candidate optimization.",
        "- Raw optima are included as extrapolation diagnostics rather than presumed deployment recommendations.",
        "",
        "## Raw optimum audit",
        "",
        compact.to_markdown(index=False, floatfmt=".6f"),
        "",
        (
            "The pathology is objective/model specific. The Table-9 separate-head raw optimum places 0.832 of "
            "one phase on one bucket and reaches 238 simulated epochs; scalar-eta raw optima reach 77-100 epochs. "
            "The one-phase raw optima are less extreme at about 24-25 maximum epochs, but still extrapolative."
        ),
        "",
        "## Validation panel",
        "",
        "- One-phase: aggregate KL coefficients 0 (raw), 0.05, and 0.1.",
        (
            "- Two-phase: each model's raw optimum plus a decoupled path with aggregate KL 0.05 fixed and "
            "phase-information budgets 0.005, 0.02, and 0.05."
        ),
        "- Objectives: eval/uncheatable_eval/bpb and native 51-component Table-9 macro BPB.",
        "- All 22 candidates use one shared data seed; there are no repeats in this screening panel.",
        "",
        "## Interpretation guardrail",
        "",
        (
            "A regularized candidate transfers only if its observed 3e18 metric improves. Large surrogate-predicted "
            "gains, especially on Table-9 separate heads, are not evidence by themselves because prior phase-path "
            "experiments show severe optimum-region optimism."
        ),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument(
        "--aggregate-kl-values",
        default=",".join(str(value) for value in AGGREGATE_KL_VALUES),
    )
    parser.add_argument(
        "--regularized-aggregate-kl-values",
        default=",".join(str(value) for value in REGULARIZED_AGGREGATE_KL_VALUES),
    )
    parser.add_argument(
        "--phase-information-budgets",
        default=",".join(str(value) for value in PHASE_INFORMATION_BUDGETS),
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_kl_values = parse_float_tuple(args.aggregate_kl_values)
    regularized_aggregate_kl_values = parse_float_tuple(args.regularized_aggregate_kl_values)
    phase_information_budgets = parse_float_tuple(args.phase_information_budgets)
    if 0.0 not in aggregate_kl_values:
        raise ValueError("The one-phase sweep must include the raw optimum at aggregate KL=0")

    records: list[dict[str, object]] = []
    model_records: dict[str, Any] = {}
    for objective in OBJECTIVES:
        print(f"Fitting {objective}", flush=True)
        datasets = load_datasets(objective)
        _packet, domains, natural, token_counts, target_budget, _folds = objective_data.load_objective(objective)
        if domains != list(datasets["1p"].domains) or domains != list(datasets["2p"].domains):
            raise ValueError(f"Domain order mismatch for {objective}")

        one_phase_model = fit_model(datasets["1p"], objective, ONE_PHASE_VARIANT, "1p")
        two_phase_models = {
            name: fit_model(datasets["2p"], objective, variant, "2p") for name, variant in TWO_PHASE_MODELS.items()
        }
        model_records[f"{objective}/shared_1p"] = {
            "variant": ONE_PHASE_VARIANT.name,
            "selection": asdict(one_phase_model.selection),
        }
        for name, model in two_phase_models.items():
            model_records[f"{objective}/{name}"] = {
                "variant": model.variant.name,
                "selection": asdict(model.selection),
            }

        aggregates: dict[float, np.ndarray] = {}
        for aggregate_kl in aggregate_kl_values:
            result = matched.optimize(
                scalar_predictor(one_phase_model),
                optimizer_dataset(datasets["1p"]),
                natural,
                aggregate_kl,
                "1p",
            )
            aggregate = np.asarray(result.weights[0], dtype=float)
            aggregates[aggregate_kl] = aggregate
            candidate = f"bfgrp_{OBJECTIVE_ABBREVIATIONS[objective]}_1p_akl{float_tag(aggregate_kl)}"
            record = candidate_record(
                candidate=candidate,
                objective=objective,
                model_name="shared_1p",
                policy="1p",
                model=one_phase_model,
                weights=result.weights,
                natural=natural,
                token_counts=token_counts,
                target_budget=target_budget,
                aggregate_kl=aggregate_kl,
                phase_information_budget=0.0,
                successful_starts=result.successful_starts,
            )
            mixture = per_component.mixture_frame(
                domains=domains,
                natural=natural,
                weights=result.weights,
                token_counts=token_counts,
                target_budget=target_budget,
            )
            write_candidate(args.output_dir, args.gcs_output_dir, candidate, mixture, upload=args.upload)
            records.append(record)

        for model_name, model in two_phase_models.items():
            raw = matched.optimize(
                scalar_predictor(model),
                optimizer_dataset(datasets["2p"]),
                natural,
                0.0,
                "2p",
            )
            candidate = f"bfgrp_{OBJECTIVE_ABBREVIATIONS[objective]}_{model_name}_2p_raw"
            record = candidate_record(
                candidate=candidate,
                objective=objective,
                model_name=model_name,
                policy="2p",
                model=model,
                weights=raw.weights,
                natural=natural,
                token_counts=token_counts,
                target_budget=target_budget,
                aggregate_kl=None,
                phase_information_budget=None,
                successful_starts=raw.successful_starts,
            )
            mixture = per_component.mixture_frame(
                domains=domains,
                natural=natural,
                weights=raw.weights,
                token_counts=token_counts,
                target_budget=target_budget,
            )
            write_candidate(args.output_dir, args.gcs_output_dir, candidate, mixture, upload=args.upload)
            records.append(record)

            for aggregate_kl in regularized_aggregate_kl_values:
                aggregate = aggregates[aggregate_kl]
                for phase_budget in phase_information_budgets:
                    result = decoupled.generic_optimize_fixed_aggregate(
                        scalar_predictor(model),
                        aggregate,
                        phase_budget,
                    )
                    candidate = (
                        f"bfgrp_{OBJECTIVE_ABBREVIATIONS[objective]}_{model_name}_2p_"
                        f"akl{float_tag(aggregate_kl)}_e{float_tag(phase_budget)}"
                    )
                    record = candidate_record(
                        candidate=candidate,
                        objective=objective,
                        model_name=model_name,
                        policy="2p",
                        model=model,
                        weights=result.weights,
                        natural=natural,
                        token_counts=token_counts,
                        target_budget=target_budget,
                        aggregate_kl=aggregate_kl,
                        phase_information_budget=phase_budget,
                        successful_starts=result.successful_starts,
                    )
                    mixture = per_component.mixture_frame(
                        domains=domains,
                        natural=natural,
                        weights=result.weights,
                        token_counts=token_counts,
                        target_budget=target_budget,
                    )
                    write_candidate(args.output_dir, args.gcs_output_dir, candidate, mixture, upload=args.upload)
                    records.append(record)

    manifest = pd.DataFrame(records).sort_values(["objective", "policy", "model", "candidate"])
    manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    (args.output_dir / "fitted_models.json").write_text(json.dumps(model_records, indent=2, allow_nan=False) + "\n")
    render_diagnostics(manifest, args.output_dir)
    write_report(manifest, args.output_dir)
    print(manifest.to_string(index=False), flush=True)
    print(f"Wrote {len(manifest)} candidates to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
