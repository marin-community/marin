# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "fsspec", "gcsfs", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Follow-up diagnostics for two-phase Table-9 DSP optimism.

This script intentionally does not launch training. It checks two questions that
matter before paying for more 3e18 validations:

1. Do simple phase-aware residual corrections still help when whole diagnostic
   groups are held out?
2. Can those corrections retrodict the already validated two-phase proposals,
   or are nearest observed neighbors more reliable out of support?
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_table9_phase_residual_corrections_300m as residuals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmix_reference_deletion_augmented_300m as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_olmo_base_easy_top_level_dsp_300m as top_level_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_INPUT_DIR = REFERENCE_OUTPUTS / "olmo_base_easy_extra_300m_heldout_eval_20260630"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "table9_phase_trust_region_followups_20260630"
DEFAULT_AGGREGATE_DSP_MODEL = residuals.DEFAULT_AGGREGATE_DSP_MODEL
TARGET_COL = residuals.TARGET_COL
BASELINE_PRED_COL = residuals.BASELINE_PRED_COL
RIDGE_ALPHAS = residuals.RIDGE_ALPHAS
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ProposalSpec:
    name: str
    path: Path
    actual_bpb: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aggregate-dsp-model", type=Path, default=DEFAULT_AGGREGATE_DSP_MODEL)
    return parser.parse_args()


def proposal_specs() -> list[ProposalSpec]:
    proposal_dir = REFERENCE_OUTPUTS / "table9_dsp_phase_functional_form_20260630" / "proposal_screen" / "mixtures"
    return [
        ProposalSpec(
            name="one_phase_dsp_effexp_kl0p1",
            path=REFERENCE_OUTPUTS
            / "olmo_base_easy_one_phase_model_sweeps_300m_20260628"
            / "dsp_one_phase_effexp_linear_reg0p0001_kl0p1"
            / "proposed_mixture_weights.csv",
            actual_bpb=1.070728420274658,
            source="one_phase_validation",
        ),
        ProposalSpec(
            name="two_phase_effexp_l2_0p01_kl0p5",
            path=proposal_dir / "effective_exposure_l2_0p01_kl_0p5.csv",
            actual_bpb=1.0988919535825066,
            source="phase_split_validation",
        ),
        ProposalSpec(
            name="two_phase_split_l2_0p01_kl0p3",
            path=proposal_dir / "split_saturation_penalty_l2_0p01_kl_0p3.csv",
            actual_bpb=1.085228613919854,
            source="phase_split_validation",
        ),
        ProposalSpec(
            name="two_phase_split_l2_0p01_kl0p4",
            path=proposal_dir / "split_saturation_penalty_l2_0p01_kl_0p4.csv",
            actual_bpb=1.0958575091227092,
            source="phase_split_validation",
        ),
        ProposalSpec(
            name="two_phase_split_l2_0p01_kl0p4_repeat",
            path=proposal_dir / "split_saturation_penalty_l2_0p01_kl_0p4.csv",
            actual_bpb=1.097619091512261,
            source="phase_split_validation_repeat",
        ),
    ]


def group_folds(groups: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = groups.dropna().unique()
    split_count = min(5, len(unique_groups))
    if split_count < 2:
        raise ValueError("Need at least two diagnostic groups for GroupKFold")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in GroupKFold(n_splits=split_count).split(np.zeros(len(groups)), groups=groups):
        folds.append((np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)))
    return folds


def load_weight_csv(path: Path, domains: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    if "domain" not in frame.columns:
        raise ValueError(f"{path} has no domain column")
    indexed = frame.set_index("domain")
    missing = sorted(set(domains).difference(indexed.index))
    if missing:
        raise ValueError(f"{path} is missing domains: {missing[:8]}")
    weights = np.stack(
        [
            indexed.loc[domains, "phase_0_weight"].to_numpy(dtype=float),
            indexed.loc[domains, "phase_1_weight"].to_numpy(dtype=float),
        ],
        axis=0,
    )
    return dsp.normalize_weights(weights[None, :, :])[0]


def proposal_frame(domains: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for spec in proposal_specs():
        weights = load_weight_csv(spec.path, domains)
        row: dict[str, float | str] = {
            "run_name": spec.name,
            "diagnostic_group": "validated_proposal",
            "diagnostic_family": spec.source,
            TARGET_COL: spec.actual_bpb,
            "proposal_path": str(spec.path),
        }
        for phase in (0, 1):
            for domain_idx, domain in enumerate(domains):
                row[f"phase_{phase}_{domain}"] = float(weights[phase, domain_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def build_packet_for_frame(frame: pd.DataFrame, domains: list[str]) -> dsp.PacketData:
    _signal, columns, _domains, _natural = base.load_raw_signal_panel()
    token_counts = base.load_domain_token_counts(domains)
    return top_level_dsp.build_dsp_packet(frame, columns, domains, token_counts, TARGET_COL)


def feature_bundle(frame: pd.DataFrame, model: dsp.FittedDSPModel) -> tuple[dsp.PacketData, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    domains = list(model.domain_names)
    packet = build_packet_for_frame(frame, domains)
    baseline_row = frame["run_name"].eq("baseline_proportional")
    if int(baseline_row.sum()) == 1:
        natural = packet.w[int(np.flatnonzero(baseline_row)[0]), 0].copy()
    else:
        natural = dsp.normalize_weights(np.asarray([base.load_domain_token_counts(domains)], dtype=float))[0]
    global_features = residuals.global_phase_features(packet, natural)
    family_features = residuals.family_phase_features(packet, domains)
    domain_features = residuals.domain_phase_features(packet, domains)
    return packet, natural, global_features, family_features, domain_features


def shared_model_prediction(model: dsp.FittedDSPModel, packet: dsp.PacketData) -> np.ndarray:
    return dsp.predict(model, packet.w)


def full_fit_residual_prediction(
    *,
    train_y: np.ndarray,
    train_base_pred: np.ndarray,
    train_features: pd.DataFrame,
    proposal_base_pred: np.ndarray,
    proposal_features: pd.DataFrame,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    model.fit(train_features.to_numpy(dtype=float), train_y - train_base_pred)
    return proposal_base_pred + model.predict(proposal_features.to_numpy(dtype=float))


def full_fit_blend_prediction(
    *,
    train_y: np.ndarray,
    train_features: pd.DataFrame,
    proposal_features: pd.DataFrame,
) -> np.ndarray:
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    model.fit(train_features.to_numpy(dtype=float), train_y)
    return model.predict(proposal_features.to_numpy(dtype=float))


def full_fit_conservative_penalty(
    *,
    train_y: np.ndarray,
    train_base_pred: np.ndarray,
    train_risk: np.ndarray,
    proposal_base_pred: np.ndarray,
    proposal_risk: np.ndarray,
) -> np.ndarray:
    risk_mean = float(np.mean(train_risk))
    risk_std = float(np.std(train_risk) + 1e-12)
    train_risk_std = (train_risk - risk_mean) / risk_std
    proposal_risk_std = (proposal_risk - risk_mean) / risk_std
    lambdas = np.asarray([0.0, 0.001, 0.002, 0.003, 0.005, 0.008, 0.012, 0.016, 0.02, 0.03, 0.05])
    best_lambda = 0.0
    best_key: tuple[float, float] | None = None
    for lam in lambdas:
        pred = train_base_pred + lam * train_risk_std
        regret = residuals.global_regret_at_k(train_y, pred, 1)
        optimism, _rmse = residuals.lower_tail_optimism(train_y, pred)
        key = (regret, optimism)
        if best_key is None or key < best_key:
            best_key = key
            best_lambda = float(lam)
    return proposal_base_pred + best_lambda * proposal_risk_std


def nearest_observed_predictions(train_frame: pd.DataFrame, proposal: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
    weight_cols = [f"phase_{phase}_{domain}" for phase in (0, 1) for domain in domains]
    train_weights = train_frame[weight_cols].to_numpy(dtype=float)
    proposal_weights = proposal[weight_cols].to_numpy(dtype=float)
    neighbor = NearestNeighbors(n_neighbors=1, metric="manhattan")
    neighbor.fit(train_weights)
    distances, indices = neighbor.kneighbors(proposal_weights)
    rows: list[dict[str, float | str]] = []
    for proposal_idx, train_idx in enumerate(indices[:, 0]):
        rows.append(
            {
                "run_name": str(proposal.iloc[proposal_idx]["run_name"]),
                "nearest_observed_run_name": str(train_frame.iloc[int(train_idx)]["run_name"]),
                "nearest_observed_bpb": float(train_frame.iloc[int(train_idx)][TARGET_COL]),
                "nearest_observed_l1_distance": float(distances[proposal_idx, 0]),
                "nearest_observed_mean_phase_tv": float(0.5 * distances[proposal_idx, 0] / 2.0),
            }
        )
    return pd.DataFrame(rows)


def write_retrodiction_plot(path: Path, frame: pd.DataFrame) -> None:
    fig = go.Figure()
    for variant, group in frame.groupby("variant", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group["actual_bpb"],
                y=group["predicted_bpb"],
                mode="markers+text",
                name=variant,
                text=group["run_name"],
                textposition="top center",
                hovertemplate="run=%{text}<br>actual=%{x:.5f}<br>pred=%{y:.5f}<extra></extra>",
            )
        )
    lo = min(float(frame["actual_bpb"].min()), float(frame["predicted_bpb"].min()))
    hi = max(float(frame["actual_bpb"].max()), float(frame["predicted_bpb"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y=x", line={"dash": "dash", "color": "#64748b"}))
    fig.update_layout(
        title="Retrodiction on already validated Table-9 proposals",
        xaxis_title="Validated Table-9 macro BPB",
        yaxis_title="Predicted Table-9 macro BPB",
        template="plotly_white",
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_groupkfold_plot(path: Path, summary: pd.DataFrame) -> None:
    metrics = ["fold_mean_regret_at_1", "lower_tail_optimism", "rmse", "spearman"]
    fig = go.Figure()
    for idx, metric in enumerate(metrics):
        ordered = summary.sort_values(metric, ascending=metric != "spearman")
        fig.add_trace(
            go.Bar(
                x=ordered["variant"],
                y=ordered[metric],
                name=metric,
                visible=idx == 0,
                hovertemplate=f"{metric}=%{{y:.6f}}<extra></extra>",
            )
        )
    buttons = []
    for idx, metric in enumerate(metrics):
        visible = [False] * len(metrics)
        visible[idx] = True
        buttons.append({"label": metric, "method": "update", "args": [{"visible": visible}, {"yaxis.title.text": metric}]})
    fig.update_layout(
        title="Strict diagnostic-group-heldout Table-9 phase variants",
        xaxis_title="Variant",
        yaxis_title=metrics[0],
        template="plotly_white",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "y": 1.16}],
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel, _predictions = residuals.load_frames(args.input_dir)
    model = dsp.model_from_json(json.loads(args.aggregate_dsp_model.read_text()))
    domains = list(model.domain_names)

    panel_packet, _natural, global_features, family_features, domain_features = feature_bundle(panel, model)
    y = panel[TARGET_COL].to_numpy(dtype=float)
    base_pred = panel[BASELINE_PRED_COL].to_numpy(dtype=float)
    collapsed_pred = residuals.collapsed_dsp_prediction(args.aggregate_dsp_model, panel_packet)
    folds = group_folds(panel["diagnostic_group"])
    collapsed_features = pd.concat(
        [
            pd.DataFrame(
                {
                    "base_pred": base_pred,
                    "collapsed_pred": collapsed_pred,
                    "base_minus_collapsed_pred": base_pred - collapsed_pred,
                }
            ),
            global_features,
        ],
        axis=1,
    )

    variant_predictions: dict[str, tuple[np.ndarray, str]] = {
        "baseline_aggregate_dsp": (base_pred, "Existing expanded-panel aggregate effective-exposure DSP OOF prediction."),
        "residual_global_phase_ridge": (
            residuals.residual_ridge_oof(y=y, base_pred=base_pred, features=global_features, folds=folds),
            "Group-heldout ridge correction using global phase diagnostics.",
        ),
        "residual_family_phase_ridge": (
            residuals.residual_ridge_oof(
                y=y,
                base_pred=base_pred,
                features=pd.concat([global_features, family_features], axis=1),
                folds=folds,
            ),
            "Group-heldout ridge correction using global and family phase diagnostics.",
        ),
        "residual_domain_phase_ridge": (
            residuals.residual_ridge_oof(
                y=y,
                base_pred=base_pred,
                features=pd.concat([global_features, domain_features], axis=1),
                folds=folds,
            ),
            "Group-heldout ridge correction using global and per-domain phase diagnostics.",
        ),
        "collapsed_prediction_blend_ridge": (
            residuals.blend_ridge_oof(y=y, features=collapsed_features, folds=folds),
            "Group-heldout blend of current and collapsed DSP predictions.",
        ),
        "conservative_phase_risk_penalty": (
            residuals.conservative_penalty_oof(
                y=y,
                base_pred=base_pred,
                risk=global_features["log_exposure_phase_l2"].to_numpy(dtype=float)
                + global_features["phase_tv"].to_numpy(dtype=float),
                folds=folds,
            ),
            "Group-heldout fold-tuned positive phase-risk penalty.",
        ),
    }
    grouped_summary = pd.DataFrame.from_records(
        [
            asdict(residuals.summarize_variant(variant=variant, frame=panel, pred=pred, folds=folds, notes=notes))
            for variant, (pred, notes) in variant_predictions.items()
        ]
    ).sort_values(["fold_mean_regret_at_1", "lower_tail_optimism", "rmse"])

    proposals = proposal_frame(domains)
    proposal_packet, _proposal_natural, proposal_global, proposal_family, proposal_domain = feature_bundle(proposals, model)
    proposal_base_pred = shared_model_prediction(model, proposal_packet)
    proposal_collapsed_pred = residuals.collapsed_dsp_prediction(args.aggregate_dsp_model, proposal_packet)
    proposal_collapsed_features = pd.concat(
        [
            pd.DataFrame(
                {
                    "base_pred": proposal_base_pred,
                    "collapsed_pred": proposal_collapsed_pred,
                    "base_minus_collapsed_pred": proposal_base_pred - proposal_collapsed_pred,
                }
            ),
            proposal_global,
        ],
        axis=1,
    )
    proposal_actual = proposals[TARGET_COL].to_numpy(dtype=float)
    retrodiction_predictions: dict[str, np.ndarray] = {
        "shared_aggregate_dsp": proposal_base_pred,
        "nearest_observed": nearest_observed_predictions(panel, proposals, domains)["nearest_observed_bpb"].to_numpy(dtype=float),
        "residual_global_phase_ridge": full_fit_residual_prediction(
            train_y=y,
            train_base_pred=base_pred,
            train_features=global_features,
            proposal_base_pred=proposal_base_pred,
            proposal_features=proposal_global,
        ),
        "residual_family_phase_ridge": full_fit_residual_prediction(
            train_y=y,
            train_base_pred=base_pred,
            train_features=pd.concat([global_features, family_features], axis=1),
            proposal_base_pred=proposal_base_pred,
            proposal_features=pd.concat([proposal_global, proposal_family], axis=1),
        ),
        "residual_domain_phase_ridge": full_fit_residual_prediction(
            train_y=y,
            train_base_pred=base_pred,
            train_features=pd.concat([global_features, domain_features], axis=1),
            proposal_base_pred=proposal_base_pred,
            proposal_features=pd.concat([proposal_global, proposal_domain], axis=1),
        ),
        "collapsed_prediction_blend_ridge": full_fit_blend_prediction(
            train_y=y,
            train_features=collapsed_features,
            proposal_features=proposal_collapsed_features,
        ),
        "conservative_phase_risk_penalty": full_fit_conservative_penalty(
            train_y=y,
            train_base_pred=base_pred,
            train_risk=global_features["log_exposure_phase_l2"].to_numpy(dtype=float)
            + global_features["phase_tv"].to_numpy(dtype=float),
            proposal_base_pred=proposal_base_pred,
            proposal_risk=proposal_global["log_exposure_phase_l2"].to_numpy(dtype=float)
            + proposal_global["phase_tv"].to_numpy(dtype=float),
        ),
    }
    nearest = nearest_observed_predictions(panel, proposals, domains)
    retrodiction_rows: list[dict[str, float | str]] = []
    for variant, pred in retrodiction_predictions.items():
        for idx, spec in enumerate(proposal_specs()):
            retrodiction_rows.append(
                {
                    "variant": variant,
                    "run_name": spec.name,
                    "actual_bpb": float(proposal_actual[idx]),
                    "predicted_bpb": float(pred[idx]),
                    "actual_minus_predicted": float(proposal_actual[idx] - pred[idx]),
                    "abs_error": float(abs(proposal_actual[idx] - pred[idx])),
                    "nearest_observed_bpb": float(nearest.iloc[idx]["nearest_observed_bpb"]),
                    "nearest_observed_run_name": str(nearest.iloc[idx]["nearest_observed_run_name"]),
                    "nearest_observed_l1_distance": float(nearest.iloc[idx]["nearest_observed_l1_distance"]),
                    "proposal_path": str(spec.path),
                }
            )
    retrodiction = pd.DataFrame(retrodiction_rows)
    retrodiction_summary = (
        retrodiction.groupby("variant", sort=False)
        .agg(
            mean_abs_error=("abs_error", "mean"),
            max_abs_error=("abs_error", "max"),
            mean_optimism=("actual_minus_predicted", "mean"),
            max_optimism=("actual_minus_predicted", "max"),
        )
        .reset_index()
        .sort_values("mean_abs_error")
    )

    grouped_summary.to_csv(args.output_dir / "group_heldout_phase_variant_summary.csv", index=False)
    retrodiction.to_csv(args.output_dir / "validated_proposal_retrodiction.csv", index=False)
    retrodiction_summary.to_csv(args.output_dir / "validated_proposal_retrodiction_summary.csv", index=False)
    write_groupkfold_plot(args.output_dir / "group_heldout_phase_variant_summary.html", grouped_summary)
    write_retrodiction_plot(args.output_dir / "validated_proposal_retrodiction.html", retrodiction)

    result = {
        "best_group_heldout_variant_by_regret_at_1": str(grouped_summary.iloc[0]["variant"]),
        "best_group_heldout_regret_at_1": float(grouped_summary.iloc[0]["fold_mean_regret_at_1"]),
        "best_retrodiction_variant_by_mae": str(retrodiction_summary.iloc[0]["variant"]),
        "best_retrodiction_mean_abs_error": float(retrodiction_summary.iloc[0]["mean_abs_error"]),
        "shared_aggregate_retrodiction_mean_abs_error": float(
            retrodiction_summary.loc[
                retrodiction_summary["variant"].eq("shared_aggregate_dsp"), "mean_abs_error"
            ].iloc[0]
        ),
    }
    (args.output_dir / "trust_region_followup_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(grouped_summary.to_string(index=False), flush=True)
    print(retrodiction_summary.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
