# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scipy", "scikit-learn"]
# ///
"""Diagnose two-phase Table-9 DSP failure modes before adding model terms."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import fit_olmix_reference_deletion_augmented_300m as base  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "table9_two_phase_dsp_failure_modes_20260630"
DEFAULT_MODEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_table9_macro_dsp_300m_20260625"
    / "dsp_effective_exposure"
    / "table9_macro_bpb"
    / "linear_reg_0.0001"
    / "model.json"
)
DEFAULT_EXPANDED_PANEL = (
    REFERENCE_OUTPUTS
    / "olmo_base_easy_extra_300m_heldout_eval_20260630"
    / "expanded_300m_table9_diagnostic_panel.csv"
)
DEFAULT_VALIDATION_RESULTS = (
    REFERENCE_OUTPUTS
    / "table9_dsp_phase_functional_form_20260630"
    / "validation_results_wandb_probe.csv"
)
DEFAULT_KL_SUMMARY = (
    REFERENCE_OUTPUTS
    / "table9_dsp_phase_functional_form_20260630"
    / "proposal_screen"
    / "phase_variant_kl_proposal_summary.csv"
)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
TARGET_COL = "table9_macro_bpb"


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    path: Path
    actual_bpb: float
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--expanded-panel", type=Path, default=DEFAULT_EXPANDED_PANEL)
    parser.add_argument("--validation-results", type=Path, default=DEFAULT_VALIDATION_RESULTS)
    parser.add_argument("--kl-summary", type=Path, default=DEFAULT_KL_SUMMARY)
    return parser.parse_args()


def candidate_specs() -> list[CandidateSpec]:
    return [
        CandidateSpec(
            name="one_phase_dsp_effexp_kl0p1",
            path=REFERENCE_OUTPUTS
            / "olmo_base_easy_one_phase_model_sweeps_300m_20260628"
            / "dsp_one_phase_effexp_linear_reg0p0001_kl0p1"
            / "proposed_mixture_weights.csv",
            actual_bpb=1.070728420274658,
            notes="Best current one-phase Table-9 DSP validation at 3e18.",
        ),
        CandidateSpec(
            name="two_phase_split_l2_0p01_kl0p3",
            path=REFERENCE_OUTPUTS
            / "table9_dsp_phase_functional_form_20260630"
            / "proposal_screen"
            / "mixtures"
            / "split_saturation_penalty_l2_0p01_kl_0p3.csv",
            actual_bpb=1.085228613919854,
            notes="Best current split two-phase Table-9 validation at 3e18.",
        ),
        CandidateSpec(
            name="two_phase_effexp_l2_0p01_kl0p5",
            path=REFERENCE_OUTPUTS
            / "table9_dsp_phase_functional_form_20260630"
            / "proposal_screen"
            / "mixtures"
            / "effective_exposure_l2_0p01_kl_0p5.csv",
            actual_bpb=1.0988919535825066,
            notes="Validated phase-split panel effective-exposure comparison.",
        ),
    ]


def load_weights(path: Path, domains: list[str]) -> np.ndarray:
    frame = pd.read_csv(path)
    by_domain = frame.set_index("domain")
    missing = sorted(set(domains).difference(by_domain.index))
    if missing:
        raise ValueError(f"{path} is missing domains: {missing[:8]}")
    weights = np.stack(
        [
            by_domain.loc[domains, "phase_0_weight"].to_numpy(dtype=float),
            by_domain.loc[domains, "phase_1_weight"].to_numpy(dtype=float),
        ],
        axis=0,
    )
    return dsp.normalize_weights(weights[None, :, :])[0]


def score_candidates(model: dsp.FittedDSPModel, domains: list[str], output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for spec in candidate_specs():
        weights = load_weights(spec.path, domains)
        pred = float(dsp.predict(model, weights[None, :, :])[0])
        phase_tv = float(0.5 * np.abs(weights[0] - weights[1]).sum())
        rows.append(
            {
                "candidate": spec.name,
                "model_predicted_bpb": pred,
                "actual_3e18_bpb": spec.actual_bpb,
                "actual_minus_predicted": spec.actual_bpb - pred,
                "phase_tv": phase_tv,
                "path": str(spec.path),
                "notes": spec.notes,
            }
        )
    frame = pd.DataFrame(rows)
    one = frame[frame["candidate"].eq("one_phase_dsp_effexp_kl0p1")].iloc[0]
    split = frame[frame["candidate"].eq("two_phase_split_l2_0p01_kl0p3")].iloc[0]
    branch = {
        "one_phase_predicted_bpb": float(one["model_predicted_bpb"]),
        "split_predicted_bpb": float(split["model_predicted_bpb"]),
        "one_phase_actual_bpb": float(one["actual_3e18_bpb"]),
        "split_actual_bpb": float(split["actual_3e18_bpb"]),
        "model_preferred_split": bool(float(split["model_predicted_bpb"]) < float(one["model_predicted_bpb"])),
        "realized_split_minus_one_phase": float(split["actual_3e18_bpb"]) - float(one["actual_3e18_bpb"]),
        "interpretation": (
            "shared effective-exposure surrogate prefers the off-diagonal split despite worse validation"
            if float(split["model_predicted_bpb"]) < float(one["model_predicted_bpb"])
            else "shared effective-exposure surrogate does not prefer the split; selection/optimizer path should be audited"
        ),
    }
    (output_dir / "claimed_vs_realized_branch.json").write_text(json.dumps(branch, indent=2, sort_keys=True) + "\n")
    return frame


def validated_optimism_table(validation_path: Path, kl_summary_path: Path) -> pd.DataFrame:
    validation = pd.read_csv(validation_path)
    validation = validation[validation["macro"].notna()].copy()
    rows = [
        {
            "validated_name": "t9_dsp_effexp_table9_l2_0p01_kl0p5_3e18",
            "variant_key": "effective_exposure",
            "linear_reg": 0.01,
            "kl_reg": 0.5,
        },
        {
            "validated_name": "t9_dsp_split_table9_l2_0p01_kl0p3_3e18",
            "variant_key": "split_saturation_penalty",
            "linear_reg": 0.01,
            "kl_reg": 0.3,
        },
        {
            "validated_name": "t9_dsp_split_table9_l2_0p01_kl0p4_3e18",
            "variant_key": "split_saturation_penalty",
            "linear_reg": 0.01,
            "kl_reg": 0.4,
        },
        {
            "validated_name": "t9_dsp_split_table9_l2_0p01_kl0p4_repeat_3e18",
            "variant_key": "split_saturation_penalty",
            "linear_reg": 0.01,
            "kl_reg": 0.4,
        },
    ]
    map_frame = pd.DataFrame(rows)
    actuals = validation[["name", "macro"]].rename(columns={"name": "validated_name", "macro": "actual_bpb"})
    kl = pd.read_csv(kl_summary_path)
    merged = map_frame.merge(actuals, on="validated_name", how="inner", validate="one_to_many").merge(
        kl,
        on=["variant_key", "linear_reg", "kl_reg"],
        how="inner",
        validate="many_to_one",
    )
    merged["actual_minus_predicted"] = merged["actual_bpb"] - merged["predicted_objective"]
    merged["predicted_minus_actual"] = merged["predicted_objective"] - merged["actual_bpb"]
    return merged


def write_optimism_plot(path: Path, frame: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["mean_phase_tv_to_proportional"],
            y=frame["actual_minus_predicted"],
            mode="markers+text",
            text=frame["validated_name"],
            textposition="top center",
            marker={
                "size": 12,
                "color": frame["max_simulated_epoch"],
                "colorscale": "RdYlGn_r",
                "showscale": True,
                "colorbar": {"title": "max epoch"},
            },
            hovertemplate=(
                "candidate=%{text}<br>TV to prop=%{x:.3f}<br>actual-pred=%{y:.3f}"
                "<br>max epoch=%{marker.color:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Validated Table-9 DSP optimism vs extrapolation",
        xaxis_title="Mean phase TV to proportional",
        yaxis_title="Actual BPB - predicted BPB (positive = optimistic)",
        template="plotly_white",
    )
    fig.write_html(path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def qsplit_within_aggregate_diagnostic(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_path, low_memory=False)
    phase_columns = [column for column in panel.columns if column.startswith("phase_0_")]
    domains = [column.removeprefix("phase_0_") for column in phase_columns]
    weights = panel[[f"phase_{phase}_{domain}" for phase in (0, 1) for domain in domains]].to_numpy(dtype=float)
    weights = dsp.normalize_weights(weights.reshape(len(panel), 2, len(domains)))
    aggregate = np.einsum("p,npd->nd", base.PHASE_FRACTIONS, weights)
    panel = panel.copy()
    panel["phase_tv"] = 0.5 * np.abs(weights[:, 0, :] - weights[:, 1, :]).sum(axis=1)
    pred_path = DEFAULT_EXPANDED_PANEL.parent / "expanded_300m_table9_cv_predictions.csv"
    pred = pd.read_csv(pred_path)[["run_name", "aggregate_effective_exposure_dsp_l2_0p0001"]]
    panel = panel.merge(pred, on="run_name", how="left", validate="one_to_one")
    qsplit = panel[panel["diagnostic_group"].eq("old_280_qsplit_signal")].copy()
    offdiag = qsplit[qsplit["phase_tv"].gt(1e-6)].copy()
    cluster_count = min(16, max(2, len(offdiag) // 12))
    labels = KMeans(n_clusters=cluster_count, random_state=0, n_init=25).fit_predict(aggregate[offdiag.index])
    offdiag["aggregate_cluster"] = labels
    rows: list[dict[str, float | int | str]] = []
    for cluster, group in offdiag.groupby("aggregate_cluster", sort=True):
        if len(group) < 8:
            continue
        rho = float(
            spearmanr(group[TARGET_COL], group["aggregate_effective_exposure_dsp_l2_0p0001"]).statistic
            if group[TARGET_COL].std() > 0.0 and group["aggregate_effective_exposure_dsp_l2_0p0001"].std() > 0.0
            else np.nan
        )
        rows.append(
            {
                "aggregate_cluster": int(cluster),
                "n": int(len(group)),
                "spearman_pred_actual": rho,
                "actual_bpb_min": float(group[TARGET_COL].min()),
                "actual_bpb_range": float(group[TARGET_COL].max() - group[TARGET_COL].min()),
                "pred_bpb_range": float(
                    group["aggregate_effective_exposure_dsp_l2_0p0001"].max()
                    - group["aggregate_effective_exposure_dsp_l2_0p0001"].min()
                ),
                "phase_tv_mean": float(group["phase_tv"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = dsp.model_from_json(json.loads(args.model.read_text()))
    domains = list(model.domain_names)

    candidate_scores = score_candidates(model, domains, args.output_dir)
    optimism = validated_optimism_table(args.validation_results, args.kl_summary)
    qsplit_clusters = qsplit_within_aggregate_diagnostic(args.expanded_panel)

    candidate_scores.to_csv(args.output_dir / "claimed_vs_realized_candidate_scores.csv", index=False)
    optimism.to_csv(args.output_dir / "validated_optimism_vs_extrapolation.csv", index=False)
    qsplit_clusters.to_csv(args.output_dir / "qsplit_within_aggregate_cluster_ranking.csv", index=False)
    write_optimism_plot(args.output_dir / "validated_optimism_vs_extrapolation.html", optimism)

    summary = {
        "candidate_scores": candidate_scores.to_dict(orient="records"),
        "optimism_spearman_vs_phase_tv": float(
            spearmanr(optimism["mean_phase_tv_to_proportional"], optimism["actual_minus_predicted"]).statistic
        )
        if len(optimism) >= 3
        else None,
        "optimism_spearman_vs_max_epoch": float(
            spearmanr(optimism["max_simulated_epoch"], optimism["actual_minus_predicted"]).statistic
        )
        if len(optimism) >= 3
        else None,
        "qsplit_cluster_mean_spearman": float(qsplit_clusters["spearman_pred_actual"].mean())
        if not qsplit_clusters.empty
        else None,
        "qsplit_cluster_weighted_spearman": float(
            np.average(qsplit_clusters["spearman_pred_actual"], weights=qsplit_clusters["n"])
        )
        if not qsplit_clusters.empty
        else None,
    }
    (args.output_dir / "failure_mode_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
