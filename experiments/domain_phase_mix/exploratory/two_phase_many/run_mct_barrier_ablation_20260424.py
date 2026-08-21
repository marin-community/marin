#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ablate the MCT-LRQ compatibility barrier.

This is a local diagnostic for the Session-12 MCT-LRQ structural model.  It
keeps the LRQ anchor and tuned Chinchilla-style exponents fixed, then compares:

1. the archived hand-coded compatibility barrier,
2. the same law with no barrier, and
3. a generic observed-support deployment prior that is used only for optimum
   search diagnostics.

The goal is to determine whether the hand-coded P(w) term is carrying the
result or can be removed/replaced by a more general deployment regularizer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MCT_CODE_DIR = Path(
    "/tmp/chatgpt_pro_session_12_review/"
    "5_joint_mixture_scale_structural_v31_mct/"
    "joint_mixture_scale_structural_v31_mct/code"
)

MCT_BALANCED_EXPONENTS = (0.148968, 0.209383, 0.009859, 1.043436)
MCT_DROP_EXPONENTS = (0.154791, 0.146425, 0.014295, 1.063376)
MCT_PAIR_WEIGHT = 4.0
FAMILY_BARRIER_THRESHOLD = 0.85
FAMILY_BARRIER_STRENGTH = 3.5
FAMILY_COLLAPSE_THRESHOLD = 0.98
FAMILY_COLLAPSE_STRENGTH = 200.0
DOMAIN_COLLAPSE_THRESHOLD = 0.60
DOMAIN_COLLAPSE_STRENGTH = 1.0


def import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def jsonify(x):
    if isinstance(x, dict):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class SupportDistanceOptimumWrapper:
    """Wrap a fitted model with a generic observed-support prior for optima.

    The wrapper deliberately leaves predictions on observed rows unchanged by
    forwarding `predict_all` to the base model.  The support penalty is only
    active when `predict_custom` is called on arbitrary mixtures, which is how
    the existing raw optimum probe searches the simplex.
    """

    def __init__(self, base_model, support_weights: np.ndarray, tv_radius: float, penalty_strength: float):
        self.base_model = base_model
        self.model_id = f"{base_model.model_id}_support_tv"
        self.support_weights = np.asarray(support_weights, dtype=float)
        self.tv_radius = float(tv_radius)
        self.penalty_strength = float(penalty_strength)

        for name in [
            "data",
            "ff",
            "anchor_feature_count",
            "scale_param_count",
            "donor_constant_count",
            "barrier_constant_count",
            "fitted_param_count",
            "total_constant_count",
            "pair_weight",
            "scale_coef",
            "dA",
            "dB",
            "dC",
        ]:
            if hasattr(base_model, name):
                setattr(self, name, getattr(base_model, name))

    def nearest_support_tv(self, W: np.ndarray) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        out = np.empty(W.shape[0], dtype=float)
        chunk = 512
        for start in range(0, W.shape[0], chunk):
            part = W[start : start + chunk]
            tv = 0.25 * np.abs(part[:, None, :, :] - self.support_weights[None, :, :, :]).sum(axis=(2, 3))
            out[start : start + chunk] = tv.min(axis=1)
        return out

    def offset(self, W: np.ndarray) -> np.ndarray:
        excess = np.maximum(0.0, self.nearest_support_tv(W) - self.tv_radius)
        return self.penalty_strength * excess**2

    def predict_all(self):
        return self.base_model.predict_all()

    def predict_custom(self, W: np.ndarray, N: np.ndarray | float, D: np.ndarray | float) -> np.ndarray:
        return self.base_model.predict_custom(W, N, D) + self.offset(W)

    def head_values(self, W: np.ndarray) -> pd.DataFrame:
        return self.base_model.head_values(W)

    def artifact(self) -> dict:
        art = self.base_model.artifact()
        art["model_id"] = self.model_id
        art["support_distance_wrapper"] = {
            "formula": "base_prediction + penalty_strength * relu(nearest_observed_phase_mean_tv(w) - tv_radius)^2",
            "tv_radius": self.tv_radius,
            "penalty_strength": self.penalty_strength,
            "support_count": int(self.support_weights.shape[0]),
            "prediction_protocol": (
                "predict_all forwards the base model; this wrapper is for raw-optimum deployment diagnostics"
            ),
        }
        return art


class FamilyConcentrationBarrierPowerLaw:
    """Generic family-concentration barrier around the MCT-LRQ law.

    This keeps the MCT functional form but replaces the hand-coded
    p0-reasoning/p1-tech compatibility barrier with a simpler prior:

        P_family(w) = strength * sum_phase relu(max_family_share_phase - threshold)^2.

    It is still a prior, but it is not specific to a particular family pair.
    """

    def __init__(
        self,
        cbs,
        s2_mod,
        data,
        ff,
        model_id: str,
        exponents,
        threshold: float = FAMILY_BARRIER_THRESHOLD,
        strength: float = FAMILY_BARRIER_STRENGTH,
        domain_threshold: float | None = None,
        domain_strength: float = 0.0,
    ):
        self.base = cbs.CompatibilityBarrierPowerLaw(
            s2_mod,
            model_id=model_id,
            data=data,
            ff=ff,
            strength=0.0,
            anchor_kind="lrq_scarcity",
            exponents=exponents,
            pair_weight=MCT_PAIR_WEIGHT,
            ridge_anchor=1e-4,
            ridge_scale=1e-5,
            head_A="constant",
            head_B="family",
            head_C="constant",
        )
        self.model_id = model_id
        self.data = data
        self.ff = ff
        self.threshold = float(threshold)
        self.strength = float(strength)
        self.domain_threshold = None if domain_threshold is None else float(domain_threshold)
        self.domain_strength = float(domain_strength)
        self.family_barrier_constant_count = 2 + (
            2 if self.domain_threshold is not None and self.domain_strength > 0 else 0
        )

    def offset(self, W: np.ndarray) -> np.ndarray:
        fam = self.ff.family_shares(W)
        p0_max = fam[:, :3].max(axis=1)
        p1_max = fam[:, 3:].max(axis=1)
        excess = np.maximum(0.0, p0_max - self.threshold) ** 2 + np.maximum(0.0, p1_max - self.threshold) ** 2
        penalty = self.strength * excess
        if self.domain_threshold is not None and self.domain_strength > 0:
            max_domain = np.maximum(W[:, 0, :].max(axis=1), W[:, 1, :].max(axis=1))
            penalty = penalty + self.domain_strength * np.maximum(0.0, max_domain - self.domain_threshold) ** 2
        return penalty

    def fit(self, train_mask: np.ndarray):
        original_offset = self.base.offset
        self.base.offset = self.offset
        try:
            self.base.barrier_constant_count = self.family_barrier_constant_count
            self.base.fit(train_mask)
        finally:
            self.base.offset = original_offset
        for name, value in self.base.__dict__.items():
            if name not in {"offset", "strength"}:
                setattr(self, name, value)
        self.model_id = self.base.model_id
        self.threshold = float(self.threshold)
        self.strength = float(self.strength)
        self.domain_strength = float(self.domain_strength)
        self.total_constant_count = int(
            self.fitted_param_count + self.donor_constant_count + self.family_barrier_constant_count
        )
        self.barrier_constant_count = self.family_barrier_constant_count
        return self

    def predict_custom(self, W: np.ndarray, N: np.ndarray | float, D: np.ndarray | float) -> np.ndarray:
        return self.base.predict_custom(W, N, D)

    def predict_all(self):
        return self.base.predict_all()

    def head_values(self, W: np.ndarray) -> pd.DataFrame:
        return self.base.head_values(W)

    def artifact(self) -> dict:
        art = self.base.artifact()
        art["model_id"] = self.model_id
        art["barrier"] = {
            "kind": "family_concentration",
            "formula": (
                "strength * (relu(max_phase0_family_share - threshold)^2 + relu(max_phase1_family_share - threshold)^2)"
            ),
            "threshold": self.threshold,
            "strength": self.strength,
            "domain_threshold": self.domain_threshold,
            "domain_strength": self.domain_strength,
            "constant_count": self.family_barrier_constant_count,
        }
        art["total_constant_count_including_donor_and_barrier_constants"] = self.total_constant_count
        return art


def md_table(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = ".6f") -> str:
    if cols is not None:
        df = df[cols].copy()
    if len(df) == 0:
        return "_empty_"
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def make_powerlaw(cbs, s2_mod, data, ff, model_id: str, train_mask: np.ndarray, exponents, strength: float):
    return cbs.CompatibilityBarrierPowerLaw(
        s2_mod,
        model_id=model_id,
        data=data,
        ff=ff,
        strength=strength,
        anchor_kind="lrq_scarcity",
        exponents=exponents,
        pair_weight=MCT_PAIR_WEIGHT,
        ridge_anchor=1e-4,
        ridge_scale=1e-5,
        head_A="constant",
        head_B="family",
        head_C="constant",
    ).fit(train_mask)


def metric_row(cbs, data, model_name: str, fit_protocol: str, split: str, mask: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "model": model_name,
        "fit_protocol": fit_protocol,
        "split": split,
        **cbs.metric_dict(data.y[mask], pred[mask]),
    }


def evaluate_models(cbs, data, seed_models: dict[str, object], leave_models: dict[str, object]):
    rows = []
    preds = []
    for name, model in seed_models.items():
        pred = model.predict_all()
        for split, mask in [
            ("seed7_train", data.seed7_train),
            ("seed7_holdout", data.seed7_holdout),
            ("fixed340_holdout", data.fixed340),
            ("random_supplement", data.random_supplement),
        ]:
            rows.append(metric_row(cbs, data, name, "seed7", split, mask, pred))
        preds.append(cbs.prediction_frame(data, data.runs, name, "seed7", pred))
    for name, model in leave_models.items():
        pred = model.predict_all()
        for split, mask in [("non900_train", data.all900_train), ("all900_leave_scale_out", data.all900_holdout)]:
            rows.append(metric_row(cbs, data, name, "leave900out", split, mask, pred))
        preds.append(cbs.prediction_frame(data, data.runs, name, "leave900out", pred))
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def summarize_barrier_values(data, models: dict[str, object]) -> pd.DataFrame:
    rows = []
    masks = {
        "all_rows": np.ones(len(data.y), dtype=bool),
        "seed7_train": data.seed7_train,
        "seed7_holdout": data.seed7_holdout,
        "fixed340_holdout": data.fixed340,
        "random_supplement": data.random_supplement,
        "all900_holdout": data.all900_holdout,
    }
    for name, model in models.items():
        vals = model.offset(data.W) if hasattr(model, "offset") else np.zeros(len(data.y))
        for split, mask in masks.items():
            sub = vals[mask]
            rows.append(
                {
                    "model": name,
                    "split": split,
                    "n": len(sub),
                    "nonzero_count": int(np.sum(sub > 1e-12)),
                    "nonzero_frac": float(np.mean(sub > 1e-12)) if len(sub) else np.nan,
                    "max_offset": float(np.max(sub)) if len(sub) else np.nan,
                    "mean_offset": float(np.mean(sub)) if len(sub) else np.nan,
                    "p95_offset": float(np.quantile(sub, 0.95)) if len(sub) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def parameter_counts(models: dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        rows.append(
            {
                "model": name,
                "fitted_param_count_counting_exponents": int(getattr(model, "fitted_param_count", 0)),
                "total_constant_count": int(getattr(model, "total_constant_count", 0)),
                "anchor_feature_count_including_intercept": int(getattr(model, "anchor_feature_count", 0)),
                "scale_param_count": int(getattr(model, "scale_param_count", 0)),
                "donor_constant_count": int(getattr(model, "donor_constant_count", 0)),
                "barrier_constant_count": int(getattr(model, "barrier_constant_count", 0)),
            }
        )
    return pd.DataFrame(rows)


def plot_metric_bars(core: pd.DataFrame, outpath: Path):
    pivot = core.pivot_table(index="model", columns="split", values="rmse", aggfunc="first")
    order = [
        c
        for c in ["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_leave_scale_out"]
        if c in pivot.columns
    ]
    pivot = pivot[order]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(pivot))
    width = 0.18
    cmap = plt.get_cmap("RdYlGn_r")
    for i, split in enumerate(order):
        ax.bar(
            x + (i - (len(order) - 1) / 2) * width,
            pivot[split],
            width=width,
            label=split,
            color=cmap(i / max(1, len(order) - 1)),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=18, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Barrier ablation prediction RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_optimum_families(opt: pd.DataFrame, outpath: Path):
    sub = opt[
        (opt["target_scale"] == "340M/10.4B")
        & (opt["opt_kind"].isin(["raw_random_search", "top8actual_hull_random_search"]))
    ].copy()
    family_cols = [
        "p0_broad_text_share",
        "p0_tech_code_share",
        "p0_reasoning_share",
        "p1_broad_text_share",
        "p1_tech_code_share",
        "p1_reasoning_share",
    ]
    labels = [x.replace("_share", "").replace("_", "\n") for x in family_cols]
    fig, axes = plt.subplots(1, len(sub), figsize=(max(9, 2.2 * len(sub)), 4.4), sharey=True)
    if len(sub) == 1:
        axes = [axes]
    cmap = plt.get_cmap("RdYlGn_r")
    for ax, (_, row) in zip(axes, sub.iterrows(), strict=False):
        vals = [float(row[c]) for c in family_cols]
        colors = [cmap(v) for v in np.linspace(0.15, 0.85, len(vals))]
        ax.bar(np.arange(len(vals)), vals, color=colors)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{row['model']}\n{row['opt_kind']}", fontsize=8)
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].set_ylabel("family share")
    fig.suptitle("340M/10.4B optimum family shares")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_pred_actual(predictions: pd.DataFrame, outpath: Path):
    splits = [
        ("fixed340_holdout", "Fixed 340M"),
        ("all900_holdout", "All 900M"),
    ]
    models = list(predictions["model"].drop_duplicates())
    fig, axes = plt.subplots(len(splits), len(models), figsize=(4.2 * len(models), 4.0 * len(splits)), squeeze=False)
    for r, (mask_col, split_label) in enumerate(splits):
        for c, model in enumerate(models):
            ax = axes[r, c]
            fit = "leave900out" if mask_col == "all900_holdout" else "seed7"
            df = predictions[(predictions["model"] == model) & (predictions["fit_protocol"] == fit)]
            bg = df[~df[mask_col]]
            fg = df[df[mask_col]]
            ax.scatter(
                bg["actual_bpb"], bg["pred_bpb"], s=12, alpha=0.15, color="0.55", label="background" if c == 0 else None
            )
            ax.scatter(fg["actual_bpb"], fg["pred_bpb"], s=34, alpha=0.9, label=split_label if c == 0 else None)
            lo = min(float(df["actual_bpb"].min()), float(df["pred_bpb"].min()))
            hi = max(float(df["actual_bpb"].max()), float(df["pred_bpb"].max()))
            ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
            ax.set_title(model, fontsize=9)
            ax.set_xlabel("actual BPB")
            if c == 0:
                ax.set_ylabel(f"{split_label}\npredicted BPB")
            ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_report(
    outdir: Path,
    metrics: pd.DataFrame,
    drop_summary: pd.DataFrame,
    opt: pd.DataFrame,
    barrier_summary: pd.DataFrame,
    params: pd.DataFrame,
    monotone: pd.DataFrame,
):
    core = metrics[
        metrics["split"].isin(["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_leave_scale_out"])
    ].copy()
    metric_cols = [
        "model",
        "fit_protocol",
        "split",
        "n",
        "rmse",
        "spearman",
        "bias_pred_minus_actual",
        "slope_pred_on_actual",
        "std_ratio",
        "low_tail_rmse",
    ]
    opt_cols = [
        "model",
        "target_scale",
        "opt_kind",
        "predicted_bpb",
        "hard_corner_flag",
        "phase1_tech_collapse_flag",
        "any_family_collapse_flag",
        "nearest_observed_phase_mean_tv",
        "barrier_value",
        "p0_broad_text_share",
        "p0_tech_code_share",
        "p0_reasoning_share",
        "p1_broad_text_share",
        "p1_tech_code_share",
        "p1_reasoning_share",
    ]
    lines = [
        "# MCT-LRQ Barrier Ablation",
        "",
        "Date: 2026-04-24",
        "",
        "## Conclusion",
        "",
        (
            "`P(w)` is not carrying the predictive result. On the ordinary held-out modeling rows its offset "
            "is zero, and removing it barely changes seed-7, fixed-340M, random-supplement, or leave-900M RMSE."
        ),
        "",
        (
            "However, `P(w)` is carrying an important raw-optimum safety role. Without any barrier, the raw "
            "optimum collapses onto an observed but pathological 60M mixture with phase-1 almost entirely "
            "tech. A generic observed-support TV prior does not fix this, because the pathological point is "
            "itself in observed support."
        ),
        "",
        (
            "The original barrier is therefore suspicious: it is active on only one observed training row, and "
            "that row is exactly the pathological raw optimum selected by the no-barrier model. This looks more "
            "like an ad hoc deployment safety prior than a clean law component."
        ),
        "",
        (
            "The generic replacements tested here did not solve this cleanly. A broad `0.85` "
            "family-concentration barrier is too active and damages prediction. A near-family-collapse barrier "
            "keeps prediction mostly intact but still allows unobserved hard corners. Adding max-domain-collapse "
            "also degrades transfer and still does not produce a clean raw-simplex optimum under this fit."
        ),
        "",
        (
            "Recommendation: use the barrier-free MCT law only as a predictive structural ablation, not as a "
            "raw-optimum deployment law. If raw optima matter, keep the original `P(w)` labeled as an ad hoc "
            "prior, or use constrained/hull deployment. Do not claim that current MCT has solved raw optimum "
            "quality in a clean way."
        ),
        "",
        "## Variants",
        "",
        md_table(params),
        "",
        (
            "The `support_tv` variant wraps the barrier-free model only for raw optimum search. It does not "
            "change predictions on observed rows; it adds a generic penalty for candidate mixtures farther "
            "than TV `0.15` from observed support. It is included as a negative control because it cannot "
            "reject an observed pathological mixture."
        ),
        "",
        (
            "`family_barrier` uses a broad `max_family_share > 0.85` prior and is too invasive. "
            "`family_collapse` uses a sparse `max_family_share > 0.98` prior. `generic_collapse` adds a "
            "`max_domain_weight > 0.60` prior on top. Neither sparse generic variant is a clean replacement "
            "for the original hand-coded barrier."
        ),
        "",
        "## Predictive Metrics",
        "",
        md_table(core.sort_values(["model", "fit_protocol", "split"]), cols=metric_cols),
        "",
        "## Fixed-340M Same-Mixture Drops",
        "",
        md_table(
            drop_summary.sort_values(["model", "drop_pair"]),
            cols=[
                "model",
                "drop_pair",
                "n",
                "actual_drop_mean",
                "pred_drop_mean",
                "drop_error_mean",
                "drop_ratio_mean",
                "drop_ratio_median",
                "drop_rmse",
            ],
        ),
        "",
        "## Barrier / Offset Activity On Observed Rows",
        "",
        md_table(
            barrier_summary.sort_values(["model", "split"]),
            cols=["model", "split", "n", "nonzero_count", "nonzero_frac", "max_offset", "mean_offset", "p95_offset"],
        ),
        "",
        "## Raw And Constrained Optima",
        "",
        md_table(
            opt[(opt["target_scale"].isin(["340M/10.4B", "900M/24B"]))].sort_values(
                ["model", "target_scale", "opt_kind"]
            ),
            cols=opt_cols,
        ),
        "",
        "## Monotonicity Grid",
        "",
        md_table(monotone),
        "",
        "## Artifact Map",
        "",
        "- `csv/metric_summary.csv`: split metrics for every variant.",
        "- `csv/fixed340_drop_summary.csv`: same-mixture multiplier drop ratios.",
        "- `csv/offset_activity.csv`: observed-row offset activity for the barrier and support-TV wrapper.",
        "- `csv/optimum_diagnostics.csv`: raw/hull/trustblend optimum probes.",
        "- `plots/pred_actual_barrier_ablation.png`: fixed-340M and 900M predicted-vs-actual panels.",
        "- `plots/rmse_barrier_ablation.png`: compact RMSE comparison.",
        "- `plots/optimum_family_shares_340m.png`: 340M optimum family shares.",
        "- `code/`: the local ablation runner plus the Session-12 MCT helper code used for reproduction.",
    ]
    (outdir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-root",
        type=Path,
        default=Path(
            "experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v31"
        ),
    )
    parser.add_argument(
        "--mct-code-dir",
        type=Path,
        default=MCT_CODE_DIR,
        help="Directory containing the Session-12 MCT archive code.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/mct_barrier_ablation_20260424"
        ),
    )
    args = parser.parse_args()

    code_dir = args.mct_code_dir.resolve()
    if not (code_dir / "cbs_lrq_base.py").exists():
        raise FileNotFoundError(f"Missing MCT code at {code_dir}; expected cbs_lrq_base.py")

    outdir = args.outdir.resolve()
    for subdir in ["csv", "plots", "models", "code"]:
        (outdir / subdir).mkdir(parents=True, exist_ok=True)

    cbs = import_from_path("mct_cbs_lrq_base", code_dir / "cbs_lrq_base.py")
    packet_root = cbs.packet_root_from_arg(args.packet_root)
    s2_mod = cbs.import_s2(packet_root)
    data = s2_mod.load_packet(packet_root)
    base_ff = s2_mod.FeatureFactory(data)
    ff = cbs.LRQFeatureFactory(base_ff)

    seed_models = {
        "mct_lrq74_balanced_barrier5": make_powerlaw(
            cbs, s2_mod, data, ff, "mct_lrq74_balanced_barrier5", data.seed7_train, MCT_BALANCED_EXPONENTS, strength=5.0
        ),
        "mct_lrq71_balanced_family_barrier": (
            FamilyConcentrationBarrierPowerLaw(
                cbs, s2_mod, data, ff, "mct_lrq71_balanced_family_barrier", MCT_BALANCED_EXPONENTS
            ).fit(data.seed7_train)
        ),
        "mct_lrq71_balanced_family_collapse": (
            FamilyConcentrationBarrierPowerLaw(
                cbs,
                s2_mod,
                data,
                ff,
                "mct_lrq71_balanced_family_collapse",
                MCT_BALANCED_EXPONENTS,
                threshold=FAMILY_COLLAPSE_THRESHOLD,
                strength=FAMILY_COLLAPSE_STRENGTH,
            ).fit(data.seed7_train)
        ),
        "mct_lrq73_balanced_generic_collapse": (
            FamilyConcentrationBarrierPowerLaw(
                cbs,
                s2_mod,
                data,
                ff,
                "mct_lrq73_balanced_generic_collapse",
                MCT_BALANCED_EXPONENTS,
                threshold=FAMILY_COLLAPSE_THRESHOLD,
                strength=FAMILY_COLLAPSE_STRENGTH,
                domain_threshold=DOMAIN_COLLAPSE_THRESHOLD,
                domain_strength=DOMAIN_COLLAPSE_STRENGTH,
            ).fit(data.seed7_train)
        ),
        "mct_lrq69_balanced_no_barrier": make_powerlaw(
            cbs,
            s2_mod,
            data,
            ff,
            "mct_lrq69_balanced_no_barrier",
            data.seed7_train,
            MCT_BALANCED_EXPONENTS,
            strength=0.0,
        ),
        "mct_lrq69_drop_no_barrier": make_powerlaw(
            cbs, s2_mod, data, ff, "mct_lrq69_drop_no_barrier", data.seed7_train, MCT_DROP_EXPONENTS, strength=0.0
        ),
    }
    support = data.W[data.seed7_train]
    seed_models["mct_lrq69_balanced_support_tv"] = SupportDistanceOptimumWrapper(
        seed_models["mct_lrq69_balanced_no_barrier"], support_weights=support, tv_radius=0.15, penalty_strength=0.25
    )

    leave_models = {
        "mct_lrq74_balanced_barrier5": make_powerlaw(
            cbs, s2_mod, data, ff, "mct_lrq74_balanced_barrier5", data.all900_train, MCT_BALANCED_EXPONENTS, strength=5.0
        ),
        "mct_lrq71_balanced_family_barrier": (
            FamilyConcentrationBarrierPowerLaw(
                cbs, s2_mod, data, ff, "mct_lrq71_balanced_family_barrier", MCT_BALANCED_EXPONENTS
            ).fit(data.all900_train)
        ),
        "mct_lrq71_balanced_family_collapse": (
            FamilyConcentrationBarrierPowerLaw(
                cbs,
                s2_mod,
                data,
                ff,
                "mct_lrq71_balanced_family_collapse",
                MCT_BALANCED_EXPONENTS,
                threshold=FAMILY_COLLAPSE_THRESHOLD,
                strength=FAMILY_COLLAPSE_STRENGTH,
            ).fit(data.all900_train)
        ),
        "mct_lrq73_balanced_generic_collapse": (
            FamilyConcentrationBarrierPowerLaw(
                cbs,
                s2_mod,
                data,
                ff,
                "mct_lrq73_balanced_generic_collapse",
                MCT_BALANCED_EXPONENTS,
                threshold=FAMILY_COLLAPSE_THRESHOLD,
                strength=FAMILY_COLLAPSE_STRENGTH,
                domain_threshold=DOMAIN_COLLAPSE_THRESHOLD,
                domain_strength=DOMAIN_COLLAPSE_STRENGTH,
            ).fit(data.all900_train)
        ),
        "mct_lrq69_balanced_no_barrier": make_powerlaw(
            cbs,
            s2_mod,
            data,
            ff,
            "mct_lrq69_balanced_no_barrier",
            data.all900_train,
            MCT_BALANCED_EXPONENTS,
            strength=0.0,
        ),
        "mct_lrq69_drop_no_barrier": make_powerlaw(
            cbs, s2_mod, data, ff, "mct_lrq69_drop_no_barrier", data.all900_train, MCT_DROP_EXPONENTS, strength=0.0
        ),
    }
    leave_models["mct_lrq69_balanced_support_tv"] = SupportDistanceOptimumWrapper(
        leave_models["mct_lrq69_balanced_no_barrier"],
        support_weights=data.W[data.all900_train],
        tv_radius=0.15,
        penalty_strength=0.25,
    )

    metrics, predictions = evaluate_models(cbs, data, seed_models, leave_models)
    drop_detail, drop_summary, beta_detail = cbs.fixed340_drop_tables(data, seed_models)
    opt = cbs.optimum_diagnostics(seed_models, data, ff)
    monotone = s2_mod.monotonicity_grid(seed_models, data, ff)
    barrier_summary = summarize_barrier_values(data, seed_models)
    params = parameter_counts(seed_models)

    metrics.to_csv(outdir / "csv" / "metric_summary.csv", index=False)
    predictions.to_csv(outdir / "csv" / "row_predictions.csv", index=False)
    drop_detail.to_csv(outdir / "csv" / "fixed340_drop_pairs.csv", index=False)
    drop_summary.to_csv(outdir / "csv" / "fixed340_drop_summary.csv", index=False)
    beta_detail.to_csv(outdir / "csv" / "fixed340_beta_triples.csv", index=False)
    opt.to_csv(outdir / "csv" / "optimum_diagnostics.csv", index=False)
    monotone.to_csv(outdir / "csv" / "monotonicity_grid.csv", index=False)
    barrier_summary.to_csv(outdir / "csv" / "offset_activity.csv", index=False)
    params.to_csv(outdir / "csv" / "parameter_counts.csv", index=False)

    for name, model in seed_models.items():
        artifact = model.artifact() if hasattr(model, "artifact") else {"model_id": name}
        (outdir / "models" / f"{name}_seed7.json").write_text(json.dumps(jsonify(artifact), indent=2), encoding="utf-8")

    shutil.copy2(Path(__file__), outdir / "code" / Path(__file__).name)
    for helper in ["cbs_lrq_base.py", "run_mct_lrq_law.py"]:
        shutil.copy2(code_dir / helper, outdir / "code" / helper)

    core = metrics[
        metrics["split"].isin(["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_leave_scale_out"])
    ]
    plot_metric_bars(core, outdir / "plots" / "rmse_barrier_ablation.png")
    plot_pred_actual(predictions, outdir / "plots" / "pred_actual_barrier_ablation.png")
    plot_optimum_families(opt, outdir / "plots" / "optimum_family_shares_340m.png")

    write_report(outdir, metrics, drop_summary, opt, barrier_summary, params, monotone)

    manifest = {
        "packet_root": str(packet_root),
        "mct_code_dir": str(code_dir),
        "outdir": str(outdir),
        "selected_clean_structural_candidate": "mct_lrq69_balanced_no_barrier_for_prediction_only",
        "recommendation": (
            "The hand-coded P(w) can be ablated for prediction but not for raw optima. Current generic "
            "barriers are not clean replacements; use constrained deployment or keep P(w) explicitly labeled "
            "as an ad hoc prior."
        ),
    }
    (outdir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
