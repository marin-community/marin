# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Paper plot for the simple no-cross MCT-LRQ joint scaling law."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PAPER_ROOT = Path(__file__).resolve().parent
IMG_DIR = PAPER_ROOT / "img"
TWO_PHASE_MANY_ROOT = PAPER_ROOT.parent / "two_phase_many"
ANALYSIS_DATASET_ROOT = TWO_PHASE_MANY_ROOT / "analysis_dataset"
V31_PACKET_ROOT = TWO_PHASE_MANY_ROOT / "chatgpt_pro_hybrid_data_mixing_packet_v31"
STRUCTURAL_S2_SCRIPT = (
    V31_PACKET_ROOT / "DO_NOT_READ_FIRST_REFERENCE_FORMS" / "structural_s2" / "code" / "run_joint_mixture_scale_law.py"
)
CBS_LRQ_SCRIPT = (
    TWO_PHASE_MANY_ROOT / "reference_outputs" / "mct_family_n_head_ablation_20260426" / "code" / "cbs_lrq_base.py"
)

OUTPUT_STEM = IMG_DIR / "f10_joint_law_prediction"
OUTPUT_CSV = IMG_DIR / "f10_joint_law_prediction_points.csv"
OUTPUT_METRICS_CSV = IMG_DIR / "f10_joint_law_prediction_metrics.csv"
OUTPUT_FORMULA_CSV = IMG_DIR / "f10_joint_law_prediction_formula.csv"

MODEL_NAME = "no_cross"
TEXT_COLOR = "#232B32"
GRID_COLOR = "#E6E2DA"
AXIS_COLOR = "#A8A29A"
REFERENCE_COLOR = "#8F6B38"
TRAIN_EDGE_COLOR = "#938D84"
HELDOUT_EDGE_COLOR = "#1A9850"

LEAVEOUT_PROTOCOL = "leave900out"
MCT_DROP_ALPHA = 0.154791
MCT_DROP_BETA = 0.146425
MCT_PAIR_WEIGHT = 4.0
METHOD_LABELS = {
    "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_1_2b_24b": "GRP no-L2",
    "baseline_olmix_loglinear_uncheatable_bpb": "Olmix",
    "baseline_proportional": "Proportional",
    "baseline_unimax": "UniMax",
}
METHOD_MARKERS = {"GRP no-L2": "D", "Olmix": "P", "Proportional": "s", "UniMax": "^"}
METHOD_ORDER = ("GRP no-L2", "Olmix", "Proportional", "UniMax")


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": AXIS_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_current_packet_data():
    s2_mod = _import_module("f10_structural_s2", STRUCTURAL_S2_SCRIPT)
    cbs = _import_module("f10_cbs_lrq_base", CBS_LRQ_SCRIPT)
    runs = pd.read_csv(ANALYSIS_DATASET_ROOT / "nd_scale_runs.csv", low_memory=False)
    payload = np.load(ANALYSIS_DATASET_ROOT / "nd_scale_packet.npz", allow_pickle=True)
    weights = np.asarray(payload["weights"], dtype=float)
    labels = np.asarray(payload["primary_y"], dtype=float)
    mask = np.asarray(payload["primary_y_mask"], dtype=bool)
    if not np.all(mask):
        raise ValueError("F10 expects all canonical modeling rows to have primary BPB labels.")

    domains = [str(value) for value in payload["domain_names"]]
    scale_names = np.asarray([str(value) for value in payload["scale_names"]], dtype=object)
    scale_index = np.asarray(payload["scale_index"], dtype=int)
    scale_by_row = scale_names[scale_index]
    mixture_ids = np.asarray([str(value) for value in payload["mixture_ids"]], dtype=object)
    registry_keys = np.asarray([str(value) for value in payload["registry_run_keys"]], dtype=object)
    model_sizes = np.asarray(payload["model_sizes"], dtype=float)
    realized_tokens = np.asarray(payload["realized_train_tokens"], dtype=float)
    multipliers = np.asarray(payload["target_budget_multipliers"], dtype=float)
    sim_epoch = np.asarray(payload["simulated_epoch_multipliers"], dtype=float)

    all900_holdout = scale_by_row == "1_2b_24b"
    all900_train = ~all900_holdout
    reference_mask = (scale_by_row == "300m_6b") & np.isclose(multipliers, 1.0)
    if not np.any(reference_mask):
        raise ValueError("Cannot find 300M/6B anchor rows for F10 fit.")

    data = s2_mod.PacketData(
        root=V31_PACKET_ROOT,
        runs=runs,
        W=weights,
        y=labels,
        domains=domains,
        scale_names_by_row=scale_by_row,
        scale_index=scale_index,
        mixture_ids=mixture_ids,
        registry_keys=registry_keys,
        N=model_sizes,
        D=realized_tokens,
        mu=multipliers,
        sim_epoch=sim_epoch,
        base_epoch=np.median(sim_epoch[np.isclose(multipliers, 1.0)], axis=0),
        seed7_train=np.ones(len(labels), dtype=bool),
        seed7_holdout=np.zeros(len(labels), dtype=bool),
        fixed340=np.zeros(len(labels), dtype=bool),
        random_supplement=np.zeros(len(labels), dtype=bool),
        all900_train=all900_train,
        all900_holdout=all900_holdout,
        N0=float(np.unique(model_sizes[reference_mask])[0]),
        D0=float(np.unique(realized_tokens[reference_mask])[0]),
    )
    return s2_mod, cbs, data


class NoCrossLRQPowerLaw:
    """LRQ anchor plus monotone Chinchilla Approach 3 scale terms."""

    def __init__(self, s2_mod, data, ff) -> None:
        self.s2_mod = s2_mod
        self.data = data
        self.ff = ff
        self.alpha = MCT_DROP_ALPHA
        self.beta = MCT_DROP_BETA
        self.pair_weight = MCT_PAIR_WEIGHT
        self.ridge_anchor = 1e-4
        self.ridge_scale = 1e-5
        self.anchor_kind = "lrq_scarcity"

    def scale_terms(self) -> tuple[np.ndarray, np.ndarray]:
        u_n = (self.data.N / self.data.N0) ** (-self.alpha) - 1.0
        u_d = (self.data.D / self.data.D0) ** (-self.beta) - 1.0
        return u_n, u_d

    def make_pairs(self, train_mask: np.ndarray) -> list[tuple[int, int]]:
        frame = pd.DataFrame(
            {
                "idx": np.arange(len(train_mask)),
                "scale": self.data.scale_names_by_row,
                "mixture_id": self.data.mixture_ids,
                "D": self.data.D,
            }
        )
        frame = frame[np.asarray(train_mask, dtype=bool)]
        pairs: list[tuple[int, int]] = []
        for _, group in frame.groupby(["scale", "mixture_id"]):
            if group["D"].nunique() < 2:
                continue
            rows = group.sort_values("D")["idx"].to_numpy()
            for first in range(len(rows)):
                for second in range(first + 1, len(rows)):
                    pairs.append((int(rows[first]), int(rows[second])))
        return pairs

    def fit(self, train_mask: np.ndarray):
        self.train_mask = np.asarray(train_mask, dtype=bool).copy()
        x_anchor, anchor_names = self.ff.anchor_features(self.data.W, self.anchor_kind)
        anchor_rows = self.train_mask & (self.data.scale_names_by_row == "300m_6b") & np.isclose(self.data.mu, 1.0)
        anchor_coef, anchor_stats = self.s2_mod.ridge_fit(
            x_anchor[anchor_rows], self.data.y[anchor_rows], ridge=self.ridge_anchor
        )
        anchor_pred = self.s2_mod.ridge_predict_from_stats(x_anchor, anchor_coef, anchor_stats)

        head_n = self.ff.scale_head_features(self.data.W, "constant")
        head_d = self.ff.scale_head_features(self.data.W, "family")
        u_n, u_d = self.scale_terms()
        design = np.column_stack([head_n * u_n[:, None], head_d * u_d[:, None]])
        target = self.data.y - anchor_pred

        rows = np.where(self.train_mask)[0]
        fit_design = design[rows]
        fit_target = target[rows]
        pair_design: list[np.ndarray] = []
        pair_target: list[float] = []
        for first, second in self.make_pairs(self.train_mask):
            weight = np.sqrt(self.pair_weight)
            pair_design.append(weight * (design[first] - design[second]))
            pair_target.append(weight * (target[first] - target[second]))
        if pair_design:
            fit_design = np.vstack([fit_design, np.asarray(pair_design)])
            fit_target = np.concatenate([fit_target, np.asarray(pair_target)])
        if self.ridge_scale > 0:
            fit_design = np.vstack([fit_design, np.sqrt(self.ridge_scale) * np.eye(design.shape[1])])
            fit_target = np.concatenate([fit_target, np.zeros(design.shape[1])])

        result = lsq_linear(fit_design, fit_target, bounds=(0.0, np.inf), max_iter=2000, lsmr_tol="auto")
        self.anchor_coef = anchor_coef
        self.anchor_stats = anchor_stats
        self.anchor_names = anchor_names
        self.anchor_pred = anchor_pred
        self.scale_coef = np.clip(result.x, 0.0, np.inf)
        self.d_n = head_n.shape[1]
        self.d_d = head_d.shape[1]
        self.fit_success = bool(result.success)
        self.fit_cost = float(result.cost)
        self.pair_count = len(pair_design)
        self.anchor_feature_count = x_anchor.shape[1] + 1
        self.scale_param_count = len(self.scale_coef)
        self.fitted_param_count = self.anchor_feature_count + self.scale_param_count + 2
        self.total_constant_count = self.fitted_param_count + 9
        return self

    def predict_all(self) -> np.ndarray:
        head_n = self.ff.scale_head_features(self.data.W, "constant")
        head_d = self.ff.scale_head_features(self.data.W, "family")
        u_n, u_d = self.scale_terms()
        design = np.column_stack([head_n * u_n[:, None], head_d * u_d[:, None]])
        return self.anchor_pred + design @ self.scale_coef


def _fit_leave900_current() -> pd.DataFrame:
    s2_mod, cbs, data = _load_current_packet_data()
    base_ff = s2_mod.FeatureFactory(data)
    ff = cbs.LRQFeatureFactory(base_ff)
    model = NoCrossLRQPowerLaw(s2_mod, data, ff).fit(data.all900_train)
    predictions = model.predict_all()
    out = cbs.prediction_frame(data, data.runs, MODEL_NAME, LEAVEOUT_PROTOCOL, predictions)
    out["predicted_bpb"] = out["pred_bpb"]
    out["residual"] = out["residual_pred_minus_actual"]
    out["all900_holdout"] = out["all900_holdout"].astype(bool)
    out["plot_role"] = np.where(out["all900_holdout"], "Held-out 1.2B/24B", "Fit rows (non-1.2B)")
    out["method_label"] = out["mixture_id"].map(METHOD_LABELS).fillna(out["mixture_id"])
    out["method_order"] = out["method_label"].map({name: index for index, name in enumerate(METHOD_ORDER)}).fillna(99)
    out["fit_rows"] = int(data.all900_train.sum())
    out["heldout_rows"] = int(data.all900_holdout.sum())
    out["params_fitted"] = int(model.fitted_param_count)
    out["params_total"] = int(model.total_constant_count)
    out["alpha"] = float(model.alpha)
    out["beta"] = float(model.beta)
    out["N0"] = float(data.N0)
    out["D0"] = float(data.D0)
    out["A_scalar"] = float(model.scale_coef[0])
    b_coef = model.scale_coef[model.d_n :]
    b_names = ["intercept"] + [f"p{phase}_{family}" for phase in range(2) for family in ff.family_names]
    if len(b_coef) != len(b_names):
        raise ValueError(f"Unexpected B-head width: got {len(b_coef)}, expected {len(b_names)}")
    for name, value in zip(b_names, b_coef, strict=True):
        out[f"B_{name}"] = float(value)
    out["fit_success"] = bool(model.fit_success)
    out["pair_count"] = int(model.pair_count)
    return out


def _plot_identity(ax: plt.Axes, values: pd.Series) -> None:
    low = float(values.min())
    high = float(values.max())
    pad = 0.04 * (high - low)
    low -= pad
    high += pad
    ax.plot([low, high], [low, high], color=REFERENCE_COLOR, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)


def _plot_predictions(ax: plt.Axes, predictions: pd.DataFrame) -> None:
    all_values = pd.concat([predictions["actual_bpb"], predictions["predicted_bpb"]], ignore_index=True)
    _plot_identity(ax, all_values)

    train = predictions.loc[~predictions["all900_holdout"].astype(bool)]
    heldout = predictions.loc[predictions["all900_holdout"].astype(bool)].copy()
    cmap = plt.get_cmap("RdYlGn_r")
    norm = plt.Normalize(float(predictions["actual_bpb"].min()), float(predictions["actual_bpb"].max()))
    train_scatter = ax.scatter(
        train["actual_bpb"],
        train["predicted_bpb"],
        s=14,
        c=train["actual_bpb"],
        cmap=cmap,
        norm=norm,
        edgecolors=TRAIN_EDGE_COLOR,
        linewidths=0.12,
        alpha=0.16,
        label="Fit rows (non-900M)",
        zorder=2,
    )

    for label in METHOD_ORDER:
        split = heldout.loc[heldout["method_label"].eq(label)]
        if not split.empty:
            ax.scatter(
                split["actual_bpb"],
                split["predicted_bpb"],
                s=95,
                c=split["actual_bpb"],
                cmap=cmap,
                norm=norm,
                marker=METHOD_MARKERS[label],
                edgecolors=HELDOUT_EDGE_COLOR,
                linewidths=1.05,
                label=f"Held-out {label}",
                zorder=4,
            )

    heldout_spearman = float(spearmanr(heldout["actual_bpb"], heldout["predicted_bpb"]).statistic)
    train_spearman = float(spearmanr(train["actual_bpb"], train["predicted_bpb"]).statistic)
    ax.text(
        0.03,
        0.96,
        (
            f"fit rows: {len(train)}\nheld out: {len(heldout)}\n"
            f"1.2B Spearman: {heldout_spearman:.2f}\nfit Spearman: {train_spearman:.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": AXIS_COLOR, "alpha": 0.88},
    )
    ax.set_title("Leave-1.2B-out BPB predictions", fontsize=11.5, pad=7)
    ax.set_xlabel("Actual BPB")
    ax.set_ylabel("Predicted BPB")
    return train_scatter


def _add_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#AFAAA2",
            markeredgecolor=TRAIN_EDGE_COLOR,
            alpha=0.35,
            markersize=5.8,
            label="Fit rows (non-1.2B)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor=HELDOUT_EDGE_COLOR,
            markeredgewidth=0.9,
            markersize=6.2,
            label="Held-out 1.2B/24B",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        edgecolor=AXIS_COLOR,
        fontsize=7.7,
    )


def _method_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[label],
            color="none",
            markerfacecolor="white",
            markeredgecolor=HELDOUT_EDGE_COLOR,
            markeredgewidth=0.9,
            markersize=6.4,
            label=label,
        )
        for label in METHOD_ORDER
    ]


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()
    predictions = _fit_leave900_current()
    predictions.to_csv(OUTPUT_CSV, index=False)
    train = predictions.loc[~predictions["all900_holdout"]]
    heldout = predictions.loc[predictions["all900_holdout"]]
    metrics = pd.DataFrame(
        [
            {
                "split": "non900_train",
                "n": len(train),
                "spearman": float(spearmanr(train["actual_bpb"], train["predicted_bpb"]).statistic),
                "rmse": float(np.sqrt(np.mean((train["predicted_bpb"] - train["actual_bpb"]) ** 2))),
                "bias_pred_minus_actual": float((train["predicted_bpb"] - train["actual_bpb"]).mean()),
            },
            {
                "split": "all900_leave_scale_out",
                "n": len(heldout),
                "spearman": float(spearmanr(heldout["actual_bpb"], heldout["predicted_bpb"]).statistic),
                "rmse": float(np.sqrt(np.mean((heldout["predicted_bpb"] - heldout["actual_bpb"]) ** 2))),
                "bias_pred_minus_actual": float((heldout["predicted_bpb"] - heldout["actual_bpb"]).mean()),
            },
        ]
    )
    metrics.to_csv(OUTPUT_METRICS_CSV, index=False)
    formula = predictions[
        [
            "params_fitted",
            "params_total",
            "alpha",
            "beta",
            "N0",
            "D0",
            "A_scalar",
            "B_intercept",
            "B_p0_broad_text",
            "B_p0_tech_code",
            "B_p0_reasoning",
            "B_p1_broad_text",
            "B_p1_tech_code",
            "B_p1_reasoning",
            "fit_success",
            "pair_count",
        ]
    ].drop_duplicates()
    if len(formula) != 1:
        raise ValueError(f"Expected exactly one formula row, found {len(formula)}")
    formula.to_csv(OUTPUT_FORMULA_CSV, index=False)

    fig, ax = plt.subplots(1, 1, figsize=(5.25, 4.15), constrained_layout=False)
    scatter = _plot_predictions(ax, predictions)
    method_legend = ax.legend(
        handles=_method_handles(),
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        edgecolor=AXIS_COLOR,
        fontsize=7.8,
    )
    ax.add_artist(method_legend)
    _add_legend(ax)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
    colorbar.set_label("Actual BPB", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.90)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_METRICS_CSV}")
    print(f"Wrote {OUTPUT_FORMULA_CSV}")


if __name__ == "__main__":
    main()
