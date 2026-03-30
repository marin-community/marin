# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the CCPairTotal-RetainedTotal follow-up and plot its optimum."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    PENALTY_KIND_GROUP_LOG_THRESHOLD,
    PREMIUM_MODE_GLOBAL,
    SIGNAL_KIND_RETAINED_TOTAL,
    SIGNAL_KIND_THRESHOLD_TOTAL,
    CCPairTotalStructuredSurrogate,
    PacketData,
    evaluate_cc_model,
    evaluate_cc_pairtotal_model,
    foldwise_regret,
    load_two_phase_many_packet,
    optimize_cc_globalpremium_model,
    optimize_cc_pairtotal_model,
    regression_metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
COMPARISON_CSV = SCRIPT_DIR / "diversity_followup_comparison.csv"
OPTIMUM_CSV = SCRIPT_DIR / "ccpairtotal_retainedtotal_optimum_full.csv"
COEFFICIENTS_CSV = SCRIPT_DIR / "ccpairtotal_retainedtotal_coefficients.csv"
REPEATED_CV_CSV = SCRIPT_DIR / "ccpairtotal_retainedtotal_repeated_cv.csv"
REPORT_MD = SCRIPT_DIR / "diversity_followup_report.md"
SUMMARY_JSON = SCRIPT_DIR / "diversity_followup_summary.json"
PLOT_PNG = SCRIPT_DIR / "ccpairtotal_retainedtotal_optimum_comparison.png"

CCGLOBALPREMIUM_THRESHOLD_PARAMS = {
    "signal_kind": SIGNAL_KIND_THRESHOLD_TOTAL,
    "alpha": 8.0,
    "eta": 5.0,
    "lam": 0.0,
    "sig_tau": 0.25,
    "premium_mode": PREMIUM_MODE_GLOBAL,
    "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
    "tau": 2.0,
    "reg": 0.01,
}
CCGLOBALPREMIUM_RETAINEDTOTAL_PARAMS = {
    "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
    "alpha": 8.0,
    "eta": 3.0,
    "lam": 1.0,
    "sig_tau": 0.0,
    "premium_mode": PREMIUM_MODE_GLOBAL,
    "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
    "tau": 1.0,
    "reg": 0.01,
}
CCPAIRTOTAL_RETAINEDTOTAL_PARAMS = {
    "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
    "group_signal_kind": "log_after_sum",
    "premium_mode": PREMIUM_MODE_GLOBAL,
    "diversity_mode": "none",
    "alpha": 5.693767311270728,
    "eta": 6.323564464532408,
    "lam": 0.004606280004722357,
    "tau": 1.3976070420563144,
    "reg": 2.1562923313580245e-06,
    "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
}


def _entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _effective_support(weights: np.ndarray) -> float:
    return float(np.exp(_entropy(weights)))


def _support_stats(phase0: np.ndarray, phase1: np.ndarray) -> dict[str, float | int]:
    return {
        "phase0_below_1e6": int(np.sum(phase0 < 1e-6)),
        "phase1_below_1e6": int(np.sum(phase1 < 1e-6)),
        "phase0_below_1e4": int(np.sum(phase0 < 1e-4)),
        "phase1_below_1e4": int(np.sum(phase1 < 1e-4)),
        "phase0_entropy": _entropy(phase0),
        "phase1_entropy": _entropy(phase1),
        "phase0_effn": _effective_support(phase0),
        "phase1_effn": _effective_support(phase1),
        "phase0_max": float(np.max(phase0)),
        "phase1_max": float(np.max(phase1)),
    }


def _top_domains(
    data: PacketData, weights: np.ndarray, phase_idx: int, *, top_k: int = 12
) -> list[dict[str, float | str]]:
    multipliers = data.c0 if phase_idx == 0 else data.c1
    frame = pd.DataFrame({"domain": data.domain_names, "weight": weights, "epochs": weights * multipliers})
    return frame.sort_values(["weight", "epochs"], ascending=False).head(top_k).to_dict(orient="records")


def _domain_plot_order(domain_names: list[str]) -> list[int]:
    cc_topics = sorted(
        {
            domain_name[len("dolma3_cc/") : -len("_high")]
            for domain_name in domain_names
            if domain_name.startswith("dolma3_cc/") and domain_name.endswith("_high")
        }
    )
    name_to_index = {domain_name: idx for idx, domain_name in enumerate(domain_names)}
    cc_order: list[int] = []
    for topic in cc_topics:
        cc_order.append(name_to_index[f"dolma3_cc/{topic}_high"])
        cc_order.append(name_to_index[f"dolma3_cc/{topic}_low"])
    cc_indices = set(cc_order)
    non_cc_order = [idx for idx in range(len(domain_names)) if idx not in cc_indices]
    return non_cc_order + cc_order


def _display_domain_label(domain_name: str) -> str:
    if domain_name.startswith("dolma3_cc/"):
        topic, quality = domain_name[len("dolma3_cc/") :].rsplit("_", maxsplit=1)
        return f"cc/{topic} {quality}"
    return domain_name.replace("dolma3_", "").replace("dolmino_", "")


def _plot_optimum(
    *,
    data: PacketData,
    optimum_phase0: np.ndarray,
    optimum_phase1: np.ndarray,
    best_name: str,
    best_weights: np.ndarray,
) -> None:
    ordered_indices = _domain_plot_order(data.domain_names)
    ordered_labels = [_display_domain_label(data.domain_names[idx]) for idx in ordered_indices]
    schedules = [
        ("CCPairTotal optimum", np.stack([optimum_phase0, optimum_phase1], axis=0), plt.get_cmap("RdYlGn_r")(0.1)),
        (f"Best observed\n{best_name}", best_weights, plt.get_cmap("RdYlGn_r")(0.7)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(18, 15), sharey=True)
    bar_height = 0.34
    for phase_idx, ax in enumerate(axes):
        y = np.arange(len(ordered_indices))
        for schedule_idx, (label, schedule, color) in enumerate(schedules):
            ax.barh(
                y + (schedule_idx - 0.5) * bar_height,
                schedule[phase_idx, ordered_indices],
                height=bar_height,
                color=color,
                alpha=0.95,
                edgecolor="none",
                label=label if phase_idx == 0 else None,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(ordered_labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mixture weight")
        ax.set_title(f"Phase {phase_idx}")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

    axes[0].legend(loc="lower right", frameon=False)
    fig.suptitle("CCPairTotal-RetainedTotal optimum vs best observed run", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PLOT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _comparison_row(
    *,
    model_name: str,
    fit_row: dict[str, object],
    phase0: np.ndarray,
    phase1: np.ndarray,
    predicted_optimum_value: float,
) -> dict[str, object]:
    row = {
        "model": model_name,
        "n_params": int(fit_row["reported_n_params"]),
        "train_rmse": float(fit_row["train_rmse"]),
        "train_r2": float(fit_row["train_r2"]),
        "train_spearman": float(fit_row["train_spearman"]),
        "cv_rmse": float(fit_row["cv_rmse"]),
        "cv_r2": float(fit_row["cv_r2"]),
        "cv_spearman": float(fit_row["cv_spearman"]),
        "cv_regret_at_1": float(fit_row["cv_regret_at_1"]),
        "cv_foldmean_regret_at_1": float(fit_row["cv_foldmean_regret_at_1"]),
        "predicted_optimum_value": predicted_optimum_value,
    }
    row.update(_support_stats(phase0, phase1))
    return row


def main() -> None:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)

    threshold_row, threshold_model = evaluate_cc_model(
        data,
        "CCGlobalPremium-Threshold",
        CCGLOBALPREMIUM_THRESHOLD_PARAMS,
    )
    threshold_result, threshold_phase0, threshold_phase1 = optimize_cc_globalpremium_model(threshold_model, data, seed=0)

    retained_row, retained_model = evaluate_cc_model(
        data,
        "CCGlobalPremium-RetainedTotal",
        CCGLOBALPREMIUM_RETAINEDTOTAL_PARAMS,
    )
    retained_result, retained_phase0, retained_phase1 = optimize_cc_globalpremium_model(retained_model, data, seed=0)

    pairtotal_row, pairtotal_model = evaluate_cc_pairtotal_model(
        data,
        "CCPairTotal-RetainedTotal",
        CCPAIRTOTAL_RETAINEDTOTAL_PARAMS,
    )
    optimum_result, optimum_phase0, optimum_phase1 = optimize_cc_pairtotal_model(pairtotal_model, data, seed=0)

    repeated_cv = []
    for seed in range(10):
        cv_pred = pairtotal_model.cv_predict(data.w, data.y, n_splits=5, seed=seed)
        metrics = regression_metrics(data.frame, data.name_col, data.y, cv_pred)
        foldwise = foldwise_regret(
            data.frame,
            data.name_col,
            data.w,
            data.y,
            lambda wtr, ytr, inner_seed: (
                lambda wte: CCPairTotalStructuredSurrogate(data, CCPAIRTOTAL_RETAINEDTOTAL_PARAMS)
                .fit(wtr, ytr)
                .predict(wte)
            ),
            seed=seed,
        )
        repeated_cv.append(
            {
                "seed": seed,
                "cv_rmse": float(metrics["rmse"]),
                "cv_r2": float(metrics["r2"]),
                "cv_spearman": float(metrics["spearman"]),
                "cv_regret_at_1": float(metrics["regret_at_1"]),
                "cv_foldmean_regret_at_1": float(foldwise["cv_foldmean_regret_at_1"]),
            }
        )
    repeated_cv_frame = pd.DataFrame(repeated_cv)
    repeated_cv_frame.to_csv(REPEATED_CV_CSV, index=False)

    comparison = pd.DataFrame(
        [
            _comparison_row(
                model_name="CCGlobalPremium-Threshold",
                fit_row=threshold_row,
                phase0=threshold_phase0,
                phase1=threshold_phase1,
                predicted_optimum_value=float(threshold_result.fun),
            ),
            _comparison_row(
                model_name="CCGlobalPremium-RetainedTotal",
                fit_row=retained_row,
                phase0=retained_phase0,
                phase1=retained_phase1,
                predicted_optimum_value=float(retained_result.fun),
            ),
            _comparison_row(
                model_name="CCPairTotal-RetainedTotal",
                fit_row=pairtotal_row,
                phase0=optimum_phase0,
                phase1=optimum_phase1,
                predicted_optimum_value=float(optimum_result.fun),
            ),
        ]
    )
    comparison.to_csv(COMPARISON_CSV, index=False)

    optimum_frame = pd.DataFrame(
        {
            "domain": data.domain_names,
            "phase0_weight": optimum_phase0,
            "phase0_epochs": optimum_phase0 * data.c0,
            "phase1_weight": optimum_phase1,
            "phase1_epochs": optimum_phase1 * data.c1,
        }
    )
    optimum_frame.to_csv(OPTIMUM_CSV, index=False)
    pairtotal_model.coef_table().sort_values("coef", ascending=False).to_csv(COEFFICIENTS_CSV, index=False)

    best_idx = int(np.argmin(data.y))
    best_name = str(data.frame.iloc[best_idx][data.name_col])
    best_weights = data.w[best_idx]
    distances = average_phase_tv_distance(data.w, np.stack([optimum_phase0, optimum_phase1], axis=0)[None, :, :])
    nearest_idx = int(np.argmin(distances))
    train_metrics_json = json.dumps(
        {k.removeprefix("train_"): v for k, v in pairtotal_row.items() if k.startswith("train_")},
        indent=2,
        sort_keys=True,
    )
    cv_metrics_json = json.dumps(
        {k.removeprefix("cv_"): v for k, v in pairtotal_row.items() if k.startswith("cv_")},
        indent=2,
        sort_keys=True,
    )
    report = f"""# Diversity follow-up: grouped pair-total surrogate

Selected model:
- `CCPairTotal-RetainedTotal`

Selected parameters:
```json
{json.dumps(CCPAIRTOTAL_RETAINEDTOTAL_PARAMS, indent=2, sort_keys=True)}
```

Train metrics:
```json
{train_metrics_json}
```

5-fold CV metrics (seed 0 split):
```json
{cv_metrics_json}
```

10x repeated 5-fold CV:
- mean CV RMSE: {repeated_cv_frame["cv_rmse"].mean():.6f}
- std CV RMSE: {repeated_cv_frame["cv_rmse"].std(ddof=1):.6f}
- mean CV fold-mean Regret@1: {repeated_cv_frame["cv_foldmean_regret_at_1"].mean():.6f}
- std CV fold-mean Regret@1: {repeated_cv_frame["cv_foldmean_regret_at_1"].std(ddof=1):.6f}

Predicted optimum:
- predicted bpb: {float(optimum_result.fun):.6f}
- nearest observed run: {data.frame.iloc[nearest_idx][data.name_col]}
- nearest observed TV distance: {float(distances[nearest_idx]):.6f}
- phase 0 max weight: {float(np.max(optimum_phase0)):.6f}
- phase 1 max weight: {float(np.max(optimum_phase1)):.6f}
- phase 0 effective support: {_effective_support(optimum_phase0):.3f}
- phase 1 effective support: {_effective_support(optimum_phase1):.3f}
- phase 0 weights below 1e-6: {int(np.sum(optimum_phase0 < 1e-6))}
- phase 1 weights below 1e-6: {int(np.sum(optimum_phase1 < 1e-6))}
- phase 0 weights below 1e-4: {int(np.sum(optimum_phase0 < 1e-4))}
- phase 1 weights below 1e-4: {int(np.sum(optimum_phase1 < 1e-4))}

Observed best run:
```json
{json.dumps({
    "run_name": best_name,
    "value": float(data.y[best_idx]),
    **_support_stats(best_weights[0], best_weights[1]),
}, indent=2, sort_keys=True)}
```

Top phase-0 domains:
```json
{json.dumps(_top_domains(data, optimum_phase0, 0), indent=2)}
```

Top phase-1 domains:
```json
{json.dumps(_top_domains(data, optimum_phase1, 1), indent=2)}
```
"""
    REPORT_MD.write_text(report)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "selected_model": "CCPairTotal-RetainedTotal",
                "params": CCPAIRTOTAL_RETAINEDTOTAL_PARAMS,
                "train": {k.removeprefix("train_"): v for k, v in pairtotal_row.items() if k.startswith("train_")},
                "cv": {k.removeprefix("cv_"): v for k, v in pairtotal_row.items() if k.startswith("cv_")},
                "predicted_optimum_value": float(optimum_result.fun),
                "nearest_observed_run_name": str(data.frame.iloc[nearest_idx][data.name_col]),
                "nearest_observed_tv_distance": float(distances[nearest_idx]),
                "support": _support_stats(optimum_phase0, optimum_phase1),
            },
            indent=2,
            sort_keys=True,
        )
    )

    _plot_optimum(
        data=data,
        optimum_phase0=optimum_phase0,
        optimum_phase1=optimum_phase1,
        best_name=best_name,
        best_weights=best_weights,
    )

    print(comparison.to_string(index=False))
    print(f"Plot: {PLOT_PNG}")


if __name__ == "__main__":
    main()
