# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test whether simple postprocessing maps the reproduced LCB optimum to Grug-MoE v4."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
REPRO_DIR = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "collaborator_grug_v4_aggregate_repro_20260525/sent_raw_metric_matrix_300m_zip"
)
DASHBOARD_WEIGHTS = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "grug_moe_mix_dashboard_20260517/grug_moe_mix_weights_long.csv"
)
EPOCH_METADATA = REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/two_phase_many_epoch_metadata.csv"
OUTPUT_DIR = (
    REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "collaborator_grug_v4_aggregate_repro_20260525/postprocess_tests"
)


def normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot normalize non-positive weights")
    return weights / total


def metrics(candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    corr = float(np.corrcoef(candidate, target)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean((candidate - target) ** 2))),
        "mae": float(np.mean(np.abs(candidate - target))),
        "tv": 0.5 * float(np.abs(candidate - target).sum()),
        "corr": corr,
        "max_abs": float(np.abs(candidate - target).max()),
    }


def combined_metrics(
    candidate0: np.ndarray,
    candidate1: np.ndarray,
    target0: np.ndarray,
    target1: np.ndarray,
) -> dict[str, float]:
    phase0 = metrics(candidate0, target0)
    phase1 = metrics(candidate1, target1)
    flat_candidate = np.concatenate([candidate0, candidate1])
    flat_target = np.concatenate([target0, target1])
    flat = metrics(flat_candidate, flat_target)
    return {
        "phase0_tv": phase0["tv"],
        "phase1_tv": phase1["tv"],
        "mean_phase_tv": 0.5 * (phase0["tv"] + phase1["tv"]),
        "phase0_rmse": phase0["rmse"],
        "phase1_rmse": phase1["rmse"],
        "flat_rmse": flat["rmse"],
        "flat_mae": flat["mae"],
        "flat_corr": flat["corr"],
        "flat_max_abs": flat["max_abs"],
    }


def grid_best(
    family: str,
    domains: list[str],
    transform: Callable[..., tuple[np.ndarray, np.ndarray]],
    target0: np.ndarray,
    target1: np.ndarray,
    grid: list[dict[str, float]],
) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    best: dict[str, object] | None = None
    for params in grid:
        candidate0, candidate1 = transform(**params)
        row = {"family": family, **params, **combined_metrics(candidate0, candidate1, target0, target1)}
        rows.append(row)
        if best is None or row["mean_phase_tv"] < best["mean_phase_tv"]:
            best = {**row, "candidate0": candidate0, "candidate1": candidate1}
    assert best is not None
    detail = pd.DataFrame(
        {
            "family": family,
            "domain": domains,
            "candidate_w0": best["candidate0"],
            "candidate_w1": best["candidate1"],
            "target_w0": target0,
            "target_w1": target1,
        }
    )
    detail["candidate_total"] = detail["candidate_w0"] + detail["candidate_w1"]
    detail["target_total"] = detail["target_w0"] + detail["target_w1"]
    detail["abs_total_diff"] = (detail["candidate_total"] - detail["target_total"]).abs()
    best_no_arrays = {key: value for key, value in best.items() if key not in {"candidate0", "candidate1"}}
    return best_no_arrays, detail.sort_values("abs_total_diff", ascending=False)


def top_overlap(candidate_total: np.ndarray, target_total: np.ndarray, k: int = 8) -> int:
    candidate_top = set(np.argsort(-candidate_total)[:k])
    target_top = set(np.argsort(-target_total)[:k])
    return len(candidate_top & target_top)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = pd.read_csv(REPRO_DIR / "mixture_weights.csv")
    dashboard = pd.read_csv(DASHBOARD_WEIGHTS)
    metadata = pd.read_csv(EPOCH_METADATA).set_index("domain_name")

    lcb = weights[weights["label"].eq("lcb_optimum")].set_index("domain")
    prop = weights[weights["label"].eq("proportional")].set_index("domain")
    training_domains = sorted(set(lcb.index) | set(prop.index))
    v4 = (
        dashboard[
            dashboard["track"].eq("grug_moe_mix_v4")
            & dashboard["hidden_dim"].eq(512)
            & dashboard["domain"].isin(training_domains)
        ]
        .pivot(index="domain", columns="phase", values="weight")
        .fillna(0.0)
    )
    domains = training_domains
    lcb = lcb.reindex(domains).fillna(0.0)
    prop = prop.reindex(domains).fillna(0.0)
    v4 = v4.reindex(domains).fillna(0.0)
    c0 = metadata.loc[domains, "phase_0_epoch_multiplier"].to_numpy(dtype=np.float64)
    c1 = metadata.loc[domains, "phase_1_epoch_multiplier"].to_numpy(dtype=np.float64)

    q0 = normalize(lcb["w0"].to_numpy(dtype=np.float64))
    q1 = normalize(lcb["w1"].to_numpy(dtype=np.float64))
    p0 = normalize(prop["w0"].to_numpy(dtype=np.float64))
    p1 = normalize(prop["w1"].to_numpy(dtype=np.float64))
    target0 = normalize(v4["phase_0"].to_numpy(dtype=np.float64))
    target1 = normalize(v4["phase_1"].to_numpy(dtype=np.float64))
    uniform = np.full(len(domains), 1.0 / len(domains))

    eps_grid = [1e-12, 1e-8, 1e-6, 1e-4, 1e-3]
    alpha_grid = np.linspace(0.0, 1.0, 501)
    power_grid = np.linspace(0.0, 3.0, 601)
    cap_grid = np.linspace(0.02, 0.25, 461)

    results = []
    details = []

    def add_family(
        name: str,
        transform: Callable[..., tuple[np.ndarray, np.ndarray]],
        grid: list[dict[str, float]],
    ) -> None:
        best, detail = grid_best(name, domains, transform, target0, target1, grid)
        candidate_total = detail.sort_values("domain")["candidate_total"].to_numpy(dtype=np.float64)
        target_total = detail.sort_values("domain")["target_total"].to_numpy(dtype=np.float64)
        best["top8_overlap"] = top_overlap(candidate_total, target_total, k=8)
        results.append(best)
        details.append(detail)

    add_family(
        "arithmetic_blend_proportional_lcb_shared_alpha",
        lambda alpha: (normalize((1.0 - alpha) * p0 + alpha * q0), normalize((1.0 - alpha) * p1 + alpha * q1)),
        [{"alpha": float(alpha)} for alpha in alpha_grid],
    )
    add_family(
        "arithmetic_blend_proportional_lcb_phase_alpha",
        lambda alpha0, alpha1: (
            normalize((1.0 - alpha0) * p0 + alpha0 * q0),
            normalize((1.0 - alpha1) * p1 + alpha1 * q1),
        ),
        [{"alpha0": float(a0), "alpha1": float(a1)} for a0 in alpha_grid[::5] for a1 in alpha_grid[::5]],
    )
    add_family(
        "geometric_softmax_proportional_lcb_shared_alpha",
        lambda alpha, eps: (
            normalize(np.exp((1.0 - alpha) * np.log(p0) + alpha * np.log(q0 + eps))),
            normalize(np.exp((1.0 - alpha) * np.log(p1) + alpha * np.log(q1 + eps))),
        ),
        [{"alpha": float(alpha), "eps": eps} for alpha in alpha_grid for eps in eps_grid],
    )
    add_family(
        "power_softmax_lcb_only",
        lambda power, eps: (
            normalize((q0 + eps) ** power),
            normalize((q1 + eps) ** power),
        ),
        [{"power": float(power), "eps": eps} for power in power_grid for eps in eps_grid],
    )
    add_family(
        "additive_floor_lcb_only",
        lambda floor: (
            normalize(q0 + floor),
            normalize(q1 + floor),
        ),
        [{"floor": float(floor)} for floor in np.linspace(0.0, 0.05, 501)],
    )
    add_family(
        "cap_lcb_then_renormalize",
        lambda cap: (
            normalize(np.minimum(q0, cap)),
            normalize(np.minimum(q1, cap)),
        ),
        [{"cap": float(cap)} for cap in cap_grid],
    )
    add_family(
        "epoch_space_arithmetic_blend_shared_alpha",
        lambda alpha: (
            normalize(((1.0 - alpha) * (p0 * c0) + alpha * (q0 * c0)) / c0),
            normalize(((1.0 - alpha) * (p1 * c1) + alpha * (q1 * c1)) / c1),
        ),
        [{"alpha": float(alpha)} for alpha in alpha_grid],
    )
    add_family(
        "arithmetic_blend_uniform_lcb_shared_alpha",
        lambda alpha: (
            normalize((1.0 - alpha) * uniform + alpha * q0),
            normalize((1.0 - alpha) * uniform + alpha * q1),
        ),
        [{"alpha": float(alpha)} for alpha in alpha_grid],
    )
    add_family(
        "geometric_softmax_uniform_lcb_shared_alpha",
        lambda alpha, eps: (
            normalize(np.exp((1.0 - alpha) * np.log(uniform) + alpha * np.log(q0 + eps))),
            normalize(np.exp((1.0 - alpha) * np.log(uniform) + alpha * np.log(q1 + eps))),
        ),
        [{"alpha": float(alpha), "eps": eps} for alpha in alpha_grid for eps in eps_grid],
    )

    results_df = pd.DataFrame(results).sort_values("mean_phase_tv")
    results_df.to_csv(OUTPUT_DIR / "postprocess_family_best_fits.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(OUTPUT_DIR / "postprocess_family_best_fit_details.csv", index=False)

    baseline_rows = []
    for name, w0, w1 in [
        ("proportional", p0, p1),
        ("lcb", q0, q1),
        ("uniform", uniform, uniform),
    ]:
        row = {"family": name, **combined_metrics(w0, w1, target0, target1)}
        row["top8_overlap"] = top_overlap(w0 + w1, target0 + target1, k=8)
        baseline_rows.append(row)
    pd.DataFrame(baseline_rows).sort_values("mean_phase_tv").to_csv(OUTPUT_DIR / "baseline_distances.csv", index=False)

    summary = {
        "best_family": results_df.iloc[0].to_dict(),
        "baseline_distances": baseline_rows,
        "n_domains": len(domains),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Grug v4 LCB Postprocessing Tests",
        "",
        "Goal: test whether dashboard v4 can be reconstructed by simple postprocessing of the reproduced LCB optimum.",
        "",
        "## Best Families",
        "",
    ]
    for _, row in results_df.head(8).iterrows():
        params = {
            key: row[key]
            for key in row.index
            if key
            not in {
                "family",
                "phase0_tv",
                "phase1_tv",
                "mean_phase_tv",
                "phase0_rmse",
                "phase1_rmse",
                "flat_rmse",
                "flat_mae",
                "flat_corr",
                "flat_max_abs",
                "top8_overlap",
            }
            and not pd.isna(row[key])
        }
        lines.append(
            f"- `{row['family']}`: mean phase TV `{row['mean_phase_tv']:.4f}`, "
            f"flat corr `{row['flat_corr']:.4f}`, top-8 overlap `{int(row['top8_overlap'])}/8`, "
            f"params `{params}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "- A simple softmax/temperature transform would need to reach a substantially lower TV than "
                "proportional itself to be convincing."
            ),
            (
                "- Families that interpolate from proportional can look moderately correlated with v4 because "
                "v4 remains closer to proportional than to the raw LCB endpoint."
            ),
            (
                "- Domain-level residuals in `postprocess_family_best_fit_details.csv` should be inspected "
                "before accepting a high correlation as a reconstruction."
            ),
            "",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines))
    print(results_df.head(10).to_string(index=False))
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
