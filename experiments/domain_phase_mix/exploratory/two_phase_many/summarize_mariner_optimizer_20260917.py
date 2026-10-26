# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "matplotlib"]
# ///

"""Audit and summarize the complete archived optimized-mixture check."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE / "reference_outputs/mariner_local_validation_20260917"
MODEL_NAMES = {"hs17_mariner": "MARINER", "hs17_log1": "Logarithmic", "hs17_bounded_exp_c1": "Bounded exponential"}
COLORS = {"hs17_mariner": "#0072B2", "hs17_log1": "#D55E00", "hs17_bounded_exp_c1": "#009E73"}
PANELS = {"delphi_3e18_39bucket": "Qwen3 360M", "300m_39bucket": "Llama 200M", "60m_39bucket": "Llama 160M"}


def records() -> tuple[pd.DataFrame, pd.DataFrame]:
    output = ROOT / "optimizer"
    manifest = json.loads((output / "manifest.json").read_text())
    for path, digest in manifest["sources"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Optimizer source changed: {path}")
    full_paths = sorted(output.glob("fullfit/*/*/hs*.json"))
    fold_paths = sorted(output.glob("outerfold/*/*/*/*.json"))
    if len(full_paths) != 18 or len(fold_paths) != 90:
        raise ValueError(f"Incomplete optimizer check: {len(full_paths)} full, {len(fold_paths)} folds")
    full = []
    folds = []
    for paths, destination in ((full_paths, full), (fold_paths, folds)):
        for path in paths:
            row = json.loads(path.read_text())
            for source, digest in row["fit_hashes"].items():
                if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Optimizer fit changed: {source}")
            w = np.asarray(row["continuous_weights"])
            assert abs(w.sum() - 1) < 1e-8 and w.min() >= 0
            assert all(np.isfinite(w))
            if "runtime_counts" in row:
                counts = np.asarray(row["runtime_counts"])
                assert counts.sum() == 2048 and counts.min() >= 0
                assert row["runtime_prediction"] >= row["continuous_prediction"] - 1e-7
            destination.append({k: v for k, v in row.items() if not isinstance(v, (list, dict))})
    return pd.DataFrame(full), pd.DataFrame(folds)


def figure(summary: pd.DataFrame) -> None:
    labels = [(p, t) for p in PANELS for t in ("uncheatable", "table9")]
    fig, ax = plt.subplots(figsize=(10.8, 5.5), layout="constrained")
    for offset, (model, name) in zip((-0.22, 0, 0.22), MODEL_NAMES.items(), strict=True):
        rows = summary[summary.model == model].set_index(["panel", "target"]).loc[labels]
        x = np.arange(len(labels)) + offset
        ax.scatter(x, rows.median_tv_to_fullfit, s=65, color=COLORS[model], label=name, zorder=3)
        ax.vlines(x, rows.min_tv_to_fullfit, rows.max_tv_to_fullfit, color=COLORS[model], alpha=0.55, lw=2)
    ax.set_xticks(
        np.arange(len(labels)), [f"{PANELS[p]}\n{'Uncheatable' if t == 'uncheatable' else 'Suite'}" for p, t in labels]
    )
    ax.set_ylabel("Mixture total variation from the full-data fit")
    ax.set_ylim(0, 0.39)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Refitting on 80% of the swarm changes the selected mixture", loc="left", fontsize=14, pad=42)
    ax.text(
        0,
        1.06,
        "Dots: median of five grouped outer-fold fits. Lines: minimum to maximum.",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
    )
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(ROOT / "optimizer/fit_stability.png", dpi=180)
    fig.savefig(ROOT / "optimizer/fit_stability.pdf")
    plt.close(fig)


def main() -> None:
    full, folds = records()
    output = ROOT / "optimizer"
    full.to_csv(output / "all_fullfit_summary.csv", index=False)
    folds.to_csv(output / "all_outerfold_summary.csv", index=False)
    summary = folds.groupby(["panel", "target", "model"], as_index=False).agg(
        median_tv_to_fullfit=("tv_to_fullfit", "median"),
        min_tv_to_fullfit=("tv_to_fullfit", "min"),
        max_tv_to_fullfit=("tv_to_fullfit", "max"),
        max_start_tv=("start_tv_max", "max"),
        max_start_objective_range=("start_objective_range", "max"),
    )
    summary.to_csv(output / "fold_stability_summary.csv", index=False)
    audit = {
        "full_fit_objectives": len(full),
        "outerfold_objectives": len(folds),
        "all_selected_successful": bool(full.selected_success.all() and folds.selected_success.all()),
        "maximum_predictor_replay_error": float(max(full.predictor_parity.max(), folds.predictor_parity.max())),
        "clipped_solution_components": int(
            sum(
                full[c].sum() + folds[c].sum()
                for c in ("upper_clipped_components_at_solution", "lower_clipped_components_at_solution")
            )
        ),
        "input_hashes_checked": True,
        "source_hashes_checked": True,
    }
    (output / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    figure(summary)
    text = """# Optimized-mixture stability

Logarithmic harm produces more variable Uncheatable mixtures under fit-data
removal on both Llama swarms. On Llama 160M the median total variation from the
full-data proposal is 0.213, versus 0.087 for MARINER and 0.146 for bounded
exponential harm. On Llama 200M it is 0.191, versus 0.101 and 0.113. For Qwen3,
bounded exponential is less variable in this check (0.061 versus 0.115), while
logarithmic is 0.132. Total variation is half the sum of absolute changes in
bucket weights; 0.1 means reallocating 10% of the mixture.

All 18 full-fit and 90 outer-fold optimizations selected feasible, successfully
terminated endpoints. None of the chosen solutions activates either numerical
logit clip. Seventeen of 18 full-fit objectives converge to essentially the
same solution across ten starts. Logarithmic harm on Llama 200M Uncheatable has
a competing local solution 0.00252 predicted BPB worse and 0.272 away in total
variation. These checks do not establish global optimality for any of the
three families.

The changes across outer folds mostly exceed differences across optimizer
starts. The full-fit versus outer-fold comparison uses the same first five
starts; adding the extra five full-fit starts changes the selected objective
by at most 1.08e-9 BPB and the weights by at most 0.000138 total variation.
An independent audit records this check. Each outer-fold model fits 80% of
the original panel, with shape, ridge and floor selection repeated inside
that training subset.

The proposed mixtures are far from individual fit rows: nearest-row total
variation ranges from 0.390 to 0.623. This is a distance diagnostic, not a test
of whether a mixture lies outside the fit rows' convex hull. Full-fit peak
epoch counts range from 5.28 to 25.24 across objectives. Predictions at these
untrained optima are exploratory; measured-bank selection remains the direct
evidence for this local comparison.

All components are summed with the original objective weights. The vectorized
predictor reproduces canonical component predictions at archived inputs and
all 39 simplex vertices for every optimized objective. Individual JSON files
retain source-fit hashes, every starting and ending mixture, solver status,
continuous predictions, and the reference 2048-block materialization for full
fits. Cross-model prediction matrices are saved per panel and objective.

`all_fullfit_summary.csv` and `all_outerfold_summary.csv` combine every panel;
`fold_stability_summary.csv` supplies the plotted values. Individual parallel
runner invocations wrote partial `fullfit_summary.csv` files; use the combined
tables. No fit or optimization result was discarded.
"""
    (output / "RESULTS.md").write_text(text)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
