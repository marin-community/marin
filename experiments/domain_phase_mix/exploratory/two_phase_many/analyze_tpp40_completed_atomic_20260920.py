# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "matplotlib==3.10.8"]
# ///
"""Compare completed TPP40 and original Delphi policies without fitting a surrogate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent / "reference_outputs"
ROOT = BASE / "tpp40_frontier_gap_20260909"
INPUT = ROOT / "refresh_20260920"
OUTPUT = INPUT / "analysis"
LOW_COMPONENTS = BASE / "delphi_3e18_observed_components_20260724/observed_component_panel.csv"
LOW_SPECS = BASE / "delphi_augmented_swarm_3e18_20260714/run_specs.json"
ALPHA_LOW = 2400 / 3007
ALPHA_HIGH = 21856 / 27336
DISPLAY = {
    "ao3_english": "AO3",
    "arxiv_computer_science": "arXiv CS",
    "arxiv_physics": "arXiv Physics",
    "bbc_news": "BBC",
    "github_cpp": "GitHub C++",
    "github_python": "GitHub Python",
    "wikipedia_english": "Wikipedia",
    "uncheatable": "Uncheatable aggregate",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_data() -> tuple[pd.DataFrame, list[str], dict]:
    snapshot = json.loads((INPUT / "snapshot.json").read_text())
    for name, expected in snapshot["sha256"].items():
        assert sha(INPUT / name) == expected, name
    high = pd.read_csv(INPUT / "inputs/outcomes.csv").sort_values("order").reset_index(drop=True)
    low_all = pd.read_csv(LOW_COMPONENTS)
    low = low_all[low_all.panel.eq("two_phase_fit")].set_index("source_row_name")
    assert low.index.is_unique and high.order.is_unique and high.source_run.is_unique
    low = low.loc[high.source_run].reset_index(drop=True)
    specs = json.loads((INPUT / "inputs/run_specs.json").read_text())
    low_specs = json.loads(LOW_SPECS.read_text())
    domains = tuple(sorted(specs[0]["phase_weights"]["phase_0"]))
    vectors = np.array([[[s["phase_weights"][f"phase_{phase}"][d] for d in domains] for phase in (0, 1)] for s in specs])
    fields = (
        "source_run_name",
        "data_seed",
        "trainer_seed",
        "model_hidden_dim",
        "model_layers",
        "total_trainable_params",
        "non_embedding_params",
        "batch_size",
        "simulated_epoch_target_budget",
    )
    max_weight_error = 0.0
    for index, order in enumerate(high.order):
        source, previous = specs[order], low_specs[order]
        assert source["run_order"] == previous["run_order"] == order
        assert all(source[key] == previous[key] for key in fields), order
        assert source["phase_weights"] == previous["phase_weights"]
        assert source["source_run_name"] == high.loc[index, "source_run"]
        observed = np.array([[low.loc[index, f"phase_{phase}_{d}"] for d in domains] for phase in (0, 1)])
        error = float(np.max(np.abs(observed - vectors[order])))
        max_weight_error = max(max_weight_error, error)
        assert error < 1e-12, (order, error)
    assignment = json.loads((INPUT / "inputs/assignment.json").read_text())["assignments"]
    east = set(assignment["completed"]) | set(assignment["east5"])
    europe = set(assignment["europe"])
    assert not east & europe and east | europe == set(range(280))
    for row in high.itertuples():
        assert row.region == ("east5" if row.order in east else "europe")
    tied = np.max(np.abs(vectors[:, 0] - vectors[:, 1]), axis=1) < 1e-12
    assert np.array_equal(high.tied, tied[high.order])
    weights = vectors / vectors.sum(axis=2, keepdims=True)
    aggregate_low = ALPHA_LOW * weights[:, 0] + (1 - ALPHA_LOW) * weights[:, 1]
    aggregate_high = ALPHA_HIGH * weights[:, 0] + (1 - ALPHA_HIGH) * weights[:, 1]
    distances = np.abs(aggregate_high[~tied, None] - aggregate_high[None, tied]).sum(axis=2) / 2
    objectives = pd.read_csv(INPUT / "inputs/objectives.csv")
    objective = objectives[objectives.objective.eq("uncheatable")]
    columns = list(objective.component)
    tasks = [c.split("/")[-2] for c in columns] + ["uncheatable"]
    joined = high[["order", "run_name", "source_run", "source", "region", "tied"]].copy()
    for column, task in zip(columns, tasks[:-1], strict=True):
        joined[f"low::{task}"] = low[column].to_numpy()
        joined[f"high::{task}"] = high[column].to_numpy()
    joined["low::uncheatable"] = low[columns].to_numpy() @ objective.weight.to_numpy()
    joined["high::uncheatable"] = high[columns].to_numpy() @ objective.weight.to_numpy()
    high_error = float(np.max(abs(joined["high::uncheatable"] - high["eval/uncheatable_eval/bpb"])))
    low_error = float(np.max(abs(joined["low::uncheatable"] - low.uncheatable_bpb)))
    assert max(high_error, low_error) < 3e-6
    assert np.isfinite(joined.filter(regex=r"^(low|high)::").to_numpy()).all()
    audit = {
        "observed_start_utc": snapshot["started_utc"],
        "observed_end_utc": snapshot["finished_utc"],
        "coverage": snapshot["coverage"],
        "historical_weight_max_error": max_weight_error,
        "historical_source_seed_architecture_matches": len(joined),
        "assignment_region_matches": len(joined),
        "planned_tied": int(tied.sum()),
        "planned_asymmetric": int((~tied).sum()),
        "planned_exact_aggregate_counterparts": int((distances.min(axis=1) < 1e-10).sum()),
        "planned_nearest_tied_minimum_tv": float(distances.min()),
        "planned_aggregate_drift_max_tv": float(np.max(abs(aggregate_high - aggregate_low).sum(axis=1) / 2)),
        "alpha_low": ALPHA_LOW,
        "alpha_high": ALPHA_HIGH,
        "tpp_low": low_specs[0]["realized_train_tokens"] / low_specs[0]["total_trainable_params"],
        "tpp_high": specs[0]["realized_train_tokens"] / specs[0]["total_trainable_params"],
        "high_aggregate_reconstruction_error": high_error,
        "low_aggregate_reconstruction_error": low_error,
        "new_completed_orders": sorted(set(joined.order) - set(pd.read_csv(ROOT / "inputs/outcomes.csv").order)),
        "native_atomic_coverage": (
            high.filter(regex=r"^eval/(paloma|uncheatable_eval)/[^/]+/bpb$").notna().sum().to_dict()
        ),
        "inference": "Descriptive menu and paired-policy horizon comparisons; no phase-only or seed-noise inference.",
        "method_requirement": (
            "Surrogate fitting must work with two-phase endpoints alone; matched ties are optional validation."
        ),
    }
    return joined, tasks, audit


def summaries(frame: pd.DataFrame, tasks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    menu, relative, balanced = [], [], []
    for scope in ("all", "east5", "europe"):
        cell = frame if scope == "all" else frame[frame.region.eq(scope)]
        tied = cell[cell.tied]
        asymmetric = cell[~cell.tied]
        rng = np.random.default_rng(20260920)
        samples = np.array([rng.choice(len(asymmetric), len(tied), replace=False) for _ in range(5000)])
        for task in tasks:
            record: dict[str, str | int | float] = {
                "scope": scope,
                "task": task,
                "tied_count": len(tied),
                "asymmetric_count": len(asymmetric),
            }
            for horizon in ("low", "high"):
                column = f"{horizon}::{task}"
                best_tied, best_asymmetric = tied.loc[tied[column].idxmin()], asymmetric.loc[asymmetric[column].idxmin()]
                record.update(
                    {
                        f"{horizon}_best_tied": best_tied[column],
                        f"{horizon}_best_asymmetric": best_asymmetric[column],
                        f"{horizon}_best_tied_order": int(best_tied.order),
                        f"{horizon}_best_asymmetric_order": int(best_asymmetric.order),
                        f"{horizon}_gap": best_tied[column] - best_asymmetric[column],
                        f"{horizon}_asymmetric_better_than_best_tied": int(
                            (asymmetric[column] < best_tied[column]).sum()
                        ),
                    }
                )
                sampled_gaps = best_tied[column] - asymmetric[column].to_numpy()[samples].min(axis=1)
                balanced.append(
                    {
                        "scope": scope,
                        "task": task,
                        "horizon": horizon,
                        "candidates_per_class": len(tied),
                        "catalog_subsets": len(samples),
                        "median_gap": float(np.median(sampled_gaps)),
                        "fraction_subsets_asymmetric_wins": float(np.mean(sampled_gaps > 0)),
                        "gap_q05": float(np.quantile(sampled_gaps, 0.05)),
                        "gap_q95": float(np.quantile(sampled_gaps, 0.95)),
                    }
                )
            record["gap_change"] = float(record["high_gap"]) - float(record["low_gap"])
            menu.append(record)
            for baseline_order in (0, 1, 240):
                baseline = frame[frame.order.eq(baseline_order)].iloc[0]
                for row in asymmetric.to_dict("records"):
                    low_gain = baseline[f"low::{task}"] - row[f"low::{task}"]
                    high_gain = baseline[f"high::{task}"] - row[f"high::{task}"]
                    relative.append(
                        {
                            "scope": scope,
                            "task": task,
                            "order": row["order"],
                            "baseline_order": baseline_order,
                            "baseline_region": baseline.region,
                            "policy_region": row["region"],
                            "same_region_high": baseline.region == row["region"],
                            "low_baseline_advantage": low_gain,
                            "high_baseline_advantage": high_gain,
                            "advantage_change": high_gain - low_gain,
                        }
                    )
    return pd.DataFrame(menu), pd.DataFrame(relative), pd.DataFrame(balanced)


def plot_comparison(menu: pd.DataFrame, tasks: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), sharex=True, sharey=True)
    colors = [plt.get_cmap("RdYlGn_r")(value) for value in (0.08, 0.92)]
    y = np.arange(len(tasks))
    for ax, scope, title in zip(
        axes,
        ("all", "east5", "europe"),
        ("All completed policies", "East5-assigned policies", "Europe-assigned policies"),
        strict=True,
    ):
        rows = menu[menu.scope.eq(scope)].set_index("task").loc[tasks]
        ax.axvline(0, color="#969696", linewidth=1)
        for i, (_, row) in enumerate(rows.iterrows()):
            ax.plot([row.low_gap, row.high_gap], [i, i], color="#b8b8b8", linewidth=1.5, zorder=1)
        ax.scatter(rows.low_gap, y, color=colors[0], s=42, label="Delphi 3e18 (TPP 4.4)", zorder=2)
        ax.scatter(rows.high_gap, y, color=colors[1], s=42, label="Delphi TPP 40", zorder=3)
        ax.set_title(
            f"{title}\n{int(rows.tied_count.iloc[0])} tied / {int(rows.asymmetric_count.iloc[0])} asymmetric",
            fontsize=10,
        )
        ax.set_yticks(y, [DISPLAY[t] for t in tasks])
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.56, 0.97))
    fig.suptitle("Observed two-phase advantage over the best tied candidate", fontsize=13, y=1.01)
    fig.supxlabel("Best tied BPB - best asymmetric BPB; positive favors two phases", fontsize=10, y=0.09)
    fig.text(
        0.5,
        0.015,
        "Same menu at both horizons. Comparisons combine mixture and timing; they are not matched phase effects.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.17, right=0.99, bottom=0.19, top=0.81, wspace=0.12)
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"atomic_menu_gaps.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), INPUT / "snapshot.json", LOW_COMPONENTS, LOW_SPECS, ROOT / "inputs/outcomes.csv"]
    identity = {str(p): sha(p) for p in paths}
    manifest = OUTPUT / "manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        assert previous["inputs"] == identity, "Analysis identity changed; preserve the prior results."
        assert all(sha(OUTPUT / n) == h for n, h in previous["outputs"].items())
        print("Verified completed atomic horizon analysis.")
        return
    frame, tasks, audit = load_data()
    menu, relative, balanced = summaries(frame, tasks)
    frame.to_csv(OUTPUT / "matched_horizon_endpoints.csv", index=False)
    menu.to_csv(OUTPUT / "atomic_menu_comparison.csv", index=False)
    relative.to_csv(OUTPUT / "baseline_relative_policy_changes.csv", index=False)
    relative.groupby(["scope", "task", "baseline_order", "same_region_high"]).agg(
        n=("order", "size"),
        mean_change=("advantage_change", "mean"),
        median_change=("advantage_change", "median"),
        fraction_positive=("advantage_change", lambda x: float((x > 0).mean())),
    ).reset_index().to_csv(OUTPUT / "baseline_relative_summary.csv", index=False)
    balanced.to_csv(OUTPUT / "equal_menu_size_sensitivity.csv", index=False)
    write_json(OUTPUT / "validation.json", audit)
    plot_comparison(menu, tasks)
    write_json(
        manifest,
        {
            "inputs": identity,
            "outputs": {p.name: sha(p) for p in sorted(OUTPUT.iterdir()) if p.is_file() and p != manifest},
            "model_fits": 0,
            "lm_jobs": 0,
        },
    )
    print(menu[menu.scope.eq("all")][["task", "low_gap", "high_gap", "gap_change"]].to_string(index=False))


if __name__ == "__main__":
    main()
