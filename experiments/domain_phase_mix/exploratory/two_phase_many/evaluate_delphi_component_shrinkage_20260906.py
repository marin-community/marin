# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score skill-weighted aggregation of per-component surrogates on the frozen Delphi selection benchmark.

The Table-9 mean averages 51 components whose out-of-fold predictability differs widely (paper Figure 7). The
surrogate's pick is the argmin of the predicted mean, so a component the surrogate cannot predict contributes
prediction noise to the ordering and no signal. Each rule below replaces component c's bank prediction by
alpha_c * prediction + (1 - alpha_c) * panel mean, with alpha_c a function of the component's out-of-fold R^2 on
the panel (five outer folds; no bank information), and the aggregate is the usual weighted mean. Panel out-of-fold
scores use nested alphas (from the other four folds). Rules are pre-specified; nothing is selected on the bank.

Reads the per-component shards of an existing evaluation directory (default: the link-variants screen, which holds
WSPU, the bounded link, the CV-floor link and the link with hub interactions) and scores every method x rule with
`score_delphi_selection_20260906`, paired against plain WSPU over source blocks.

usage: uv run python evaluate_delphi_component_shrinkage_20260906.py [--shards-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_delphi_selection_20260906 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    score_delphi_selection_20260906 as scorer,
)

DEFAULT_SHARDS = SCRIPT_DIR / "reference_outputs" / "delphi_link_variants_selection_20260906"
DEFAULT_OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_component_shrinkage_selection_20260906"
WSPU = "weibull_softplus_unscaled"
OUTER_FOLDS = (0, 1, 2, 3, 4)
SHORT = {
    WSPU: "wspu",
    "weibull_softplus_unscaled@log_deficit_bounded_link": "link",
    "weibull_softplus_unscaled@log_deficit_bounded_link_floor_cv": "link_floor_cv",
    "weibull_softplus_unscaled@log_deficit_bounded_link_total_hub": "link_hub",
}


def _clip_r2(r2: np.ndarray) -> np.ndarray:
    return np.clip(r2, 0.0, 1.0)


RULES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "plain": lambda r2: np.ones_like(r2),
    "r2_soft": _clip_r2,
    "r2_square": lambda r2: _clip_r2(r2) ** 2,
    "r2_ge_0.25": lambda r2: (r2 >= 0.25).astype(float),
    "r2_ge_0.5": lambda r2: (r2 >= 0.5).astype(float),
    "r2_ge_0.75": lambda r2: (r2 >= 0.75).astype(float),
    "r2_top_half": lambda r2: (stats.rankdata(-r2, method="ordinal") <= np.ceil(len(r2) / 2)).astype(float),
}


@dataclasses.dataclass(frozen=True)
class ComponentFits:
    """Per-component predictions of one method on one target: out-of-fold on the panel, full fit on the bank."""

    oof: np.ndarray  # (280, components), out-of-fold across the five outer folds
    fold_of_row: np.ndarray  # (280,), which outer fold held each row out
    bank: np.ndarray  # (bank rows, components), fold -1 fit
    fold_bank: dict[int, np.ndarray]  # per outer fold, (bank rows, components)


def load_fits(shards: Path, method: str, target: str, components: int, rows: int) -> ComponentFits:
    oof = np.full((rows, components), np.nan)
    fold_of_row = np.full(rows, -1)
    fold_bank = {}
    for fold in OUTER_FOLDS:
        parts = []
        for component in range(components):
            shard = benchmark.read_npz(shards / "baseline_shards" / method / target / f"r0_f{fold}_c{component}.npz")
            oof[shard["test"], component] = shard["prediction"]
            fold_of_row[shard["test"]] = fold
            parts.append(shard["bank_prediction"])
        fold_bank[fold] = np.stack(parts, axis=1)
    bank = np.stack(
        [
            benchmark.read_npz(shards / "baseline_shards" / method / target / f"r0_f-1_c{component}.npz")[
                "bank_prediction"
            ]
            for component in range(components)
        ],
        axis=1,
    )
    if np.isnan(oof).any() or (fold_of_row < 0).any():
        raise ValueError(f"{method}/{target}: incomplete out-of-fold coverage")
    return ComponentFits(oof, fold_of_row, bank, fold_bank)


def r_squared(oof: np.ndarray, outcomes: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Out-of-fold R^2 per component over the given panel rows (1 - MSE / variance)."""
    residual = oof[rows] - outcomes[rows]
    centred = outcomes[rows] - outcomes[rows].mean(axis=0, keepdims=True)
    return 1.0 - (residual**2).sum(axis=0) / (centred**2).sum(axis=0)


def shrunk_aggregate(
    predictions: np.ndarray, alpha: np.ndarray, centre: np.ndarray, aggregation: np.ndarray
) -> np.ndarray:
    return (alpha[None, :] * predictions + (1.0 - alpha)[None, :] * centre[None, :]) @ aggregation


def collect(shards: Path, output: Path, methods: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = benchmark.read_npz(output / "inputs" / "panel.npz")
    records = []
    skill_rows = []
    for target in benchmark.TARGETS:
        bank = benchmark.read_npz(output / "inputs" / f"{target}_bank_features.npz")
        outcomes = data[f"{target}_outcomes"]
        aggregation = data[f"{target}_aggregation_weights"]
        names = data[f"{target}_components"]
        all_rows = np.arange(len(outcomes))
        for method in methods:
            fits = load_fits(shards, method, target, outcomes.shape[1], len(outcomes))
            full_r2 = r_squared(fits.oof, outcomes, all_rows)
            skill_rows.extend(
                {
                    "target": target,
                    "method": method,
                    "component": str(name),
                    "aggregation_weight": float(weight),
                    "oof_r2": float(value),
                    "oof_spearman": float(stats.spearmanr(fits.oof[:, index], outcomes[:, index]).statistic),
                    "panel_sd": float(outcomes[:, index].std(ddof=1)),
                    "oof_rmse": float(np.sqrt(np.mean((fits.oof[:, index] - outcomes[:, index]) ** 2))),
                }
                for index, (name, weight, value) in enumerate(zip(names, aggregation, full_r2, strict=True))
            )
            for rule, function in RULES.items():
                label = f"{SHORT[method]}:{rule}"
                alpha = function(full_r2)
                centre = outcomes.mean(axis=0)
                final = shrunk_aggregate(fits.bank, alpha, centre, aggregation)
                replicates = []
                for fold in OUTER_FOLDS:
                    held = np.flatnonzero(fits.fold_of_row == fold)
                    others = np.flatnonzero(fits.fold_of_row != fold)
                    nested_alpha = function(r_squared(fits.oof, outcomes, others))
                    nested_centre = outcomes[others].mean(axis=0)
                    panel_prediction = shrunk_aggregate(fits.oof[held], nested_alpha, nested_centre, aggregation)
                    replicates.append(shrunk_aggregate(fits.fold_bank[fold], nested_alpha, nested_centre, aggregation))
                    records.extend(
                        {
                            "method": label,
                            "target": target,
                            "population": "panel_oof",
                            "repeat": 0,
                            "fold": fold,
                            "row_id": str(int(row)),
                            "prediction": float(value),
                            "uncertainty": np.nan,
                            "active_components": int(np.count_nonzero(nested_alpha)),
                        }
                        for row, value in zip(held, panel_prediction, strict=True)
                    )
                spread = np.std(np.stack(replicates), axis=0, ddof=1)
                records.extend(
                    {
                        "method": label,
                        "target": target,
                        "population": "external_development",
                        "repeat": 0,
                        "fold": -1,
                        "row_id": str(row),
                        "prediction": float(value),
                        "uncertainty": float(error),
                        "active_components": int(np.count_nonzero(alpha)),
                    }
                    for row, value, error in zip(bank["coordinate_id"], final, spread, strict=True)
                )
    predictions = pd.DataFrame(records)
    skill = pd.DataFrame(skill_rows)
    predictions.to_csv(output / "predictions.csv", index=False)
    skill.to_csv(output / "component_skill.csv", index=False)
    return predictions, skill


def summarise(output: Path, metrics: pd.DataFrame, contrasts: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "target",
        "method",
        "regret_at_1",
        "best_of_5_regret",
        "best_of_10_regret",
        "selected_rank",
        "rows",
        "selected_id",
        "optimism",
        "rmse",
        "spearman",
    ]
    table = metrics[
        metrics.stratum.eq("optima") & metrics.policy.eq("point") & metrics.population.eq("external_development")
    ][columns].copy()
    active = (
        predictions[predictions.population.eq("external_development")]
        .groupby(["target", "method"])
        .active_components.first()
    )
    table["active_components"] = [active[(t, m)] for t, m in zip(table.target, table.method, strict=True)]
    panel = metrics[metrics.population.eq("panel_oof") & metrics.fold.eq(-1)].set_index(["target", "method"])
    table["panel_oof_rmse"] = [panel.loc[(t, m), "rmse"] for t, m in zip(table.target, table.method, strict=True)]
    table["panel_oof_spearman"] = [
        panel.loc[(t, m), "spearman"] for t, m in zip(table.target, table.method, strict=True)
    ]
    regret = contrasts[contrasts.metric.eq("regret_at_1")].set_index(["target", "candidate"])
    table["block_regret_vs_wspu"] = [
        (
            f"{regret.loc[(t, m), 'mean_delta']:+.4f} "
            f"[{regret.loc[(t, m), 'ci_low']:+.4f}, {regret.loc[(t, m), 'ci_high']:+.4f}]"
            if (t, m) in regret.index
            else "reference"
        )
        for t, m in zip(table.target, table.method, strict=True)
    ]
    table = table.sort_values(
        ["target", "method"], key=lambda s: s.map({"uncheatable": 0, "table9": 1}) if s.name == "target" else s
    )
    table.to_csv(output / "comparison.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--methods", default=",".join(SHORT), help="comma-separated registry ids with shards")
    args = parser.parse_args()
    methods = tuple(args.methods.split(","))
    if methods[0] != WSPU:
        raise ValueError("WSPU must come first; its plain rule is the paired reference")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not (args.output_dir / "inputs").exists():
        shutil.copytree(args.shards_dir / "inputs", args.output_dir / "inputs")
        shutil.copy(args.shards_dir / "input_hashes.json", args.output_dir / "input_hashes.json")
    benchmark.verify_inputs(args.output_dir)
    predictions, skill = collect(args.shards_dir, args.output_dir, methods)
    # The plain WSPU rule is the paired reference, so it carries the reference's registry id.
    predictions = predictions.assign(method=predictions.method.replace({"wspu:plain": WSPU}))
    scorer.METHODS = tuple(predictions.method.unique())
    metrics, assignments, loso = scorer.score_predictions(args.output_dir, predictions)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    assignments.to_csv(args.output_dir / "source_blocks.csv", index=False)
    loso.to_csv(args.output_dir / "source_disjoint_selection.csv", index=False)
    contrasts = scorer.paired_sources(metrics, loso)
    contrasts.to_csv(args.output_dir / "paired_source_contrasts.csv", index=False)
    table = summarise(args.output_dir, metrics, contrasts, predictions)
    pd.set_option("display.width", 250)
    print("component out-of-fold R^2 by target and method (quantiles):")
    print(skill.groupby(["target", "method"]).oof_r2.describe()[["count", "min", "25%", "50%", "75%", "max"]].round(3))
    print(
        table[
            [
                "target",
                "method",
                "active_components",
                "regret_at_1",
                "best_of_5_regret",
                "best_of_10_regret",
                "selected_rank",
                "optimism",
                "rmse",
                "spearman",
                "panel_oof_rmse",
                "panel_oof_spearman",
                "block_regret_vs_wspu",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps({"shards_dir": str(args.shards_dir), "methods": methods, "rules": list(RULES)}, indent=2)
    )


if __name__ == "__main__":
    main()
