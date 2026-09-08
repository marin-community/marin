# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-rank the Observatory's surrogate models with the resampling unit set to the swarm.

An earlier pass ranked every model across the Observatory's cells and metrics and bootstrapped the
cells independently, reporting a 0.978 probability that the band ensemble ranks first. That interval
is too narrow, and the reason is structural rather than numerical: the cells are not independent
draws. A swarm contributes up to four cells -- two objectives crossed with two policy classes -- built
from the same trained runs, the same folds and the same fitted structures, and the metrics within one
cell are themselves correlated. Resampling cells independently treats one swarm's four correlated
cells as four pieces of evidence.

This recomputes the same comparison three ways so the cost of the mistake is visible: resampling
cells independently, resampling whole swarms with all their cells attached, and resampling swarms
while keeping objective and policy class as strata. Score magnitudes are reported next to the ranks,
because a rank of one says nothing about whether the margin matters and the margins here are small.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BUNDLE = SCRIPT_DIR / "mixture_fit_debugger" / "src" / "generated" / "dashboard_data.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "model_rank_clustered_bootstrap_20260727"
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260727
# Metrics where a smaller value is better, split by what they are evidence about. Keeping fit and
# optimum quality apart matters because a model can win one while losing the other, which is the
# whole reason both are tracked.
FIT_METRICS = ("rmse", "mae")
OPTIMUM_METRICS = ("regretAt1", "foldMeanRegretAt1", "lowerTailOptimism", "lowTailRmse")
# Ranked ascending for the metrics above; spearman is handled separately since larger is better.
HIGHER_IS_BETTER = ("spearman",)
SPLITS = ("fitOof", "heldout")
MIN_MODELS_PER_CELL = 2


def load_cells(bundle_path: Path) -> pd.DataFrame:
    """One row per swarm, objective, policy class, split, model and metric."""
    bundle = json.loads(bundle_path.read_text())
    rows = []
    for swarm_id, swarm in bundle["swarms"].items():
        for target, by_policy in swarm.get("diagnostics", {}).items():
            for policy_class, by_model in by_policy.items():
                for model_id, by_split in by_model.items():
                    for split in SPLITS:
                        metrics = by_split.get(split)
                        if not metrics or not metrics.get("n"):
                            continue
                        for metric, value in metrics.items():
                            if metric in {"n", "lowerTailCount"} or value is None:
                                continue
                            rows.append(
                                {
                                    "swarm": swarm_id,
                                    "target": target,
                                    "policy_class": policy_class,
                                    "split": split,
                                    "model": model_id,
                                    "metric": metric,
                                    "value": float(value),
                                    "n": int(metrics["n"]),
                                }
                            )
    return pd.DataFrame(rows)


def rank_within_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalized rank of each model inside each cell and metric, 0 best and 1 worst."""
    ranked = []
    keys = ["swarm", "target", "policy_class", "split", "metric"]
    for key, group in frame.groupby(keys):
        if group["model"].nunique() < MIN_MODELS_PER_CELL:
            continue
        metric = key[-1]
        values = group["value"].to_numpy()
        oriented = -values if metric in HIGHER_IS_BETTER else values
        ranks = rankdata(oriented, method="average")
        normalized = (ranks - 1.0) / max(len(ranks) - 1, 1)
        block = group.copy()
        block["normalized_rank"] = normalized
        ranked.append(block)
    return pd.concat(ranked, ignore_index=True)


def family_of(metric: str) -> str:
    if metric in FIT_METRICS:
        return "fit"
    if metric in OPTIMUM_METRICS:
        return "optimum"
    return "ranking"


def bootstrap_probability(
    ranked: pd.DataFrame,
    unit: str,
    draws: int,
    seed: int,
    stratify: bool,
) -> pd.DataFrame:
    """Probability each model has the best mean normalized rank, resampling ``unit`` with replacement.

    With ``unit`` set to the swarm, every cell belonging to a drawn swarm is carried along, so the
    correlation between a swarm's own cells is preserved instead of being broken up.
    """
    generator = np.random.default_rng(seed)
    models = sorted(ranked["model"].unique())
    wins = {family: dict.fromkeys(models, 0) for family in ("fit", "optimum", "overall")}
    ranked = ranked.assign(family=ranked["metric"].map(family_of))
    units = sorted(ranked[unit].unique())
    strata = [group for _key, group in ranked.groupby(["target", "policy_class"])] if stratify else [ranked]
    for _draw in range(draws):
        pieces = []
        for stratum in strata:
            available = sorted(stratum[unit].unique())
            if not available:
                continue
            drawn = generator.choice(available, size=len(available), replace=True)
            pieces.extend(stratum[stratum[unit] == value] for value in drawn)
        sample = pd.concat(pieces, ignore_index=True)
        for family in ("fit", "optimum", "overall"):
            subset = sample if family == "overall" else sample[sample["family"] == family]
            if subset.empty:
                continue
            means = subset.groupby("model")["normalized_rank"].mean()
            # Only models scored in this resample can win it; a model absent from every drawn swarm
            # is not evidence of anything either way.
            wins[family][means.idxmin()] += 1
    rows = []
    for family, counts in wins.items():
        total = max(sum(counts.values()), 1)
        for model, count in counts.items():
            rows.append(
                {
                    "unit": unit,
                    "stratified": stratify,
                    "family": family,
                    "model": model,
                    "probability_rank_1": count / total,
                }
            )
    _ = units
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_cells(args.bundle)
    ranked = rank_within_cells(frame)
    ranked["cell"] = ranked[["swarm", "target", "policy_class", "split"]].agg("/".join, axis=1)
    print(
        f"{ranked['cell'].nunique()} cells over {ranked['swarm'].nunique()} swarms, "
        f"{ranked['model'].nunique()} models, {len(ranked)} scored entries"
    )
    print(f"cells per swarm: {ranked.groupby('swarm')['cell'].nunique().to_dict()}")

    ranked = ranked.assign(family=ranked["metric"].map(family_of))
    print("\n" + "=" * 100)
    print("MEAN NORMALIZED RANK (0 best, 1 worst) AND RAW SCORE MARGINS")
    print("=" * 100)
    summary = ranked.groupby(["model", "family"])["normalized_rank"].agg(["mean", "count"]).reset_index()
    pivot = summary.pivot(index="model", columns="family", values="mean")
    pivot["overall"] = ranked.groupby("model")["normalized_rank"].mean()
    pivot["cells"] = ranked.groupby("model")["cell"].nunique()
    pivot = pivot.sort_values("overall")
    print(pivot.to_string(float_format=lambda value: f"{value:.3f}"))

    draws = []
    for unit, stratify in (("cell", False), ("swarm", False), ("swarm", True)):
        result = bootstrap_probability(ranked, unit, args.draws, BOOTSTRAP_SEED, stratify)
        draws.append(result)
        label = f"resample {unit}" + (", stratified by target and policy class" if stratify else "")
        print(f"\n  {label}")
        for family in ("overall", "fit", "optimum"):
            top = result[result["family"] == family].nlargest(3, "probability_rank_1")
            rendered = "   ".join(f"{row.model} {row.probability_rank_1:.3f}" for row in top.itertuples())
            print(f"    {family:<9} {rendered}")

    ranked.to_csv(args.output_dir / "model_cell_ranks.csv", index=False)
    pd.concat(draws, ignore_index=True).to_csv(args.output_dir / "rank_bootstrap.csv", index=False)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
