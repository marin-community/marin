# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Bank value of coarse grid rules for the successor: restrict the shape grid or the link, keep inner CV inside.

Each rule is a mask over the successor's (shape, ridge, link) grid. Per component the inner-CV argmin inside the
mask is taken (from the tables written by ``single_phase_round3_shape_scan_20260903.py``), the bank prediction is
aggregated, and selection metrics are computed on the archive stratum. The rule set itself is then evaluated out
of sample: archive sources are split in half, the rule with the best regret on one half is scored on the other.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)

DOSE_SOURCE = "conditional_epoch_dose_response"
SEED = 20_260_908


def rule_masks(shapes: list[dict], ridges: np.ndarray, links: list[str]) -> dict[str, np.ndarray]:
    rate = np.array([shape["rate"] for shape in shapes])
    power = np.array([shape["power"] for shape in shapes])
    threshold = np.array([shape["threshold"] for shape in shapes])
    identity = np.array([link == "identity" for link in links])
    bounded = ~identity
    full = np.ones((len(shapes), len(ridges), len(links)), dtype=bool)

    def combine(shape_mask=None, ridge_mask=None, link_mask=None):
        mask = full.copy()
        if shape_mask is not None:
            mask &= shape_mask[:, None, None]
        if ridge_mask is not None:
            mask &= ridge_mask[None, :, None]
        if link_mask is not None:
            mask &= link_mask[None, None, :]
        return mask

    return {
        "frozen (identity, full grid)": combine(link_mask=identity),
        "link by CV": combine(),
        "bounded link": combine(link_mask=bounded),
        "threshold >= 3": combine(threshold >= 3, link_mask=identity),
        "threshold >= 4": combine(threshold >= 4, link_mask=identity),
        "threshold >= 5": combine(threshold >= 5, link_mask=identity),
        "threshold <= 2": combine(threshold <= 2, link_mask=identity),
        "threshold >= 4, link by CV": combine(threshold >= 4),
        "rate <= 0.25": combine(rate <= 0.25, link_mask=identity),
        "rate >= 0.5": combine(rate >= 0.5, link_mask=identity),
        "power = 1": combine(power == 1.0, link_mask=identity),
        "power <= 0.5": combine(power <= 0.5, link_mask=identity),
        "ridge >= 0.01": combine(ridge_mask=ridges >= 0.01, link_mask=identity),
        "ridge = 0": combine(ridge_mask=ridges == 0.0, link_mask=identity),
        "ridge >= 0.1, threshold >= 4": combine(threshold >= 4, ridges >= 0.1, identity),
    }


def rule_prediction(
    component_grid: np.ndarray, component_cv: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    prediction = np.zeros(component_grid.shape[-1])
    for index in range(component_grid.shape[0]):
        table = np.where(mask, component_cv[index], np.inf)
        flat = int(np.argmin(table))
        shape_index, ridge_index, link_index = np.unravel_index(flat, table.shape)
        prediction += weights[index] * component_grid[index, shape_index, ridge_index, link_index].astype(float)
    return prediction


def score(loss: np.ndarray, guess: np.ndarray) -> dict[str, float]:
    order = np.argsort(guess, kind="stable")
    frontier = int(np.argmin(loss))
    quartile = loss <= np.quantile(loss, 0.25)
    return {
        "regret_at_1": float(loss[order[0]] - loss.min()),
        "top5_regret": float(loss[order[:5]].min() - loss.min()),
        "frontier_predicted_rank": float(stats.rankdata(guess, method="average")[frontier]),
        "bias": float(np.mean(guess - loss)),
        "rmse": float(np.sqrt(np.mean((guess - loss) ** 2))),
        "spearman_best_quartile": harness._safe_spearman(loss[quartile], guess[quartile]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR / "heldout_round3_corrected")
    parser.add_argument("--model", default="weibull_softplus_unscaled")
    parser.add_argument("--splits", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    metric_rows, split_rows = [], []
    for target in ("uncheatable", "table9"):
        payload = np.load(
            args.output_dir / f"shape_scan_{args.model.replace('@', '_')}_{target}.npz", allow_pickle=False
        )
        shapes = [ast.literal_eval(text) for text in payload["shapes"]]
        masks = rule_masks(shapes, payload["ridges"], list(payload["links"]))
        measured, sources = payload["measured"], payload["sources"].astype(str)
        archive = np.array([DOSE_SOURCE not in source for source in sources])
        predictions = {
            name: rule_prediction(
                payload["component_grid"], payload["component_cv"], payload["aggregation_weights"], mask
            )
            for name, mask in masks.items()
        }
        frozen = payload["inner_cv"]
        check = np.abs(predictions["frozen (identity, full grid)"] - frozen).max()
        print(f"{target}: frozen rule reproduces the inner-CV model to {check:.2e}", flush=True)
        loss = measured[archive]
        samples = rng.integers(0, archive.sum(), size=(args.bootstrap, archive.sum()))
        frozen_boot = np.array([score(loss[s], frozen[archive][s])["regret_at_1"] for s in samples])
        for name, guess in predictions.items():
            row = {"target": target, "rule": name, "grid_cells": int(masks[name].sum())}
            row.update(score(loss, guess[archive]))
            boot = np.array([score(loss[s], guess[archive][s])["regret_at_1"] for s in samples])
            diff = boot - frozen_boot
            row.update(
                {
                    "regret_diff_vs_frozen": float(diff.mean()),
                    "diff_ci_low": float(np.quantile(diff, 0.025)),
                    "diff_ci_high": float(np.quantile(diff, 0.975)),
                    "share_better": float(np.mean(diff < 0)),
                }
            )
            metric_rows.append(row)
        memberships = [frozenset(token.strip() for token in source.split(";") if token.strip()) for source in sources]
        archive_sources = sorted(
            {source for membership, keep in zip(memberships, archive, strict=True) if keep for source in membership}
        )
        names = list(predictions)
        for split in range(args.splits):
            shuffled = list(archive_sources)
            rng.shuffle(shuffled)
            half = set(shuffled[: len(shuffled) // 2])
            in_half = np.array([membership <= half for membership in memberships])
            outside = np.array([not (membership & half) for membership in memberships])
            select_mask, test_mask = archive & in_half, archive & outside
            if select_mask.sum() < 10 or test_mask.sum() < 10:
                continue
            ranked = sorted(
                names,
                key=lambda name: (
                    score(measured[select_mask], predictions[name][select_mask])["regret_at_1"],
                    score(measured[select_mask], predictions[name][select_mask])["frontier_predicted_rank"],
                ),
            )
            chosen = ranked[0]
            chosen_score = score(measured[test_mask], predictions[chosen][test_mask])
            frozen_score = score(measured[test_mask], frozen[test_mask])
            split_rows.append(
                {
                    "target": target,
                    "split": split,
                    "chosen": chosen,
                    "chosen_regret": chosen_score["regret_at_1"],
                    "frozen_regret": frozen_score["regret_at_1"],
                    "chosen_frontier_rank": chosen_score["frontier_predicted_rank"],
                    "frozen_frontier_rank": frozen_score["frontier_predicted_rank"],
                }
            )
    metrics = pd.DataFrame(metric_rows)
    splits = pd.DataFrame(split_rows)
    metrics.to_csv(args.output_dir / f"grid_rules_{args.model.replace('@', '_')}.csv", index=False)
    splits.to_csv(args.output_dir / f"grid_rules_split_check_{args.model.replace('@', '_')}.csv", index=False)
    pd.set_option("display.width", 250)
    for target in ("uncheatable", "table9"):
        print(f"\n=== {target} / archive stratum, rules (per-component inner CV inside the rule)")
        print(metrics[metrics["target"].eq(target)].round(4).to_string(index=False))
        subset = splits[splits["target"].eq(target)]
        diff = subset["chosen_regret"] - subset["frozen_regret"]
        print(
            f"split-half rule selection: chosen regret {subset['chosen_regret'].mean():.4f} "
            f"vs frozen {subset['frozen_regret'].mean():.4f}; "
            f"difference {diff.mean():+.4f} [{diff.quantile(0.025):+.4f}, {diff.quantile(0.975):+.4f}]; "
            f"chosen better {np.mean(diff < 0):.2f}, tie {np.mean(diff == 0):.2f}; "
            f"frontier rank {subset['chosen_frontier_rank'].mean():.1f} vs {subset['frozen_frontier_rank'].mean():.1f}"
        )
        print("most chosen:", subset["chosen"].value_counts().head(4).to_dict())


if __name__ == "__main__":
    main()
