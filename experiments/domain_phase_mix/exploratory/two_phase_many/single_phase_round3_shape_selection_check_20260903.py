# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Out-of-sample value of choosing the successor's shared shape on the bank.

Uses the prediction grid written by ``single_phase_round3_shape_scan_20260903.py``. Archive sources are split in
half at random; the (shape, ridge, link) row with the best regret (ties broken by frontier rank, then best-of-5) on
one half is scored on the other half, and compared with the frozen inner-CV model on the same half. Repeated over
random splits so the selection step itself is evaluated out of sample.
"""

from __future__ import annotations

import argparse
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
SEED = 20_260_907


def score(loss: np.ndarray, guess: np.ndarray) -> tuple[float, float, float]:
    order = np.argsort(guess, kind="stable")
    frontier = int(np.argmin(loss))
    return (
        float(loss[order[0]] - loss.min()),
        float(stats.rankdata(guess, method="average")[frontier]),
        float(loss[order[:5]].min() - loss.min()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR / "heldout_round3_corrected")
    parser.add_argument("--model", default="weibull_softplus_unscaled")
    parser.add_argument("--splits", type=int, default=200)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    rows = []
    for target in ("uncheatable", "table9"):
        payload = np.load(
            args.output_dir / f"shape_scan_{args.model.replace('@', '_')}_{target}.npz", allow_pickle=False
        )
        grid, cv, measured = payload["grid"], payload["inner_cv"], payload["measured"]
        sources = payload["sources"].astype(str)
        archive = np.array([DOSE_SOURCE not in source for source in sources])
        flat = grid.reshape(-1, grid.shape[-1])
        labels = [
            f"{shape}|ridge={ridge}|{link}"
            for shape in payload["shapes"]
            for ridge in payload["ridges"]
            for link in payload["links"]
        ]
        finite = np.isfinite(flat).all(axis=1)
        memberships = [frozenset(token.strip() for token in source.split(";") if token.strip()) for source in sources]
        archive_sources = sorted(
            {source for membership, keep in zip(memberships, archive, strict=True) if keep for source in membership}
        )
        for split in range(args.splits):
            shuffled = list(archive_sources)
            rng.shuffle(shuffled)
            half = set(shuffled[: len(shuffled) // 2])
            select_mask = archive & np.array([membership <= half for membership in memberships])
            test_mask = archive & np.array([not (membership & half) for membership in memberships])
            if select_mask.sum() < 10 or test_mask.sum() < 10:
                continue
            loss_select, loss_test = measured[select_mask], measured[test_mask]
            keys = []
            for index in np.flatnonzero(finite):
                keys.append((*score(loss_select, flat[index, select_mask]), index))
            keys.sort()
            chosen = keys[0][3]
            fixed = score(loss_test, flat[chosen, test_mask])
            frozen = score(loss_test, cv[test_mask])
            rows.append(
                {
                    "target": target,
                    "split": split,
                    "chosen": labels[chosen],
                    "test_size": int(test_mask.sum()),
                    "fixed_regret": fixed[0],
                    "frozen_regret": frozen[0],
                    "fixed_frontier_rank": fixed[1],
                    "frozen_frontier_rank": frozen[1],
                    "fixed_top5": fixed[2],
                    "frozen_top5": frozen[2],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / f"shape_selection_check_{args.model.replace('@', '_')}.csv", index=False)
    pd.set_option("display.width", 250)
    for target, subset in table.groupby("target"):
        diff = subset["fixed_regret"] - subset["frozen_regret"]
        print(
            f"{target}: {len(subset)} splits | regret fixed {subset['fixed_regret'].mean():.4f} vs frozen "
            f"{subset['frozen_regret'].mean():.4f} | difference {diff.mean():+.4f} "
            f"[{diff.quantile(0.025):+.4f}, {diff.quantile(0.975):+.4f}] | fixed better {np.mean(diff < 0):.2f}, "
            f"tie {np.mean(diff == 0):.2f} | frontier rank fixed {subset['fixed_frontier_rank'].mean():.1f} vs frozen "
            f"{subset['frozen_frontier_rank'].mean():.1f} | best-of-5 fixed {subset['fixed_top5'].mean():.4f} vs "
            f"frozen {subset['frozen_top5'].mean():.4f}"
        )
        print("  most often chosen:", subset["chosen"].value_counts().head(4).to_dict())


if __name__ == "__main__":
    main()
