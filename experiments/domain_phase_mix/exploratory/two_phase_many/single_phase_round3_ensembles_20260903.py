# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Selection value of every frozen model and of fixed prediction ensembles on the Delphi bank.

Reads the per-coordinate predictions written by ``single_phase_round3_heldout_selection_20260903.py`` and scores,
on the archive stratum, each model plus fixed ensembles (mean prediction, mean z-score, mean rank) of named model
sets, with a paired bootstrap against the reference model.
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
SEED = 20_260_909
SETS = {
    "successor+links": (
        "weibull_softplus_unscaled",
        "weibull_softplus_unscaled@log_deficit_bounded_link",
        "weibull_softplus_unscaled@link_by_cv",
    ),
    "top5_tabular": (
        "weibull_softplus_unscaled",
        "weibull_softplus_unscaled@log_deficit_bounded_link",
        "weibull_softplus_unscaled@link_by_cv",
        "dsp_total_exposure_concentration",
        "bucket_family_power_grp",
    ),
    "successor+dsp+olmix": ("weibull_softplus_unscaled", "dsp_total_exposure", "olmix_loglinear_taskwise"),
}


def score(loss: np.ndarray, guess: np.ndarray) -> dict[str, float]:
    order = np.argsort(guess, kind="stable")
    frontier = int(np.argmin(loss))
    quartile = loss <= np.quantile(loss, 0.25)
    return {
        "regret_at_1": float(loss[order[0]] - loss.min()),
        "top5_regret": float(loss[order[:5]].min() - loss.min()),
        "frontier_predicted_rank": float(stats.rankdata(guess, method="average")[frontier]),
        "spearman_best_quartile": harness._safe_spearman(loss[quartile], guess[quartile]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR / "heldout_round3_corrected")
    parser.add_argument("--reference", default="weibull_softplus_unscaled")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    predictions = pd.read_csv(args.output_dir / "predictions.csv")
    rng = np.random.default_rng(SEED)
    rows = []
    for target, subset in predictions.groupby("target"):
        wide = subset.pivot(index="coordinate_id", columns="model", values="prediction")
        meta = subset.drop_duplicates("coordinate_id").set_index("coordinate_id").loc[wide.index]
        archive = np.array([DOSE_SOURCE not in source for source in meta["sources"].astype(str)])
        loss = meta["measured_mean_bpb"].to_numpy(float)[archive]
        candidates: dict[str, np.ndarray] = {model: wide[model].to_numpy(float)[archive] for model in wide.columns}
        for name, members in SETS.items():
            block = wide.loc[:, list(members)].to_numpy(float)[archive]
            candidates[f"ens_mean[{name}]"] = block.mean(axis=1)
            candidates[f"ens_zscore[{name}]"] = ((block - block.mean(axis=0)) / block.std(axis=0)).mean(axis=1)
            candidates[f"ens_rank[{name}]"] = stats.rankdata(block, axis=0).mean(axis=1)
        samples = rng.integers(0, len(loss), size=(args.bootstrap, len(loss)))
        reference = candidates[args.reference]
        reference_boot = np.array([score(loss[s], reference[s])["regret_at_1"] for s in samples])
        for name, guess in candidates.items():
            if not np.isfinite(guess).all():
                continue
            row = {"target": target, "candidate": name, "bank_size": len(loss)}
            row.update(score(loss, guess))
            boot = np.array([score(loss[s], guess[s])["regret_at_1"] for s in samples])
            diff = boot - reference_boot
            row.update(
                {
                    "regret_diff": float(diff.mean()),
                    "diff_ci_low": float(np.quantile(diff, 0.025)),
                    "diff_ci_high": float(np.quantile(diff, 0.975)),
                    "share_better": float(np.mean(diff < 0)),
                }
            )
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "ensembles.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    for target in ("uncheatable", "table9"):
        print(
            f"\n=== {target} / archive stratum (paired bootstrap vs {args.reference}; "
            "difference < 0 favours the candidate)"
        )
        print(
            table[table["target"].eq(target)]
            .sort_values(["regret_at_1", "frontier_predicted_rank"])
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
