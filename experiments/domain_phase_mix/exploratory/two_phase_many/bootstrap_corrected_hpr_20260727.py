# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resample the corrected-HPR fix comparison, because every fix's effect is smaller than one sigma.

The ablation reports point differences of 0.05 to 0.8 run sigma. At that size a point estimate says
nothing: the question is whether the sign is stable under resampling, not what the third decimal is.
This holds the out-of-fold predictions fixed and resamples the *policy rows* -- the unit that varies
independently, since each row is one trained run -- to get a paired interval on the RMSE difference
against baseline.

Paired on the same resampled rows, so the correlation between two models' errors on the same policy
does not inflate the interval. Reported as the fraction of resamples where the fix wins, which is
directly the confidence that its sign is real.

Two fixes are excluded from the fit comparison because arithmetic already settled them:
deduplicated ledgers changes nothing on a partition with no singleton families, and the identifiable
hierarchy is a reparameterization whose predictions are unchanged to seven digits. Neither needs an
interval; they are correctness fixes whose justification is the rank of the design, not its error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    corrected_hpr_model_20260727 as corrected,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    evaluate_corrected_hpr_20260727 as evaluation,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "corrected_hpr_bootstrap_20260727"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260727
ARMS = (
    ("baseline", corrected.Corrections()),
    ("recency_kernel", corrected.Corrections(transition=corrected.TransitionForm.RECENCY_KERNEL)),
    ("tied_invariant", corrected.Corrections(transition=corrected.TransitionForm.TIED_INVARIANT)),
    ("normalized_family_ledger", corrected.Corrections(normalized_family_ledger=True)),
    ("bounded_returns", corrected.Corrections(bounded_returns=True)),
    ("smooth_phase_cost", corrected.Corrections(smooth_phase_cost=True)),
    (
        "recency_kernel+normalized_family_ledger",
        corrected.Corrections(
            transition=corrected.TransitionForm.RECENCY_KERNEL,
            normalized_family_ledger=True,
            identifiable_hierarchy=True,
            deduplicated_ledgers=True,
        ),
    ),
)


def best_oof_prediction(dataset, dataset_id, corrections, configs) -> np.ndarray:
    """Out-of-fold prediction for the configuration this correction set would select."""
    splits = bench.split_indices(dataset, dataset_id, np.arange(dataset.n), bench.SCREEN_SEED)
    best_error, best_prediction = float("inf"), None
    for config in configs:
        prediction = corrected.corrected_oof_prediction(dataset, config, corrections, splits)
        error = float(np.sqrt(np.mean((prediction - dataset.target) ** 2)))
        if error < best_error:
            best_error, best_prediction = error, prediction
    assert best_prediction is not None
    return best_prediction


def paired_bootstrap(
    observed: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Interval on the RMSE difference, resampling policy rows and pairing the two models on them."""
    generator = np.random.default_rng(seed)
    baseline_error = (baseline - observed) ** 2
    candidate_error = (candidate - observed) ** 2
    differences = np.empty(draws, dtype=float)
    for draw in range(draws):
        rows = generator.integers(0, len(observed), len(observed))
        differences[draw] = np.sqrt(candidate_error[rows].mean()) - np.sqrt(baseline_error[rows].mean())
    return {
        "rmse_delta": float(np.sqrt(candidate_error.mean()) - np.sqrt(baseline_error.mean())),
        "delta_p05": float(np.quantile(differences, 0.05)),
        "delta_p95": float(np.quantile(differences, 0.95)),
        "fraction_improving": float(np.mean(differences < 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=6)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = evaluation.config_grid(args.num_shapes)
    rows = []
    for dataset_id in evaluation.DATASETS:
        dataset = bench.load_dataset(dataset_id)
        sigma = evaluation.RUN_SIGMA[dataset_id.value]
        predictions = {
            name: best_oof_prediction(dataset, dataset_id, corrections, configs) for name, corrections in ARMS
        }
        print(f"\n{dataset_id.value}   (run sigma {sigma:.6f}, {dataset.n} policies)")
        for name, _corrections in ARMS:
            if name == "baseline":
                continue
            interval = paired_bootstrap(
                dataset.target,
                predictions["baseline"],
                predictions[name],
                args.draws,
                BOOTSTRAP_SEED,
            )
            stable = interval["delta_p95"] < 0.0 or interval["delta_p05"] > 0.0
            verdict = ("improves" if interval["rmse_delta"] < 0 else "worsens") if stable else "not resolved"
            rows.append({"dataset": dataset_id.value, "arm": name, **interval, "sign_resolved": stable})
            print(
                f"  {name:<40} {interval['rmse_delta'] / sigma:+.2f} sigma  "
                f"[{interval['delta_p05'] / sigma:+.2f}, {interval['delta_p95'] / sigma:+.2f}]  "
                f"P(better) {interval['fraction_improving']:.3f}  -> {verdict}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "corrected_hpr_bootstrap.csv", index=False)

    print("\n" + "=" * 100)
    print("HOW MANY OF THE FOUR CELLS RESOLVE THE SIGN OF EACH FIX")
    print("=" * 100)
    for arm, group in table.groupby("arm"):
        resolved = group[group["sign_resolved"]]
        improving = resolved[resolved["rmse_delta"] < 0]
        print(
            f"  {arm:<40} resolved {len(resolved)}/{len(group)}   "
            f"improving {len(improving)}   worsening {len(resolved) - len(improving)}"
        )
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
