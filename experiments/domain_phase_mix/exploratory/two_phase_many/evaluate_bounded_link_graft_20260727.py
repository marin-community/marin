# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test whether the bounded log-deficit link transfers from compact retained state onto HPR.

The compact retained-state model is the only surrogate in the Observatory whose *relative* rank
improves from in-sample to held out on every metric -- RMSE 0.633 to 0.282, Spearman 0.581 to 0.299,
regret@1 0.538 to 0.295 -- and it leads lower-tail optimism outright at 0.164 against 0.349 for the
same model without the bound. Since the two differ only in the link, the bound is the whole
difference, and it is worth asking whether it is a property of that design block or a portable idea.

The link fits ``log(target - floor)`` and predicts ``floor + exp(eta)``, with the floor pinned at 0.95
of the smallest observed target on the fitting rows. An additive head can predict below any entropy
floor; that is the mechanism behind out-of-support optimism, because an optimizer walks toward a region
the model calls arbitrarily good and the panel never contradicted it. Under the bounded link that
region is unreachable rather than penalized.

Ordinary out-of-fold error is the wrong instrument for this, because random folds keep the good
policies in the training set and the bound never binds. The test that does bind is **censored
extrapolation**: hold back the best policies by observed target, fit on the remainder, then ask how
much better the model claims the held-back policies are than they actually are. That signed error is
out-of-support optimism measured directly rather than proxied, and it is the quantity a deployment
proposal actually risks.
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

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "bounded_link_graft_20260727"
# Fractions of the panel censored from the top, so the model must predict into a region strictly
# better than anything it was shown.
CENSOR_FRACTIONS = (0.10, 0.20, 0.30)
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260727
ARMS = (
    ("baseline", corrected.Corrections()),
    ("bounded_link", corrected.Corrections(bounded_link=True)),
    ("bounded_returns", corrected.Corrections(bounded_returns=True)),
    (
        "recency+ledger+identifiable",
        corrected.Corrections(
            transition=corrected.TransitionForm.RECENCY_KERNEL,
            identifiable_hierarchy=True,
            normalized_family_ledger=True,
            deduplicated_ledgers=True,
        ),
    ),
    (
        "recency+ledger+identifiable+bounded_link",
        corrected.Corrections(
            transition=corrected.TransitionForm.RECENCY_KERNEL,
            identifiable_hierarchy=True,
            normalized_family_ledger=True,
            deduplicated_ledgers=True,
            bounded_link=True,
        ),
    ),
)


def select_config(dataset, dataset_id, corrections, configs, rows: np.ndarray) -> bench.Config:
    """Configuration with the lowest cross-validated error on ``rows`` only."""
    splits = bench.split_indices(dataset, dataset_id, rows, bench.SCREEN_SEED)
    best_config, best_error = configs[0], float("inf")
    for config in configs:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in splits:
            prediction[test] = corrected.fit_corrected(dataset, config, corrections, train).predict(
                dataset.weights[test]
            )
        mask = np.isfinite(prediction)
        error = float(np.sqrt(np.mean((prediction[mask] - dataset.target[mask]) ** 2)))
        if error < best_error:
            best_config, best_error = config, error
    return best_config


def censored_extrapolation(dataset, dataset_id, corrections, configs, fraction: float) -> dict[str, float]:
    """Fit on the worse policies, then measure optimism about the better ones held back.

    Selection also happens inside the retained rows, so the censored policies inform neither the
    coefficients nor the configuration nor the floor.
    """
    order = np.argsort(dataset.target)
    held_back = order[: max(1, round(fraction * dataset.n))]
    retained = np.setdiff1d(np.arange(dataset.n), held_back)
    config = select_config(dataset, dataset_id, corrections, configs, retained)
    model = corrected.fit_corrected(dataset, config, corrections, retained)
    predicted = model.predict(dataset.weights[held_back])
    observed = dataset.target[held_back]
    # Positive optimism means the model claims the policy is better than it is, which is the direction
    # that misleads a proposal. Signed mean and the worst single case are both reported because a
    # proposal acts on one policy, not on an average.
    optimism = observed - predicted
    return {
        "censor_fraction": fraction,
        "held_back": len(held_back),
        "mean_optimism": float(np.mean(optimism)),
        "max_optimism": float(np.max(optimism)),
        "censored_rmse": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "predicted_below_observed_min": float(np.min(predicted) - float(np.min(dataset.target))),
    }


def paired_bootstrap_optimism(
    observed: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Interval on the difference in mean optimism, resampling the censored policies."""
    generator = np.random.default_rng(seed)
    base = observed - baseline
    other = observed - candidate
    differences = np.asarray(
        [
            float(np.mean(other[rows]) - np.mean(base[rows]))
            for rows in (generator.integers(0, len(observed), len(observed)) for _ in range(draws))
        ]
    )
    return {
        "optimism_delta": float(np.mean(other) - np.mean(base)),
        "delta_p05": float(np.quantile(differences, 0.05)),
        "delta_p95": float(np.quantile(differences, 0.95)),
        "fraction_less_optimistic": float(np.mean(differences < 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = evaluation.config_grid(args.num_shapes)
    rows = []
    for dataset_id in evaluation.DATASETS:
        dataset = bench.load_dataset(dataset_id)
        sigma = evaluation.RUN_SIGMA[dataset_id.value]
        print(f"\n{dataset_id.value}   (run sigma {sigma:.6f})")
        splits = bench.split_indices(dataset, dataset_id, np.arange(dataset.n), bench.SCREEN_SEED)
        for arm, corrections in ARMS:
            best_error, best_summary = float("inf"), None
            for config in configs:
                prediction = corrected.corrected_oof_prediction(dataset, config, corrections, splits)
                error = float(np.sqrt(np.mean((prediction - dataset.target) ** 2)))
                if error < best_error:
                    best_error = error
                    best_summary = bench.metric_summary(dataset.target, prediction)
            assert best_summary is not None
            censored = [
                censored_extrapolation(dataset, dataset_id, corrections, configs, fraction)
                for fraction in CENSOR_FRACTIONS
            ]
            for record in censored:
                rows.append({"dataset": dataset_id.value, "arm": arm, **record, "oof_rmse": best_error})
            print(
                f"  {arm:<42} oof rmse {best_error:.6f}  "
                f"tail optimism {best_summary['lower_tail_optimism']:.6f}  "
                f"regret@1 {best_summary['regret_at_1']:.6f}"
            )
            for record in censored:
                print(
                    f"    censor top {record['censor_fraction']:.0%} ({record['held_back']:>2} policies): "
                    f"mean optimism {record['mean_optimism'] / sigma:+6.2f} sigma  "
                    f"worst {record['max_optimism'] / sigma:+6.2f} sigma  "
                    f"predicts {record['predicted_below_observed_min'] / sigma:+7.2f} sigma vs panel best"
                )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "bounded_link_graft.csv", index=False)

    print("\n" + "=" * 100)
    print("CENSORED-EXTRAPOLATION OPTIMISM, EACH ARM AGAINST BASELINE (negative is less optimistic)")
    print("=" * 100)
    for dataset_name, group in table.groupby("dataset"):
        sigma = evaluation.RUN_SIGMA[dataset_name]
        base = group[group["arm"] == "baseline"].set_index("censor_fraction")
        print(f"\n  {dataset_name}")
        for arm, _corrections in ARMS:
            if arm == "baseline":
                continue
            arm_rows = group[group["arm"] == arm].set_index("censor_fraction")
            deltas = [
                (arm_rows.loc[fraction, "mean_optimism"] - base.loc[fraction, "mean_optimism"]) / sigma
                for fraction in CENSOR_FRACTIONS
                if fraction in arm_rows.index and fraction in base.index
            ]
            improved = sum(1 for delta in deltas if delta < 0)
            rendered = "  ".join(f"{delta:+.2f}" for delta in deltas)
            print(f"    {arm:<42} {rendered}   less optimistic in {improved}/{len(deltas)} censor levels")
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
