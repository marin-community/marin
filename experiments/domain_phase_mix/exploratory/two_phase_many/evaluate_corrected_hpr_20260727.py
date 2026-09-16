# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score each HPR audit fix on fit and on optimum quality, under an honest selection protocol.

Two questions, kept apart because they have different answers.

**Does each fix help?** Every correction is applied alone and then all together, over the same
configuration grid and the same folds, so any change in error is attributable to one structural
change rather than to a bundle. A fix that repairs an invariance while costing fit is still worth
knowing about; the point is to measure the price, not to assume there is none.

**Is the reported error real?** The current protocol screens shapes, keeps the best few, searches
structural configurations, then reports out-of-fold error with the winner held fixed. The outer folds
never repeat the selection, so the reported number is post-selection and optimistically biased. This
runs the selection *inside* each outer fold as well, and reports both, because the gap between them
is the size of the bias and is itself the interesting quantity.

Optimum quality is reported alongside fit and never collapsed into it. ``regret_at_1`` is the real
cost of deploying the model's favourite panel policy, which is the closest thing to deployment regret
that can be measured without new training. The lower-tail statistics say whether the model is
optimistic exactly where an optimizer would push, which is the failure mode a good aggregate RMSE
hides.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
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

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "corrected_hpr_evaluation_20260727"
VARIANT = bench.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
DATASETS = (
    bench.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    bench.DatasetId.THREE_HUNDRED_M_TABLE9,
    bench.DatasetId.DELPHI_3E18_UNCHEATABLE,
    bench.DatasetId.DELPHI_3E18_TABLE9,
)
OUTER_SPLITS = 5
NESTED_SEED = 20260727
# Run-to-run standard deviation of the targets, for reading effect sizes as multiples of noise.
RUN_SIGMA = {
    "300m_uncheatable": 0.000993,
    "300m_table9": 0.0031,
    "delphi_3e18_uncheatable": 0.000913,
    "delphi_3e18_table9": 0.003772,
}


def correction_variants() -> list[corrected.Corrections]:
    """Baseline, each fix alone, both transition forms, then every fix together under each form."""
    switches = [field.name for field in fields(corrected.Corrections) if field.name != "transition"]
    variants = [corrected.Corrections()]
    variants.extend(corrected.Corrections(**{name: True}) for name in switches)
    everything = dict.fromkeys(switches, True)
    for form in corrected.TransitionForm:
        if form is corrected.TransitionForm.LEGACY:
            continue
        variants.append(corrected.Corrections(transition=form))
        variants.append(corrected.Corrections(transition=form, **everything))
    variants.append(corrected.Corrections(**everything))
    return variants


def config_grid(num_shapes: int) -> list[bench.Config]:
    shapes = bench.family_grp.shape_candidates(bench.family_grp.Variant.BUCKET_RESOLVED, num_shapes)
    return [
        bench.Config(VARIANT, index, shape, l2, residual, 0.0, 0.0)
        for index, shape in enumerate(shapes)
        for l2 in bench.L2_GRID
        for residual in bench.RESIDUAL_SHRINK_GRID
    ]


def select_config(
    dataset,
    dataset_id,
    corrections: corrected.Corrections,
    configs: list[bench.Config],
    rows: np.ndarray,
) -> bench.Config:
    """Pick the configuration with the lowest inner out-of-fold error on ``rows`` only."""
    inner = bench.split_indices(dataset, dataset_id, rows, bench.SCREEN_SEED)
    observed = dataset.target
    best_config, best_error = configs[0], float("inf")
    for config in configs:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in inner:
            prediction[test] = corrected.fit_corrected(dataset, config, corrections, train).predict(
                dataset.weights[test]
            )
        mask = np.isfinite(prediction)
        error = float(np.sqrt(np.mean((prediction[mask] - observed[mask]) ** 2)))
        if error < best_error:
            best_config, best_error = config, error
    return best_config


def post_selection_scores(
    dataset,
    dataset_id,
    corrections: corrected.Corrections,
    configs: list[bench.Config],
) -> tuple[dict[str, float | int], bench.Config]:
    """The current protocol: select on all rows, then report out-of-fold error for that winner."""
    splits = bench.split_indices(dataset, dataset_id, np.arange(dataset.n), bench.SCREEN_SEED)
    best_config, best_error, best_prediction = configs[0], float("inf"), None
    for config in configs:
        prediction = corrected.corrected_oof_prediction(dataset, config, corrections, splits)
        error = float(np.sqrt(np.mean((prediction - dataset.target) ** 2)))
        if error < best_error:
            best_config, best_error, best_prediction = config, error, prediction
    assert best_prediction is not None
    return bench.metric_summary(dataset.target, best_prediction), best_config


def nested_scores(
    dataset,
    dataset_id,
    corrections: corrected.Corrections,
    configs: list[bench.Config],
) -> dict[str, float | int]:
    """Selection repeated inside every outer fold, so no held-out row informs its own predictor."""
    outer = bench.split_indices(dataset, dataset_id, np.arange(dataset.n), NESTED_SEED)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in outer:
        config = select_config(dataset, dataset_id, corrections, configs, train)
        prediction[test] = corrected.fit_corrected(dataset, config, corrections, train).predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete nested prediction")
    return bench.metric_summary(dataset.target, prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=6)
    parser.add_argument("--nested", action="store_true", help="Also run the fully nested protocol.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = config_grid(args.num_shapes)
    variants = correction_variants()
    print(f"{len(configs)} configurations x {len(variants)} correction sets x {len(DATASETS)} datasets")

    rows = []
    for dataset_id in DATASETS:
        dataset = bench.load_dataset(dataset_id)
        print(f"\n{dataset_id.value}")
        for corrections in variants:
            summary, winner = post_selection_scores(dataset, dataset_id, corrections, configs)
            diagnostics = corrected.design_diagnostics(dataset, winner, corrections)
            record = {
                "dataset": dataset_id.value,
                "corrections": corrections.label(),
                "protocol": "post_selection",
                **summary,
                **diagnostics,
                "exponent": winner.shape.exponent,
                "late_multiplier": winner.shape.late_multiplier,
                "forgetting_rate": winner.shape.forgetting_rate,
                "penalty_threshold": winner.shape.penalty_threshold,
                "l2": winner.l2,
                "residual_shrink": winner.residual_shrink,
            }
            rows.append(record)
            print(
                f"  {corrections.label():<30} rmse {summary['rmse']:.6f}  regret@1 {summary['regret_at_1']:.6f}  "
                f"rank {diagnostics['rank']}/{diagnostics['columns']}  edof {diagnostics['effective_dof']:.1f}"
            )
            if args.nested:
                nested = nested_scores(dataset, dataset_id, corrections, configs)
                rows.append(
                    {
                        "dataset": dataset_id.value,
                        "corrections": corrections.label(),
                        "protocol": "nested",
                        **nested,
                    }
                )
                print(
                    f"  {'  -> nested':<30} rmse {nested['rmse']:.6f}  regret@1 {nested['regret_at_1']:.6f}  "
                    f"(post-selection bias {summary['rmse'] - nested['rmse']:+.6f})"
                )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "corrected_hpr_scores.csv", index=False)

    print("\n" + "=" * 100)
    print("FIT AND OPTIMUM QUALITY BY FIX, AS A DELTA AGAINST BASELINE (negative is better)")
    print("=" * 100)
    for dataset_name, group in table[table["protocol"] == "post_selection"].groupby("dataset"):
        sigma = RUN_SIGMA[dataset_name]
        base = group[group["corrections"] == "baseline"].iloc[0]
        print(f"\n  {dataset_name}   (baseline rmse {base['rmse']:.6f}, regret@1 {base['regret_at_1']:.6f})")
        ordered = group[group["corrections"] != "baseline"].sort_values("rmse")
        for _, row in ordered.iterrows():
            rmse_delta = row["rmse"] - base["rmse"]
            regret_delta = row["regret_at_1"] - base["regret_at_1"]
            print(
                f"    {row['corrections']:<30} rmse {rmse_delta:+.6f} ({rmse_delta / sigma:+.2f} sigma)   "
                f"regret@1 {regret_delta:+.6f} ({regret_delta / sigma:+.2f} sigma)"
            )
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
