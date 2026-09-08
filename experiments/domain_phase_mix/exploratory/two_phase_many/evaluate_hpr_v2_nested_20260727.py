# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-run the HPR corrections against the exact promoted baseline, fully nested.

The first pass at scoring these corrections used a 6-shape grid and a single-stage sweep, then reported
post-selection out-of-fold error. The promoted Observatory fit does something different: it screens 14
shapes against a bucket-resolved baseline over the ridge grid, keeps the best three shape indices, and
only then sweeps 90 structural configurations over those three. Comparing a correction under one
protocol against a baseline fitted under another confounds the correction with the protocol, so none of
those numbers settle anything.

This reproduces the promoted protocol exactly -- same shape count, same top-three screen, same ridge
and residual grids, same fold construction and screen seed -- and then runs the *whole* two-stage
selection inside each outer fold. Both numbers are reported, because the gap between them is the
post-selection bias and is worth knowing separately from the corrections.

Only nested candidates are compared, so each row differs from the one above it by one change:

1. ``original`` -- the promoted structure, reproduced.
2. ``identifiable`` -- the collinear family base replaced by a full-rank hierarchical penalty. Not an
   algebraic equivalent: the original penalty carries a constraint that can bind, so this is a
   different partial-pooling prior that happens to be indistinguishable in fit.
3. ``identifiable+ledger`` -- and the family overexposure ledger normalized by its proportional
   reference so one threshold means the same thing in both ledgers.
4. ``identifiable+ledger+recency`` -- and the transition law rewritten on normalized elapsed time.
5. ``identifiable+ledger+recency+bounded_link`` -- and the head fitted on a bounded log deficit.

Two diagnostics accompany the fit numbers. Condition number is reported under a stated convention and
across the whole shape library rather than as a single headline value, since it moves with the selected
shape while the rank deficiency does not. And the model's favourite *observed* policy is bootstrapped
over the fitting rows -- a narrower quantity than raw-optimum stability, since the configuration stays
fixed and the argmin cannot leave the panel, but a necessary condition all the same.
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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    corrected_hpr_model_20260727 as corrected,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "hpr_v2_nested_20260727"
VARIANT = bench.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
DATASETS = (
    bench.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    bench.DatasetId.THREE_HUNDRED_M_TABLE9,
    bench.DatasetId.DELPHI_3E18_UNCHEATABLE,
    bench.DatasetId.DELPHI_3E18_TABLE9,
)
# Exactly the promoted Observatory settings.
PROMOTED_SHAPE_COUNT = 12
PROMOTED_TOP_SHAPES = 3
NESTED_SEED = 20260727
OPTIMUM_BOOTSTRAP_DRAWS = 200
OPTIMUM_BOOTSTRAP_SEED = 20260727
# Singular values below this fraction of the largest are treated as structurally zero when forming the
# condition number, so the reported figure is the conditioning of the identified subspace rather than
# an artefact of the rank deficiency.
SINGULAR_TOLERANCE = 1e-12
RUN_SIGMA = {
    "300m_uncheatable": 0.000993,
    "300m_table9": 0.0031,
    "delphi_3e18_uncheatable": 0.000913,
    "delphi_3e18_table9": 0.003772,
}
ARMS = (
    ("original", corrected.Corrections()),
    ("identifiable", corrected.Corrections(identifiable_hierarchy=True, deduplicated_ledgers=True)),
    (
        "identifiable+ledger",
        corrected.Corrections(identifiable_hierarchy=True, deduplicated_ledgers=True, normalized_family_ledger=True),
    ),
    (
        "identifiable+ledger+recency",
        corrected.Corrections(
            transition=corrected.TransitionForm.RECENCY_KERNEL,
            identifiable_hierarchy=True,
            deduplicated_ledgers=True,
            normalized_family_ledger=True,
        ),
    ),
    (
        "identifiable+ledger+recency+bounded_link",
        corrected.Corrections(
            transition=corrected.TransitionForm.RECENCY_KERNEL,
            identifiable_hierarchy=True,
            deduplicated_ledgers=True,
            normalized_family_ledger=True,
            bounded_link=True,
        ),
    ),
)


def promoted_shapes() -> tuple[bench.family_grp.Shape, ...]:
    """The promoted shape library, deduplicated exactly as the Observatory does."""
    candidates = bench.family_grp.shape_candidates(bench.family_grp.Variant.BUCKET_RESOLVED, PROMOTED_SHAPE_COUNT)
    return tuple(dict.fromkeys(candidates))


def two_stage_selection(
    dataset,
    dataset_id,
    corrections: corrected.Corrections,
    shapes: tuple[bench.family_grp.Shape, ...],
    rows: np.ndarray,
) -> bench.Config:
    """Screen shapes on a bucket-resolved baseline, keep the top three, then sweep structure.

    This is the promoted selector's shape, reimplemented against the corrected model so the two are
    comparable. Everything is computed from ``rows`` alone, which is what makes it safe to call inside
    an outer fold.
    """
    splits = bench.split_indices(dataset, dataset_id, rows, bench.SCREEN_SEED)

    def cross_validated(config: bench.Config) -> float:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in splits:
            prediction[test] = corrected.fit_corrected(dataset, config, corrections, train).predict(
                dataset.weights[test]
            )
        mask = np.isfinite(prediction)
        return float(np.sqrt(np.mean((prediction[mask] - dataset.target[mask]) ** 2)))

    best_by_shape: dict[int, float] = {}
    for index, shape in enumerate(shapes):
        for l2 in bench.L2_GRID:
            error = cross_validated(bench.Config(bench.Variant.BUCKET_RESOLVED, index, shape, l2, 1.0, 0.0, 0.0))
            best_by_shape[index] = min(best_by_shape.get(index, float("inf")), error)
    top = [index for index, _error in sorted(best_by_shape.items(), key=lambda item: item[1])[:PROMOTED_TOP_SHAPES]]

    best_config, best_error = None, float("inf")
    for index in top:
        for l2 in bench.L2_GRID:
            for residual in bench.RESIDUAL_SHRINK_GRID:
                config = bench.Config(VARIANT, index, shapes[index], l2, residual, 0.0, 0.0)
                error = cross_validated(config)
                if error < best_error:
                    best_config, best_error = config, error
    assert best_config is not None
    return best_config


def nested_prediction(dataset, dataset_id, corrections, shapes) -> np.ndarray:
    """Prediction where the full two-stage selection is repeated inside every outer fold."""
    outer = bench.split_indices(dataset, dataset_id, np.arange(dataset.n), NESTED_SEED)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in outer:
        config = two_stage_selection(dataset, dataset_id, corrections, shapes, train)
        prediction[test] = corrected.fit_corrected(dataset, config, corrections, train).predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete nested prediction")
    return prediction


def conditioning_across_shapes(dataset, corrections, shapes) -> dict[str, float]:
    """Rank deficiency and conditioning over the whole shape library, under a stated convention.

    Convention: singular values of the column-centered design, dropping any below
    ``SINGULAR_TOLERANCE`` times the largest, ratio of the surviving extremes. Rank deficiency is
    invariant across the library; the condition number is not, so its range is reported.
    """
    numbers, deficiencies = [], []
    for index, shape in enumerate(shapes):
        config = bench.Config(VARIANT, index, shape, 0.0, 1.0, 0.0, 0.0)
        design = corrected.build_corrected_design(dataset, config, corrections)
        centered = design.values - design.values.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        surviving = singular[singular > singular.max() * SINGULAR_TOLERANCE]
        numbers.append(float(surviving.max() / surviving.min()))
        deficiencies.append(int(design.values.shape[1] - np.linalg.matrix_rank(design.values)))
    return {
        "columns": int(design.values.shape[1]),
        "deficiency_min": min(deficiencies),
        "deficiency_max": max(deficiencies),
        "condition_p10": float(np.quantile(numbers, 0.10)),
        "condition_median": float(np.median(numbers)),
        "condition_p90": float(np.quantile(numbers, 0.90)),
    }


def panel_argmin_coefficient_stability(
    dataset,
    dataset_id,
    corrections,
    config,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """How far the model's favourite *panel* policy moves when only the coefficients are resampled.

    Deliberately narrow, and named for what it measures rather than for what one might want it to
    measure. The configuration is held at the one selected on the complete panel, the bootstrap
    perturbs only the fitting rows and hence the linear head, and the argmin is taken over the
    observed policies rather than by continuously optimizing the surrogate over the simplex. So this
    is not raw-optimum stability: it does not repeat hyperparameter selection and it cannot move to a
    policy the panel never ran. It is still worth reporting, because a model whose favourite observed
    policy is unstable under resampling cannot support a proposal either -- it is a necessary
    condition, not a sufficient one.
    """
    generator = np.random.default_rng(seed)
    rows = np.arange(dataset.n)
    chosen = []
    for _draw in range(draws):
        sample = generator.integers(0, dataset.n, dataset.n)
        model = corrected.fit_corrected(dataset, config, corrections, sample)
        chosen.append(int(np.argmin(model.predict(dataset.weights))))
    counts = np.bincount(chosen, minlength=dataset.n)
    modal = int(np.argmax(counts))
    targets = dataset.target[np.asarray(chosen)]
    _ = rows, dataset_id
    return {
        "modal_argmin_share": float(counts[modal] / draws),
        "distinct_argmins": int((counts > 0).sum()),
        "argmin_target_spread": float(targets.max() - targets.min()),
        "argmin_target_mean_excess": float(targets.mean() - dataset.target.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--draws", type=int, default=OPTIMUM_BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shapes = promoted_shapes()
    print(f"promoted shape library: {len(shapes)} shapes, top {PROMOTED_TOP_SHAPES} retained after screening")
    print(f"screen grid {len(shapes) * len(bench.L2_GRID)} baseline configs, then ")
    print(f"{PROMOTED_TOP_SHAPES * len(bench.L2_GRID) * len(bench.RESIDUAL_SHRINK_GRID)} structural configs\n")

    rows = []
    for dataset_id in DATASETS:
        dataset = bench.load_dataset(dataset_id)
        sigma = RUN_SIGMA[dataset_id.value]
        print(f"\n{dataset_id.value}   (run sigma {sigma:.6f})")
        for arm, corrections in ARMS:
            config = two_stage_selection(dataset, dataset_id, corrections, shapes, np.arange(dataset.n))
            splits = bench.split_indices(dataset, dataset_id, np.arange(dataset.n), bench.SCREEN_SEED)
            post = bench.metric_summary(
                dataset.target,
                corrected.corrected_oof_prediction(dataset, config, corrections, splits),
            )
            nested = bench.metric_summary(dataset.target, nested_prediction(dataset, dataset_id, corrections, shapes))
            conditioning = conditioning_across_shapes(dataset, corrections, shapes)
            stability = panel_argmin_coefficient_stability(
                dataset, dataset_id, corrections, config, args.draws, OPTIMUM_BOOTSTRAP_SEED
            )
            rows.append(
                {
                    "dataset": dataset_id.value,
                    "arm": arm,
                    "shape_index": config.shape_index,
                    "l2": config.l2,
                    "residual_shrink": config.residual_shrink,
                    **{f"post_{key}": value for key, value in post.items()},
                    **{f"nested_{key}": value for key, value in nested.items()},
                    **conditioning,
                    **stability,
                }
            )
            print(
                f"  {arm:<42} post {post['rmse']:.6f}  nested {nested['rmse']:.6f}  "
                f"bias {(post['rmse'] - nested['rmse']) / sigma:+.2f}s  "
                f"nested regret@1 {nested['regret_at_1']:.6f}"
            )
            print(
                f"    {'':<40} deficiency {conditioning['deficiency_min']}-{conditioning['deficiency_max']} "
                f"of {conditioning['columns']}  condition {conditioning['condition_p10']:.3g}"
                f"-{conditioning['condition_p90']:.3g} (median {conditioning['condition_median']:.3g})"
            )
            print(
                f"    {'':<40} argmin stable in {stability['modal_argmin_share'] * 100:.0f}% of draws, "
                f"{stability['distinct_argmins']} distinct choices, "
                f"chosen policies span {stability['argmin_target_spread'] / sigma:.1f}s"
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "hpr_v2_nested.csv", index=False)

    print("\n" + "=" * 104)
    print("NESTED FIT AND OPTIMUM STABILITY AGAINST THE REPRODUCED BASELINE (negative rmse delta is better)")
    print("=" * 104)
    for dataset_name, group in table.groupby("dataset"):
        sigma = RUN_SIGMA[dataset_name]
        base = group[group["arm"] == "original"].iloc[0]
        print(f"\n  {dataset_name}   (baseline nested rmse {base['nested_rmse']:.6f})")
        for _, row in group.iterrows():
            if row["arm"] == "original":
                continue
            print(
                f"    {row['arm']:<42} nested {(row['nested_rmse'] - base['nested_rmse']) / sigma:+.2f}s   "
                f"regret@1 {(row['nested_regret_at_1'] - base['nested_regret_at_1']) / sigma:+.2f}s   "
                f"argmin stability {base['modal_argmin_share'] * 100:.0f}% -> {row['modal_argmin_share'] * 100:.0f}%"
            )
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
