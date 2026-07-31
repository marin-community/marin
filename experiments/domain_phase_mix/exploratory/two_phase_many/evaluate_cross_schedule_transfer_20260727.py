# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test whether the recency-kernel transition law transfers across learning-rate schedules.

The corrected transition law is justified by an invariance rather than by fit: on the four dolma fit
panels it is roughly a wash, improving the two 300M cells and costing 0.5-0.8 run sigma on Delphi.
Its actual claim is that it can be carried to a run with a different phase boundary, which no
single-boundary panel can test -- every row of those panels splits 80/20, so the legacy law's
boundary dependence is invisible in-sample.

The two StarCoder swarms are the test that does discriminate. They share both domains, the same target
metric, and the same construction, and differ only in learning-rate schedule and phase split: 50/50
cosine against 80/20 warmup-stable-decay. Under the legacy law they select very different transition
parameters, which is the signature of a parameter absorbing a schedule rather than describing a
mechanism. If the recency kernel is doing what it claims, two things should improve: predictions should
carry across schedules better, and the *selected shape* should stop depending on which schedule it was
selected on.

Transfer is scored by rank correlation first, because the two schedules put the target on different
absolute levels and a rank statistic is immune to that offset. RMSE is reported after an
intercept-only recalibration on the target schedule, which is the weakest calibration that makes the
comparison meaningful; anything more would hide the transfer failure it is meant to measure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    corrected_hpr_model_20260727 as corrected,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "cross_schedule_transfer_20260727"
VARIANT = bench.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
N_SPLITS = 5
SELECT_SEED = 20260727
# Both StarCoder swarms have two domains, so every family is a singleton and the duplicate-ledger
# defect is live here in a way it is not on the three-family dolma panels. Deduplication is therefore
# switched on for every arm; leaving it off would compare two rank-deficient designs.
ARMS = (
    ("legacy", corrected.TransitionForm.LEGACY),
    ("recency_kernel", corrected.TransitionForm.RECENCY_KERNEL),
    ("tied_invariant", corrected.TransitionForm.TIED_INVARIANT),
)


def as_family_dataset(raw) -> bench.family_grp.Dataset:
    """Convert a StarCoder surface, giving every domain its own family.

    The shared converter hardcodes the 300M domain ordering and rejects anything else. StarCoder has
    two domains and no semantic grouping to impose, so singleton families are the only honest
    partition -- which is also why the duplicate-ledger defect is live on these swarms.
    """
    return bench.family_grp.Dataset(
        frame=raw.frame,
        target=np.asarray(raw.y, dtype=float),
        weights=np.asarray(raw.weights, dtype=float),
        c0=np.asarray(raw.c0, dtype=float),
        c1=np.asarray(raw.c1, dtype=float),
        domains=tuple(raw.domain_names),
        family_names=tuple(raw.domain_names),
        family_members=tuple(np.array([index]) for index in range(len(raw.domain_names))),
        quality=np.full(raw.m, -1, dtype=int),
    )


def load_schedules() -> dict[str, bench.family_grp.Dataset]:
    """The two StarCoder surfaces, converted to the family dataset the model expects."""
    from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: PLC0415
        export_mixture_fit_observatory as observatory,
    )

    cosine = observatory.load_cosine_starcoder()
    wsd80 = observatory.load_wsd80_starcoder(cosine)
    return {
        "cosine_50_50": as_family_dataset(cosine),
        "wsd80_80_20": as_family_dataset(wsd80),
    }


def config_grid(num_shapes: int) -> list[bench.Config]:
    shapes = bench.family_grp.shape_candidates(bench.family_grp.Variant.BUCKET_RESOLVED, num_shapes)
    return [
        bench.Config(VARIANT, index, shape, l2, residual, 0.0, 0.0)
        for index, shape in enumerate(shapes)
        for l2 in bench.L2_GRID
        for residual in bench.RESIDUAL_SHRINK_GRID
    ]


def select_config(dataset, corrections, configs: list[bench.Config]) -> tuple[bench.Config, float]:
    """Lowest cross-validated error on this schedule alone, with its score."""
    splits = bench.family_grp.kfold_indices(np.arange(dataset.n), N_SPLITS, SELECT_SEED)
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
    return best_config, best_error


def shape_landscape(dataset, corrections, configs: list[bench.Config]) -> dict[int, float]:
    """Best cross-validated error achievable at each shape index on one schedule.

    Two schedules agreeing on a single argmin could be luck with fourteen candidates. Comparing the
    whole ranking is the robust version of the question: a transition law that describes a mechanism
    rather than absorbing a schedule should induce the same ordering over shapes on both.
    """
    splits = bench.family_grp.kfold_indices(np.arange(dataset.n), N_SPLITS, SELECT_SEED)
    best: dict[int, float] = {}
    for config in configs:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in splits:
            prediction[test] = corrected.fit_corrected(dataset, config, corrections, train).predict(
                dataset.weights[test]
            )
        mask = np.isfinite(prediction)
        error = float(np.sqrt(np.mean((prediction[mask] - dataset.target[mask]) ** 2)))
        current = best.get(config.shape_index)
        if current is None or error < current:
            best[config.shape_index] = error
    return best


def transfer_scores(source, target, config, corrections) -> dict[str, float]:
    """Fit on the source schedule, predict the target schedule.

    Only the intercept is refitted on the target, which absorbs the difference in absolute level
    between two learning-rate schedules without letting the coefficients re-learn the new schedule.
    """
    model = corrected.fit_corrected(source, config, corrections, np.arange(source.n))
    predicted = model.predict(target.weights)
    observed = np.asarray(target.target, dtype=float)
    recentered = predicted + (observed.mean() - predicted.mean())
    return {
        "transfer_spearman": float(spearmanr(observed, predicted).statistic),
        "transfer_rmse_recentered": float(np.sqrt(np.mean((recentered - observed) ** 2))),
        "transfer_regret_at_1": float(observed[int(np.argmin(predicted))] - observed.min()),
        "target_spread": float(observed.max() - observed.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=14)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    schedules = load_schedules()
    configs = config_grid(args.num_shapes)
    for name, dataset in schedules.items():
        late = corrected.late_token_fraction(dataset)
        print(
            f"{name:<14} n={dataset.n:<5} domains={dataset.domains}  families={dataset.family_names}  "
            f"late token fraction {late.min():.3f}-{late.max():.3f}"
        )
    print(f"\n{len(configs)} configurations per arm\n")

    rows = []
    for arm, form in ARMS:
        corrections = corrected.Corrections(transition=form, deduplicated_ledgers=True)
        selected = {}
        for name, dataset in schedules.items():
            config, error = select_config(dataset, corrections, configs)
            selected[name] = config
            rows.append(
                {
                    "arm": arm,
                    "role": "within_schedule",
                    "source": name,
                    "target": name,
                    "shape_index": config.shape_index,
                    "late_multiplier": config.shape.late_multiplier,
                    "forgetting_rate": config.shape.forgetting_rate,
                    "exponent": config.shape.exponent,
                    "cv_rmse": error,
                }
            )
        names = list(schedules)
        agree = selected[names[0]].shape_index == selected[names[1]].shape_index
        print(f"{arm}")
        print(
            f"  shape selected on {names[0]}: index {selected[names[0]].shape_index} "
            f"(eta {selected[names[0]].shape.late_multiplier:.4f}, "
            f"lambda {selected[names[0]].shape.forgetting_rate:.4g})"
        )
        print(
            f"  shape selected on {names[1]}: index {selected[names[1]].shape_index} "
            f"(eta {selected[names[1]].shape.late_multiplier:.4f}, "
            f"lambda {selected[names[1]].shape.forgetting_rate:.4g})"
        )
        print(f"  schedules agree on the selected shape: {'YES' if agree else 'NO'}")
        landscapes = {name: shape_landscape(schedules[name], corrections, configs) for name in names}
        shared = sorted(set(landscapes[names[0]]) & set(landscapes[names[1]]))
        landscape_agreement = float(
            spearmanr(
                [landscapes[names[0]][index] for index in shared],
                [landscapes[names[1]][index] for index in shared],
            ).statistic
        )
        rows.append(
            {
                "arm": arm,
                "role": "shape_landscape",
                "source": names[0],
                "target": names[1],
                "landscape_spearman": landscape_agreement,
                "shapes_compared": len(shared),
            }
        )
        print(
            f"  shape-ranking agreement across schedules: spearman {landscape_agreement:+.4f} "
            f"over {len(shared)} shapes"
        )
        for source_name, target_name in ((names[0], names[1]), (names[1], names[0])):
            scores = transfer_scores(
                schedules[source_name],
                schedules[target_name],
                selected[source_name],
                corrections,
            )
            rows.append(
                {
                    "arm": arm,
                    "role": "cross_schedule",
                    "source": source_name,
                    "target": target_name,
                    "shape_index": selected[source_name].shape_index,
                    **scores,
                }
            )
            print(
                f"  {source_name} -> {target_name}: spearman {scores['transfer_spearman']:+.4f}  "
                f"rmse {scores['transfer_rmse_recentered']:.6f}  "
                f"regret@1 {scores['transfer_regret_at_1']:.6f} "
                f"(target spread {scores['target_spread']:.6f})"
            )
        print()

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "cross_schedule_transfer.csv", index=False)

    print("=" * 96)
    print("CROSS-SCHEDULE TRANSFER, EACH ARM AGAINST THE LEGACY LAW")
    print("=" * 96)
    crossed = table[table["role"] == "cross_schedule"]
    baseline = crossed[crossed["arm"] == "legacy"].set_index(["source", "target"])
    for arm, _form in ARMS:
        if arm == "legacy":
            continue
        group = crossed[crossed["arm"] == arm].set_index(["source", "target"])
        print(f"\n  {arm}")
        for key in group.index:
            print(
                f"    {key[0]} -> {key[1]}: "
                f"spearman {baseline.loc[key, 'transfer_spearman']:+.4f} -> "
                f"{group.loc[key, 'transfer_spearman']:+.4f}   "
                f"rmse {baseline.loc[key, 'transfer_rmse_recentered']:.6f} -> "
                f"{group.loc[key, 'transfer_rmse_recentered']:.6f}   "
                f"regret@1 {baseline.loc[key, 'transfer_regret_at_1']:.6f} -> "
                f"{group.loc[key, 'transfer_regret_at_1']:.6f}"
            )
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
