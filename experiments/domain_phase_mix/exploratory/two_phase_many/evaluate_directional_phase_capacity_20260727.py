# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test whether family-level signed phase directions supply real phase-order capacity.

The recency-kernel correction turns out to be effective exposure with a single fitted late-to-early
ratio -- verified to machine precision, gamma = 5.0816 shared by every bucket -- times an
aggregate-only curvature. That makes it rank-one in the phase tilt: it can say late loading helps
uniformly, and it cannot say one bucket belongs late while another belongs early. So it is a
schedule-invariance correction and not a phase-order mechanism, and phase-order capacity has to come
from somewhere else.

The candidate is one signed coefficient per family on that family's summed tilt, with a genuinely
unconstrained sign. Three things decide whether it is real capacity or decoration.

Are the coefficients nonzero once the solver is free to set them either way? Are their signs stable
under resampling, since a direction that flips sign between bootstrap draws is noise wearing a
mechanism's clothes? And does the term help specifically where phase order is the only thing varying --
measured on two-phase rows alone, since tied rows carry no tilt and dilute the comparison.
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
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    evaluate_hpr_v2_nested_20260727 as nested,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "directional_phase_capacity_20260727"
DATASETS = (
    bench.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    bench.DatasetId.THREE_HUNDRED_M_TABLE9,
    bench.DatasetId.DELPHI_3E18_UNCHEATABLE,
    bench.DatasetId.DELPHI_3E18_TABLE9,
)
BOOTSTRAP_DRAWS = 300
BOOTSTRAP_SEED = 20260727
TIED_TOLERANCE = 1e-9
# A sign is called stable when it agrees with the full-panel sign in at least this share of resamples.
SIGN_STABILITY_THRESHOLD = 0.90
BASE = corrected.Corrections(
    transition=corrected.TransitionForm.RECENCY_KERNEL,
    identifiable_hierarchy=True,
    deduplicated_ledgers=True,
    normalized_family_ledger=True,
)
DIRECTIONAL = corrected.Corrections(
    transition=corrected.TransitionForm.RECENCY_KERNEL,
    identifiable_hierarchy=True,
    deduplicated_ledgers=True,
    normalized_family_ledger=True,
    smooth_phase_cost=True,
)


def two_phase_rows(dataset) -> np.ndarray:
    """Rows whose two phase mixtures actually differ, which are the only ones carrying a tilt."""
    tilt = np.abs(dataset.weights[:, 1, :] - dataset.weights[:, 0, :]).sum(axis=1)
    return np.flatnonzero(tilt > TIED_TOLERANCE)


def directional_coefficients(dataset, config, corrections, indices: np.ndarray) -> dict[str, float]:
    """Fitted signed direction per family, keyed by family name."""
    design = corrected.build_corrected_design(dataset, config, corrections)
    model = corrected.fit_corrected(dataset, config, corrections, indices)
    return {
        name.split(":", 1)[1]: float(model.coefficients[index])
        for index, name in enumerate(design.names)
        if name.startswith("phase_direction:")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shapes = nested.promoted_shapes()
    rows = []
    for dataset_id in DATASETS:
        dataset = bench.load_dataset(dataset_id)
        sigma = nested.RUN_SIGMA[dataset_id.value]
        tilted = two_phase_rows(dataset)
        print(f"\n{dataset_id.value}   ({len(tilted)} two-phase rows of {dataset.n}, run sigma {sigma:.6f})")

        errors = {}
        for arm, corrections in (("no_direction", BASE), ("with_direction", DIRECTIONAL)):
            config = nested.two_stage_selection(dataset, dataset_id, corrections, shapes, np.arange(dataset.n))
            splits = bench.split_indices(dataset, dataset_id, np.arange(dataset.n), bench.SCREEN_SEED)
            prediction = corrected.corrected_oof_prediction(dataset, config, corrections, splits)
            residual = prediction - dataset.target
            errors[arm] = {
                "config": config,
                "all_rmse": float(np.sqrt(np.mean(residual**2))),
                "two_phase_rmse": float(np.sqrt(np.mean(residual[tilted] ** 2))),
            }
            print(
                f"  {arm:<16} oof rmse all {errors[arm]['all_rmse']:.6f}   "
                f"two-phase rows only {errors[arm]['two_phase_rmse']:.6f}"
            )

        delta_all = errors["with_direction"]["all_rmse"] - errors["no_direction"]["all_rmse"]
        delta_tilt = errors["with_direction"]["two_phase_rmse"] - errors["no_direction"]["two_phase_rmse"]
        print(
            f"  adding the direction: all rows {delta_all / sigma:+.2f}s, " f"two-phase rows {delta_tilt / sigma:+.2f}s"
        )

        config = errors["with_direction"]["config"]
        full = directional_coefficients(dataset, config, DIRECTIONAL, np.arange(dataset.n))
        generator = np.random.default_rng(BOOTSTRAP_SEED)
        draws: dict[str, list[float]] = {name: [] for name in full}
        for _draw in range(args.draws):
            sample = generator.integers(0, dataset.n, dataset.n)
            for name, value in directional_coefficients(dataset, config, DIRECTIONAL, sample).items():
                draws[name].append(value)
        print("  fitted family directions (positive means late loading predicts higher BPB, i.e. worse):")
        for name, value in full.items():
            series = np.asarray(draws[name])
            nonzero = float(np.mean(np.abs(series) > 1e-10))
            agree = float(np.mean(np.sign(series) == np.sign(value))) if value != 0.0 else float("nan")
            stable = agree >= SIGN_STABILITY_THRESHOLD
            rows.append(
                {
                    "dataset": dataset_id.value,
                    "family": name,
                    "coefficient": value,
                    "bootstrap_nonzero_share": nonzero,
                    "bootstrap_sign_agreement": agree,
                    "sign_stable": bool(stable),
                    "delta_rmse_two_phase": delta_tilt,
                    "delta_rmse_all": delta_all,
                }
            )
            print(
                f"    {name:<16} {value:+.6f}   nonzero in {nonzero * 100:5.1f}% of draws   "
                f"sign agrees {agree * 100:5.1f}%   -> {'STABLE' if stable else 'not stable'}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "directional_phase_capacity.csv", index=False)

    print("\n" + "=" * 100)
    print("IS THERE IDENTIFIABLE FAMILY-LEVEL PHASE-ORDER CAPACITY?")
    print("=" * 100)
    print(f"  families with a nonzero fitted direction : {int((table['coefficient'].abs() > 1e-10).sum())}/{len(table)}")
    print(f"  families with a bootstrap-stable sign    : {int(table['sign_stable'].sum())}/{len(table)}")
    helped = table.drop_duplicates("dataset")
    print(
        f"  cells where the direction improves two-phase rows: "
        f"{int((helped['delta_rmse_two_phase'] < 0).sum())}/{len(helped)}"
    )
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
