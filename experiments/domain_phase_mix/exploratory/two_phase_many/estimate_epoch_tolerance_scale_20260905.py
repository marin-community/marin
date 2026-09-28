# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate a scale-dependent repetition tolerance from two many-bucket swarms with a shared WSPU shape.

The Qwen3 360M/1.6B swarm (``delphi_3e18_39bucket``) is the reference. A second swarm at another token
budget (the Llama 200M/6B swarm ``300m_39bucket``, and optionally the 160M/1.2B swarm) is fitted with the
same WSPU shape (rate, power, threshold) but with its materialized epochs divided by a free scale ``s``:
the swarm's response at ``E`` is modelled as the reference response at ``E / s``. Heads are fitted per
component per swarm as in the benchmark (NNLS over the ridge grid, three mixture-blocked inner folds). The
selection objective is the inner-CV RMSE in units of each component's swarm SD, averaged over components,
summed over swarms; the profile over ``s`` at the best shared shape gives the tolerance ratio, repeated over
fold seeds.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_fits_20260905 as fits,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_observatory_registry_20260902 as registry,
)

LOGGER = logging.getLogger("epoch_tolerance")
MODULE_NAME = "experiments.domain_phase_mix.exploratory.two_phase_many.estimate_epoch_tolerance_scale_20260905"
MODEL_ID = "weibull_softplus_unscaled"
REFERENCE_PANEL = "delphi_3e18_39bucket"
SCALED_PANELS = ("300m_39bucket", "60m_39bucket")
PANEL_BUDGETS = {"delphi_3e18_39bucket": 1.6e9, "300m_39bucket": 6.0e9, "60m_39bucket": 1.2e9}
PANEL_LABELS = {
    "delphi_3e18_39bucket": "Qwen3 360M/1.6B",
    "300m_39bucket": "Llama 200M/6B",
    "60m_39bucket": "Llama 160M/1.2B",
}
TARGETS = ("uncheatable", "table9")
SCALE_GRID = tuple(float(2.0**k) for k in np.arange(-2.0, 4.01, 0.5))
FOLD_SEEDS = (0, 1, 2)
INNER_FOLDS = benchmark.INNER_FOLDS
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "epoch_tolerance_scale_20260905"


def scaled_features(features: models.Features, scale: float) -> models.Features:
    """The same swarm with every materialized epoch count divided by ``scale``."""
    if scale == 1.0:
        return features
    return dataclasses.replace(
        features, exposures=features.exposures / scale, label=f"{features.label}|epoch_scale:{scale:.6g}"
    )


def inner_folds(panel: benchmark.BenchPanel, seed: int) -> models.InnerFolds:
    rows = np.arange(panel.rows)
    labels = benchmark.olmix_benchmark.block_labels(
        panel.features.weights, INNER_FOLDS, benchmark.HELDOUT_INNER_SEED + 7 * seed
    )
    return tuple((rows[labels != index], rows[labels == index]) for index in range(INNER_FOLDS))


def cv_rmse_matrix(panel_name: str, target: str, shape_index: int, scales: tuple[float, ...], seed: int) -> np.ndarray:
    """Inner-CV RMSE (best ridge) per component and scale for one shape: shape (len(scales), components)."""
    panel = benchmark.load_panel(panel_name)
    entry = registry.ENTRY_BY_ID[MODEL_ID]
    group = panel.group(target)
    base = dataclasses.replace(registry.apply_transform(panel.features, entry), component=group.components[0])
    model = entry.build(base)
    shape = model.candidate_shapes(base)[shape_index]
    spec = model.head_for(shape)
    folds = inner_folds(panel, seed)
    responses = group.outcomes
    count = responses.shape[1]
    result = np.full((len(scales), count), np.inf)
    validation_total = sum(len(validation) for _train, validation in folds)
    for scale_index, scale in enumerate(scales):
        design = model.design(scaled_features(base, scale), shape)
        best = np.full(count, np.inf)
        for ridge in model.ridge_grid:
            errors = np.zeros(count)
            alive = np.ones(count, dtype=bool)
            for train, validation in folds:
                system = fits.prepare_nonnegative(design.values[train], design.ridge, ridge, spec)
                for column in range(count):
                    if not alive[column]:
                        continue
                    head = fits.solve_prepared(system, responses[train, column], spec)
                    prediction = models.predict_head(head, design.values[validation], spec)
                    if not np.isfinite(prediction).all():
                        alive[column] = False
                        continue
                    errors[column] += float(np.sum((prediction - responses[validation, column]) ** 2))
            rmse = np.where(alive, np.sqrt(errors / validation_total), np.inf)
            best = np.minimum(best, rmse)
        result[scale_index] = best
    return result


def run_target(target: str, seed: int, workers: int, scaled_panels: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Normalised CV objective per (shape, scale) for the reference swarm (scale 1) and each scaled swarm."""
    reference = benchmark.load_panel(REFERENCE_PANEL)
    entry = registry.ENTRY_BY_ID[MODEL_ID]
    shapes = entry.build(reference.features).candidate_shapes(reference.features)
    module = importlib.import_module(MODULE_NAME)
    objectives: dict[str, np.ndarray] = {}
    for panel_name in (REFERENCE_PANEL, *scaled_panels):
        panel = benchmark.load_panel(panel_name)
        group = panel.group(target)
        spread = group.outcomes.std(axis=0, ddof=1)
        scales = (1.0,) if panel_name == REFERENCE_PANEL else SCALE_GRID
        matrices = Parallel(n_jobs=workers, backend="loky", batch_size=2)(
            delayed(module.cv_rmse_matrix)(panel_name, target, index, scales, seed) for index in range(len(shapes))
        )
        # objective[shape, scale]: mean over components of RMSE / swarm SD
        objectives[panel_name] = np.stack([(matrix / spread[None, :]).mean(axis=1) for matrix in matrices])
        LOGGER.info("%s %s seed %d: %d shapes x %d scales done", target, panel_name, seed, len(shapes), len(scales))
    return objectives


def summarize(
    target: str, seed: int, objectives: dict[str, np.ndarray], shapes: tuple[models.Shape, ...]
) -> tuple[list[dict], list[dict]]:
    reference = objectives[REFERENCE_PANEL][:, 0]
    profile_rows, summary_rows = [], []
    for panel_name, matrix in objectives.items():
        if panel_name == REFERENCE_PANEL:
            continue
        joint = reference[:, None] + matrix  # shape x scale
        best_shape_per_scale = joint.argmin(axis=0)
        profile = joint.min(axis=0)
        for scale_index, scale in enumerate(SCALE_GRID):
            shape = shapes[best_shape_per_scale[scale_index]]
            profile_rows.append(
                {
                    "target": target,
                    "seed": seed,
                    "panel": panel_name,
                    "scale": scale,
                    "joint_objective": float(profile[scale_index]),
                    "scaled_panel_objective": float(matrix[best_shape_per_scale[scale_index], scale_index]),
                    "reference_objective": float(reference[best_shape_per_scale[scale_index]]),
                    **{f"shape_{k}": v for k, v in shape.items()},
                }
            )
        best_flat = int(joint.argmin())
        best_shape_index, best_scale_index = divmod(best_flat, joint.shape[1])
        own_best_shape = int(matrix[:, list(SCALE_GRID).index(1.0)].argmin())
        reference_best_shape = int(reference.argmin())
        summary_rows.append(
            {
                "target": target,
                "seed": seed,
                "panel": panel_name,
                "budget_tokens": PANEL_BUDGETS[panel_name],
                "budget_ratio": PANEL_BUDGETS[panel_name] / PANEL_BUDGETS[REFERENCE_PANEL],
                "best_scale": SCALE_GRID[best_scale_index],
                "best_joint_objective": float(joint[best_shape_index, best_scale_index]),
                "joint_objective_at_scale_1": float(profile[list(SCALE_GRID).index(1.0)]),
                "shared_shape": json.dumps(shapes[best_shape_index], sort_keys=True),
                "reference_own_shape": json.dumps(shapes[reference_best_shape], sort_keys=True),
                "scaled_panel_own_shape_at_scale_1": json.dumps(shapes[own_best_shape], sort_keys=True),
                "scaled_panel_objective_own_shape": float(matrix[own_best_shape, list(SCALE_GRID).index(1.0)]),
                "scaled_panel_objective_shared_shape_best_scale": float(matrix[best_shape_index, best_scale_index]),
            }
        )
    return profile_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--targets", nargs="*", default=list(TARGETS))
    parser.add_argument("--panels", nargs="*", default=list(SCALED_PANELS))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(FOLD_SEEDS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    importlib.import_module(MODULE_NAME).execute(args)


def execute(args: argparse.Namespace) -> None:
    reference = benchmark.load_panel(REFERENCE_PANEL)
    shapes = registry.ENTRY_BY_ID[MODEL_ID].build(reference.features).candidate_shapes(reference.features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles, summaries = [], []
    for target in args.targets:
        for seed in args.seeds:
            objectives = run_target(target, seed, args.workers, tuple(args.panels))
            profile_rows, summary_rows = summarize(target, seed, objectives, shapes)
            profiles.extend(profile_rows)
            summaries.extend(summary_rows)
            pd.DataFrame(profiles).to_csv(args.output_dir / "profile.csv", index=False)
            pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)
    summary = pd.DataFrame(summaries)
    print(
        summary[
            [
                "target",
                "seed",
                "panel",
                "budget_ratio",
                "best_scale",
                "best_joint_objective",
                "joint_objective_at_scale_1",
                "shared_shape",
                "reference_own_shape",
                "scaled_panel_own_shape_at_scale_1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
