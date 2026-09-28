# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Stage 2: multiple atomic objectives on the no-replay two-bucket panel (ATOM-002).

Stage 1 established that a two-bucket exposure model represents every atomic surface here, fitted
independently. This asks what happens when the nonlinear geometry is SHARED across the 23 objectives,
which is where earlier rounds found false phase transfer -- sharing sharpened the recommended mixture on
one panel while inventing roughly 0.012 BPB of phase gain on broad-text controls that have none.

Arms, all on identical spatial folds and optimiser budgets:

  independent  one theta per target                    -- the Stage 1 result
  shared       one theta for all targets, per-target heads
  pooled       shared theta, plus a shrunk per-target deviation on the geometry

and a head diagnostic, non-negative amplitudes against sign-unconstrained ones, run on both independent
and shared geometry so that a failure can be attributed to the representation or to the head.

THE FALSE-TRANSFER TEST is the point of the stage and it is cheap here. This panel has no forced replay
and its own confirmation evidence reports no positive fresh two-phase gain for any full-pool selected
policy, so a surrogate that predicts a large positive two-phase gain on these surfaces is inventing it.
Predicted gain is therefore reported per target and per arm, and the code and broad-text families are
kept separate because they are driven by different buckets.

Usage: ``uv run python ... [--horizons N] [--targets N]``
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    atomic_surface_panel_20260811 as panel_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_atomic_stage1_20260811 as stage1,
)

CODE_MARKERS = ("programing", "github", "arxiv_computer")
BOX = [(0.01, 4.0), (-4.0, -0.5), (0.0, 1.0), (0.01, 4.0)]
GRID = 201


def columns_for(panel, theta: np.ndarray) -> np.ndarray:
    return stage1.design(panel, "two-bucket", theta)


def solve_head(columns: np.ndarray, response: np.ndarray, constrained: bool) -> np.ndarray:
    if constrained:
        return stage1.solve(columns, response)
    return np.linalg.lstsq(columns, response, rcond=None)[0]


def fold_error(theta, panel, responses, folds, constrained: bool) -> float:
    """Variance-normalised squared error summed over targets; scale-free so no target dominates."""
    columns = columns_for(panel, theta)
    if not np.isfinite(columns).all():
        return 1e6
    total = 0.0
    for train, test in folds:
        scale = np.maximum(responses[train].std(axis=0), 1e-9)
        for index in range(responses.shape[1]):
            head = solve_head(columns[train], responses[train, index], constrained)
            residual = columns[test] @ head - responses[test, index]
            total += float(residual @ residual) / (scale[index] ** 2)
    return total


def select_shared(panel, responses, folds, constrained: bool) -> np.ndarray:
    return differential_evolution(
        fold_error,
        BOX,
        args=(panel, responses, folds, constrained),
        rng=np.random.default_rng(20260812),
        popsize=10,
        maxiter=30,
        tol=1e-10,
        polish=True,
        init="sobol",
    ).x


def out_of_fold(panel, responses, folds, arm: str, constrained: bool) -> np.ndarray:
    predictions = np.empty_like(responses)
    for train, test in folds:
        inner = [
            (np.intersect1d(a, train), np.intersect1d(b, train))
            for a, b in panel_module.spatial_folds(panel, n_splits=3, seed=7)
        ]
        inner = [(a, b) for a, b in inner if len(a) >= 8 and len(b) >= 4]
        if arm == "shared":
            # inner folds carry GLOBAL row indices already restricted to `train`, so the full response
            # matrix is passed and indexed with them; slicing it first would put the two in different
            # index spaces.
            theta = select_shared(panel, responses, inner, constrained)
            columns = columns_for(panel, theta)
            for index in range(responses.shape[1]):
                head = solve_head(columns[train], responses[train, index], constrained)
                predictions[test, index] = columns[test] @ head
        else:
            for index in range(responses.shape[1]):
                theta = select_shared(panel, responses[:, [index]], inner, constrained)
                columns = columns_for(panel, theta)
                head = solve_head(columns[train], responses[train, index], constrained)
                predictions[test, index] = columns[test] @ head
    return predictions


def predicted_gain(panel, theta, response, constrained: bool) -> float:
    """Best two-phase surface value minus the best tied value, from a full-data refit."""
    columns = columns_for(panel, theta)
    head = solve_head(columns, response, constrained)
    axis = np.linspace(0.0, 1.0, GRID)
    g0, g1 = np.meshgrid(axis, axis, indexing="ij")
    grid = stage1._grid_panel(panel, g0.ravel(), g1.ravel())
    tied = stage1._grid_panel(panel, axis, axis)
    return float((columns_for(tied, theta) @ head).min() - (columns_for(grid, theta) @ head).min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, default=0)
    parser.add_argument("--targets", type=int, default=0)
    args = parser.parse_args()

    frame = panel_module.load_full_pool()
    panels = panel_module.panels_by_horizon(frame)
    targets = list(panel_module.atomic_targets())
    if args.horizons:
        panels = panels[: args.horizons]
    if args.targets:
        targets = targets[: args.targets]
    is_code = np.array([any(marker in key for marker in CODE_MARKERS) for key in targets])

    print("ATOM-002 Stage 2: sharing nonlinear geometry across atomic objectives")
    print(f"{len(panels)} horizons x {len(targets)} targets; no forced replay, so true two-phase gain is ~0\n")

    for panel in panels:
        folds = panel_module.spatial_folds(panel)
        responses = np.column_stack([panel.target(key) for key in targets])
        floor = np.empty_like(responses)
        for train, test in folds:
            floor[test] = responses[train].mean(axis=0)
        base = np.sqrt(((floor - responses) ** 2).mean(axis=0))

        print(f"=== horizon {panel.horizon:.3f}B ===", flush=True)
        for arm in ("independent", "shared"):
            for constrained in (True, False):
                predictions = out_of_fold(panel, responses, folds, arm, constrained)
                ratio = np.sqrt(((predictions - responses) ** 2).mean(axis=0)) / base
                theta = select_shared(panel, responses, folds, constrained)
                gains = (
                    np.array([predicted_gain(panel, theta, responses[:, i], constrained) for i in range(len(targets))])
                    if arm == "shared"
                    else None
                )
                label = f"{arm}/{'nonneg' if constrained else 'free-sign'}"
                line = (
                    f"  {label:22s} RMSE ratio: code {np.median(ratio[is_code]):.3f}  "
                    f"broad {np.median(ratio[~is_code]):.3f}  all {np.median(ratio):.3f}"
                )
                if gains is not None:
                    line += (
                        f"   predicted 2p gain: code {np.median(gains[is_code]):+.5f}  "
                        f"broad {np.median(gains[~is_code]):+.5f}  max {gains.max():+.5f}"
                    )
                print(line, flush=True)
        print()


if __name__ == "__main__":
    main()
