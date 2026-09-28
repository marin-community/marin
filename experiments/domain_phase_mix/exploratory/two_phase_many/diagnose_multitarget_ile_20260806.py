# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Post-hoc diagnostics for the multi-target interference-evidence round.

Everything here was run after the main stages, to explain results rather than to produce them. It is
kept separate so that the frozen harness and its recorded protocol hashes stay untouched, and so the
numbers quoted in the report can be regenerated.

Six checks, in the order they were needed:

`rank`      why the unconstrained solve is non-unique. The head carries a family-level column and a
            per-bucket departure column for the same evidence, and the family column is the exact mean
            of its members. The design is rank-deficient by exactly the number of families.
`signed`    whether the head's non-negativity constraint binds on anything that matters. It does not:
            relaxing it changes neither the selected shape nor the fit.
`backbone`  what the aggregate response alone can reach, fitted on tied coordinates only, where the
            phase channel is inert by construction. Reported against the panel's training-seed noise.
`sweep`     how predicted gain and predicted optimum move with the interference rate at that backbone.
`corner`    what the surface actually measured where the swept model wants to put its optimum.
`fibers`    predicted-versus-observed best contrast along the eight frozen fixed-aggregate fibers. This
            is the diagnostic the single predicted optimum was hiding: a model can land near the
            optimum and still have the sign of the contrast wrong on the fiber that carries the whole
            two-phase advantage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as expanded,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_multitarget_interference_evidence_20260806 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    interference_evidence_model_20260806 as ile,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    multitarget_ile_panel300m_20260806 as panel300m,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    multitarget_ile_wsd80_20260806 as wsd_stage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import starcoder_wsd80_panel_20260728 as wsd80  # noqa: E402

PRIMARY = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
LAW = ile.InterferenceLaw.RECENCY_EXPOSURE
SWEEP_RATES = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
OPTIMUM_GRID = 201


def _primary(targets) -> np.ndarray:
    return targets.values[:, targets.index(PRIMARY)]


def rank() -> None:
    panel, _targets = wsd_stage.load_targets()
    geometry = wsd_stage.geometry()
    design = ile.design_matrix(panel.weights, geometry, ile.Shape(0.5, 2.0, LAW, curvature=1.0))
    print(f"WSD80 design {design.shape}  rank {np.linalg.matrix_rank(design)}")

    dataset = expanded.load_300m("uncheatable")
    base = expanded.geometry_300m(dataset)
    geometry = ile.Geometry(
        c0=dataset.c0,
        c1=dataset.c1,
        phase_1_fraction=1.0 - base.phase_0_fraction,
        family_index=dataset.family_index,
    )
    design = ile.design_matrix(dataset.weights, geometry, ile.Shape(0.5, 2.0, LAW, curvature=1.0))
    print(f"300M  design {design.shape}  rank {np.linalg.matrix_rank(design)}")

    families = geometry.n_families
    family_columns = design[:, 1 : 1 + families]
    bucket_columns = -design[:, 1 + 2 * families :]
    rebuilt = np.column_stack(
        [-bucket_columns[:, geometry.family_index == family].mean(axis=1) for family in np.unique(geometry.family_index)]
    )
    print(f"family evidence columns minus the mean of their bucket columns: {np.abs(family_columns - rebuilt).max()}")


def _fit(design: np.ndarray, response: np.ndarray, geometry: ile.Geometry, rows: np.ndarray, signed: bool) -> np.ndarray:
    penalty = ile._penalty_rows(geometry, 1e-3)
    augmented = np.vstack([design[rows], penalty])
    target = np.concatenate([response[rows], np.zeros(len(penalty))])
    lower, upper = ile._head_bounds(design.shape[1], geometry)
    if signed:
        lower = np.full(design.shape[1], -ile.AMPLITUDE_LIMIT)
        upper = np.full(design.shape[1], ile.AMPLITUDE_LIMIT)
        lower[0], upper[0] = -np.inf, np.inf
    return lsq_linear(augmented, target, bounds=(lower, upper), method="trf", max_iter=500).x


def signed() -> None:
    panel, targets = wsd_stage.load_targets()
    geometry = wsd_stage.geometry()
    response = _primary(targets)
    indices = np.arange(len(response))
    interior = wsd_stage.interior_mask(panel)
    outer = harness.wsd80_folds("random", panel.weights, indices, 3, 0)
    shapes = ile.shape_grid(law=LAW, curvature_grid=ile.CURVATURE_GRID)

    for relaxed in (False, True):
        best = (np.inf, shapes[0])
        for shape in shapes:
            design = ile.design_matrix(panel.weights, geometry, shape)
            squared = 0.0
            for train, test in outer:
                residual = design[test] @ _fit(design, response, geometry, train, relaxed) - response[test]
                squared += float(residual @ residual)
            if squared < best[0]:
                best = (squared, shape)
        shape = best[1]
        design = ile.design_matrix(panel.weights, geometry, shape)
        out_of_fold = np.empty_like(response)
        for train, test in outer:
            out_of_fold[test] = design[test] @ _fit(design, response, geometry, train, relaxed)
        coefficients = _fit(design, response, geometry, indices, relaxed)
        label = "signed" if relaxed else "nonneg"
        print(f"{label:7s} shape rho={shape.rho} mu={shape.interference} nu={shape.curvature}")
        print(f"        interior OOF RMSE {np.sqrt(np.mean((out_of_fold - response)[interior] ** 2)):.6f}")
        print(f"        family evidence amplitudes {np.round(coefficients[1:3], 4)}")


def backbone() -> ile.Shape:
    panel, targets = wsd_stage.load_targets()
    geometry = wsd_stage.geometry()
    response = _primary(targets)
    tied = np.flatnonzero(np.isclose(panel.phase_0[:, 1], panel.phase_1[:, 1]))
    scored = []
    for shape in ile.shape_grid(law=LAW, curvature_grid=ile.CURVATURE_GRID):
        design = ile.design_matrix(panel.weights, geometry, shape)
        head = ile.solve_head(design[tied], response[tied], geometry, 1e-4)
        residual = design[tied] @ ile.coefficient_vector(head) - response[tied]
        scored.append((float(np.sqrt(np.mean(residual**2))), shape.rho, shape))
    score, _rho, shape = min(scored)
    sigma = wsd80.training_seed_sigma(wsd80.load_fiber_replicates())
    print(f"tied coordinates: {len(tied)}")
    print(f"best tied RMSE {score:.6f} at rho={shape.rho} nu={shape.curvature}")
    print(f"training-seed sigma {sigma:.6f}  ->  {score / sigma:.1f} seed sigma")
    return shape


def sweep(shape: ile.Shape) -> None:
    panel, targets = wsd_stage.load_targets()
    geometry = wsd_stage.geometry()
    response = _primary(targets)
    axis = np.linspace(0.0, 1.0, OPTIMUM_GRID)
    phase_0, phase_1 = np.meshgrid(axis, axis, indexing="ij")
    grid = wsd_stage.grid_weights(phase_0.ravel(), phase_1.ravel())
    tied_grid = wsd_stage.grid_weights(axis, axis)

    print(" rate   in-sample RMSE   predicted gain   predicted optimum")
    for rate in SWEEP_RATES:
        local = ile.Shape(shape.rho, rate, LAW, curvature=shape.curvature)
        design = ile.design_matrix(panel.weights, geometry, local)
        coefficients = ile.coefficient_vector(ile.solve_head(design, response, geometry, 1e-4))
        residual = design @ coefficients - response
        prediction = ile.design_matrix(grid, geometry, local) @ coefficients
        tied_prediction = ile.design_matrix(tied_grid, geometry, local) @ coefficients
        best = int(np.argmin(prediction))
        print(
            f"{rate:5.1f}   {np.sqrt(np.mean(residual**2)):.6f}      "
            f"{tied_prediction.min() - prediction.min():+.6f}      "
            f"({phase_0.ravel()[best]:.3f}, {phase_1.ravel()[best]:.3f})"
        )
    print("\nobserved: gain +0.009594 at (0.100, 0.500)")


FIBER_AGGREGATES = (0.18, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80)
FIBER_GRID = 401


def fibers() -> None:
    """Predicted-versus-observed gain along the eight frozen fixed-aggregate fibers.

    The single predicted optimum is a weak summary: a model can land near it while getting the whole
    fiber profile wrong. The frozen evaluation ladder asks for the profile, and in particular for the
    best contrast to change sign around the tied optimum at aggregate 0.30 -- below it the two-phase
    policy should load code late, above it the advantage should vanish or reverse.
    """
    panel, targets = wsd_stage.load_targets()
    geometry = wsd_stage.geometry()
    response = _primary(targets)
    indices = np.arange(len(response))
    outer = harness.wsd80_folds("random", panel.weights, indices, 3, 0)

    beta_0, beta_1 = wsd80.PHASE_0_FRACTION, wsd80.PHASE_1_FRACTION
    observed_aggregate = beta_0 * panel.phase_0[:, 1] + beta_1 * panel.phase_1[:, 1]

    for law, curvatures in (
        (ile.InterferenceLaw.ABSOLUTE, (np.inf,)),
        (ile.InterferenceLaw.SHARE_DROP, (np.inf,)),
        (ile.InterferenceLaw.RECENCY_EXPOSURE, ile.CURVATURE_GRID),
    ):
        shapes = ile.shape_grid(law=law, curvature_grid=curvatures)
        designs = harness.build_designs(panel.weights, geometry, shapes)
        single = harness.MultiTarget(
            names=(PRIMARY,),
            values=response[:, None],
            observed=np.ones((len(response), 1), dtype=bool),
            family=("wsd80",),
            family_share=np.array([1.0]),
        )
        scores = harness.fold_scores(designs, single, geometry, outer, ile.HEAD_RIDGE_GRID)
        chosen = harness.choose(scores, list(designs), ile.HEAD_RIDGE_GRID, single, "independent")[0]
        design = designs[chosen.shape]
        head = ile.solve_head(design, response, geometry, chosen.ridge)
        coefficients = ile.coefficient_vector(head)

        print(f"\n{law}: rho={chosen.shape.rho} mu={chosen.shape.interference} nu={chosen.shape.curvature}")
        print("  aggregate   observed best contrast   predicted best contrast   predicted fiber gain")
        for aggregate in FIBER_AGGREGATES:
            # Every (p0, p1) on the fiber, clipped to the simplex.
            phase_1 = np.linspace(0.0, 1.0, FIBER_GRID)
            phase_0 = (aggregate - beta_1 * phase_1) / beta_0
            feasible = (phase_0 >= 0.0) & (phase_0 <= 1.0)
            phase_0, phase_1 = phase_0[feasible], phase_1[feasible]
            prediction = (
                ile.design_matrix(wsd_stage.grid_weights(phase_0, phase_1), geometry, chosen.shape) @ coefficients
            )
            tied = np.argmin(np.abs(phase_1 - phase_0))
            best = int(np.argmin(prediction))

            near = np.abs(observed_aggregate - aggregate) < 0.01
            if near.sum() >= 2:
                rows = np.flatnonzero(near)
                observed_best = rows[int(np.argmin(response[rows]))]
                observed_contrast = panel.phase_1[observed_best, 1] - panel.phase_0[observed_best, 1]
                observed_text = f"{observed_contrast:+.3f} (n={near.sum()})"
            else:
                observed_text = f"    n/a (n={near.sum()})"
            print(
                f"     {aggregate:.2f}       {observed_text:>18s}          {phase_1[best] - phase_0[best]:+.3f}"
                f"                {prediction[tied] - prediction[best]:+.6f}"
            )


CODE_TARGETS = (
    PRIMARY,
    "eval/uncheatable_eval/github_python-llama3/bpb",
    "eval/uncheatable_eval/github_cpp-llama3/bpb",
)
GATE_DRAWS = 4000
GATE_SEED = 20260806


def _regret(observed: np.ndarray, prediction: np.ndarray, rows: np.ndarray, k: int) -> float:
    """Observed BPB of the k best-predicted policies in `rows`, minus the best observed there."""
    ranked = rows[np.argsort(prediction[rows])][:k]
    return float(np.min(observed[ranked]) - np.min(observed[rows]))


def gate() -> None:
    """The registered multi-target gate, which the main harness never actually tested.

    The gate asks whether sharing one nonlinear shape improves at least two code-target *selection or
    gain* diagnostics beyond paired-bootstrap uncertainty. The harness bootstraps out-of-fold RMSE, a
    different quantity: a model can predict every policy better on average and still pick a worse one.
    Regret at 1 and at 5 are the row-level selection diagnostics, so they are what gets resampled here,
    paired across fitting modes on identical draws.
    """
    panel, targets = wsd_stage.load_targets()
    interior = np.flatnonzero(wsd_stage.interior_mask(panel))
    predictions = pd.read_csv(
        SCRIPT_DIR
        / "reference_outputs"
        / "multitarget_interference_evidence_20260806"
        / "wsd80_out_of_fold_predictions.csv"
    )
    rng = np.random.default_rng(GATE_SEED)
    draws = [rng.choice(interior, size=len(interior), replace=True) for _ in range(GATE_DRAWS)]

    rows = []
    for (law, protocol), block in predictions.groupby(["law", "protocol"]):
        modes = {m: b.sort_values("row") for m, b in block.groupby("mode")}
        if not {"joint", "independent"} <= set(modes):
            continue
        for name in CODE_TARGETS:
            observed = targets.values[:, targets.index(name)]
            joint = modes["joint"][name].to_numpy(float)
            independent = modes["independent"][name].to_numpy(float)
            for k in (1, 5):
                point = _regret(observed, joint, interior, k) - _regret(observed, independent, interior, k)
                sample = np.array([_regret(observed, joint, d, k) - _regret(observed, independent, d, k) for d in draws])
                rows.append(
                    {
                        "law": str(law).replace("InterferenceLaw.", ""),
                        "protocol": protocol,
                        "target": name.replace("eval/", ""),
                        "k": k,
                        "joint_minus_independent": point,
                        "ci_low": float(np.quantile(sample, 0.025)),
                        "ci_high": float(np.quantile(sample, 0.975)),
                        "improves_beyond_uncertainty": bool(np.quantile(sample, 0.975) < 0.0),
                    }
                )
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    passing = frame[frame.improves_beyond_uncertainty]
    print()
    print(f"gate needs at least two code-target diagnostics improving beyond uncertainty; {len(passing)} do")
    if len(passing):
        print(passing[["law", "protocol", "target", "k", "joint_minus_independent"]].to_string(index=False))


def corner() -> None:
    panel, targets = wsd_stage.load_targets()
    response = _primary(targets)
    phase_0, phase_1 = panel.phase_0[:, 1], panel.phase_1[:, 1]
    interior = wsd_stage.interior_mask(panel)
    best = int(np.argmin(np.where(interior, response, np.inf)))
    print(f"best interior observed: ({phase_0[best]:.3f}, {phase_1[best]:.3f}) BPB {response[best]:.6f}")
    low = phase_1 < 0.05
    print(f"rows with phase-1 code share below 0.05: n={low.sum()}, best BPB {response[low].min():.6f}")
    distance = np.hypot(phase_0 - 0.265, phase_1)
    print("nearest measured coordinates to (0.265, 0.000):")
    for row in np.argsort(distance)[:6]:
        print(f"   ({phase_0[row]:.3f}, {phase_1[row]:.3f})  BPB {response[row]:.6f}  interior={bool(interior[row])}")


def constraint_binding() -> None:
    """How often the unconstrained solve leaves the head's bounds. Explained by `rank`, not by the data."""
    panel, targets = wsd_stage.load_targets()
    geometry = wsd_stage.geometry()
    dataset = expanded.load_300m("uncheatable")
    base = expanded.geometry_300m(dataset)
    panel_geometry = ile.Geometry(
        c0=dataset.c0,
        c1=dataset.c1,
        phase_1_fraction=1.0 - base.phase_0_fraction,
        family_index=dataset.family_index,
    )
    panel_targets = panel300m.build_targets(dataset.frame.reset_index(drop=True))

    for name, weights, geo, tgt in (
        ("WSD80", panel.weights, geometry, targets),
        ("300M", dataset.weights, panel_geometry, panel_targets),
    ):
        for shape in (ile.Shape(0.5, 2.0, LAW, curvature=1.0), ile.Shape(1.0, 0.0, LAW, curvature=np.inf)):
            design = ile.design_matrix(weights, geo, shape)
            lower, upper = ile._head_bounds(design.shape[1], geo)
            penalty = ile._penalty_rows(geo, 1e-3)
            columns = [j for j in range(tgt.n_targets) if tgt.observed[:, j].all()]
            augmented = np.vstack([design, penalty])
            stacked = np.vstack([tgt.values[:, columns], np.zeros((len(penalty), len(columns)))])
            solved, *_ = np.linalg.lstsq(augmented, stacked, rcond=None)
            outside = ~np.all((solved >= lower[:, None] - 1e-12) & (solved <= upper[:, None] + 1e-12), axis=0)
            print(
                f"{name} rho={shape.rho} mu={shape.interference} nu={shape.curvature}: "
                f"{outside.sum()}/{len(columns)} outside bounds"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "check", choices=("rank", "signed", "backbone", "sweep", "corner", "binding", "fibers", "gate", "all")
    )
    arguments = parser.parse_args()

    if arguments.check in ("rank", "all"):
        print("=== rank ===")
        rank()
    if arguments.check in ("binding", "all"):
        print("\n=== constraint binding ===")
        constraint_binding()
    if arguments.check in ("signed", "all"):
        print("\n=== signed versus non-negative amplitudes ===")
        signed()
    if arguments.check in ("backbone", "sweep", "all"):
        print("\n=== aggregate backbone on tied coordinates ===")
        shape = backbone()
        if arguments.check in ("sweep", "all"):
            print("\n=== interference sweep at that backbone ===")
            sweep(shape)
    if arguments.check in ("corner", "all"):
        print("\n=== what the surface measured near the swept optimum ===")
        corner()
    if arguments.check in ("fibers", "all"):
        print("\n=== eight frozen fixed-aggregate fibers ===")
        fibers()
    if arguments.check in ("gate", "all"):
        print("\n=== registered multi-target gate: paired bootstrap of selection regret ===")
        gate()


if __name__ == "__main__":
    main()
