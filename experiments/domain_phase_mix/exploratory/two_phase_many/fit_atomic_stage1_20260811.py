# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Stage 1: independent atomic fits on the no-replay two-bucket panel (ATOM-001).

Each of the 23 atomic BPB targets is fitted independently at each of the 4 horizons -- 92 independent
problems -- with nothing shared and no replay term available. If the response mechanism is adequate this
should already work; if it does not, the mechanism is inadequate and adding replay structure would only
conceal that.

Candidates, all on identical spatial folds:

  intercept   the floor any fit must clear
  exposure    benefit as a power law in exact StarCoder epochs, read at a fitted horizon
  two-bucket  the same, plus a readout of the COMPLEMENT bucket's exposure, one amplitude each
  two-horizon both buckets read at TWO horizons, which is what makes untied optima representable
  phase-split StarCoder exposure with the two phases carrying separate amplitudes
  quadratic   a plain quadratic surface in the two mixture shares, as an EXPRESSIVITY REFERENCE only

The exposure family uses EXACT physical epochs from the panel, not a fitted phase blend. That matters
here: the committed GEN damage feature is defined on a blend and is therefore not a valid no-replay
ablation, so it is absent rather than disabled.

`phase-split` is worth testing on this panel specifically. A previous round found per-phase separation
untestable because the phase-0 share of materialized epochs was a global constant fixed by the schedule;
here it varies across the whole unit interval, so the two phases are genuinely distinguishable.

The quadratic is a reference, never a candidate: it has no mechanism and cannot extrapolate, but it
bounds how much structure the surface actually contains.

Usage: ``uv run python ... [--targets N]``
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution, nnls  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    atomic_surface_panel_20260811 as panel_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_starcoder_wsd80_full_pool_atomic_surface_explorer_20260811 as explorer,
)

EPOCH_FLOOR = 1e-4  # epochs; the readout is regularised, never divided by zero
GRID = 201


def exposure(panel, horizon: float) -> np.ndarray:
    """Materialized StarCoder epochs as seen by a channel weighting the decay phase by `horizon`."""
    return (1.0 - horizon) * panel.epochs_phase_0 + horizon * panel.epochs_phase_1


def design(panel, name: str, theta: np.ndarray) -> np.ndarray:
    """Columns for one candidate. Column 0 is always the intercept."""
    ones = np.ones(len(panel.frame))
    if name == "intercept":
        return ones[:, None]
    if name == "quadratic":
        p0, p1 = panel.phase_0, panel.phase_1
        return np.column_stack([ones, p0, p1, p0 * p0, p1 * p1, p0 * p1])
    gamma, offset_log, horizon = theta[0], theta[1], theta[2]
    offset = 10.0**offset_log
    if name == "exposure":
        readout = (exposure(panel, horizon) + offset) ** -gamma
        return np.column_stack([ones, readout])
    if name == "two-bucket":
        complement = (1.0 - horizon) * panel.complement_epochs_phase_0 + horizon * panel.complement_epochs_phase_1
        return np.column_stack(
            [
                ones,
                (exposure(panel, horizon) + offset) ** -gamma,
                (complement + offset) ** -theta[3],
            ]
        )
    if name == "two-horizon":
        # SINGLE-INDEX IS THE PROBLEM THIS SOLVES. With one shared horizon, both bucket exposures are
        # monotone in the same scalar u = (1-phi)*f0*p0 + phi*f1*p1 -- measured rank correlation |rho| =
        # 1.000 -- so predicted loss depends on the policy only through u. Every untied policy then shares
        # its u with some tied policy, and the model cannot express ANY two-phase advantage. That is fatal
        # here: 37 of 92 empirical atomic optima are untied, and 77 of 92 are interior rather than corners.
        # Reading each bucket at TWO horizons gives two independent linear directions in (p0, p1), the
        # minimum needed for an untied optimum to be representable at all.
        second = theta[4]
        complement_a = (1.0 - horizon) * panel.complement_epochs_phase_0 + horizon * panel.complement_epochs_phase_1
        complement_b = (1.0 - second) * panel.complement_epochs_phase_0 + second * panel.complement_epochs_phase_1
        return np.column_stack(
            [
                ones,
                (exposure(panel, horizon) + offset) ** -gamma,
                (exposure(panel, second) + offset) ** -gamma,
                (complement_a + offset) ** -theta[3],
                (complement_b + offset) ** -theta[3],
            ]
        )
    if name == "phase-split":
        early = (panel.epochs_phase_0 + offset) ** -gamma
        late = (panel.epochs_phase_1 + offset) ** -gamma
        return np.column_stack([ones, early, late])
    raise ValueError(f"unknown candidate {name!r}")


def bounds_for(name: str) -> list[tuple[float, float]]:
    if name in ("intercept", "quadratic"):
        return []
    box = [(0.01, 4.0), (-4.0, -0.5), (0.0, 1.0)]  # gamma, log offset, horizon
    if name in ("two-bucket", "two-horizon"):
        box.append((0.01, 4.0))  # complement readout exponent
    if name == "two-horizon":
        box.append((0.0, 1.0))  # second horizon
    return box


def solve(columns: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Intercept free, every mechanism column non-negative, as in the committed model."""
    if columns.shape[1] == 1:
        return np.array([response.mean()])
    basis = columns[:, :1] / np.linalg.norm(columns[:, :1])
    rest = columns[:, 1:] - basis @ (basis.T @ columns[:, 1:])
    target = response - basis @ (basis.T @ response)
    scale = np.maximum(np.linalg.norm(rest, axis=0), 1e-300)
    amplitudes, _ = nnls(rest / scale, target)
    amplitudes = amplitudes / scale
    intercept = float(np.mean(response - columns[:, 1:] @ amplitudes))
    return np.concatenate([[intercept], amplitudes])


def fit_predict(panel, name: str, response: np.ndarray, folds) -> np.ndarray:
    """Out-of-fold predictions, selecting theta inside each training fold only."""
    predictions = np.empty(len(response))
    box = bounds_for(name)
    for train, test in folds:
        if box:

            def objective(theta, train=train):
                columns = design(panel, name, theta)
                if not np.isfinite(columns).all():
                    return 1e6
                inner = panel_module.spatial_folds(panel, n_splits=3, seed=7)
                total = 0.0
                for a, b in inner:
                    a = np.intersect1d(a, train)
                    b = np.intersect1d(b, train)
                    if len(a) < 8 or len(b) < 4:
                        continue
                    coefficients = solve(columns[a], response[a])
                    residual = columns[b] @ coefficients - response[b]
                    total += float(residual @ residual)
                return total

            theta = differential_evolution(
                objective,
                box,
                rng=np.random.default_rng(20260811),
                popsize=12,
                maxiter=40,
                tol=1e-10,
                polish=True,
                init="sobol",
            ).x
        else:
            theta = np.array([])
        columns = design(panel, name, theta)
        predictions[test] = columns[test] @ solve(columns[train], response[train])
    return predictions


def surface_optimum(panel, name: str, response: np.ndarray):
    """Recommended optimum from a full-data refit, plus the implied two-phase gain."""
    box = bounds_for(name)
    if box:

        def objective(theta):
            columns = design(panel, name, theta)
            if not np.isfinite(columns).all():
                return 1e6
            total = 0.0
            for a, b in panel_module.spatial_folds(panel, n_splits=3, seed=7):
                coefficients = solve(columns[a], response[a])
                residual = columns[b] @ coefficients - response[b]
                total += float(residual @ residual)
            return total

        theta = differential_evolution(
            objective,
            box,
            rng=np.random.default_rng(20260811),
            popsize=12,
            maxiter=40,
            tol=1e-10,
            polish=True,
            init="sobol",
        ).x
    else:
        theta = np.array([])
    columns = design(panel, name, theta)
    coefficients = solve(columns, response)

    axis = np.linspace(0.0, 1.0, GRID)
    g0, g1 = np.meshgrid(axis, axis, indexing="ij")
    flat0, flat1 = g0.ravel(), g1.ravel()
    grid = _grid_panel(panel, flat0, flat1)
    values = design(grid, name, theta) @ coefficients
    tied = _grid_panel(panel, axis, axis)
    tied_values = design(tied, name, theta) @ coefficients
    cell = int(np.argmin(values))
    return (float(flat0[cell]), float(flat1[cell])), float(tied_values.min() - values.min()), theta


class _GridPanel:
    """A panel-shaped view over hypothetical policies, with exposure scaled from the real rows."""

    def __init__(self, phase_0, phase_1, epochs_0, epochs_1, frame):
        self.phase_0, self.phase_1 = phase_0, phase_1
        self.epochs_phase_0, self.epochs_phase_1 = epochs_0, epochs_1
        self.frame = frame


def _grid_panel(panel, flat0, flat1):
    # Epochs are linear in the phase share at fixed horizon, so the per-unit rate is read off the data.
    rate_0 = float(np.max(panel.epochs_phase_0) / max(np.max(panel.phase_0), 1e-12))
    rate_1 = float(np.max(panel.epochs_phase_1) / max(np.max(panel.phase_1), 1e-12))
    capacity = float(panel.frame["nemotron_max_total_epochs"].to_numpy(float)[0]) if len(panel.frame) else 0.0
    grid = _GridPanel(flat0, flat1, rate_0 * flat0, rate_1 * flat1, np.empty((len(flat0), 0)))
    grid.complement_epochs_phase_0 = explorer.PHASE_0_FRACTION * capacity * (1.0 - flat0)
    grid.complement_epochs_phase_1 = (1.0 - explorer.PHASE_0_FRACTION) * capacity * (1.0 - flat1)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=int, default=0, help="limit target count for a smoke run")
    args = parser.parse_args()

    frame = panel_module.load_full_pool()
    panels = panel_module.panels_by_horizon(frame)
    targets = panel_module.atomic_targets()
    if args.targets:
        targets = targets[: args.targets]
    names = ("intercept", "two-bucket", "two-horizon", "quadratic")

    print("ATOM-001 Stage 1: independent atomic fits, no replay, two buckets, one objective at a time")
    print(f"{len(panels)} horizons x {len(targets)} targets, spatial leave-region-out folds")
    print("zero-replay assertion passed at load; exposure uses EXACT materialized epochs\n")

    beats = {name: 0 for name in names}
    ratios = {name: [] for name in names}
    # Whether a candidate can even PLACE an untied optimum is a structural question separate from fit:
    # a single-index model cannot, whatever its error. 37 of 92 empirical optima here are untied.
    untied = {name: 0 for name in names}
    distance = {name: [] for name in names}
    empirical_untied = 0
    for p in panels:
        folds = panel_module.spatial_folds(p)
        print(f"=== horizon {p.horizon:.3f}B ===", flush=True)
        for key in targets:
            y = p.target(key)
            best = int(np.argmin(y))
            truth = (float(p.phase_0[best]), float(p.phase_1[best]))
            empirical_untied += not np.isclose(truth[0], truth[1])
            line = []
            base = None
            for name in names:
                rmse = float(np.sqrt(np.mean((fit_predict(p, name, y, folds) - y) ** 2)))
                if name == "intercept":
                    base = rmse
                ratio = rmse / base
                ratios[name].append(ratio)
                beats[name] += ratio < 1.0
                line.append(f"{name} {ratio:.3f}")
                if name != "intercept":
                    where, _gain, _theta = surface_optimum(p, name, y)
                    untied[name] += not np.isclose(where[0], where[1], atol=1e-6)
                    distance[name].append(float(np.hypot(where[0] - truth[0], where[1] - truth[1])))
            print(f"  {key.split('/')[-2][:34]:34s} " + "  ".join(line), flush=True)
        print()

    print("SUMMARY, out-of-fold RMSE as a ratio to the intercept (lower is better)")
    for name in names:
        r = np.array(ratios[name])
        print(
            f"  {name:11s} median {np.median(r):.3f}  mean {r.mean():.3f}  "
            f"beats intercept on {beats[name]}/{len(r)} target-horizons"
        )


if __name__ == "__main__":
    main()
