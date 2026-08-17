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
DAMAGE_KNEE = 105.0  # excess epochs at which repetition harm saturates; measured on 300M, not fitted


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
    if name in ("two-bucket-split-damage", "two-horizon-split-damage"):
        # The incumbent damage term, and NOTHING else, with its attribution split along the path
        # (ATOM-012). Total materialized epochs are a pure function of the aggregate share here -- the
        # ratio spans 105.44 to 105.84 across the whole m400 panel -- so charging damage on the total
        # makes it near-constant on aggregate-matched pairs and unable to express a phase preference by
        # itself. Splitting the ATTRIBUTION of the same law into what was incurred before the phase
        # boundary and what was added after gives the two increments separate non-negative amplitudes,
        # which is the statement that repetition damage taken early is partly repaired by what follows.
        # The pair sums back to the incumbent's single column exactly, so equal amplitudes reproduce it
        # and this adds one head column and NO shared parameter.
        base = "two-bucket-damage" if name.startswith("two-bucket") else "two-horizon-damage"
        columns = design(panel, base, theta)
        early = _hill(panel.epochs_phase_0, theta[4])
        return np.column_stack([columns[:, :-1], early, columns[:, -1] - early])
    if name == "two-bucket-damage":
        # The repetition conditions activate a term the no-replay panel could not test. Damage is charged
        # on EXACT materialized epochs -- the panel supplies them per phase -- in the bounded Hill form,
        # so it saturates rather than diverging. The knee is the 105 excess epochs measured elsewhere in
        # this project, and m400 is the first condition whose maximum exposure, 106.11, actually reaches
        # it, so this panel can finally see the saturation rather than assuming it.
        complement = (1.0 - horizon) * panel.complement_epochs_phase_0 + horizon * panel.complement_epochs_phase_1
        return np.column_stack(
            [
                ones,
                (exposure(panel, horizon) + offset) ** -gamma,
                (complement + offset) ** -theta[3],
                _hill(panel.epochs_phase_0 + panel.epochs_phase_1, theta[4]),
            ]
        )
    if name == "two-horizon-damage":
        second = theta[5]
        complement_a = (1.0 - horizon) * panel.complement_epochs_phase_0 + horizon * panel.complement_epochs_phase_1
        complement_b = (1.0 - second) * panel.complement_epochs_phase_0 + second * panel.complement_epochs_phase_1
        return np.column_stack(
            [
                ones,
                (exposure(panel, horizon) + offset) ** -gamma,
                (exposure(panel, second) + offset) ** -gamma,
                (complement_a + offset) ** -theta[3],
                (complement_b + offset) ** -theta[3],
                _hill(panel.epochs_phase_0 + panel.epochs_phase_1, theta[4]),
            ]
        )
    if name == "gated":
        # ORDER WITHOUT WEIGHTING. Every exposure-weighting scheme tried reduces to linear indices in
        # (p0, p1), which tie every untied policy to a tied one -- see ATOM-003. A gate is different in
        # kind: late exposure is absorbed only to the extent that groundwork was laid EARLY, so the term
        # is a PRODUCT of a phase-1 quantity with a function of phase-0 quantities and cannot be written
        # as a function of any single weighted sum.
        #
        # The data motivates the specific gate. The most common untied atomic optima here are
        # (0.000, 0.050), (0.000, 0.025) and (0.000, 0.100) -- no code early, a little code late -- which
        # is what "build general capability first, then specialise" looks like. So the groundwork variable
        # is the COMPLEMENT bucket's early exposure, and it gates the absorption of late StarCoder.
        beta, kappa_log = theta[4], theta[5]
        kappa = 10.0**kappa_log
        groundwork = panel.complement_epochs_phase_0
        gate = groundwork**beta / (groundwork**beta + kappa**beta)
        absorbed = panel.epochs_phase_0 + panel.epochs_phase_1 * gate
        complement = (1.0 - horizon) * panel.complement_epochs_phase_0 + horizon * panel.complement_epochs_phase_1
        return np.column_stack(
            [
                ones,
                (exposure(panel, horizon) + offset) ** -gamma,
                (absorbed + offset) ** -gamma,
                (complement + offset) ** -theta[3],
            ]
        )
    if name == "phase-split":
        early = (panel.epochs_phase_0 + offset) ** -gamma
        late = (panel.epochs_phase_1 + offset) ** -gamma
        return np.column_stack([ones, early, late])
    if name in ("path-exposure", "path-damage", "path-tied"):
        return _path_design(panel, name, theta)
    raise ValueError(f"unknown candidate {name!r}")


def _path_design(panel, name: str, theta: np.ndarray) -> np.ndarray:
    """Path-attributed columns: split the ATTRIBUTION of each law, never its argument (ATOM-012).

    A run traverses cumulative StarCoder exposure 0 -> E0 -> E0+E1. Benefit and damage are both functions
    of cumulative exposure, so per-phase treatment cannot mean evaluating either law separately on E0 and
    on E1 -- that would assert f(E0+E1) = f(E0) + f(E1), which is false for every saturating law. What is
    well defined is the INCREMENT each phase contributes along the path, and those increments carry
    separate amplitudes.

    Two facts make this the right shape here, both exact rather than fitted. The complement readout is
    antiparallel to the StarCoder readout at every horizon, because PHASE_0_FRACTION/(1-PHASE_0_FRACTION)
    and the epoch-rate ratio r0/r1 are both exactly 4, so `two-bucket` is a one-dimensional index at any
    theta and predicts zero two-phase gain by construction. And total epochs are a pure function of the
    aggregate share, so a damage law charged on the total is near-constant on aggregate-matched pairs.
    Splitting along the path breaks both degeneracies: H(E0) and H(E0+E1)-H(E0) point at (1.000, 0.014)
    and (0.182, 0.983), near-orthogonal where every incumbent column lies on one line.

    Each pair sums back to the unsplit law exactly, so equal amplitudes reproduce the total-exposure
    model and `path-tied` is the same model with only this degree of freedom removed.

    ONLY THE REPLAYED BUCKET IS SPLIT, and that is a measurement rather than a preference. The complement
    pool's per-phase exposures span 0 to 1.03e-3 epochs, so a readout of them is constant to within 0.7%
    and splitting it adds two directions that are null in everything but noise: the head answered with
    amplitudes of 66.7 and 199.3 against an intercept of -73.3, and out-of-fold error rose above the
    response's own spread. StarCoder traverses 0 to 85 epochs and is the bucket being repeated, so it is
    the only one with a path worth attributing. The complement keeps the incumbent's readout at a fitted
    horizon, unchanged, so that the comparison isolates the split.

    The readout is written in the retained form `(1 + x/scale)^-gamma`. AN EARLIER VERSION OF THIS NOTE
    CLAIMED THAT FORM MATTERS BECAUSE IT IS EXACTLY 1 AT ZERO EXPOSURE WHERE THE OFFSET FORM SPIKES, AND
    THAT IS WRONG: `(1 + x/s)^-gamma = s^gamma * (x + s)^-gamma` identically, so the two are the same
    column up to a constant that the free head amplitude absorbs, and swapping them cannot change any
    fit. Verified by construction and numerically -- the ratio is constant to ten digits. The changes
    that DID move the two-bucket results were made at the same time and are the real levers: the scale
    bound widened from [1e-4, 10^-0.5] to [1e-2, 10^2], and only the replayed bucket being split. The
    form is kept because the scale then reads directly as a saturation point in epochs.
    """
    ones = np.ones(len(panel.frame))
    gamma, scale, gamma_c, scale_c, horizon = theta[0], 10.0 ** theta[1], theta[2], 10.0 ** theta[3], theta[4]
    e0 = panel.epochs_phase_0
    e_total = e0 + panel.epochs_phase_1
    complement = (1.0 - horizon) * panel.complement_epochs_phase_0 + horizon * panel.complement_epochs_phase_1
    complement_readout = _retained(complement, scale_c, gamma_c)
    if name == "path-tied":
        return np.column_stack([ones, _retained(e_total, scale, gamma), complement_readout])
    early = _retained(e0, scale, gamma)
    columns = [ones, early, _retained(e_total, scale, gamma) - early, complement_readout]
    if name == "path-damage":
        # The knee is the 105 excess epochs measured elsewhere; m400's maximum of 106.11 is the first
        # condition that reaches it, so the saturation is observed rather than assumed.
        harm = _hill(e0, theta[5])
        columns.extend([harm, _hill(e_total, theta[5]) - harm])
    return np.column_stack(columns)


def _retained(epochs: np.ndarray, scale: float, gamma: float) -> np.ndarray:
    return (1.0 + epochs / scale) ** -gamma


def _hill(epochs: np.ndarray, tau: float) -> np.ndarray:
    excess = np.maximum(epochs - 1.0, 0.0) / DAMAGE_KNEE
    powered = excess**tau
    return powered / (1.0 + powered)


def bounds_for(name: str) -> list[tuple[float, float]]:
    if name in ("intercept", "quadratic"):
        return []
    if name in ("path-exposure", "path-tied", "path-damage"):
        # StarCoder's phase structure comes from the path rather than from a fitted readout time; the
        # horizon here belongs to the complement readout alone, which is not split. Each pool gets its
        # own retained scale, since a saturation scale is a property of the pool and the two differ by
        # five orders of magnitude in epochs.
        box = [(0.01, 4.0), (-2.0, 2.0), (0.01, 4.0), (-5.0, -1.0), (0.0, 1.0)]
        return [*box, (0.2, 10.0)] if name == "path-damage" else box
    box = [(0.01, 4.0), (-4.0, -0.5), (0.0, 1.0)]  # gamma, log offset, horizon
    damage = ("two-bucket-damage", "two-horizon-damage", "two-bucket-split-damage", "two-horizon-split-damage")
    if name in ("two-bucket", "two-horizon", "gated", *damage):
        box.append((0.01, 4.0))  # complement readout exponent
    if name in damage:
        box.append((0.2, 10.0))  # repetition-damage exponent
    if name in ("two-horizon-damage", "two-horizon-split-damage"):
        box.append((0.0, 1.0))  # second horizon
    if name == "two-horizon":
        box.append((0.0, 1.0))  # second horizon
    if name == "gated":
        box.extend([(0.3, 12.0), (-6.0, -1.0)])  # gate sharpness, log gate scale in epochs
    if name in ("two-bucket-split-damage", "two-horizon-split-damage"):
        box.append((-6.0, 2.0))  # log shrinkage of the split's departure from the unsplit model
    return box


def shrink_for(name: str, theta: np.ndarray):
    """Which amplitudes are shrunk together, and how hard (ATOM-015).

    The split-damage models generalise the unsplit ones, and the difference between the two damage
    amplitudes is the whole of that generalisation. Shrinking that difference therefore shrinks toward
    the nested model rather than toward zero, which is the meaningful null here: the unshrunk fits
    reproduce the ordering of the fresh two-phase gain but over-disperse its magnitude 5.7-fold, with
    every error at the longest horizon in the same direction, which is what an unregularised head does to
    a local gradient.
    """
    if name == "two-bucket-split-damage":
        return ((2, 3),), 10.0 ** theta[-1]
    if name == "two-horizon-split-damage":
        return ((4, 5),), 10.0 ** theta[-1]
    return None


def solve(columns: np.ndarray, response: np.ndarray, shrink=None) -> np.ndarray:
    """Intercept free, every mechanism column non-negative, as in the committed model.

    `shrink` is `(pairs, weight)` from `shrink_for`: a ridge on the DIFFERENCE within each amplitude
    pair, applied as extra rows so that non-negativity is still enforced by the same solve. A weight of
    zero recovers the unshrunk fit exactly.
    """
    if columns.shape[1] == 1:
        return np.array([response.mean()])
    basis = columns[:, :1] / np.linalg.norm(columns[:, :1])
    rest = columns[:, 1:] - basis @ (basis.T @ columns[:, 1:])
    target = response - basis @ (basis.T @ response)
    scale = np.maximum(np.linalg.norm(rest, axis=0), 1e-300)
    design, target = rest / scale, target
    if shrink is not None:
        # Penalise the departure's CONTRIBUTION, not its amplitude. Writing the split as
        # a_e*f + a_l*(g - f) = a_l*g + (a_e - a_l)*f, the whole departure from the nested model is the
        # term (a_e - a_l)*f, whose size in response units is |a_e - a_l| times f's residualised norm.
        # Penalising the bare amplitude difference instead would make the weight mean different things at
        # different fitted damage exponents, since a steeply suppressed column buys large amplitudes --
        # the fits here run to 15.5 against columns of order 1e-3 -- so a weight tuned on one fit would
        # not transfer to the next. In the normalised amplitudes b = a * scale that penalty is a row
        # sqrt(weight) * (e_i - e_j * scale_i / scale_j).
        pairs, weight = shrink
        rows = np.zeros((len(pairs), design.shape[1]))
        for row, (first, second) in zip(rows, pairs, strict=True):
            row[first] = np.sqrt(weight)
            row[second] = -np.sqrt(weight) * scale[first] / scale[second]
        design = np.vstack([design, rows])
        target = np.concatenate([target, np.zeros(len(pairs))])
    amplitudes, _ = nnls(design, target)
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
                popsize=10,
                maxiter=20,
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
            popsize=10,
            maxiter=20,
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
    names = ("intercept", "two-bucket", "two-horizon", "gated")

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
