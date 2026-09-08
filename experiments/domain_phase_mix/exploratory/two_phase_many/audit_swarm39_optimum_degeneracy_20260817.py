# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Is the 39-bucket predicted optimum degenerate, and which term makes it so (ATOM-017)?

Fits on every available row for a scale -- the two-phase fit panel and the coordinate-disjoint heldout
panel together -- then finds the model's own argmin over the policy space and asks whether that argmin is
a policy at all. `argmin` is the criterion here, taken before any deployment-side regularisation, so a
flat or vertex-seeking surface is a defect of the model rather than something a range penalty should hide.

The design is separable by construction: every column is a family sum or a per-bucket term, so predicted
loss is `intercept + sum_i h_i(p0_i, p1_i)` with the only coupling being the two simplex constraints. Two
consequences the audit uses. First, the optimum can be searched directly in the 78-dimensional policy
space rather than over a grid. Second, the SHAPE of each `h_i` decides whether the optimum is interior:
a decreasing convex readout of exposure has diminishing returns and pulls interior, while any term LINEAR
in the weights is minimised at a simplex vertex and any CONCAVE term pushes to the boundary. The
committed design carries a free-sign late-share term that is exactly linear in the phase-1 weights.

Degeneracy is measured, not asserted, by restarting the search from many Dirichlet draws:

  loss spread    how much predicted loss varies across restarts; near zero means they all found the floor
  policy spread  how far apart those equally-good policies are; large with a flat loss IS the degeneracy
  support size   buckets carrying more than 1% weight, against 39 available
  top-1 weight   how much of a phase sits on its single heaviest bucket

Usage: ``uv run python ... [--scale delphi_3e18] [--variants blended,split] [--restarts 24]``
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_swarm39_split_damage_20260817 as split_damage,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    general_mixture_surrogate_20260809 as gen,
)


def pooled_panel(scale: str, target: str):
    """Every non-sealed row for this scale: the two-phase fit panel plus the disjoint heldout panel.

    The heldout panel is used for FITTING here, which is deliberate -- the question is whether the model's
    argmin is a sane policy, and starving the fit would confound a degenerate optimum with a noisy one.
    Nothing below reports held-out accuracy, so nothing below is inflated by this.
    """
    fit, held = swarm39.load_scale(scale)
    weights = np.concatenate([np.stack([p.phase0, p.phase1], axis=1) for p in (fit, held)])
    response = np.concatenate([fit.targets[target], held.targets[target]])
    # Not every row carries every target -- at 300M the Uncheatable macro is missing on 17 heldout rows --
    # and a missing outcome is not a fittable one. Dropping them beats letting NaN reach the solver, where
    # it surfaces as an opaque failure inside the parameter search rather than as a data problem.
    observed = np.isfinite(response)
    panel = gen.Panel(weights[observed], fit.c0, fit.c1, fit.family_index)
    return panel, response[observed], fit


def loss_curvature(panel: gen.Panel, shape, amplitudes, offsets, variant: str) -> dict[str, float]:
    """Per-bucket second differences of h_i along each phase, which decide interior versus vertex.

    A positive second difference is a convex direction with diminishing returns and admits an interior
    optimum; a negative one is concave and drives the solution to a face of the simplex; exactly zero is
    linear and drives it to a vertex.
    """
    curvature = {}
    for phase, label in ((0, "phase_0"), (1, "phase_1")):
        second = []
        for bucket in range(panel.weights.shape[2]):
            grid = np.linspace(0.0, 0.25, 9)
            values = []
            for value in grid:
                probe = np.zeros((1, 2, panel.weights.shape[2]))
                probe[0, phase, bucket] = value
                probe[0, 1 - phase, bucket] = 0.0
                trial = gen.Panel(probe, panel.epochs_early, panel.epochs_late, panel.family_index)
                free, constrained = gen.design(trial, shape, variant)
                values.append(float((free @ offsets + constrained @ amplitudes).item()))
            values = np.array(values)
            span = max(np.abs(values - values[0]).max(), 1e-30)
            second.append(float(np.diff(values, 2).mean() / span))
        curvature[label] = float(np.median(second))
        curvature[label + "_convex_fraction"] = float(np.mean(np.array(second) > 1e-6))
    return curvature


def find_optimum(panel: gen.Panel, shape, amplitudes, offsets, variant: str, start: np.ndarray):
    """Minimise predicted loss over two simplices from one starting policy."""
    buckets = panel.weights.shape[2]

    def unpack(vector):
        probe = np.clip(vector.reshape(1, 2, buckets), 0.0, 1.0)
        return gen.Panel(probe, panel.epochs_early, panel.epochs_late, panel.family_index)

    def objective(vector):
        trial = unpack(vector)
        free, constrained = gen.design(trial, shape, variant)
        value = float((free @ offsets + constrained @ amplitudes).item())
        return value if np.isfinite(value) else 1e6

    constraints = [
        {"type": "eq", "fun": lambda v, phase=phase: v.reshape(2, buckets)[phase].sum() - 1.0} for phase in (0, 1)
    ]
    result = minimize(
        objective,
        start.ravel(),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * (2 * buckets),
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-12},
    )
    policy = np.clip(result.x.reshape(2, buckets), 0.0, 1.0)
    policy = policy / policy.sum(axis=1, keepdims=True)
    return policy, objective(policy.ravel())


def describe(policies: np.ndarray, losses: np.ndarray) -> dict[str, float]:
    """Degeneracy summary across restarts."""
    best = losses.min()
    near = policies[losses <= best + 1e-6 * max(abs(best), 1.0)]
    spread = 0.0
    if len(near) > 1:
        distances = [
            0.5 * np.abs(near[i] - near[j]).sum() / 2.0 for i in range(len(near)) for j in range(i + 1, len(near))
        ]
        spread = float(np.median(distances))
    top = policies[int(np.argmin(losses))]
    return {
        "loss_spread": float(losses.max() - losses.min()),
        "restarts_at_floor": len(near),
        "policy_spread": spread,
        "support_phase_0": int((top[0] > 0.01).sum()),
        "support_phase_1": int((top[1] > 0.01).sum()),
        "top1_phase_0": float(top[0].max()),
        "top1_phase_1": float(top[1].max()),
        "phase_tv": float(0.5 * np.abs(top[1] - top[0]).sum()),
    }


def envelope(panel: gen.Panel, policy: np.ndarray, response: np.ndarray) -> dict[str, float]:
    """Where the predicted optimum sits relative to the policies that were actually run.

    A unique interior argmin is still worthless if it lies in unexplored space, so this reports the total
    variation distance to the nearest observed policy and how good that neighbour actually was. The
    two-bucket work scored exactly this quantity: what you get by deploying the model's recommendation,
    approximated by the closest thing anyone has measured.
    """
    distance = 0.5 * np.abs(panel.weights - policy[None, :, :]).sum(axis=2).mean(axis=1)
    nearest = int(np.argmin(distance))
    return {
        "distance": float(distance[nearest]),
        "nearest_bpb": float(response[nearest]),
        "nearest_percentile": float((response < response[nearest]).mean()),
        "median_pairwise": float(np.median(0.5 * np.abs(panel.weights[:, 0] - panel.weights[:, 1]).sum(axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="delphi_3e18")
    parser.add_argument("--target", default=swarm39.UNCHEATABLE)
    parser.add_argument("--variants", default="blended,split")
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    panel, response, reference = pooled_panel(args.scale, args.target)
    buckets = panel.weights.shape[2]
    print(f"ATOM-017 optimum degeneracy: {args.scale}, target {args.target}")
    print(f"fitting on ALL {len(response)} rows x {buckets} buckets, {panel.n_families} families")
    print(f"observed best row: {response.min():.5f}   panel sd {response.std():.5f}\n")

    rng = np.random.default_rng(20260817 + args.seed)
    starts = [rng.dirichlet(np.ones(buckets), size=2) for _ in range(args.restarts)]
    starts.append(np.repeat(reference.proportional[None, :], 2, axis=0))

    optima = []
    for variant in args.variants.split(","):
        fitted = split_damage.fit_variant(panel, response, variant, args.seed)
        _vector, shape, offsets, amplitudes = fitted
        predicted = split_damage.predict(panel, fitted, variant)
        results = [find_optimum(panel, shape, amplitudes, offsets, variant, start) for start in starts]
        policies = np.array([policy for policy, _ in results])
        losses = np.array([loss for _, loss in results])
        summary = describe(policies, losses)
        curvature = loss_curvature(panel, shape, amplitudes, offsets, variant)
        print(f"--- {variant} (in-sample RMSE {np.sqrt(np.mean((predicted - response) ** 2)):.5f}) ---")
        print(f"  predicted loss at its own optimum {losses.min():.5f}, best observed row {response.min():.5f}")
        print(
            f"  restarts {len(losses)}, loss spread across them {summary['loss_spread']:.2e}, "
            f"{summary['restarts_at_floor']} at the floor"
        )
        print(f"  POLICY spread among equally-good restarts (total variation) {summary['policy_spread']:.4f}")
        print(
            f"  support: {summary['support_phase_0']}/{buckets} buckets early, "
            f"{summary['support_phase_1']}/{buckets} late;  heaviest bucket carries "
            f"{summary['top1_phase_0']:.3f} early, {summary['top1_phase_1']:.3f} late"
        )
        print(f"  phase total variation of the optimum {summary['phase_tv']:.4f}")
        print(
            f"  curvature of per-bucket loss: median second difference {curvature['phase_0']:+.2e} early, "
            f"{curvature['phase_1']:+.2e} late"
        )
        print(
            f"  buckets convex (interior-admitting) in phase 0: "
            f"{curvature['phase_0_convex_fraction']:.0%}, phase 1: {curvature['phase_1_convex_fraction']:.0%}"
        )
        best = policies[int(np.argmin(losses))]
        where = envelope(panel, best, response)
        print(
            f"  DISTANCE from the optimum to the nearest policy ever run: {where['distance']:.4f} "
            f"total variation; that neighbour scored {where['nearest_bpb']:.5f} "
            f"({where['nearest_percentile']:.0%} percentile of the panel)"
        )
        print(f"  claimed improvement over the best observed row: " f"{response.min() - losses.min():+.5f} BPB")
        optima.append(best)


def _placeholder() -> None:
    pass


if __name__ == "__main__":
    main()
