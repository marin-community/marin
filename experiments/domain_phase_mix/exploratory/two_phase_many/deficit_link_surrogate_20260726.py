# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The dual-objective surrogate: hierarchical replay on an unridged power-deficit link.

Three findings from this experiment compose into one model, and each is load-bearing.

*Zero ridge.* Ridge shrinkage of the improvement coefficients is the mechanism behind
the extrapolation bias that afflicts every Observatory baseline. All five under-rate
policies better than their training data, by +0.0037 to +0.0206 BPB on a top-censored
fit. Forcing the ridge to zero cuts the hierarchical replay bias from +0.00365 to
+0.00029 with the interval on the paired difference excluding zero and 199 of 200 draws
agreeing. The head is nonnegative least squares, so it is not unregularized in the
usual sense: the sign constraint is doing the work the ridge was doing badly.

*A deficit-parameterized response.* Predicting ``floor + deficit`` rather than the level
directly improves both in-panel fit and the ordering of good policies. The gain is not
an artifact of one design: the same pattern appears on hierarchical replay and on
bucket-family GRP, and the censored bias falls monotonically in the link exponent in all
four design-by-target cells tested.

*A floor below the achievable optimum.* Prediction is bounded below by the floor, so a
floor above the optimum makes the extrapolation bias structural rather than fitted. At a
floor fraction of 0.99 the bound sits above 7 of 28 censored truths on Uncheatable and
14 of 28 on Table-9, forcing at least +0.012 and +0.025 BPB of bias. The panel's real
headroom below its training minimum is 2.2 percent for Uncheatable and 3.6 percent for
Table-9, so ``FLOOR_FRACTION`` is set to leave several times that margin.

Hyperparameters are pinned rather than tuned on the censored metric, which would be
selection on the evaluation set. ``LINK_EXPONENT = 0.5`` is the midpoint of the family
that runs from multiplicative at 0 to additive at 1, and the sweep shows a smooth
monotone trade-off across that range rather than a spike, so the midpoint is a defensible
default and not a cherry-pick. Six configurations spanning two designs, two exponents,
and two floors all beat the incumbent on all four headline metrics at Uncheatable, which
is the reason to trust the region rather than any single cell.

Scope. The improvement is established on Uncheatable. On Table-9 no configuration
achieves the dual win: its run sigma is 3.2 times larger and fit gains there always
arrive with worse bias. This model should be read as an Uncheatable proposer.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from swarm39_harness_20260725 import Model, Panel, fit_head, grouped_splits  # noqa: E402
from swarm39_models_20260725 import _state_shapes, build_hierarchical_phase_replay  # noqa: E402

# Midpoint of the multiplicative-to-additive family. 0 is the log-deficit link and 1 is
# affine in the deficit, hence equivalent to fitting the level directly.
LINK_EXPONENT = 0.5
# Leaves roughly 15 percent headroom below the training minimum, against a measured
# achievable headroom of 2 to 4 percent, so the bound cannot forbid the optimum.
FLOOR_FRACTION = 0.85
# The sign-constrained head replaces the ridge; see the module docstring.
L2 = 0.0

N_SPLITS = 5
SPLIT_SEED = 0
DEFICIT_EPSILON = 1e-9
LINK_CLIP = 30.0


def to_link(deficit: np.ndarray, exponent: float) -> np.ndarray:
    safe = np.maximum(deficit, DEFICIT_EPSILON)
    return np.log(safe) if exponent == 0.0 else (safe**exponent - 1.0) / exponent


def from_link(eta: np.ndarray, exponent: float) -> np.ndarray:
    if exponent == 0.0:
        return np.exp(np.clip(eta, -LINK_CLIP, LINK_CLIP))
    return np.maximum(1.0 + exponent * eta, DEFICIT_EPSILON) ** (1.0 / exponent)


def base_model() -> Model:
    """The hierarchical phase-replay design, unchanged."""
    return Model("hierarchical_phase_replay", build_hierarchical_phase_replay, lambda: _state_shapes(True))


@dataclass(frozen=True)
class DeficitLinkFit:
    """A fitted surrogate, carrying everything needed to score an arbitrary policy."""

    shape: dict
    intercept: float
    coefficients: np.ndarray
    floor: float
    exponent: float
    buckets: tuple[str, ...]
    c0: np.ndarray
    c1: np.ndarray
    alpha: float
    family_index: np.ndarray
    family_names: tuple[str, ...]

    def predict_panel(self, panel: Panel) -> np.ndarray:
        design = build_hierarchical_phase_replay(panel, self.shape).matrix
        return self.floor + from_link(self.intercept + design @ self.coefficients, self.exponent)

    def predict_policy(self, phase0: np.ndarray, phase1: np.ndarray) -> np.ndarray:
        """Score arbitrary two-phase policies without needing observed targets."""
        phase0 = np.atleast_2d(np.asarray(phase0, dtype=float))
        phase1 = np.atleast_2d(np.asarray(phase1, dtype=float))
        rows = len(phase0)
        panel = Panel(
            scale="proposal",
            split="proposal",
            alpha=self.alpha,
            buckets=self.buckets,
            c0=self.c0,
            c1=self.c1,
            family_index=self.family_index,
            family_names=self.family_names,
            phase0=phase0,
            phase1=phase1,
            targets={},
            series=np.array(["proposal"] * rows),
            policy_class=np.array(["two_phase"] * rows),
            group=np.arange(rows),
            row_id=np.array([f"proposal_{i}" for i in range(rows)]),
        )
        return self.predict_panel(panel)


def _fit_head_on(panel: Panel, target: str, shape: dict, rows: np.ndarray) -> tuple[float, np.ndarray, float]:
    observed = panel.targets[target]
    use = rows & np.isfinite(observed)
    design = build_hierarchical_phase_replay(panel, shape).matrix[use]
    floor = FLOOR_FRACTION * float(np.min(observed[use]))
    intercept, coefficients = fit_head(design, to_link(observed[use] - floor, LINK_EXPONENT), L2)
    return intercept, coefficients, floor


def select_shape(panel: Panel, target: str, rows: np.ndarray | None = None) -> dict:
    """Choose the nonlinear shape by grouped out-of-fold RMSE on the fit panel alone.

    Only the shape is selected here. The link exponent, floor fraction, and ridge are
    pinned by the module constants, so nothing in the selection consults an evaluation
    set.
    """
    observed = panel.targets[target]
    available = np.isfinite(observed) if rows is None else (rows & np.isfinite(observed))
    best: tuple[float, dict] | None = None
    for shape in _state_shapes(True):
        prediction = np.full(len(observed), np.nan)
        for train, test in grouped_splits(panel, N_SPLITS, SPLIT_SEED):
            fold = train & available
            if fold.sum() < 2:
                continue
            intercept, coefficients, floor = _fit_head_on(panel, target, shape, fold)
            design = build_hierarchical_phase_replay(panel, shape).matrix[test]
            prediction[test] = floor + from_link(intercept + design @ coefficients, LINK_EXPONENT)
        finite = np.isfinite(prediction) & available
        score = float(np.sqrt(np.mean((prediction[finite] - observed[finite]) ** 2)))
        if best is None or score < best[0]:
            best = (score, shape)
    assert best is not None, "empty shape grid"
    return best[1]


def fit(panel: Panel, target: str, rows: np.ndarray | None = None) -> DeficitLinkFit:
    """Select the shape and fit the head on the supplied panel."""
    observed = panel.targets[target]
    available = np.isfinite(observed) if rows is None else (rows & np.isfinite(observed))
    shape = select_shape(panel, target, rows=available)
    intercept, coefficients, floor = _fit_head_on(panel, target, shape, available)
    return DeficitLinkFit(
        shape=shape,
        intercept=intercept,
        coefficients=coefficients,
        floor=floor,
        exponent=LINK_EXPONENT,
        buckets=panel.buckets,
        c0=panel.c0,
        c1=panel.c1,
        alpha=panel.alpha,
        family_index=panel.family_index,
        family_names=panel.family_names,
    )


def _simplex(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def kl_to(weights: np.ndarray, reference: np.ndarray) -> float:
    safe = np.clip(weights, 1e-12, None)
    return float((safe * np.log(safe / np.clip(reference, 1e-12, None))).sum())


@dataclass(frozen=True)
class Proposal:
    phase0: np.ndarray
    phase1: np.ndarray
    aggregate: np.ndarray
    predicted_bpb: float
    aggregate_kl: float
    effective_buckets: float
    max_simulated_epochs: float
    phase_tv: float


def propose(
    fitted: DeficitLinkFit,
    reference: np.ndarray,
    kl_budget: float,
    tied: bool = False,
    restarts: int = 12,
    seed: int = 20260726,
) -> Proposal:
    """Minimize predicted loss over two-phase policies inside a KL ball.

    Both phases are parameterized by softmax logits so the simplex constraints hold
    exactly, and the aggregate KL budget enters as a penalty that is raised until the
    returned policy satisfies it. ``tied`` constrains the two phases to be equal, which
    gives the single-phase optimum for comparison under an identical solver.
    """
    n = len(fitted.buckets)
    rng = np.random.default_rng(seed)
    alpha = fitted.alpha

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if tied:
            weights = _simplex(vector)
            return weights, weights
        return _simplex(vector[:n]), _simplex(vector[n:])

    def objective(vector: np.ndarray, penalty: float) -> float:
        phase0, phase1 = unpack(vector)
        aggregate = alpha * phase0 + (1.0 - alpha) * phase1
        predicted = float(fitted.predict_policy(phase0, phase1)[0])
        excess = max(0.0, kl_to(aggregate, reference) - kl_budget)
        return predicted + penalty * excess**2

    best: tuple[float, np.ndarray] | None = None
    size = n if tied else 2 * n
    for restart in range(restarts):
        start = np.log(np.clip(reference, 1e-12, None))
        start = np.concatenate([start] * (1 if tied else 2))
        if restart:
            start = start + 0.3 * rng.standard_normal(size)
        vector = start
        for penalty in (10.0, 100.0, 1000.0, 10000.0):
            result = minimize(
                objective,
                vector,
                args=(penalty,),
                method="Nelder-Mead",
                options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-10},
            )
            vector = result.x
        phase0, phase1 = unpack(vector)
        aggregate = alpha * phase0 + (1.0 - alpha) * phase1
        if kl_to(aggregate, reference) > kl_budget * 1.02:
            continue
        value = float(fitted.predict_policy(phase0, phase1)[0])
        if best is None or value < best[0]:
            best = (value, vector)

    assert best is not None, "no restart satisfied the KL budget"
    phase0, phase1 = unpack(best[1])
    aggregate = alpha * phase0 + (1.0 - alpha) * phase1
    epochs = fitted.c0 * phase0 + fitted.c1 * phase1
    return Proposal(
        phase0=phase0,
        phase1=phase1,
        aggregate=aggregate,
        predicted_bpb=best[0],
        aggregate_kl=kl_to(aggregate, reference),
        effective_buckets=float(1.0 / (aggregate**2).sum()),
        max_simulated_epochs=float(epochs.max()),
        phase_tv=float(0.5 * np.abs(phase1 - phase0).sum()),
    )
