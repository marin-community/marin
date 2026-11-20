# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hinted scaling-law fits that forecast unhinted benchmark accuracy from a model ladder.

Implements the forecasts of *Hinted Scaling Laws* (arXiv 2611.02270) and the baselines they are
compared against. The fits mirror the paper's released code
(github.com/tatsu-lab/hinted-scaling-laws at c0490e53, `experiments/forecasting/plot.py` and
`two_stage.py`): same parameterization, bounds, multi-start initializations, and unweighted
least-squares loss on cell accuracies.

- Joint law per hint method: acc(C, H) = max(F, L + (A - L) * sigmoid(b1 * (C - 20) + b2 * H + b0)),
  with C = log10 training FLOPs and H the hint strength. Its H = 0 slice is the forecast.
- Hinted ensemble: the mean of the per-method H = 0 forecasts.
- Sigmoid baseline: acc = sigmoid(a1 * C + a0) on the unhinted scores.
- Two-stage baseline: bpb = L_inf + A * FLOPs^-alpha, then acc = sigmoid(b1 * bpb + b0).

`bootstrap_forecasts` resamples benchmark problems with replacement (the same draw for every
cell and method) and refits every forecast.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

CAPABILITY_CENTER = 20.0
"""The joint law's capability axis is centred here before fitting; only the intercept depends on it."""

_JOINT_LOWER_BOUNDS = (0.0, 0.0, -np.inf, 1e-3, -50.0, 0.0)
_JOINT_UPPER_BOUNDS = (np.inf, np.inf, np.inf, 1.0, 0.999, 0.999)
_POWER_LAW_STARTS = ((1.0, 0.2, 0.26), (1e2, 0.15, 0.20), (1e4, 0.25, 0.30), (1e3, 0.18, 0.22))
_BPB_SIGMOID_STARTS = ((-5.0, 2.0), (-10.0, 5.0), (-2.0, 1.0), (-20.0, 10.0))


class HintMethod(StrEnum):
    TRUNCATION = "truncation"
    MASKING = "masking"


class ForecastKind(StrEnum):
    TRUNCATION = "truncation"
    MASKING = "masking"
    HINTED_ENSEMBLE = "hinted_ensemble"
    SIGMOID_BASELINE = "sigmoid_baseline"
    TWO_STAGE_BASELINE = "two_stage_baseline"


@dataclass(frozen=True)
class ScoreCells:
    """Per-problem scores of a model ladder at a set of (capability, hint) cells.

    Row i is one (model, hint strength) cell. Column j is one benchmark problem, the same problem
    in every row (and in every `ScoreCells` of one `LadderScores`), so problems can be resampled.
    """

    capability: np.ndarray
    """(n_cells,) log10 training FLOPs of the cell's model."""

    hint: np.ndarray
    """(n_cells,) hint strength H; 0 for unhinted cells."""

    problem_scores: np.ndarray
    """(n_cells, n_problems) per-problem accuracy in [0, 1]: the mean pass rate over the problem's
    rollouts (and over training seeds, when a ladder rung has several)."""

    def __post_init__(self):
        n_cells = self.problem_scores.shape[0]
        if self.problem_scores.ndim != 2 or n_cells == 0 or self.problem_scores.shape[1] == 0:
            raise ValueError(f"problem_scores must be a non-empty 2-D array, got {self.problem_scores.shape}")
        if self.capability.shape != (n_cells,) or self.hint.shape != (n_cells,):
            raise ValueError(
                f"capability {self.capability.shape} and hint {self.hint.shape} must have one entry per row of "
                f"problem_scores {self.problem_scores.shape}"
            )
        if not np.all(np.isfinite(self.problem_scores)):
            raise ValueError("problem_scores must be finite; drop or fill missing problems before fitting")

    @property
    def n_problems(self) -> int:
        return self.problem_scores.shape[1]

    @property
    def accuracy(self) -> np.ndarray:
        """(n_cells,) benchmark accuracy of each cell."""
        return self.problem_scores.mean(axis=1)

    def take_problems(self, problem_indices: np.ndarray) -> "ScoreCells":
        return ScoreCells(self.capability, self.hint, self.problem_scores[:, problem_indices])

    def take_cells(self, mask: np.ndarray) -> "ScoreCells":
        return ScoreCells(self.capability[mask], self.hint[mask], self.problem_scores[mask])


def concat_cells(cells: list[ScoreCells]) -> ScoreCells:
    return ScoreCells(
        np.concatenate([c.capability for c in cells]),
        np.concatenate([c.hint for c in cells]),
        np.concatenate([c.problem_scores for c in cells], axis=0),
    )


@dataclass(frozen=True)
class LadderScores:
    """One model ladder's scores on one benchmark: everything the forecasts fit."""

    unhinted: ScoreCells
    """One H = 0 cell per model. Every hint method's joint fit includes these cells."""

    hinted: dict[HintMethod, ScoreCells]
    """Per hint method, its H > 0 cells."""

    bits_per_byte: np.ndarray | None
    """(n_models,) bits per byte of each model, aligned with the rows of `unhinted`; the first stage of
    the two-stage baseline, which is skipped when this is None. Not resampled by the bootstrap."""

    def __post_init__(self):
        if not self.hinted:
            raise ValueError("at least one hint method is required")
        if np.any(self.unhinted.hint != 0):
            raise ValueError("unhinted cells must have hint == 0")
        if self.bits_per_byte is not None:
            if self.bits_per_byte.shape != self.unhinted.capability.shape:
                raise ValueError(
                    f"bits_per_byte {self.bits_per_byte.shape} must align with the unhinted cells "
                    f"{self.unhinted.capability.shape}"
                )
            if not np.all(np.isfinite(self.bits_per_byte)):
                raise ValueError("bits_per_byte must be finite")
        mismatched = [m for m, c in self.hinted.items() if c.n_problems != self.unhinted.n_problems]
        if mismatched:
            raise ValueError(f"hinted cells of {mismatched} do not share the unhinted cells' problems")

    def up_to(self, max_capability: float) -> "LadderScores":
        """The ladder restricted to models with capability <= max_capability, i.e. the fitted models."""
        limit = max_capability + 1e-9
        keep = self.unhinted.capability <= limit
        return LadderScores(
            unhinted=self.unhinted.take_cells(keep),
            hinted={m: c.take_cells(c.capability <= limit) for m, c in self.hinted.items()},
            bits_per_byte=None if self.bits_per_byte is None else self.bits_per_byte[keep],
        )

    def take_problems(self, problem_indices: np.ndarray) -> "LadderScores":
        return LadderScores(
            unhinted=self.unhinted.take_problems(problem_indices),
            hinted={m: c.take_problems(problem_indices) for m, c in self.hinted.items()},
            bits_per_byte=self.bits_per_byte,
        )


@dataclass(frozen=True)
class JointLawParams:
    """Parameters of the floored joint law.

    acc(C, H) = max(floor, lower + (upper - lower) * sigmoid(capability_slope * (C - 20) + hint_slope * H + intercept)).
    In the uncentred form sigmoid(b1 * C + b2 * H + b3), b3 = intercept - 20 * capability_slope.
    `lower` is the sigmoid's lower asymptote and may be negative; `floor` clips the curve from below.
    """

    capability_slope: float
    hint_slope: float
    intercept: float
    upper: float
    lower: float
    floor: float

    def predict(self, capability: np.ndarray | float, hint: np.ndarray | float) -> np.ndarray:
        z = (
            self.capability_slope * (np.asarray(capability, dtype=float) - CAPABILITY_CENTER)
            + self.hint_slope * np.asarray(hint, dtype=float)
            + self.intercept
        )
        return np.maximum(self.floor, self.lower + (self.upper - self.lower) * expit(z))


def _best_least_squares(
    residuals: Callable[[np.ndarray], np.ndarray],
    starts: Iterable[Sequence[float]],
    bounds: tuple[Sequence[float] | float, Sequence[float] | float] = (-np.inf, np.inf),
) -> np.ndarray:
    """Run bounded least squares from every start and return the lowest-cost solution."""
    return min((least_squares(residuals, x0=x0, bounds=bounds) for x0 in starts), key=lambda r: r.cost).x


def _joint_starts() -> Iterator[tuple[float, ...]]:
    for b1, b2, b0, (upper, lower), floor in product(
        (0.5, 2.0),
        (0.5, 2.0, 8.0),
        (-4.0, -2.0, 0.0),
        ((0.9, 0.0), (1.0, -0.1), (0.9, -0.5)),
        (0.0, 0.01, 0.02),
    ):
        yield (b1, b2, b0, upper, lower, floor)


def fit_joint_law(capability: np.ndarray, hint: np.ndarray, accuracy: np.ndarray) -> JointLawParams:
    """Fit the floored joint law by multi-start bounded least squares on cell accuracies.

    Bounds: capability_slope, hint_slope >= 0; upper in [1e-3, 1]; lower in [-50, 0.999];
    floor in [0, 0.999]. 162 starts, as in the paper's code.
    """
    c, h, y = (np.asarray(a, dtype=float) for a in (capability, hint, accuracy))

    def residuals(x: np.ndarray) -> np.ndarray:
        return JointLawParams(*x).predict(c, h) - y

    best = _best_least_squares(residuals, _joint_starts(), (_JOINT_LOWER_BOUNDS, _JOINT_UPPER_BOUNDS))
    return JointLawParams(*(float(v) for v in best))


@dataclass(frozen=True)
class SigmoidParams:
    """acc = sigmoid(slope * x + intercept)."""

    slope: float
    intercept: float

    def predict(self, x: np.ndarray | float) -> np.ndarray:
        return expit(self.slope * np.asarray(x, dtype=float) + self.intercept)


def fit_sigmoid_baseline(capability: np.ndarray, accuracy: np.ndarray) -> SigmoidParams:
    """Fit acc = sigmoid(a1 * C + a0), a1 >= 0, on unhinted scores; multi-start least squares."""
    c, y = np.asarray(capability, dtype=float), np.asarray(accuracy, dtype=float)
    c_mean = float(c.mean())
    # On floor-heavy ladders a single start lands in a poor local optimum.
    starts = ((1.0, -10.0), (1.0, -c_mean), (0.5, -0.5 * c_mean), (3.0, -3.0 * c_mean), (10.0, -10.0 * c_mean))
    best = _best_least_squares(lambda b: SigmoidParams(*b).predict(c) - y, starts, ([0.0, -np.inf], [np.inf, np.inf]))
    return SigmoidParams(*(float(v) for v in best))


@dataclass(frozen=True)
class TwoStageParams:
    """bpb = bpb_irreducible + bpb_scale * FLOPs^-bpb_exponent, then acc = accuracy_link(bpb)."""

    bpb_scale: float
    bpb_exponent: float
    bpb_irreducible: float
    accuracy_link: SigmoidParams

    def predict_bits_per_byte(self, capability: np.ndarray | float) -> np.ndarray:
        flops = 10.0 ** np.asarray(capability, dtype=float)
        return self.bpb_irreducible + self.bpb_scale * flops ** (-self.bpb_exponent)

    def predict(self, capability: np.ndarray | float) -> np.ndarray:
        return self.accuracy_link.predict(self.predict_bits_per_byte(capability))


def fit_two_stage_baseline(capability: np.ndarray, bits_per_byte: np.ndarray, accuracy: np.ndarray) -> TwoStageParams:
    """Fit compute -> bits per byte (power law, exponent in [0, 2], L_inf in [0, 1]) and
    bits per byte -> accuracy (unbounded sigmoid), each by multi-start least squares."""
    flops = 10.0 ** np.asarray(capability, dtype=float)
    bpb, y = np.asarray(bits_per_byte, dtype=float), np.asarray(accuracy, dtype=float)
    power = _best_least_squares(
        lambda p: p[2] + p[0] * flops ** (-p[1]) - bpb, _POWER_LAW_STARTS, ([0.0, 0.0, 0.0], [np.inf, 2.0, 1.0])
    )
    link = _best_least_squares(lambda b: SigmoidParams(*b).predict(bpb) - y, _BPB_SIGMOID_STARTS)
    scale, exponent, irreducible = (float(v) for v in power)
    return TwoStageParams(scale, exponent, irreducible, SigmoidParams(*(float(v) for v in link)))


@dataclass(frozen=True)
class LadderFit:
    """Every forecast law fitted to one ladder."""

    joint: dict[HintMethod, JointLawParams]
    sigmoid_baseline: SigmoidParams
    two_stage_baseline: TwoStageParams | None
    """None when the ladder has no bits per byte."""

    def forecasts(self, target_capability: float) -> dict[ForecastKind, float]:
        """Unhinted accuracy forecasts at `target_capability` (log10 training FLOPs)."""
        per_method = {ForecastKind(m.value): float(p.predict(target_capability, 0.0)) for m, p in self.joint.items()}
        forecasts = {
            **per_method,
            ForecastKind.HINTED_ENSEMBLE: float(np.mean(list(per_method.values()))),
            ForecastKind.SIGMOID_BASELINE: float(self.sigmoid_baseline.predict(target_capability)),
        }
        if self.two_stage_baseline is not None:
            forecasts[ForecastKind.TWO_STAGE_BASELINE] = float(self.two_stage_baseline.predict(target_capability))
        return forecasts


def fit_ladder(ladder: LadderScores) -> LadderFit:
    """Fit each hint method's joint law (its hinted cells plus the shared unhinted cells) and the baselines.

    The two-stage baseline is fitted only when the ladder has bits per byte.
    """
    unhinted = ladder.unhinted
    joint = {}
    for method, cells in ladder.hinted.items():
        both = concat_cells([cells, unhinted])
        joint[method] = fit_joint_law(both.capability, both.hint, both.accuracy)
    two_stage = None
    if ladder.bits_per_byte is not None:
        two_stage = fit_two_stage_baseline(unhinted.capability, ladder.bits_per_byte, unhinted.accuracy)
    return LadderFit(
        joint=joint,
        sigmoid_baseline=fit_sigmoid_baseline(unhinted.capability, unhinted.accuracy),
        two_stage_baseline=two_stage,
    )


@dataclass(frozen=True)
class BootstrapForecasts:
    """Point forecasts on the observed problems and their bootstrap distribution over problems."""

    point: dict[ForecastKind, float]
    samples: dict[ForecastKind, np.ndarray]
    """Per forecast kind, (n_resamples,) forecasts refitted on problem resamples."""

    def interval(self, kind: ForecastKind, confidence: float) -> tuple[float, float]:
        """Central percentile interval of the bootstrap forecasts."""
        tail = (1.0 - confidence) / 2.0
        lo, hi = np.quantile(self.samples[kind], [tail, 1.0 - tail])
        return float(lo), float(hi)


def bootstrap_forecasts(
    ladder: LadderScores, target_capability: float, n_resamples: int, seed: int
) -> BootstrapForecasts:
    """Forecast unhinted accuracy at `target_capability` and bootstrap every forecast over problems.

    Each resample draws n_problems problem indices with replacement, applies the same draw to every
    cell of every method, and refits all laws. Fit the ladder restricted to the models you fit on
    (`LadderScores.up_to`).
    """
    rng = np.random.default_rng(seed)
    point = fit_ladder(ladder).forecasts(target_capability)
    n_problems = ladder.unhinted.n_problems
    draws = [
        fit_ladder(ladder.take_problems(rng.integers(0, n_problems, size=n_problems))).forecasts(target_capability)
        for _ in range(n_resamples)
    ]
    samples = {kind: np.array([d[kind] for d in draws]) for kind in point}
    return BootstrapForecasts(point=point, samples=samples)
