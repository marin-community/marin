"""Adaptive curriculum sampling over prompt-dataset bins.

Training rows carry a bin assignment in their ``extra_info`` column: ``data_source`` names the
bin (e.g. ``"g0-gsm8k"``) and ``grade`` orders bins by difficulty (0 easiest). CurriculumOrder
draws dataset rows with replacement according to per-bin weights, updated each training step from
the rewards of the groups batch selection judged. A prompt group (one row's rollouts) is
"informative" when its rewards are not all equal — an all-pass or all-fail group contributes no
GRPO gradient signal.

The module splits sampling mechanics from weighting policy:

- ``CurriculumOrder`` owns the draw stream (window-unique rejection draws, RNG, checkpoint
  state) and folds each step's groups into ``BinStats``.
- ``BinStats`` holds the decayed per-bin sufficient statistics (informative/total group counts
  and solved/sample counts, whose ratio is a decayed pass-rate estimate).
- ``RowStats`` records the same evidence per dataset row (visit counts, per-visit-decayed
  pass evidence, recency), checkpointed but not yet consulted by any weight policy.
- ``WeightPolicy`` subclasses turn those statistics into per-bin weights. ``build_policy``
  normalizes the ``data.sampling`` config subtree into a policy instance once, at construction.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Dict, List

import numpy as np
from omegaconf import DictConfig

from skyrl_train.dataset import PromptDataset
from skyrl_train.rollouts.loader import JudgedGroup

EXTRA_INFO_KEY = "extra_info"
DATA_SOURCE_KEY = "data_source"
GRADE_KEY = "grade"


class SamplingKind(StrEnum):
    DOMAIN_WEIGHTED = "domain-weighted"
    NAIVE = "naive"
    THOMPSON = "thompson"
    LEARNABILITY = "learnability"
    GRADE_UNIFORM = "grade-uniform"
    GRADE_ADAPTIVE = "grade-adaptive"
    GRADE_PRIOR = "grade-prior"


class WeightingKind(StrEnum):
    """Weight curve applied to the Thompson-drawn pass rate p of each bin.

    ``pass-variance`` is p·(1−p): per-sample reward variance, peaked at p=0.5.
    ``group-informative`` is 1 − p^n − (1−p)^n with n samples per prompt: the
    probability a GRPO group has nonzero advantage variance, i.e. survives
    dynamic-sampling filtering. It is near-flat across mid difficulties and
    collapses only within ~1/n of the extremes, matching the filter's actual
    rollout-cost model.
    """

    PASS_VARIANCE = "pass-variance"
    GROUP_INFORMATIVE = "group-informative"


@dataclass(frozen=True)
class CurriculumConfig:
    """Weight-policy parameters, mirroring the ``data.sampling`` config subtree."""

    kind: SamplingKind
    decay: float = 0.95
    epsilon: float = 0.05
    reversion_mass: float = 0.0
    instance_decay: float = 0.7
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    grade_prior_strength: float = 8.0
    grade_prior_high: float = 0.85
    grade_prior_low: float = 0.05
    adaptive_exploration: float = 0.2
    adaptive_window: int = 10
    adaptive_min_informative: float = 0.1
    weighting: WeightingKind = WeightingKind.PASS_VARIANCE
    # GRPO group size (generator.n_samples_per_prompt), the n in the
    # group-informative curve. Supplied by the caller, not the config subtree.
    group_size: int | None = None

    @classmethod
    def from_dict_config(cls, cfg: DictConfig, *, group_size: int) -> "CurriculumConfig":
        return cls(
            kind=SamplingKind(cfg.kind),
            decay=cfg.decay,
            epsilon=cfg.epsilon,
            reversion_mass=cfg.reversion_mass,
            instance_decay=cfg.instance_decay,
            prior_alpha=cfg.prior_alpha,
            prior_beta=cfg.prior_beta,
            grade_prior_strength=cfg.grade_prior_strength,
            grade_prior_high=cfg.grade_prior_high,
            grade_prior_low=cfg.grade_prior_low,
            adaptive_exploration=cfg.adaptive_exploration,
            adaptive_window=cfg.adaptive_window,
            adaptive_min_informative=cfg.adaptive_min_informative,
            weighting=WeightingKind(cfg.weighting),
            group_size=group_size,
        )


@dataclass(frozen=True)
class DatasetBins:
    """Row -> bin structure extracted from a post-filter PromptDataset."""

    names: List[str]  # bin index -> data_source name, sorted by (grade, name)
    grades: np.ndarray  # bin index -> grade
    row_counts: np.ndarray  # bin index -> number of rows
    bin_rows: List[np.ndarray]  # bin index -> row indices
    row_to_bin: np.ndarray  # row index -> bin index


def dataset_bins(dataset: PromptDataset) -> DatasetBins:
    """Build per-row bin assignments from the dataset's ``extra_info`` column."""
    if EXTRA_INFO_KEY not in dataset.dataframe.column_names:
        raise ValueError(f"Curriculum sampling requires an `{EXTRA_INFO_KEY}` column in the training dataset")

    grade_by_name: Dict[str, int] = {}
    name_by_row: List[str] = []
    for row, info in enumerate(dataset.dataframe[EXTRA_INFO_KEY]):
        if not info or info.get(DATA_SOURCE_KEY) is None or info.get(GRADE_KEY) is None:
            raise ValueError(
                f"Row {row}: curriculum sampling requires `{DATA_SOURCE_KEY}` and `{GRADE_KEY}` in `{EXTRA_INFO_KEY}`"
            )
        name = info[DATA_SOURCE_KEY]
        grade = int(info[GRADE_KEY])
        if grade_by_name.setdefault(name, grade) != grade:
            raise ValueError(f"Bin `{name}` has inconsistent grades: {grade_by_name[name]} and {grade}")
        name_by_row.append(name)

    names = sorted(grade_by_name, key=lambda name: (grade_by_name[name], name))
    bin_index = {name: b for b, name in enumerate(names)}
    row_to_bin = np.array([bin_index[name] for name in name_by_row], dtype=np.int64)
    bin_rows = [np.flatnonzero(row_to_bin == b) for b in range(len(names))]
    return DatasetBins(
        names=names,
        grades=np.array([grade_by_name[name] for name in names], dtype=np.int64),
        row_counts=np.array([len(rows) for rows in bin_rows], dtype=np.int64),
        bin_rows=bin_rows,
        row_to_bin=row_to_bin,
    )


@dataclass(frozen=True)
class GroupOutcome:
    """One prompt group's rollout outcome, folded into the statistics."""

    row: int
    bin_index: int
    size: int
    solved: int
    informative: bool


class BinStats:
    """Decayed per-bin sufficient statistics, written by the curriculum order, read by policies.

    All arrays decay multiplicatively each step before the step's counts are added.
    ``reversion_mass`` adds per-step pseudo-evidence at the weight-curve peak (pass 0.5):
    a bin pinned at the epsilon floor (~8 real rollouts/step at pass 0) settles at a pass
    estimate of reversion_mass * 0.5 / (8 + reversion_mass), so its weight drifts back to
    a re-probeable level, while a heavily sampled bin's estimate is dominated by real
    counts and barely moves.
    """

    def __init__(self, bins: DatasetBins, decay: float, reversion_mass: float):
        self.bins = bins
        self.decay = decay
        self.reversion_mass = reversion_mass
        self.grade_values = np.unique(bins.grades)  # sorted ascending
        num_bins = len(bins.names)
        self.informative = np.zeros(num_bins)
        self.total = np.zeros(num_bins)
        self.solved = np.zeros(num_bins)
        self.samples = np.zeros(num_bins)
        self.step_groups = np.zeros(num_bins)

    def apply_step(self, outcomes: List[GroupOutcome]) -> None:
        num_bins = len(self.bins.names)
        step_informative = np.zeros(num_bins)
        step_total = np.zeros(num_bins)
        step_solved = np.zeros(num_bins)
        step_samples = np.zeros(num_bins)
        for outcome in outcomes:
            step_total[outcome.bin_index] += 1
            step_informative[outcome.bin_index] += outcome.informative
            step_samples[outcome.bin_index] += outcome.size
            step_solved[outcome.bin_index] += outcome.solved

        self.informative = self.decay * self.informative + step_informative + self.reversion_mass * 0.5
        self.total = self.decay * self.total + step_total + self.reversion_mass
        self.solved = self.decay * self.solved + step_solved + self.reversion_mass * 0.5
        self.samples = self.decay * self.samples + step_samples + self.reversion_mass
        self.step_groups = step_total

    def state_dict(self) -> Dict[str, Any]:
        return {
            "informative": self.informative.copy(),
            "total": self.total.copy(),
            "solved": self.solved.copy(),
            "samples": self.samples.copy(),
            "step_groups": self.step_groups.copy(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.informative = state_dict["informative"].copy()
        self.total = state_dict["total"].copy()
        self.solved = state_dict["solved"].copy()
        self.samples = state_dict["samples"].copy()
        self.step_groups = state_dict["step_groups"].copy()


class RowStats:
    """Per-row (example) rollout evidence, updated only when a row is drawn.

    Bin statistics decay per training step, which suits bins that receive rollouts every
    step; a single row is drawn orders of magnitude less often, so per-step decay would
    erase its evidence between visits. Row counts instead decay per visit: when a row
    receives a new group, its previous counts shrink by ``instance_decay`` before the
    group's counts are added, making solved/samples an EWMA pass estimate over the row's
    visit history under policy drift. ``visits`` and ``last_step`` are undecayed exposure
    and recency records. No weight policy consults these yet; they are tracked and
    checkpointed so instance-level weighting can build on recorded visit history.
    """

    def __init__(self, num_rows: int, instance_decay: float):
        self.instance_decay = instance_decay
        self.visits = np.zeros(num_rows, dtype=np.int64)
        self.samples = np.zeros(num_rows)
        self.solved = np.zeros(num_rows)
        self.last_step = np.full(num_rows, -1, dtype=np.int64)

    def observe_group(self, row: int, solved: int, size: int, step: int) -> None:
        self.visits[row] += 1
        self.samples[row] = self.instance_decay * self.samples[row] + size
        self.solved[row] = self.instance_decay * self.solved[row] + solved
        self.last_step[row] = step

    def summary_metrics(self) -> Dict[str, float]:
        """Aggregate instance-evidence health; per-row series would swamp the logger."""
        visited = self.visits > 0
        num_visited = int(visited.sum())
        out = {"curriculum/instances/visited_frac": num_visited / len(self.visits)}
        if num_visited == 0:
            return out
        pass_rate = self.solved[visited] / self.samples[visited]
        revisited = self.visits[visited] >= 2
        out["curriculum/instances/mean_pass"] = float(pass_rate.mean())
        out["curriculum/instances/mastered_frac"] = float((revisited & (pass_rate >= 0.999)).mean())
        out["curriculum/instances/dead_frac"] = float((revisited & (pass_rate <= 0.001)).mean())
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "visits": self.visits.copy(),
            "samples": self.samples.copy(),
            "solved": self.solved.copy(),
            "last_step": self.last_step.copy(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.visits = state_dict["visits"].copy()
        self.samples = state_dict["samples"].copy()
        self.solved = state_dict["solved"].copy()
        self.last_step = state_dict["last_step"].copy()


def floored_normalized(weights: np.ndarray, epsilon: float) -> np.ndarray:
    """Normalize, apply the per-bin probability floor epsilon/num_bins, renormalize."""
    weights = weights / weights.sum()
    weights = np.maximum(weights, epsilon / len(weights))
    return weights / weights.sum()


def pass_variance_curve(pass_rate: np.ndarray) -> np.ndarray:
    return pass_rate * (1.0 - pass_rate)


def group_informative_curve(group_size: int | None) -> Callable[[np.ndarray], np.ndarray]:
    if group_size is None or group_size < 2:
        raise ValueError("group-informative weighting requires group_size >= 2")

    def curve(pass_rate: np.ndarray) -> np.ndarray:
        return 1.0 - pass_rate**group_size - (1.0 - pass_rate) ** group_size

    return curve


def make_curve(weighting: WeightingKind, group_size: int | None) -> Callable[[np.ndarray], np.ndarray]:
    if weighting is WeightingKind.GROUP_INFORMATIVE:
        return group_informative_curve(group_size)
    return pass_variance_curve


def flat_prior(num_bins: int, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    return np.full(num_bins, alpha), np.full(num_bins, beta)


def grade_linear_prior(grades: np.ndarray, high: float, low: float, strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Beta priors whose mean pass rate falls linearly from ``high`` at the lowest grade
    to ``low`` at the highest, with ``strength`` pseudo-counts per bin."""
    grades = grades.astype(np.float64)
    grade_span = grades.max() - grades.min()
    fraction = (grades - grades.min()) / grade_span if grade_span > 0 else np.zeros(len(grades))
    mean = high + fraction * (low - high)
    return 1.0 + strength * mean, 1.0 + strength * (1.0 - mean)


class WeightPolicy:
    """Turns curriculum statistics into per-bin sampling weights.

    Policies are stateless by default; a policy with internal state overrides the
    observe/state hooks so the curriculum order can checkpoint it alongside the statistics.
    """

    def weights(self, stats: BinStats, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def observe_step(self, stats: BinStats) -> None:
        """Advance policy-internal state after one step's statistics have been folded in."""

    def metrics(self) -> Dict[str, float]:
        return {}

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if state_dict:
            raise ValueError(f"Unexpected policy state for {type(self).__name__}: {sorted(state_dict)}")


class RowProportional(WeightPolicy):
    """Uniform over rows: each bin weighted by its row count (the naive arm)."""

    def weights(self, stats: BinStats, rng: np.random.Generator) -> np.ndarray:
        counts = stats.bins.row_counts.astype(np.float64)
        return counts / counts.sum()


class GradeUniform(WeightPolicy):
    """Equal budget per grade, row-proportional within a grade."""

    def weights(self, stats: BinStats, rng: np.random.Generator) -> np.ndarray:
        counts = stats.bins.row_counts.astype(np.float64)
        weights = np.zeros(len(counts))
        for grade in stats.grade_values:
            mask = stats.bins.grades == grade
            weights[mask] = counts[mask] / counts[mask].sum() / len(stats.grade_values)
        return weights / weights.sum()


class ThompsonInformative(WeightPolicy):
    """Thompson draw on each bin's informative-group rate."""

    def __init__(self, prior_alpha: float, prior_beta: float, epsilon: float):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.epsilon = epsilon

    def weights(self, stats: BinStats, rng: np.random.Generator) -> np.ndarray:
        draws = rng.beta(self.prior_alpha + stats.informative, self.prior_beta + (stats.total - stats.informative))
        return floored_normalized(draws, self.epsilon)


class Directional(WeightPolicy):
    """Thompson draw on the pass rate, pushed through a difficulty weight curve.

    The learnability and grade-prior kinds are this policy with different priors: a
    flat Beta prior, or one whose mean falls linearly with grade (``grade_linear_prior``).
    """

    def __init__(
        self,
        prior_alpha: np.ndarray,
        prior_beta: np.ndarray,
        curve: Callable[[np.ndarray], np.ndarray],
        epsilon: float,
    ):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.curve = curve
        self.epsilon = epsilon

    def weights(self, stats: BinStats, rng: np.random.Generator) -> np.ndarray:
        pass_rate = rng.beta(self.prior_alpha + stats.solved, self.prior_beta + (stats.samples - stats.solved))
        return floored_normalized(self.curve(pass_rate), self.epsilon)


class GradeAdaptive(WeightPolicy):
    """Concentrates sampling on one grade level, advancing when it stops being informative.

    The current level receives 1 − exploration of the mass (row-proportional within the
    level) and every other bin shares the exploration remainder. The level advances once
    its decayed informative fraction stays below ``min_informative`` for ``window``
    consecutive steps.
    """

    def __init__(self, exploration: float, window: int, min_informative: float, grade_values: np.ndarray):
        self.exploration = exploration
        self.window = window
        self.min_informative = min_informative
        self.grade_values = grade_values
        self.level_pos = 0
        self.low_signal_steps = 0

    def observe_step(self, stats: BinStats) -> None:
        level_mask = stats.bins.grades == self.grade_values[self.level_pos]
        level_total = stats.total[level_mask].sum()
        if level_total <= 0:
            return
        informative_frac = stats.informative[level_mask].sum() / level_total
        if informative_frac < self.min_informative:
            self.low_signal_steps += 1
        else:
            self.low_signal_steps = 0
        if self.low_signal_steps >= self.window and self.level_pos < len(self.grade_values) - 1:
            self.level_pos += 1
            self.low_signal_steps = 0

    def weights(self, stats: BinStats, rng: np.random.Generator) -> np.ndarray:
        counts = stats.bins.row_counts.astype(np.float64)
        level_mask = stats.bins.grades == self.grade_values[self.level_pos]
        if level_mask.all():
            return counts / counts.sum()
        weights = np.zeros(len(counts))
        weights[level_mask] = (1.0 - self.exploration) * counts[level_mask] / counts[level_mask].sum()
        weights[~level_mask] = self.exploration * counts[~level_mask] / counts[~level_mask].sum()
        return weights / weights.sum()

    def metrics(self) -> Dict[str, float]:
        return {"curriculum/level": float(self.grade_values[self.level_pos])}

    def state_dict(self) -> Dict[str, Any]:
        return {"level_pos": self.level_pos, "low_signal_steps": self.low_signal_steps}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.level_pos = state_dict["level_pos"]
        self.low_signal_steps = state_dict["low_signal_steps"]


def build_policy(config: CurriculumConfig, bins: DatasetBins) -> WeightPolicy:
    """Normalize the config subtree into a policy instance (the only kind switch)."""
    num_bins = len(bins.names)
    match config.kind:
        case SamplingKind.NAIVE:
            return RowProportional()
        case SamplingKind.GRADE_UNIFORM:
            return GradeUniform()
        case SamplingKind.THOMPSON:
            return ThompsonInformative(config.prior_alpha, config.prior_beta, config.epsilon)
        case SamplingKind.LEARNABILITY:
            alpha, beta = flat_prior(num_bins, config.prior_alpha, config.prior_beta)
            return Directional(alpha, beta, make_curve(config.weighting, config.group_size), config.epsilon)
        case SamplingKind.GRADE_PRIOR:
            alpha, beta = grade_linear_prior(
                bins.grades, config.grade_prior_high, config.grade_prior_low, config.grade_prior_strength
            )
            return Directional(alpha, beta, make_curve(config.weighting, config.group_size), config.epsilon)
        case SamplingKind.GRADE_ADAPTIVE:
            return GradeAdaptive(
                config.adaptive_exploration,
                config.adaptive_window,
                config.adaptive_min_informative,
                np.unique(bins.grades),
            )
    raise ValueError(f"Unknown sampling kind: {config.kind}")


class CurriculumOrder:
    """Draw dataset rows with replacement from per-bin curriculum weights, learning from each step's groups.

    Draws are uniform within a bin and deterministic given the seed and draw history. Rows are distinct within
    each consecutive window of ``window_size`` draws: two groups of one row in the same batch would share a uid,
    and the rollout buffer keeps only one of them.
    """

    def __init__(self, dataset: PromptDataset, config: CurriculumConfig, *, seed: int, window_size: int):
        if window_size < 1:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self.config = config
        self.bins = dataset_bins(dataset)
        self.window_size = window_size
        self.rng = np.random.default_rng(seed)
        self.stats = BinStats(self.bins, decay=config.decay, reversion_mass=config.reversion_mass)
        self.rows = RowStats(len(dataset), instance_decay=config.instance_decay)
        self.policy = build_policy(config, self.bins)
        self.update_count = 0
        self.weights = self.policy.weights(self.stats, self.rng)
        self._window_rows: list[int] = []

    def next_index(self) -> int:
        if len(self._window_rows) == self.window_size:
            self._window_rows = []
        row = self._draw(exclude=set(self._window_rows))
        self._window_rows.append(row)
        return row

    def _draw(self, exclude: set[int]) -> int:
        for _ in range(1000):
            bin_idx = int(self.rng.choice(len(self.weights), p=self.weights))
            rows = self.bins.bin_rows[bin_idx]
            row = int(rows[self.rng.integers(len(rows))])
            if row not in exclude:
                return row
        raise RuntimeError(f"Could not draw a window-unique row after 1000 attempts (window_size={self.window_size})")

    def observe(self, groups: Sequence[JudgedGroup]) -> Dict[str, float]:
        """Fold one training step's judged groups into the statistics, redraw the bin weights, and return metrics.

        The groups include those dynamic sampling discarded: without all-pass and all-fail groups the statistics
        would be biased toward informative bins. A sample counts as solved when its reward is positive.
        """
        outcomes = [
            GroupOutcome(
                row=int(group.uid),
                bin_index=int(self.bins.row_to_bin[int(group.uid)]),
                size=len(group.rewards),
                solved=sum(reward > 0 for reward in group.rewards),
                informative=any(reward != group.rewards[0] for reward in group.rewards),
            )
            for group in groups
        ]
        self.update_count += 1
        self.stats.apply_step(outcomes)
        for outcome in outcomes:
            self.rows.observe_group(outcome.row, outcome.solved, outcome.size, self.update_count)
        self.policy.observe_step(self.stats)
        self.weights = self.policy.weights(self.stats, self.rng)
        return self.metrics()

    def metrics(self) -> Dict[str, float]:
        """Per-bin curriculum metrics for the current step."""
        out = {}
        stats = self.stats
        for bin_idx, name in enumerate(self.bins.names):
            total = stats.total[bin_idx]
            out[f"curriculum/{name}/weight"] = float(self.weights[bin_idx])
            out[f"curriculum/{name}/informative_frac"] = float(stats.informative[bin_idx] / total) if total > 0 else 0.0
            samples = stats.samples[bin_idx]
            out[f"curriculum/{name}/pass_rate"] = float(stats.solved[bin_idx] / samples) if samples > 0 else 0.0
            out[f"curriculum/{name}/groups"] = float(stats.step_groups[bin_idx])
        out.update(self.rows.summary_metrics())
        out.update(self.policy.metrics())
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "stats": self.stats.state_dict(),
            "rows": self.rows.state_dict(),
            "policy": self.policy.state_dict(),
            "weights": self.weights.copy(),
            "update_count": self.update_count,
            "rng_state": self.rng.bit_generator.state,
            "window_rows": list(self._window_rows),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.stats.load_state_dict(state_dict["stats"])
        self.rows.load_state_dict(state_dict["rows"])
        self.policy.load_state_dict(state_dict["policy"])
        self.weights = state_dict["weights"].copy()
        self.update_count = state_dict["update_count"]
        self.rng.bit_generator.state = state_dict["rng_state"]
        self._window_rows = list(state_dict["window_rows"])
