# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thompson sampling over public training prompts with delayed optimizer feedback."""

from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from skyrl_train.config.sampling import PromptAcquisition

_HASH_DIMENSION = 12
_PRIOR_PRECISION = 2.0
_CHECKPOINT_VERSION = 1
_SCORE_SPREAD = 0.05
_MIN_COST_WEIGHT = 0.65
_MAX_COST_WEIGHT = 1.5
_COST_EXPONENT = 0.6
_WORDS = re.compile(r"\w+|[^\w\s]")
_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True)
class PublicPrompt:
    uid: str
    source: str
    text: str
    family: str | None = None

    @property
    def digest(self) -> str:
        return public_prompt_digest(self.text)


def public_prompt_digest(text: str) -> str:
    """Fingerprint public text for training/evaluation overlap checks."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass(frozen=True)
class PromptFeedback:
    """The outcome of one dispatched training group, including work spent on failures."""

    lease_id: str
    uid: str
    dispatch_step: int
    sample_count: int
    verified_scores: tuple[float | None, ...] | None
    generated_tokens: int
    nonzero_advantage_tokens: int | None

    @property
    def fully_verified(self) -> bool:
        scores = self.verified_scores
        return (
            scores is not None
            and len(scores) == self.sample_count >= 2
            and all(score is not None and math.isfinite(score) and 0 <= score <= 1 for score in scores)
        )

    @property
    def useful(self) -> bool:
        """Check for full verification, score spread of at least 0.05, and nonzero training advantage."""
        if not self.fully_verified or not self.nonzero_advantage_tokens:
            return False
        scores = [score for score in self.verified_scores or () if score is not None]
        return max(scores) - min(scores) >= _SCORE_SPREAD


@lru_cache(maxsize=4096)
def _feature_bin(term: str) -> tuple[int, int]:
    digest = hashlib.blake2b(term.encode(), digest_size=2).digest()
    return digest[0] % _HASH_DIMENSION, 1 if digest[1] < 128 else -1


def public_prompt_features(text: str) -> NDArray[np.float64]:
    """Encode generic text shape and signed hashed unigrams and bigrams."""
    terms = _WORDS.findall(_DIGITS.sub("0", text.casefold()))
    lexical = np.zeros(_HASH_DIMENSION)
    for term in [*terms, *(first + " " + second for first, second in zip(terms, terms[1:]))]:
        index, sign = _feature_bin(term)
        lexical[index] += sign
    lexical /= max(1.0, float(np.linalg.norm(lexical)))
    shape = [
        math.log1p(len(text)) / 2,
        math.log1p(len(terms)) / 2,
        math.log1p(sum(character.isdigit() for character in text)) / 2,
        math.log1p(sum(not term.isalnum() and term != "_" for term in terms)) / 2,
    ]
    return np.concatenate((shape, 1.5 * lexical))


def _prompt_features(
    prompts: Sequence[PublicPrompt], sources: Sequence[str], acquisition: PromptAcquisition
) -> NDArray[np.float64]:
    source_rows = np.searchsorted(sources, [prompt.source for prompt in prompts])
    indicators = np.column_stack((np.ones(len(prompts)), np.eye(len(sources))[source_rows]))
    if acquisition is PromptAcquisition.SOURCE:
        return indicators
    features = np.stack([public_prompt_features(prompt.text) for prompt in prompts])
    for source_index in range(len(sources)):
        rows = source_rows == source_index
        features[rows] -= features[rows].mean(axis=0)
    return np.column_stack((indicators, features))


def _source_quota_boundaries(shares: NDArray[np.float64], capacities: NDArray, size: int) -> NDArray[np.float64]:
    """Cumulative expected counts after redistributing shares that exceed a source's catalog."""
    quotas = np.zeros(len(shares))
    remaining = size
    indices = np.flatnonzero(capacities)
    while len(indices):
        weights = shares[indices] / shares[indices].max()
        proposed = remaining * weights / weights.sum()
        saturated = proposed > capacities[indices]
        if not saturated.any():
            quotas[indices] = proposed
            break
        full = indices[saturated]
        quotas[full] = capacities[full]
        remaining -= int(capacities[full].sum())
        indices = indices[~saturated]
    return quotas.cumsum()[:-1]


class PromptThompsonOrder:
    """Rank distinct training prompts by sampled probability of useful optimizer input.

    Source-only ranks prompts from the same source equally. Contextual ranking also
    uses public prompt text. Cost-aware ranking gives additional weight to prompts
    predicted to spend fewer generated tokens. Within-source cost-aware ranking
    preserves the reference mixture in each window up to randomized rounding and pool capacity.
    """

    def __init__(
        self,
        prompts: Sequence[PublicPrompt],
        held_out_digests: set[str],
        reference: Mapping[str, float],
        *,
        acquisition: PromptAcquisition,
        seed: int,
        window_size: int,
        decay: float,
        epsilon: float,
        reference_tokens: int,
        groups_per_family: Mapping[str, int] | None = None,
    ):
        if not 1 <= window_size <= len(prompts):
            raise ValueError("Thompson window must fit in the training prompt catalog")
        if not 0 <= decay <= 1 or not 0 <= epsilon <= 1 or reference_tokens <= 0:
            raise ValueError("Thompson sampling requires decay and exploration in [0, 1] and positive reference tokens")
        self._rows = {prompt.uid: row for row, prompt in enumerate(prompts)}
        if any(
            not isinstance(prompt.source, str) or not prompt.source or not prompt.text.strip() for prompt in prompts
        ):
            raise ValueError("Thompson sampling requires a source and public text on each training prompt")
        self._row_sources = tuple(prompt.source for prompt in prompts)
        self._sources = sorted(set(self._row_sources))
        digests = [prompt.digest for prompt in prompts]
        if len(self._rows) != len(prompts) or len(set(digests)) != len(prompts):
            raise ValueError("Thompson sampling requires distinct training prompts")
        if held_out_digests.intersection(digests):
            raise ValueError("A held-out public prompt is present in the Thompson training catalog")

        counts = Counter(self._row_sources)
        shares = dict(reference) or dict(counts)
        if set(shares) != set(self._sources) or any(not math.isfinite(x) or x <= 0 for x in shares.values()):
            raise ValueError("Reference shares must positively weight exactly the prepared training sources")
        self._weights = np.array([shares[source] / counts[source] for source in self._row_sources])
        self._acquisition = acquisition
        self._decay = decay
        self._epsilon = epsilon
        self._reference_tokens = reference_tokens
        self._window_size = window_size
        self._window: list[int] = []
        self._groups_per_family = dict(groups_per_family or {})
        self._family_windows: dict[str, list[int]] = {}
        self._last_step = 0
        self._rng = np.random.default_rng(seed)
        self._exploration_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self._source_boundaries: NDArray[np.float64] | None = None
        if self.uses_source_passes:
            if groups_per_family:
                raise ValueError("groups_per_family cannot partition a per-source prompt pass")
            self._visits = np.zeros(len(prompts), dtype=np.int64)
            self._pass_used: set[int] = set()
            self._pass_sources: list[int] = []
            self._pass_priority = np.empty(0)
        if acquisition.within_source:
            self._source_ids = np.searchsorted(self._sources, self._row_sources)
            self._source_shares = np.array([shares[source] for source in self._sources], dtype=float)
            self._source_boundaries = _source_quota_boundaries(
                self._source_shares,
                np.array([counts[source] for source in self._sources]),
                window_size,
            )
            # Keep the source schedule paired across epsilon settings and different optimizer feedback.
            self._source_rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
            families = np.array([prompt.family for prompt in prompts], dtype=object)
            self._family_source_shares = {
                family: np.bincount(
                    self._source_ids[families == family],
                    weights=self._weights[families == family],
                    minlength=len(self._sources),
                )
                for family in self._groups_per_family
            }
        self._assigned: Counter[str] = Counter()
        self._seen: set[str] = set()

        self._features = _prompt_features(prompts, self._sources, acquisition)
        dimension = self._features.shape[1]
        self._prior = np.eye(dimension) * _PRIOR_PRECISION
        self._precision = self._prior.copy()
        self._info = np.zeros(dimension)
        self._cost_precision = self._prior.copy()
        self._cost_info = np.zeros(dimension)
        signature = (
            _CHECKPOINT_VERSION,
            acquisition,
            seed,
            window_size,
            decay,
            epsilon,
            reference_tokens,
            sorted(shares.items()),
            [(prompt.uid, prompt.source, digest) for prompt, digest in zip(prompts, digests, strict=True)],
        )
        if self._groups_per_family:
            signature += (sorted(self._groups_per_family.items()), [prompt.family for prompt in prompts])
        signature_bytes = json.dumps(signature, separators=(",", ":")).encode()
        self._signature = hashlib.sha256(signature_bytes + self._features.astype("<f8").tobytes()).hexdigest()

    @property
    def uses_source_passes(self) -> bool:
        return self._acquisition is PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE

    def next_index(self) -> int:
        if self.uses_source_passes:
            row = self.next_pass_index(())
            assert row is not None
            return row
        if not self._window:
            available = np.ones(len(self._features), dtype=bool)
            self._window = self._draw_window(available, self._window_size, self._source_boundaries)
        return self._window.pop()

    def next_pass_index(self, excluded: Collection[str]) -> int | None:
        """Choose a least-visited eligible row of the requested source, counting actual offers."""
        available = np.ones(len(self._rows), dtype=bool)
        available[[self._rows[uid] for uid in excluded]] = False
        if not available.any():
            return None
        # A pass spans source windows. Within a window, even a source that wraps
        # to its next pass cannot offer the same row twice.
        if self._pass_sources:
            unused = available.copy()
            unused[list(self._pass_used)] = False
            if unused.any():
                available = unused
            else:
                self._pass_sources.clear()
        capacities = np.bincount(self._source_ids[available], minlength=len(self._sources))
        if not self._pass_sources:
            size = min(self._window_size, int(capacities.sum()))
            boundaries = _source_quota_boundaries(self._source_shares, capacities, size)
            self._pass_sources = self._source_window(boundaries, size).tolist()[::-1]
            self._pass_priority = self._priorities()
            self._pass_used.clear()
        # A newly unavailable source can defer its slot. If all remaining
        # sources are blocked, redistribute just those slots over free rows.
        position = next(
            (i for i in reversed(range(len(self._pass_sources))) if capacities[self._pass_sources[i]]), None
        )
        if position is None:
            size = min(len(self._pass_sources), int(capacities.sum()))
            boundaries = _source_quota_boundaries(self._source_shares, capacities, size)
            self._pass_sources = self._source_window(boundaries, size).tolist()[::-1]
            position = -1
        source = self._pass_sources.pop(position)
        eligible = available & (self._source_ids == source)
        eligible &= self._visits == self._visits[eligible].min()
        row = self._choose(eligible, self._pass_priority)
        self._pass_used.add(row)
        self._visits[row] += 1
        return row

    def next_family_index(self, family: str, rows: Sequence[int], eligible: Callable[[int], bool]) -> int | None:
        """Draw from the eligible rows of a leased family."""
        window = self._family_windows.get(family, [])
        for position in range(len(window) - 1, -1, -1):
            if eligible(window[position]):
                return window.pop(position)
        eligible_rows = [row for row in rows if eligible(row)]
        if not eligible_rows:
            return None
        window_size = min(self._groups_per_family[family], len(eligible_rows))
        available = np.zeros(len(self._features), dtype=bool)
        available[eligible_rows] = True
        source_boundaries = None
        if self._acquisition is PromptAcquisition.WITHIN_SOURCE_COST_AWARE:
            capacities = np.bincount(self._source_ids[available], minlength=len(self._sources))
            # Condition the original reference row mass on family. Temporary exclusions
            # reduce source capacity without changing the reference shares.
            source_boundaries = _source_quota_boundaries(self._family_source_shares[family], capacities, window_size)
        window = self._draw_window(available, window_size, source_boundaries)
        self._family_windows[family] = window
        return window.pop()

    def assign(self, uid: str) -> str:
        source = self._row_sources[self._rows[uid]]
        self._assigned[source] += 1
        return source

    def _priorities(self) -> NDArray[np.float64]:
        coefficients = np.linalg.solve(self._precision, self._info)
        draw = np.linalg.solve(np.linalg.cholesky(self._precision).T, self._rng.standard_normal(len(coefficients)))
        priority = self._probabilities(self._features @ (coefficients + draw))
        if self._acquisition is PromptAcquisition.COST_AWARE or self._acquisition.within_source:
            log_relative_cost = self._features @ np.linalg.solve(self._cost_precision, self._cost_info)
            priority *= np.exp(
                np.clip(-_COST_EXPONENT * log_relative_cost, np.log(_MIN_COST_WEIGHT), np.log(_MAX_COST_WEIGHT))
            )
        return priority

    def _source_window(self, boundaries: NDArray[np.float64], size: int) -> NDArray:
        # A shifted unit lattice gives every source the floor or ceiling of its expected count.
        sources = np.searchsorted(boundaries, np.arange(size) + self._source_rng.random(), side="right")
        self._source_rng.shuffle(sources)
        return sources

    def _choose(self, eligible: NDArray[np.bool_], priority: NDArray[np.float64]) -> int:
        if self._exploration_rng.random() < self._epsilon:
            weights = self._weights * eligible
            return int(self._exploration_rng.choice(len(priority), p=weights / weights.sum()))
        peak = priority[eligible].max()
        return int(self._rng.choice(np.flatnonzero(eligible & (priority == peak))))

    def _draw_window(
        self,
        available: NDArray[np.bool_],
        window_size: int,
        source_boundaries: NDArray[np.float64] | None,
    ) -> list[int]:
        priority = self._priorities()
        sources = self._source_window(source_boundaries, window_size) if source_boundaries is not None else None
        window: list[int] = []
        for slot in range(window_size):
            eligible = available if sources is None else available & (self._source_ids == sources[slot])
            row = self._choose(eligible, priority)
            available[row] = False
            window.append(row)
        return window[::-1]

    @staticmethod
    def _probabilities(logits: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))

    def prediction(self, uid: str) -> tuple[float, float]:
        """Return mean-model usefulness probability and predicted generated tokens for a training row."""
        feature = self._features[self._rows[uid]]
        logit = feature @ np.linalg.solve(self._precision, self._info)
        log_relative_cost = feature @ np.linalg.solve(self._cost_precision, self._cost_info)
        return float(self._probabilities(logit)), float(self._reference_tokens * np.exp(log_relative_cost))

    def finish_step(self, step: int, groups: Sequence[PromptFeedback]) -> dict[str, float]:
        """Train on outcomes available after a completed step and report realized yield and spend."""
        if step <= self._last_step:
            raise ValueError("Thompson feedback steps must strictly advance")
        leases = {group.lease_id for group in groups}
        if len(leases) != len(groups) or self._seen.intersection(leases):
            raise ValueError("A dispatched group can contribute Thompson feedback only once")
        if any(
            not 0 < group.dispatch_step <= step
            or group.generated_tokens < 0
            or (group.nonzero_advantage_tokens is not None and group.nonzero_advantage_tokens < 0)
            for group in groups
        ):
            raise ValueError("Thompson feedback requires a prior or current training dispatch and nonnegative costs")
        labels = np.array([group.useful for group in groups], dtype=float)
        self._fit(step, groups, labels)
        self._last_step = step
        self._seen.update(leases)

        metrics = {
            "sampling/groups": float(len(groups)),
            "sampling/useful_groups": float(labels.sum()),
            "sampling/fully_verified_groups": float(sum(group.fully_verified for group in groups)),
            "sampling/optimizer_groups": float(sum(group.nonzero_advantage_tokens is not None for group in groups)),
            "sampling/generated_tokens": float(sum(group.generated_tokens for group in groups)),
            **{f"sampling/{source}/assigned_groups": float(self._assigned[source]) for source in self._sources},
        }
        self._assigned.clear()
        return metrics

    def _fit(self, step: int, groups: Sequence[PromptFeedback], labels: NDArray[np.float64]) -> None:
        features = self._features[[self._rows[group.uid] for group in groups]]
        costs = np.array([group.generated_tokens for group in groups])
        weights = np.power(self._decay, step - np.array([group.dispatch_step for group in groups], dtype=int))
        age = self._decay ** (step - self._last_step)
        self._precision = self._prior + age * (self._precision - self._prior)
        self._info *= age
        self._cost_precision = self._prior + age * (self._cost_precision - self._prior)
        self._cost_info *= age

        old_logits = features @ np.linalg.solve(self._precision, self._info)
        probability = self._probabilities(old_logits)
        curvature = probability * (1 - probability)
        self._precision += features.T @ ((weights * curvature)[:, None] * features)
        self._info += features.T @ (weights * (labels - probability + curvature * old_logits))
        log_costs = np.log((costs + 1) / (self._reference_tokens + 1))
        self._cost_precision += features.T @ (weights[:, None] * features)
        self._cost_info += features.T @ (weights * log_costs)

    def state_dict(self) -> dict[str, Any]:
        state = {
            "signature": self._signature,
            "rng": deepcopy(self._rng.bit_generator.state),
            "exploration_rng": deepcopy(self._exploration_rng.bit_generator.state),
            "assigned": dict(self._assigned),
            "window": list(self._window),
            "last_step": self._last_step,
            "seen": sorted(self._seen),
            "precision": self._precision.tolist(),
            "info": self._info.tolist(),
            "cost_precision": self._cost_precision.tolist(),
            "cost_info": self._cost_info.tolist(),
        }
        if self._acquisition.within_source:
            state["source_rng"] = deepcopy(self._source_rng.bit_generator.state)
        if self.uses_source_passes:
            state["pass"] = {
                "visits": self._visits.tolist(),
                "used": sorted(self._pass_used),
                "sources": list(self._pass_sources),
                "priority": self._pass_priority.tolist(),
            }
        if self._groups_per_family:
            state["family_windows"] = {family: list(window) for family, window in self._family_windows.items()}
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("signature") != self._signature:
            raise ValueError("Thompson checkpoint belongs to another training catalog or policy")
        if self._acquisition.within_source:
            self._source_rng.bit_generator.state = state["source_rng"]
        if self.uses_source_passes:
            saved = state["pass"]
            self._visits = np.asarray(saved["visits"], dtype=np.int64)
            self._pass_used = set(saved["used"])
            self._pass_sources = list(saved["sources"])
            self._pass_priority = np.asarray(saved["priority"], dtype=float)
        if self._groups_per_family:
            self._family_windows = {family: list(window) for family, window in state["family_windows"].items()}
        self._rng.bit_generator.state = state["rng"]
        self._exploration_rng.bit_generator.state = state["exploration_rng"]
        self._assigned = Counter(state["assigned"])
        self._window = list(state["window"])
        self._last_step = state["last_step"]
        self._seen = set(state["seen"])
        self._precision = np.asarray(state["precision"], dtype=float)
        self._info = np.asarray(state["info"], dtype=float)
        self._cost_precision = np.asarray(state["cost_precision"], dtype=float)
        self._cost_info = np.asarray(state["cost_info"], dtype=float)
