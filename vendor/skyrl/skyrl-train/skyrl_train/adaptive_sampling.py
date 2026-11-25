# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source sampling from the frequency of mixed verifier scores within a prompt group."""

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from skyrl_train.rollouts.loader import JudgedGroup


class AdaptiveSourceOrder:
    """Reweight a reference mixture by each source's posterior mean informative-group rate.

    Each fully graded prompt group supplies one observation. A group is informative when
    its verifier awarded different scores to its answers. Decay discounts older policy
    steps, and epsilon reserves a fraction of draws for the original reference mix.
    """

    def __init__(
        self,
        sources: Iterable[str],
        reference: Mapping[str, float],
        *,
        seed: int,
        window_size: int,
        decay: float,
        epsilon: float,
        prior_alpha: float,
        prior_beta: float,
    ):
        self.sources = tuple(sources)
        if not 1 <= window_size <= len(self.sources):
            raise ValueError("A draw window must fit in the filtered dataset")
        if not 0 <= decay <= 1 or not 0 <= epsilon <= 1:
            raise ValueError("adaptive-source decay and epsilon must be between 0 and 1")
        if not all(math.isfinite(x) and x > 0 for x in (prior_alpha, prior_beta)):
            raise ValueError("adaptive-source requires positive finite Beta prior parameters")
        self.pools: dict[str, list[int]] = defaultdict(list)
        for row, source in enumerate(self.sources):
            if not isinstance(source, str) or not source:
                raise ValueError(f"adaptive-source requires a source label for filtered row {row}")
            self.pools[source].append(row)
        self.names = sorted(self.pools)
        shares = reference or {name: len(rows) for name, rows in self.pools.items()}
        if set(shares) != set(self.names) or any(not math.isfinite(x) or x <= 0 for x in shares.values()):
            raise ValueError(f"Reference shares must positively weight exactly the prepared sources: {self.names}")
        total = sum(shares.values())
        self.reference = {name: shares[name] / total for name in self.names}
        self.decay = decay
        self.epsilon = epsilon
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.informative = dict.fromkeys(self.names, 0.0)
        self.graded = dict.fromkeys(self.names, 0.0)
        self.window_size = window_size
        self.window: list[int] = []
        self.proposed: Counter[str] = Counter()
        self.assigned: Counter[str] = Counter()
        self.rng = random.Random(seed)

    def probabilities(self) -> dict[str, float]:
        weighted = {
            name: self.reference[name]
            * (self.prior_alpha + self.informative[name])
            / (self.prior_alpha + self.prior_beta + self.graded[name])
            for name in self.names
        }
        total = sum(weighted.values())
        return {
            name: (1 - self.epsilon) * weighted[name] / total + self.epsilon * self.reference[name]
            for name in self.names
        }

    def next_index(self) -> int:
        if not self.window:
            # Draw sources first, then sample distinct rows within each source. Even sources
            # with fewer rows than a full window can be emphasized without rejection loops.
            weights = self.probabilities()
            counts: Counter[str] = Counter()
            available = list(self.names)
            for _ in range(self.window_size):
                source = self.rng.choices(available, weights=[weights[name] for name in available])[0]
                counts[source] += 1
                if counts[source] == len(self.pools[source]):
                    available.remove(source)
            for source, count in counts.items():
                self.window.extend(self.rng.sample(self.pools[source], count))
            self.rng.shuffle(self.window)
        row = self.window.pop()
        self.proposed[self.sources[row]] += 1
        return row

    def assign(self, uid: str) -> str:
        """Count the group actually submitted to a worker and return its sampler bin."""
        source = self.sources[int(uid)]
        self.assigned[source] += 1
        return source

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        judged: Counter[str] = Counter()
        graded: Counter[str] = Counter()
        informative: Counter[str] = Counter()
        for group in groups:
            source = self.sources[int(group.uid)]
            judged[source] += 1
            scores = group.verified_scores
            if scores is None or len(scores) < 2 or None in scores:
                continue
            graded[source] += 1
            informative[source] += int(any(score != scores[0] for score in scores))
        for name in self.names:
            self.informative[name] = self.informative[name] * self.decay + informative[name]
            self.graded[name] = self.graded[name] * self.decay + graded[name]
        metrics = {}
        for name, weight in self.probabilities().items():
            prefix = f"curriculum/{name}/"
            metrics[prefix + "weight"] = weight
            metrics[prefix + "proposed_rows"] = float(self.proposed[name])
            metrics[prefix + "assigned_groups"] = float(self.assigned[name])
            metrics[prefix + "groups"] = float(judged[name])
            metrics[prefix + "graded_groups"] = float(graded[name])
            metrics[prefix + "informative_groups"] = float(informative[name])
        self.proposed.clear()
        self.assigned.clear()
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "window": self.window.copy(),
            "proposed": self.proposed.copy(),
            "assigned": self.assigned.copy(),
            "informative": self.informative.copy(),
            "graded": self.graded.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.rng.setstate(state["rng"])
        self.window = list(state["window"])
        self.proposed = Counter(state["proposed"])
        self.assigned = Counter(state["assigned"])
        self.informative = dict(state["informative"])
        self.graded = dict(state["graded"])
