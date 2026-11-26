# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One randomized source-mix update from an independent development probe."""

import hashlib
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


CALLBACK_TYPE = "development_probe_mix"


def panel_identity(task_digests: Iterable[str]) -> str:
    """Fingerprint the order-independent set of complete development tasks."""
    return hashlib.sha256("\n".join(sorted(task_digests)).encode()).hexdigest()


@dataclass(frozen=True)
class ProbeMixDecision:
    step: int
    accuracy: float
    easy_rich_probability: float
    easy_share: float


class DevelopmentProbeMix:
    """Keep a randomized initial quota and update it once at a gross-token milestone."""

    def __init__(self, *, seed: int, panel_sha256: str):
        self.seed = seed
        self.panel_sha256 = panel_sha256
        self.initial: ProbeMixDecision | None = None
        self.updated: ProbeMixDecision | None = None

    def _draw(self, step: int, accuracy: float, probability: float) -> ProbeMixDecision:
        draw = random.Random(f"development-probe-mix/{self.seed}/{step}").random()
        return ProbeMixDecision(step, accuracy, probability, 0.8 if draw < probability else 0.2)

    def observe(self, *, step: int, accuracy: float, gross_tokens: int) -> ProbeMixDecision | None:
        """Consume a complete baseline or one intermediate probe."""
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Development probe accuracy must lie between zero and one")
        if step == 0 and self.initial is None and gross_tokens == 0:
            self.initial = self._draw(0, accuracy, 0.5)
            return self.initial
        if self.initial is None:
            raise ValueError("Development probe mix requires its initializer probe before rollout")
        if self.updated is not None:
            if self.updated.step != step:
                raise ValueError("Development probe mix accepts only one intermediate probe")
            return None
        if step <= 0 or gross_tokens <= 0:
            raise ValueError("Development probe gain requires a completed block with paid tokens")
        gain = 100 * (accuracy - self.initial.accuracy) * 100_000 / gross_tokens
        residual = (self.initial.easy_share == 0.8) - self.initial.easy_rich_probability
        # Normal prior and likelihood with standard deviations of 2 pp per 100k tokens.
        mean_over_sd = residual * gain / (2 * math.sqrt(1 + residual**2))
        probability = 0.25 + 0.25 * (1 + math.erf(mean_over_sd / math.sqrt(2)))
        self.updated = self._draw(step, accuracy, probability)
        return self.updated

    @property
    def decision(self) -> ProbeMixDecision | None:
        return self.updated or self.initial

    def state_dict(self) -> dict:
        return {
            "initial": vars(self.initial) if self.initial is not None else None,
            "updated": vars(self.updated) if self.updated is not None else None,
        }

    def load_state_dict(self, state: Mapping) -> None:
        initial = state["initial"]
        updated = state["updated"]
        self.initial = ProbeMixDecision(**initial) if initial is not None else None
        self.updated = ProbeMixDecision(**updated) if updated is not None else None
