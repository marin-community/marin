# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source quotas driven by verified transfer gain at gross-token milestones."""

from bisect import bisect_right
import hashlib
import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np


CALLBACK_TYPE = "development_probe_mix"
ACTIONS = (6, 8, 10)
WINDOW = 16
_MIX_BASE = (0.10, 0.40, 0.10)
_POSTERIOR_MASS = 0.40
INITIAL_PROBABILITIES = tuple(base + _POSTERIOR_MASS / len(ACTIONS) for base in _MIX_BASE)
_FEATURES = np.array(((-1., 1.), (0., 0.), (1., 1.)))
_PRIOR_SD = 1.0
_REWARD_SD = 0.9
_FIRST_BLOCK_REFERENCE = 1.5
_SOFTMAX_SCALE = 0.25


def panel_identity(task_digests: Iterable[str]) -> str:
    """Fingerprint the order-independent set of complete development tasks."""
    return hashlib.sha256("\n".join(sorted(task_digests)).encode()).hexdigest()


@dataclass(frozen=True)
class ProbeMixDecision:
    boundary: int
    step: int
    accuracy: float
    gross_training_tokens: int
    development_response_tokens: int
    block_gross_training_tokens: int
    transfer_gain_per_100k_train_tokens: float | None
    phase_reference_gain_per_100k: float
    action_probabilities: tuple[float, ...]
    adaptive_action_probabilities: tuple[float, ...]
    posterior_mean: tuple[float, ...]
    posterior_precision: tuple[tuple[float, ...], ...]
    uniform_draw: float
    easy_quota: int

    @property
    def easy_share(self) -> float:
        return self.easy_quota / WINDOW


class DevelopmentProbeMix:
    """Select a bounded quota and carry its probe credit across resume."""

    def __init__(self, *, seed: int, panel_sha256: str, policy: str):
        if policy not in {"adaptive", "randomized"}:
            raise ValueError("Development probe mix requires adaptive or randomized policy")
        self.seed = seed
        self.panel_sha256 = panel_sha256
        self.policy = policy
        self.decisions: list[ProbeMixDecision] = []
        self.precision = np.eye(2) / _PRIOR_SD**2
        self.information = np.zeros(2)

    @property
    def decision(self) -> ProbeMixDecision | None:
        return self.decisions[-1] if self.decisions else None

    def _probabilities(self, previous: ProbeMixDecision | None, reward: float | None, reference: float) -> tuple[float, ...]:
        if previous is None or reward is None:
            return INITIAL_PROBABILITIES
        observed = _FEATURES[ACTIONS.index(previous.easy_quota)]
        centered = observed - np.asarray(previous.action_probabilities) @ _FEATURES
        self.precision += np.outer(centered, centered) / _REWARD_SD**2
        self.information += centered * (reward - reference) / _REWARD_SD**2
        logits = _FEATURES @ np.linalg.solve(self.precision, self.information)
        preference = np.exp((logits - logits.max()) / _SOFTMAX_SCALE)
        preference /= preference.sum()
        # Keep at least 40% balanced and at least 10% for each mild tilt.
        probabilities = np.asarray(_MIX_BASE) + _POSTERIOR_MASS * preference
        return tuple(map(float, probabilities))

    def observe(
        self, *, boundary: int, step: int, accuracy: float, gross_tokens: int, development_tokens: int
    ) -> ProbeMixDecision | None:
        """Credit one complete development observation and choose the next quota."""
        previous = self.decision
        if previous is not None and (boundary, step, gross_tokens) == (
            previous.boundary, previous.step, previous.gross_training_tokens
        ):
            return None
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("Development probe accuracy must lie between zero and one")
        if development_tokens < 0 or boundary != len(self.decisions):
            raise ValueError("Development probes must complete once in milestone order")
        if previous is None and (step != 0 or gross_tokens != 0):
            raise ValueError("Development probe mix requires its initializer probe before rollout")
        if previous is not None and (step <= previous.step or gross_tokens <= previous.gross_training_tokens):
            raise ValueError("Development probe gain requires a new completed block with paid tokens")

        block_tokens = gross_tokens - previous.gross_training_tokens if previous is not None else 0
        reward = (accuracy - previous.accuracy) * 10_000_000 / block_tokens if previous is not None else None
        reference = _FIRST_BLOCK_REFERENCE if boundary == 1 else 0.0
        preference = self._probabilities(previous, reward, reference)
        probabilities = INITIAL_PROBABILITIES if self.policy == "randomized" else preference
        # Use the same draw at a given boundary even when two arms cross it at different optimizer steps.
        draw = random.Random(f"development-probe-mix/{self.seed}/{boundary}").random()
        index = bisect_right((probabilities[0], sum(probabilities[:2])), draw)
        action = ACTIONS[index]
        decision = ProbeMixDecision(
            boundary, step, accuracy, gross_tokens, development_tokens, block_tokens,
            reward, reference, probabilities, preference,
            tuple(map(float, np.linalg.solve(self.precision, self.information))),
            tuple(tuple(map(float, row)) for row in self.precision),
            draw, action
        )
        self.decisions.append(decision)
        return decision

    def state_dict(self) -> dict:
        return {
            "decisions": [asdict(decision) for decision in self.decisions],
            "precision": self.precision.tolist(),
            "information": self.information.tolist(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.decisions = [
            ProbeMixDecision(**decision)
            for decision in state["decisions"]
        ]
        self.precision = np.asarray(state["precision"], dtype=float)
        self.information = np.asarray(state["information"], dtype=float)
