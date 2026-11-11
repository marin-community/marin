"""Qualify the first naturally long update without rejecting shorter batches."""

import math
from dataclasses import dataclass
from enum import StrEnum

import torch


class QualificationStatus(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    EXHAUSTED = "exhausted"


@dataclass
class LongContextQualification:
    minimum_tokens: int
    maximum_tokens: int
    maximum_steps: int
    observed_steps: int = 0
    longest_observed_tokens: int = 0
    qualified_step: int | None = None

    @property
    def status(self) -> QualificationStatus:
        if self.qualified_step is not None:
            return QualificationStatus.PASSED
        if self.observed_steps >= self.maximum_steps:
            return QualificationStatus.EXHAUSTED
        return QualificationStatus.PENDING

    def observe(self, batch, *, step: int) -> dict | None:
        evidence = qualify_learner_batch(
            batch, minimum_tokens=self.minimum_tokens, maximum_tokens=self.maximum_tokens
        )
        self.observed_steps += 1
        self.longest_observed_tokens = max(self.longest_observed_tokens, evidence["longest_unpadded_tokens"])
        if evidence["qualifying_length"]:
            return {**evidence, "step": step}
        return None


def qualify_learner_batch(batch, *, minimum_tokens: int, maximum_tokens: int) -> dict:
    """Check token integrity; short valid batches remain eligible for training."""
    attention = batch['attention_mask'].bool()
    lengths = attention.sum(-1)
    longest = int(lengths.max())
    if longest > maximum_tokens:
        raise ValueError(f'Natural learner length {longest} exceeds qualification maximum {maximum_tokens}')
    response = batch['response_mask'].bool()
    loss = batch['loss_mask'].bool()
    if not loss.any():
        raise ValueError("Qualification requires supervised action tokens")
    if torch.any(loss & ~response):
        raise ValueError('Learner loss mask includes nonresponse tokens')
    behavior = batch['rollout_logprobs']
    if behavior is None or not torch.isfinite(behavior[loss]).all():
        raise ValueError('Qualification requires finite behavior logprobs at every action position')
    # Left padding is allowed; holes would invalidate monotonically increasing model positions.
    for mask in attention:
        positions = torch.nonzero(mask).flatten()
        if positions.numel() and not torch.equal(positions, torch.arange(positions[0], positions[-1] + 1, device=positions.device)):
            raise ValueError('Learner attention mask contains position gaps')
    return {'longest_unpadded_tokens': longest, 'qualifying_length': longest >= minimum_tokens,
            'minimum_tokens': minimum_tokens,
            'maximum_tokens': maximum_tokens, 'rows': len(lengths), 'action_tokens': int(loss.sum()),
            'behavior_logprobs_finite': True, 'attention_positions_contiguous': True}


def qualify_optimizer_status(status: dict) -> None:
    if status.get('policy_update_steps', 0) <= 0:
        raise ValueError('Qualification did not perform an optimizer update')
    for name, value in status.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f'Nonfinite first-update metric: {name}')
