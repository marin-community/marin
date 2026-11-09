"""Fail-fast evidence for the first natural long-context optimizer update."""

import math

import torch


def qualify_learner_batch(batch, *, minimum_tokens: int, maximum_tokens: int) -> dict:
    """Check unpadded lengths and every supervised token before the first forward."""
    attention = batch['attention_mask'].bool()
    lengths = attention.sum(-1)
    longest = int(lengths.max())
    if longest < minimum_tokens or longest > maximum_tokens:
        raise ValueError(f'Natural learner length {longest} is outside qualification interval {minimum_tokens}..{maximum_tokens}')
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
    return {'longest_unpadded_tokens': longest, 'minimum_tokens': minimum_tokens,
            'maximum_tokens': maximum_tokens, 'rows': len(lengths), 'action_tokens': int(loss.sum()),
            'behavior_logprobs_finite': True, 'attention_positions_contiguous': True}


def qualify_optimizer_status(status: dict) -> None:
    if status.get('policy_update_steps', 0) <= 0:
        raise ValueError('Qualification did not perform an optimizer update')
    for name, value in status.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f'Nonfinite first-update metric: {name}')
