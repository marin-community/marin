"""Qualification uses real unpadded lengths, action masks and finite learner evidence."""

import pytest
import torch

from skyrl_train.long_context_qualification import qualify_learner_batch, qualify_optimizer_status


def batch():
    return {'attention_mask': torch.tensor([[0, 0, 1, 1, 1, 1]]),
            'response_mask': torch.tensor([[1, 1, 1]]), 'loss_mask': torch.tensor([[1, 0, 1]]),
            'rollout_logprobs': torch.tensor([[-1., 0., -2.]])}


def test_padding_is_not_natural_context_and_bad_action_evidence_fails():
    evidence = qualify_learner_batch(batch(), minimum_tokens=4, maximum_tokens=8)
    assert evidence['longest_unpadded_tokens'] == 4
    with pytest.raises(ValueError, match='Natural learner length'):
        qualify_learner_batch(batch(), minimum_tokens=5, maximum_tokens=8)
    broken = batch()
    broken['rollout_logprobs'][0, 0] = torch.nan
    with pytest.raises(ValueError, match='finite behavior'):
        qualify_learner_batch(broken, minimum_tokens=4, maximum_tokens=8)
    broken = batch()
    broken['attention_mask'] = torch.tensor([[1, 0, 1, 1, 1, 1]])
    with pytest.raises(ValueError, match='position gaps'):
        qualify_learner_batch(broken, minimum_tokens=4, maximum_tokens=8)
    qualify_optimizer_status({'policy_update_steps': 1, 'raw_grad_norm': 0.1})
    with pytest.raises(ValueError, match='optimizer update'):
        qualify_optimizer_status({'policy_update_steps': 0})
    with pytest.raises(ValueError, match='Nonfinite'):
        qualify_optimizer_status({'policy_update_steps': 1, 'raw_grad_norm': float('inf')})
