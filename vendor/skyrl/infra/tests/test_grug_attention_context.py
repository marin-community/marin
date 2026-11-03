"""Lifecycle and dropout behavior of Grug's scoring attention context."""

import pytest
import torch
from torch import nn

from skyrl_train.models.grug_attention_context import grug_scoring_attention_context


@pytest.mark.parametrize("fail_forward", [False, True])
def test_scoring_restores_subtree_modes_and_preserves_outer_eval(fail_forward):
    attention = nn.Sequential(nn.Linear(2, 1, bias=False), nn.Dropout(0.0))
    model = nn.Sequential(attention, nn.Dropout(0.5)).eval()
    attention[1].train(True)
    original_modes = [module.training for module in model.modules()]

    def score():
        with torch.no_grad(), grug_scoring_attention_context(attention, attention_dropout=0.0):
            assert not model.training and not model[1].training
            assert all(module.training for module in attention.modules())
            assert not torch.is_grad_enabled()
            result = attention(torch.ones(1, 2))
            if fail_forward:
                raise RuntimeError("attention forward failed")
            return result

    if fail_forward:
        with pytest.raises(RuntimeError, match="attention forward failed"):
            score()
    else:
        assert score().shape == (1, 1)
    assert [module.training for module in model.modules()] == original_modes


def test_training_attention_preserves_backward_and_existing_subtree_modes():
    attention = nn.Sequential(nn.Linear(2, 1, bias=False), nn.Dropout(0.0)).train()
    attention[1].eval()
    original_modes = [module.training for module in attention.modules()]
    inputs = torch.tensor([[2.0, 3.0]], requires_grad=True)
    with grug_scoring_attention_context(attention, attention_dropout=0.0):
        assert [module.training for module in attention.modules()] == original_modes
        attention(inputs).sum().backward()
    torch.testing.assert_close(attention[0].weight.grad, inputs.detach())
    torch.testing.assert_close(inputs.grad, attention[0].weight.detach())
    assert [module.training for module in attention.modules()] == original_modes


@pytest.mark.parametrize("training", [False, True])
def test_nonzero_dropout_keeps_train_eval_semantics(training):
    attention = nn.Sequential(nn.Dropout(1.0)).train(training)
    inputs = torch.ones(4)
    with grug_scoring_attention_context(attention, attention_dropout=1.0):
        output = attention(inputs)
        assert all(module.training == training for module in attention.modules())
    torch.testing.assert_close(output, torch.zeros_like(inputs) if training else inputs)
    assert all(module.training == training for module in attention.modules())
