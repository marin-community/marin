"""Projection-chunking value, gradient and allocation contracts."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from skyrl_train.projected_softmax import projected_token_statistics


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("entropy_gradient", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 64])
def test_projected_statistics_match_full_projection(dtype, entropy_gradient, chunk_size):
    temperature = 0.8
    generator = torch.Generator().manual_seed(42)
    hidden = torch.randn(19, 16, generator=generator, dtype=dtype).requires_grad_()
    weight = (torch.randn(37, 16, generator=generator, dtype=dtype) / 4).requires_grad_()
    targets = torch.randint(37, (19,), generator=generator)
    coefficients = torch.randn(19, 2, generator=generator) / 19
    if not entropy_gradient:
        coefficients[:, 1] = 0
    actual = projected_token_statistics(
        hidden, weight, targets, chunk_size=chunk_size, temperature=temperature, entropy_gradient=entropy_gradient
    )
    actual_grads = torch.autograd.grad((actual * coefficients).sum(), (hidden, weight))

    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    logits = F.linear(ref_hidden, ref_weight).div_(temperature).float()
    log_softmax = logits.log_softmax(-1)
    reference = torch.stack(
        (log_softmax.gather(1, targets[:, None]).squeeze(1), -(log_softmax.exp() * log_softmax).sum(-1)), dim=1
    )
    reference_grads = torch.autograd.grad((reference * coefficients).sum(), (ref_hidden, ref_weight))
    tolerance = 2e-6 if dtype == torch.float32 else torch.finfo(dtype).eps
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=tolerance, atol=tolerance)
        relative_error = (actual_grad.float() - reference_grad.float()).norm() / reference_grad.float().norm()
        assert relative_error < tolerance


class _VocabularyAllocation(TorchDispatchMode):
    def __init__(self, vocabulary: int):
        super().__init__()
        self.vocabulary = vocabulary
        self.token_extents = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in result if isinstance(result, tuple) else (result,):
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[-1] == self.vocabulary:
                self.token_extents.append(value.numel() // self.vocabulary)
        return result


def test_projection_backward_never_materializes_full_vocabulary_sequence():
    hidden = torch.randn(101, 13, requires_grad=True)
    weight = torch.randn(43, 13, requires_grad=True)
    targets = torch.arange(101) % 43
    allocations = _VocabularyAllocation(43)
    with allocations:
        result = projected_token_statistics(hidden, weight, targets, chunk_size=7)
        (result[:, 0].sum() + 0.1 * result[:, 1].sum()).backward()
    assert result.shape == (101, 2)
    # Weight-gradient GEMMs also expose [hidden, vocabulary] views. Both that fixed
    # model dimension and the token chunk must stay independent of the 101-token input.
    assert allocations.token_extents and max(allocations.token_extents) <= 13
    assert torch.isfinite(hidden.grad).all() and torch.isfinite(weight.grad).all()


class _AccumulatingProjection(torch.autograd.Function):
    """CPU boundary substitute for native Megatron's fused fp32 main-gradient protocol."""

    @staticmethod
    def forward(ctx, hidden, weight):
        ctx.save_for_backward(hidden, weight)
        return F.linear(hidden, weight)

    @staticmethod
    def backward(ctx, gradient):
        hidden, weight = ctx.saved_tensors
        weight.main_grad.add_(gradient.float().T @ hidden.float())
        weight.grad_added_to_main_grad = True
        return gradient @ weight, torch.zeros_like(weight)


def test_chunked_projection_preserves_fp32_main_grad_without_double_counting():
    torch.manual_seed(17)
    initial_main_grad = 0.25
    hidden = torch.randn(17, 13, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(23, 13))
    weight.main_grad = torch.full_like(weight, initial_main_grad)
    weight.grad_added_to_main_grad = False
    targets = torch.arange(17) % 23
    coefficients = torch.randn(17, 2)

    def finalize(parameter):
        if not parameter.grad_added_to_main_grad:
            parameter.main_grad.add_(parameter.grad)
        parameter.grad = None

    weight.register_post_accumulate_grad_hook(finalize)
    actual = projected_token_statistics(
        hidden, weight, targets, chunk_size=5, project=_AccumulatingProjection.apply, fused_weight_gradient=True
    )
    (actual * coefficients).sum().backward()

    logits = F.linear(hidden.detach(), weight.detach()).requires_grad_()
    lp = logits.log_softmax(-1)
    reference = torch.stack((lp.gather(1, targets[:, None]).squeeze(1), -(lp.exp() * lp).sum(-1)), -1)
    grad_logits = torch.autograd.grad((reference * coefficients).sum(), logits)[0]
    expected = initial_main_grad + grad_logits.T @ hidden.detach()
    torch.testing.assert_close(weight.main_grad, expected, rtol=2e-6, atol=2e-6)
    assert weight.grad is None
