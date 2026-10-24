"""Tensors that hold a kernel's output values and take their gradient from the trainer's own computation of the
same values."""

from collections.abc import Callable

import torch


class _ValueWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Autograd wraps the returned tensor in a view whose backward is this function's; nothing is copied.
        return value

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return None, grad


def kernel_value(value: torch.Tensor, reference: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Return ``value`` with its bits unchanged. With gradients enabled, its gradient is the gradient of
    ``reference()``, which runs only then."""
    if not torch.is_grad_enabled():
        return value
    return _ValueWithGradient.apply(value, reference())
