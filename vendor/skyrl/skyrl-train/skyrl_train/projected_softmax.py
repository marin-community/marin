"""Token statistics with a recomputed, token-chunked vocabulary projection."""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from skyrl_train.stable_softmax import token_entropy, token_logprobs

Projection = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _statistics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logprobs = token_logprobs(logits.unsqueeze(0), targets.unsqueeze(0), 0, logits.shape[-1], None)
    entropy = token_entropy(logits.unsqueeze(0), None)
    return logprobs.squeeze(0), entropy.squeeze(0)


class _ProjectedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        project: Projection,
        chunk_size: int,
        temperature: float,
        entropy_gradient: bool,
        fused_weight_gradient: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(hidden, weight, targets)
        ctx.project = project
        ctx.chunk_size = chunk_size
        ctx.temperature = temperature
        ctx.entropy_gradient = entropy_gradient
        ctx.fused_weight_gradient = fused_weight_gradient
        output = torch.empty((hidden.shape[0], 2), device=hidden.device, dtype=torch.float32)
        for start in range(0, hidden.shape[0], chunk_size):
            stop = start + chunk_size
            logits = project(hidden[start:stop], weight)
            logits.div_(temperature)
            logprobs, entropy = _statistics(logits, targets[start:stop])
            output[start:stop, 0] = logprobs
            output[start:stop, 1] = entropy
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        hidden, weight, targets = ctx.saved_tensors
        hidden_gradient = torch.empty_like(hidden)
        # Megatron/TE's fused projection accumulates directly into the existing fp32 main_grad.
        # For ordinary torch layers, accumulate chunk wgrads in fp32 before casting once.
        weight_gradient = None if ctx.fused_weight_gradient else torch.zeros_like(weight, dtype=torch.float32)
        dummy_gradient = None
        for start in range(0, hidden.shape[0], ctx.chunk_size):
            stop = start + ctx.chunk_size
            with torch.enable_grad():
                chunk = hidden[start:stop].detach().requires_grad_()
                logits = ctx.project(chunk, weight)
                logits.div_(ctx.temperature)
                lp = token_logprobs(logits.unsqueeze(0), targets[start:stop].unsqueeze(0), 0, weight.shape[0], None)
                outputs = (lp.squeeze(0),)
                gradients = (grad_output[start:stop, 0],)
                if ctx.entropy_gradient:
                    outputs += (token_entropy(logits.unsqueeze(0), None).squeeze(0),)
                    gradients += (grad_output[start:stop, 1],)
                dh, dw = torch.autograd.grad(outputs, (chunk, weight), gradients)
            hidden_gradient[start:stop] = dh
            if ctx.fused_weight_gradient:
                # Return TE's dummy gradient once, so Megatron's DDP hook runs once after all
                # chunks. TE sets grad_added_to_main_grad and preserves native accumulation.
                dummy_gradient = dw
            else:
                assert weight_gradient is not None
                weight_gradient.add_(dw)
        dw = dummy_gradient if ctx.fused_weight_gradient else weight_gradient.to(weight.dtype)
        return hidden_gradient, dw, None, None, None, None, None, None


def projected_token_statistics(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int,
    temperature: float = 1.0,
    entropy_gradient: bool = True,
    project: Projection = F.linear,
    fused_weight_gradient: bool = False,
) -> torch.Tensor:
    """Return [tokens, 2] logprobs/entropy without a full token-by-vocabulary tensor.

    ``project`` must be token-local and return model-dtype logits. The Megatron adapter supplies
    its native output layer, including fp32 main-gradient accumulation. Tensor/sequence-parallel
    projections require a separate adapter; this primitive does not insert collectives.
    """
    if hidden.ndim != 2 or targets.shape != hidden.shape[:1]:
        raise ValueError("Projection requires [tokens, hidden] states and one target per token")
    if chunk_size <= 0 or temperature <= 0:
        raise ValueError("Projection chunk size and temperature must be positive")
    return _ProjectedSoftmax.apply(
        hidden, weight, targets, project, chunk_size, temperature, entropy_gradient, fused_weight_gradient
    )
