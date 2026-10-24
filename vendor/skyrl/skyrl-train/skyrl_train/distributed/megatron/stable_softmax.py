"""Token log probabilities and entropy from vocabulary-parallel logits."""

from typing import Any

import torch
import torch.distributed as dist


def _reduce(value: torch.Tensor, group: dist.ProcessGroup | None, op: dist.ReduceOp = dist.ReduceOp.SUM) -> None:
    if group is not None and dist.get_world_size(group) > 1:
        dist.all_reduce(value, op=op, group=group)


def _statistics(
    logits: torch.Tensor, group: dist.ProcessGroup | None, entropy: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x = logits.float()
    maximum, index = x.max(dim=-1, keepdim=True)
    maximum = maximum.clone()
    _reduce(maximum, group, dist.ReduceOp.MAX)
    rank = 0 if group is None else dist.get_rank(group)
    owner = torch.where(x.gather(-1, index) == maximum, rank, torch.iinfo(torch.int64).max)
    _reduce(owner, group, dist.ReduceOp.MIN)
    margins = x - maximum
    probabilities = margins.exp()
    probabilities.scatter_(-1, index, torch.where(owner == rank, 0.0, probabilities.gather(-1, index)))
    tail = probabilities.sum(dim=-1, keepdim=True)
    _reduce(tail, group)
    moment = None
    if entropy:
        moment = (probabilities * torch.where(probabilities == 0, 0.0, margins)).sum(dim=-1, keepdim=True)
        _reduce(moment, group)
    return maximum.squeeze(-1), tail.squeeze(-1), None if moment is None else moment.squeeze(-1)


def _torch_forward(
    logits: torch.Tensor,
    targets: torch.Tensor | None,
    vocab_start: int,
    vocab_end: int,
    group: dist.ProcessGroup | None,
    chunk_size: int,
    save_stats: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    entropy = targets is None
    output = torch.empty(logits.shape[:-1], dtype=torch.float32, device=logits.device)
    stats = (
        torch.empty((3 if entropy else 2, *output.shape), dtype=torch.float32, device=logits.device)
        if save_stats
        else None
    )
    for start in range(0, logits.shape[-2], chunk_size):
        stop = start + chunk_size
        x = logits[..., start:stop, :]
        maximum, tail, moment = _statistics(x, group, entropy)
        if entropy:
            assert moment is not None
            output[..., start:stop] = torch.log1p(tail) - moment / (1 + tail)
        else:
            assert targets is not None
            target = targets[..., start:stop]
            owned = (target >= vocab_start) & (target < vocab_end)
            index = (target - vocab_start).masked_fill(~owned, 0)
            selected = x.float().gather(-1, index.unsqueeze(-1)).squeeze(-1) - maximum
            selected.masked_fill_(~owned, 0)
            _reduce(selected, group)
            output[..., start:stop] = selected - torch.log1p(tail)
        if stats is not None:
            stats[0, ..., start:stop] = maximum
            stats[1, ..., start:stop] = tail
            if entropy:
                stats[2, ..., start:stop] = moment
    return output, stats


class _StableSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        targets: torch.Tensor | None,
        vocab_start: int,
        vocab_end: int,
        group: dist.ProcessGroup | None,
        chunk_size: int,
        save_stats: bool,
    ) -> torch.Tensor:
        cuda = (
            logits.is_cuda
            and logits.ndim in (2, 3)
            and logits.dtype in (torch.bfloat16, torch.float32)
            and (group is None or dist.get_world_size(group) == 1)
        )
        if cuda:
            from skyrl_train.distributed.megatron.stable_softmax_kernel import softmax_forward

            output, stats = softmax_forward(logits, targets, vocab_start, save_stats)
        else:
            output, stats = _torch_forward(logits, targets, vocab_start, vocab_end, group, chunk_size, save_stats)
        if save_stats:
            assert stats is not None
            ctx.save_for_backward(logits, stats, *(() if targets is None else (targets,)))
            ctx.entropy = targets is None
            ctx.vocab_start = vocab_start
            ctx.vocab_end = vocab_end
            ctx.chunk_size = chunk_size
            ctx.cuda = cuda
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        logits, stats, *targets = ctx.saved_tensors
        target = None if ctx.entropy else targets[0]
        if ctx.cuda:
            from skyrl_train.distributed.megatron.stable_softmax_kernel import softmax_backward

            gradient = softmax_backward(logits, target, stats, grad_output, ctx.vocab_start)
        else:
            gradient = torch.empty_like(logits)
            for start in range(0, logits.shape[-2], ctx.chunk_size):
                stop = start + ctx.chunk_size
                margins = logits[..., start:stop, :].float() - stats[0, ..., start:stop].unsqueeze(-1)
                tail = stats[1, ..., start:stop].unsqueeze(-1)
                probabilities = margins.exp() / (1 + tail)
                if ctx.entropy:
                    moment = stats[2, ..., start:stop].unsqueeze(-1)
                    grad = -probabilities * (torch.where(probabilities == 0, 0.0, margins) - moment / (1 + tail))
                else:
                    assert target is not None
                    chosen = target[..., start:stop]
                    owned = (chosen >= ctx.vocab_start) & (chosen < ctx.vocab_end)
                    index = (chosen - ctx.vocab_start).masked_fill(~owned, 0).unsqueeze(-1)
                    chosen_logprob = margins.gather(-1, index) - torch.log1p(tail)
                    grad = -probabilities
                    grad.scatter_(
                        -1,
                        index,
                        torch.where(owned.unsqueeze(-1), -torch.expm1(chosen_logprob), grad.gather(-1, index)),
                    )
                gradient[..., start:stop, :] = grad * grad_output[..., start:stop].unsqueeze(-1)
        return gradient, None, None, None, None, None, None


def token_logprobs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    group: dist.ProcessGroup | None,
    inference_only: bool = False,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Return fp32 log probabilities for targets in the global vocabulary."""
    with torch.set_grad_enabled(torch.is_grad_enabled() and not inference_only):
        return _StableSoftmax.apply(
            logits,
            targets,
            vocab_start,
            vocab_end,
            group,
            chunk_size or logits.shape[-2],
            torch.is_grad_enabled() and logits.requires_grad,
        )


def token_entropy(logits: torch.Tensor, group: dist.ProcessGroup | None, chunk_size: int | None = None) -> torch.Tensor:
    """Return fp32 entropy over the global vocabulary for each token."""
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return _StableSoftmax.apply(
        logits,
        None,
        0,
        logits.shape[-1],
        group,
        chunk_size or logits.shape[-2],
        torch.is_grad_enabled() and logits.requires_grad,
    )
