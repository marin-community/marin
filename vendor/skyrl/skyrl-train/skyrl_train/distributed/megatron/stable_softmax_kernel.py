# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
# Kernel structure from TransformerEngine v2.19 common/triton/cross_entropy.py.

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _softmax_forward(
    X,
    Y,
    Output,
    Stats,
    X_BATCH: tl.constexpr,
    X_SEQ: tl.constexpr,
    X_VOCAB: tl.constexpr,
    SEQ: tl.constexpr,
    VOCAB: tl.constexpr,
    ROWS: tl.constexpr,
    VOCAB_START: tl.constexpr,
    ENTROPY: tl.constexpr,
    SAVE_STATS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_row = X + (row // SEQ) * X_BATCH + (row % SEQ) * X_SEQ
    m = float("-inf")
    s = 0.0
    t = 0.0
    for start in range(0, VOCAB, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offsets * X_VOCAB, offsets < VOCAB, float("-inf")).to(tl.float32)
        b, index = tl.max(x, 0, return_indices=True)
        a = x - b
        e = tl.exp(a)
        others = tl.where((tl.arange(0, BLOCK) != index) & (offsets < VOCAB), e, 0.0)
        sb = tl.sum(others, 0)
        if ENTROPY:
            tb = tl.sum(others * tl.where(others == 0, 0.0, a), 0)
        next_m = tl.maximum(m, b)
        d_old = tl.where(start == 0, 0.0, m - next_m)
        d_block = b - next_m
        old_scale = tl.where(start == 0, 0.0, tl.exp(d_old))
        block_scale = tl.exp(d_block)
        if ENTROPY:
            t = old_scale * (t + d_old * (s + 1.0)) + block_scale * (tb + d_block * (sb + 1.0))
        s = old_scale * s + block_scale * sb + tl.where(b > m, old_scale, block_scale)
        m = next_m
    if ENTROPY:
        output = libdevice.log1p(s) - t / (1.0 + s)
    else:
        y = tl.load(Y + row) - VOCAB_START
        selected = tl.load(x_row + y * X_VOCAB).to(tl.float32)
        output = (selected - m) - libdevice.log1p(s)
    tl.store(Output + row, output)
    if SAVE_STATS:
        tl.store(Stats + row, m)
        tl.store(Stats + ROWS + row, s)
        if ENTROPY:
            tl.store(Stats + 2 * ROWS + row, t)


@triton.jit
def _softmax_backward(
    X,
    Y,
    Stats,
    Upstream,
    Gradient,
    X_BATCH: tl.constexpr,
    X_SEQ: tl.constexpr,
    X_VOCAB: tl.constexpr,
    G_BATCH: tl.constexpr,
    G_SEQ: tl.constexpr,
    G_VOCAB: tl.constexpr,
    SEQ: tl.constexpr,
    VOCAB: tl.constexpr,
    ROWS: tl.constexpr,
    VOCAB_START: tl.constexpr,
    ENTROPY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    x_row = X + (row // SEQ) * X_BATCH + (row % SEQ) * X_SEQ
    grad_row = Gradient + (row // SEQ) * G_BATCH + (row % SEQ) * G_SEQ
    m = tl.load(Stats + row)
    s = tl.load(Stats + ROWS + row)
    weight = tl.load(Upstream + row).to(tl.float32)
    if ENTROPY:
        expected_margin = tl.load(Stats + 2 * ROWS + row) / (1.0 + s)
    else:
        y = tl.load(Y + row) - VOCAB_START
    for start in range(0, VOCAB, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < VOCAB
        x = tl.load(x_row + offsets * X_VOCAB, mask, float("-inf")).to(tl.float32)
        a = x - m
        q = tl.exp(a) / (1.0 + s)
        if ENTROPY:
            gradient = -q * (tl.where(q == 0.0, 0.0, a) - expected_margin)
        else:
            target_gradient = tl.where(a == 0.0, s / (1.0 + s), 1.0 - q)
            gradient = tl.where(offsets == y, target_gradient, -q)
        tl.store(grad_row + offsets * G_VOCAB, gradient * weight, mask)


def softmax_forward(
    logits: torch.Tensor, targets: torch.Tensor | None, vocab_start: int, save_stats: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    entropy = targets is None
    output = torch.empty(logits.shape[:-1], dtype=torch.float32, device=logits.device)
    rows = output.numel()
    stats = (
        torch.empty((3 if entropy else 2, *output.shape), dtype=torch.float32, device=logits.device)
        if save_stats
        else None
    )
    block = min(32768, triton.next_power_of_2(logits.shape[-1]))
    _softmax_forward[(rows,)](
        logits,
        logits if targets is None else targets.contiguous(),
        output,
        output if stats is None else stats,
        logits.stride(0) if logits.ndim == 3 else 0,
        logits.stride(-2),
        logits.stride(-1),
        logits.shape[-2],
        logits.shape[-1],
        rows,
        vocab_start,
        entropy,
        save_stats,
        block,
        num_warps=32 if block >= 32768 else 4,
    )
    return output, stats


def softmax_backward(
    logits: torch.Tensor,
    targets: torch.Tensor | None,
    stats: torch.Tensor,
    grad_output: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    gradient = torch.empty_like(logits)
    block = min(32768, triton.next_power_of_2(logits.shape[-1]))
    _softmax_backward[(grad_output.numel(),)](
        logits,
        logits if targets is None else targets.contiguous(),
        stats,
        grad_output.contiguous(),
        gradient,
        logits.stride(0) if logits.ndim == 3 else 0,
        logits.stride(-2),
        logits.stride(-1),
        gradient.stride(0) if gradient.ndim == 3 else 0,
        gradient.stride(-2),
        gradient.stride(-1),
        logits.shape[-2],
        logits.shape[-1],
        grad_output.numel(),
        vocab_start,
        targets is None,
        block,
        num_warps=32 if block >= 32768 else 4,
    )
    return gradient
