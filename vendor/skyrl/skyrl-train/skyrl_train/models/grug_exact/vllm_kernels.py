"""vLLM's attention, gate GEMM, fused-MoE and log-probability kernels and the engine's expert selection, called on
the trainer's tensors so that each value equals the decode-invariant engine's."""

from __future__ import annotations

import functools
from collections.abc import Sequence

import torch

from skyrl_train.inference_engines.vllm.decode_invariant.fa3 import FLASH_ATTN_VERSION

# ``expert_outputs`` finds each expert's weights by their distance from the lowest-addressed expert weight, in units
# of this many elements. Triton then treats that stride as divisible by 16, as it does for vLLM's stacked weights.
EXPERT_OFFSET_ELEMENTS = 16
# The fused-MoE tile of ``expert_outputs``. The engine picks vLLM's default tile for each step, so this tile gives the
# engine's values where every tile gives a slot the same values, as on H100 at Grug's expert shapes.
EXPERT_TILE = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "SPLIT_K": 1}


def _fa3_causal_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> torch.Tensor:
    """FA3's causal varlen forward into ``out`` at one split, through the op ``_vllm_fa3_C.fwd``, which takes
    ``cu_seqlens_k`` and ``seqused_k`` together."""
    torch.ops._vllm_fa3_C.fwd(
        query,
        key,
        value,
        None,  # k_new
        None,  # v_new
        None,  # q_v
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # cu_seqlens_k_new
        None,  # seqused_q
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        None,  # page_table
        None,  # kv_batch_idx
        None,  # leftpad_k
        None,  # rotary_cos
        None,  # rotary_sin
        None,  # seqlens_rotary
        None,  # q_descale
        None,  # k_descale
        None,  # v_descale
        softmax_scale,
        True,  # is_causal
        -1,  # window_size_left
        -1,  # window_size_right
        0.0,  # softcap
        True,  # is_rotary_interleaved
        None,  # scheduler_metadata
        1,  # num_splits
        None,  # pack_gqa
        0,  # sm_margin
        None,  # s_aux
        1,  # cp_world_size
        0,  # cp_rank
        None,  # cp_tot_seqused_k
    )
    return out


@functools.lru_cache(maxsize=64)
def _uniform_query_start(rows: int, requests: int, device: torch.device) -> torch.Tensor:
    return torch.arange(0, (requests + 1) * rows, rows, dtype=torch.int32, device=device)


def _fa3_forward(query, key, value, *, rows: int, requests: int, window: int | None, scale: float, out=None):
    """FA3 over ``requests`` sequences of ``rows`` rows, each one varlen request from position 0, at one split."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func  # noqa: PLC0415 - vLLM is a GPU-only extra

    query_start = _uniform_query_start(rows, requests, query.device)
    return flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        max_seqlen_q=rows,
        cu_seqlens_q=query_start,
        max_seqlen_k=rows,
        cu_seqlens_k=query_start,
        softmax_scale=scale,
        causal=True,
        window_size=None if window is None else [window - 1, 0],
        fa_version=FLASH_ATTN_VERSION,
        num_splits=1,
        out=out,
    )


@functools.lru_cache(maxsize=16)
def _window_lengths(count: int, window: int, device: torch.device) -> torch.Tensor:
    return torch.full((count,), window, dtype=torch.int32, device=device)


def _fa3_window_rows_forward(query, key, value, *, rows: int, requests: int, window: int, scale: float):
    """Sliding-window FA3 over ``requests`` sequences of ``rows`` rows, each row computed as a decode-invariant engine
    computes it: the rows before the window's end as one request, each later row as a request over its window."""
    device = query.device
    output = torch.empty_like(query)
    later = rows - window
    for index in range(requests):
        first = index * rows
        _fa3_forward(
            query[first : first + window],
            key[first : first + window],
            value[first : first + window],
            rows=window,
            requests=1,
            window=window,
            scale=scale,
            out=output[first : first + window],
        )
        # Row p's keys are rows p - window + 1 .. p of the same sequence.
        window_starts = torch.arange(first + 1, first + later + 2, dtype=torch.int32, device=device)
        rows_out = slice(first + window, first + rows)
        _fa3_causal_varlen(
            query[rows_out],
            key,
            value,
            output[rows_out],
            cu_seqlens_q=_uniform_query_start(1, later, device),  # one row per request
            cu_seqlens_k=window_starts,  # each request's first key
            seqused_k=_window_lengths(later, window, device),
            max_seqlen_q=2,  # at 1, FA3 runs the requests non-causal, which blocks the keys differently
            max_seqlen_k=window,
            softmax_scale=scale,
        )
    return output


def fa3_attention_sbhd(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, window: int | None, scale: float
) -> torch.Tensor:
    """vLLM's FA3 forward on Megatron's ``[S, B, heads, dim]`` tensors, every row computed as a decode-invariant engine
    computes it; returns ``[S, B, heads * dim]``.

    Each sequence is one varlen request, so right padding changes no valid row.
    """
    sequence, batch, heads, head_dim = query.shape
    flat = [t.transpose(0, 1).reshape(batch * sequence, *t.shape[2:]).contiguous() for t in (query, key, value)]
    if window is not None and sequence > window:
        output = _fa3_window_rows_forward(*flat, rows=sequence, requests=batch, window=window, scale=scale)
    else:
        output = _fa3_forward(*flat, rows=sequence, requests=batch, window=window, scale=scale)
    return output.view(batch, sequence, heads * head_dim).transpose(0, 1).contiguous()


def topk_experts(biased_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """The ``[rows, top_k]`` experts vLLM's Grug router selects from the fp32 biased logits, in its slot order: the
    first ``top_k`` of ``torch.topk`` of ``top_k + 1``, which breaks exact ties as the engine does."""
    return torch.topk(biased_logits, k=top_k + 1, dim=-1).indices[:, :top_k]


def row_invariant_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """vLLM GrugMoE's router and head-gate GEMM: ``x @ weight.T`` in fp32, each row's result depending on that row
    and ``weight`` alone."""
    from vllm.model_executor.models.grugmoe import grug_row_invariant_gemm  # noqa: PLC0415 - GPU-only extra

    return grug_row_invariant_gemm(x, weight)


def token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Model runner V2's log-probability (``compute_token_logprobs``) of ``token_ids[i]`` under row ``i`` of the
    ``[rows, vocab]`` ``logits``; each row's value depends on that row alone."""
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs  # noqa: PLC0415 - vLLM is a GPU-only extra

    if logits.ndim != 2 or logits.stride(-1) != 1:
        raise ValueError(f"vLLM's log-probability kernel reads rows of unit vocabulary stride, got {logits.stride()}")
    return compute_token_logprobs(logits, token_ids.reshape(-1, 1))[:, 0]


@functools.lru_cache(maxsize=256)
def _offset_table(offsets: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def _expert_offset_table(weights: Sequence[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The lowest-addressed expert weight and, on ``device``, each weight's address above it in units of
    ``EXPERT_OFFSET_ELEMENTS`` elements, copied to the device once per distinct table."""
    lowest = min(weights, key=lambda weight: weight.data_ptr())
    if any(
        weight.shape != lowest.shape or weight.dtype != lowest.dtype or not weight.is_contiguous() for weight in weights
    ):
        raise ValueError("expert_outputs needs contiguous expert weights of one shape and dtype")
    unit = EXPERT_OFFSET_ELEMENTS * lowest.element_size()
    distances = [weight.data_ptr() - lowest.data_ptr() for weight in weights]
    if any(distance % unit for distance in distances):
        raise ValueError(f"expert_outputs needs expert weights at whole {unit}-byte steps above the lowest one")
    return lowest, _offset_table(tuple(distance // unit for distance in distances), device)


def _fused_moe_gemm(
    inputs: torch.Tensor,
    weights: Sequence[torch.Tensor],
    output: torch.Tensor,
    route_weights: torch.Tensor | None,
    sorted_rows: torch.Tensor,
    block_experts: torch.Tensor,
    padded_rows: torch.Tensor,
    config: dict,
) -> None:
    """vLLM's ``invoke_fused_moe_triton_kernel`` on per-expert weight tensors (see ``expert_outputs``)."""
    from vllm.model_executor.layers.fused_moe.fused_moe import (  # noqa: PLC0415 - vLLM is a GPU-only extra
        invoke_fused_moe_triton_kernel,
    )
    from vllm.triton_utils import tl  # noqa: PLC0415 - vLLM is a GPU-only extra

    base, table = _expert_offset_table(weights, inputs.device)
    rows, columns = base.shape
    # Blocks past the padded row count keep whatever moe_align_block_size left there; the kernel returns before
    # reading their expert, so any in-range value serves.
    block_offsets = table[block_experts.long().clamp(0, len(weights) - 1)]
    stacked = torch.as_strided(base.detach(), (1, rows, columns), (EXPERT_OFFSET_ELEMENTS, columns, 1))
    invoke_fused_moe_triton_kernel(
        inputs,
        stacked,
        output,
        None,
        None,
        route_weights,
        sorted_rows,
        block_offsets,
        padded_rows,
        route_weights is not None,
        1,
        config,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
    )


def expert_outputs(
    hidden: torch.Tensor,
    tokens_per_expert: Sequence[int],
    probs: torch.Tensor,
    fc1_weights: Sequence[torch.Tensor],
    fc2_weights: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Each dispatched row's routed-expert output, weighted by its route weight, as the decode-invariant engine's
    fused-MoE kernels compute one token-expert slot.

    ``hidden`` ``[rows, hidden]`` holds each local expert's rows in turn (``tokens_per_expert`` of them),
    ``probs`` the rows' fp32 route weights, and ``fc1_weights`` / ``fc2_weights`` each expert's
    ``[2 * ffn, hidden]`` gate-and-up and ``[hidden, ffn]`` down projection, contiguous, of one shape and dtype, at
    whole ``EXPERT_OFFSET_ELEMENTS``-element steps above the lowest-addressed one. A slot's value depends only on its
    row, its expert and its route weight.
    """
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (  # noqa: PLC0415 - GPU-only extra
        moe_align_block_size,
    )

    rows, width = hidden.shape
    experts = len(fc1_weights)
    gate_up_width = fc1_weights[0].shape[0]
    output = hidden.new_empty(rows, width)
    if rows == 0:
        return output
    config = EXPERT_TILE
    counts = torch.as_tensor(tokens_per_expert, dtype=torch.long, device="cpu")
    expert_of_row = torch.repeat_interleave(torch.arange(experts, dtype=torch.int32), counts)
    expert_of_row = expert_of_row.pin_memory().to(hidden.device, non_blocking=True).view(rows, 1)
    sorted_rows, block_experts, padded_rows = moe_align_block_size(expert_of_row, config["BLOCK_SIZE_M"], experts)
    gate_up = hidden.new_empty(rows, 1, gate_up_width)
    _fused_moe_gemm(hidden.contiguous(), fc1_weights, gate_up, None, sorted_rows, block_experts, padded_rows, config)
    activation = hidden.new_empty(rows, gate_up_width // 2)
    torch.ops._C.silu_and_mul(activation, gate_up.view(rows, gate_up_width))
    down = output.view(rows, 1, width)
    _fused_moe_gemm(activation, fc2_weights, down, probs.view(rows, 1), sorted_rows, block_experts, padded_rows, config)
    return output
