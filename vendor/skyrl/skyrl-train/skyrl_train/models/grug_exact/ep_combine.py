"""The routed expert outputs added in the decode-invariant engine's expert-parallel order.

vLLM's EP rank ``r`` owns experts ``[r * E / R, (r + 1) * E / R)`` and adds the row's slots it owns, in slot order,
into an fp32 zero and rounds once; NCCL's ring reduce-scatter then adds the rank partials in bf16 from rank
``home + 1`` round to ``home``, the EP rank of the engine data-parallel rank that served the row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Hidden columns per program.
HIDDEN_BLOCK = 512


@triton.jit
def _ep_combine_kernel(
    permuted_ptr,
    selected_ptr,
    engine_dp_ranks_ptr,
    pairs_through_ptr,
    output_ptr,
    hidden,
    permuted_stride,
    selected_stride,
    pairs_through_stride,
    output_stride,
    TOP_K: tl.constexpr,
    EP_SIZE: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    in_row = columns < hidden
    rounded = output_ptr.dtype.element_ty
    home = tl.load(engine_dp_ranks_ptr + token) % EP_SIZE
    home = (home + EP_SIZE) % EP_SIZE
    # The rank sum holds values of the output dtype in fp32; ranks without slots add an exact zero.
    total = tl.zeros([BLOCK], dtype=tl.float32)
    for position in tl.static_range(EP_SIZE):
        # Ring position 0 is rank home + 1; the home rank comes last.
        partial = tl.zeros([BLOCK], dtype=tl.float32)
        for slot in tl.static_range(TOP_K):
            expert = tl.load(selected_ptr + token * selected_stride + slot)
            if (expert // EXPERTS_PER_RANK + EP_SIZE - 1 - home) % EP_SIZE == position:
                row = tl.load(pairs_through_ptr + expert * pairs_through_stride + token) - 1
                value = tl.load(permuted_ptr + row.to(tl.int64) * permuted_stride + columns, mask=in_row, other=0.0)
                partial += value.to(tl.float32)
        total = (total + partial.to(rounded).to(tl.float32)).to(rounded).to(tl.float32)
    tl.store(output_ptr + token * output_stride + columns, total.to(rounded), mask=in_row)


def ep_combine(
    permuted: torch.Tensor,
    routing_map: torch.Tensor,
    selected: torch.Tensor,
    engine_dp_ranks: torch.Tensor,
    ep_size: int,
) -> torch.Tensor:
    """Each token's ``[tokens, hidden]`` routed output as vLLM's expert-parallel combine adds it.

    ``permuted`` holds the weighted expert outputs in Megatron's permuted order (by expert, then token);
    ``routing_map`` is the ``[tokens, experts]`` selection, ``selected`` the ``[tokens, top_k]`` experts in the
    engine's slot order and ``engine_dp_ranks`` the engine data-parallel rank that served each token's sequence.
    """
    tokens, experts = routing_map.shape
    top_k = selected.shape[1]
    if experts % ep_size:
        raise ValueError(f"{experts} experts do not split over {ep_size} vLLM EP ranks")
    if permuted.shape[0] != tokens * top_k:
        raise ValueError(f"{permuted.shape[0]} permuted rows for {tokens} tokens of {top_k} slots")
    hidden = permuted.shape[1]
    output = permuted.new_empty(tokens, hidden)
    if tokens == 0:
        return output
    # pairs_through[e, t] counts the selected token-expert pairs up to (e, t) in expert-major order, so slot (t, e)
    # is permuted row pairs_through[e, t] - 1.
    pairs_through = routing_map.t().contiguous().view(-1).cumsum(0, dtype=torch.int32).view(experts, tokens)
    permuted = permuted.contiguous()
    selected = selected.long().contiguous()
    grid = (tokens, triton.cdiv(hidden, HIDDEN_BLOCK))
    _ep_combine_kernel[grid](
        permuted,
        selected,
        engine_dp_ranks.long().contiguous(),
        pairs_through,
        output,
        hidden,
        permuted.stride(0),
        selected.stride(0),
        pairs_through.stride(0),
        output.stride(0),
        TOP_K=top_k,
        EP_SIZE=ep_size,
        EXPERTS_PER_RANK=experts // ep_size,
        BLOCK=HIDDEN_BLOCK,
    )
    return output
