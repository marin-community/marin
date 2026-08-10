# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""MoK-style aggregate-scheduled expert-parallel MLP."""

import jax
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.pallas.ops.gpu import blackwell_matmul_mgpu
from jaxtyping import Array, Float, Int

from levanter.grug._moe.mok_dispatch import gather_routed_tokens, scatter_routed_tokens
from levanter.grug._moe.mok_expert_gemm import TuningConfig, ragged_dot_kernel
from levanter.grug._moe.mok_megakernel import dispatch_gate_up


def default_gemm_config() -> TuningConfig:
    return TuningConfig(
        tile_m=128,
        tile_n=128,
        tile_k=64,
        max_concurrent_steps=6,
        collective=True,
        grid_tile_width=1,
        grid_minor_dim=blackwell_matmul_mgpu.MatmulDimension.N,
        epilogue_tile_n=32,
    )


def _ragged_dot(a, b, group_sizes, config):
    return ragged_dot_kernel(a, b, group_sizes, config, plgpu.LoweringSemantics.Lane)


def aggregate_expert_mlp_forward(
    x: Float[Array, "T H"],
    router_weights: Float[Array, "T K"],
    w_gate: Float[Array, "E H I"],
    w_up: Float[Array, "E H I"],
    w_down: Float[Array, "E I H"],
    shared: tuple[Float[Array, "H I"], Float[Array, "H I"], Float[Array, "I H"]],
    peer_rank: Int[Array, "C"],
    peer_token_idx: Int[Array, "C"],
    num_routed_tokens: Int[Array, ""],
    tokens_per_expert: Int[Array, "E"],
    *,
    axis_name: str,
    block_rows: int = 128,
    combine_block_rows: int = 16,
    num_comm_sms: int | None = None,
    minibatch_size: int = 4096,
    gemm_config: TuningConfig | None = None,
) -> Float[Array, "T H"]:
    """Runs aggregate dispatch, grouped expert GEMMs, and aggregate combine."""
    config = gemm_config or default_gemm_config()
    num_local_tokens, topk = router_weights.shape
    shared_gate, shared_up, shared_down = shared
    fuse_shared = num_comm_sms is not None and num_local_tokens % (config.tile_m * 2) == 0
    if num_comm_sms is None:
        x_recv = gather_routed_tokens(
            x,
            peer_rank,
            peer_token_idx,
            num_routed_tokens,
            axis_name=axis_name,
            row_divisor=topk,
            block_rows=block_rows,
        )
        gate = _ragged_dot(x_recv, w_gate, tokens_per_expert, config)
        up = _ragged_dot(x_recv, w_up, tokens_per_expert, config)
    else:
        x_recv, gate, up, hidden, y_recv, combine_recv, _, _, _, shared_y_fused = dispatch_gate_up(
            x,
            w_gate,
            w_up,
            peer_rank,
            peer_token_idx,
            num_routed_tokens,
            tokens_per_expert,
            axis_name=axis_name,
            topk=topk,
            num_comm_sms=num_comm_sms,
            minibatch_size=minibatch_size,
            gemm_config=config,
            w_down=w_down,
            shared=shared if fuse_shared else None,
        )
    if num_comm_sms is None:
        hidden = jax.nn.silu(gate) * up
        y_recv = _ragged_dot(hidden, w_down, tokens_per_expert, config)
    if num_comm_sms is None:
        contributions = scatter_routed_tokens(
            y_recv,
            peer_rank,
            peer_token_idx,
            num_routed_tokens,
            axis_name=axis_name,
            num_slots=num_local_tokens * topk,
            block_rows=combine_block_rows,
        )
    else:
        contributions = combine_recv[: num_local_tokens * topk]

    routed = contributions.reshape(num_local_tokens, topk, -1)
    routed = (routed * router_weights[:, :, None].astype(routed.dtype)).sum(axis=1)
    if fuse_shared:
        shared_y = shared_y_fused
    else:
        shared_y = (jax.nn.silu(x @ shared_gate) * (x @ shared_up)) @ shared_down
    return routed + shared_y
