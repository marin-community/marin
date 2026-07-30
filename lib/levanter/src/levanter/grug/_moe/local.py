# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Local Grug MoE backend dispatch."""

from collections.abc import Callable

import jax
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import _LOCAL_MOE_IMPLEMENTATIONS, MoeImplementation
from levanter.grug._moe.scatter import _moe_mlp_local_scatter
from levanter.grug._moe.sonic import _moe_mlp_local_sonic

_MOE_LOCAL_FNS = {
    "scatter": _moe_mlp_local_scatter,
    "sonic": _moe_mlp_local_sonic,
}


def _moe_mlp_local(
    x: Float[Array, "T H"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E H I2"],
    moe_w2: Float[Array, "E I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    implementation: MoeImplementation,
    expert_chunks: int = 1,
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
    if implementation == "sonic_cute":
        if activation_fn is not jax.nn.silu:
            raise ValueError("sonic_cute requires SiLU because its QuACK kernel fuses SwiGLU")
        # QuACK and CUTLASS DSL are installed only with the CUDA 13 GPU extra.
        from levanter.grug._moe.sonic_cute import (  # noqa: PLC0415
            _moe_mlp_local_sonic_cute,
            _moe_mlp_local_sonic_cute_chunked,
        )

        if expert_chunks > 1:
            if num_experts % expert_chunks != 0:
                raise ValueError(f"num_experts={num_experts} must be divisible by expert_chunks={expert_chunks}")
            experts_per_chunk = num_experts // expert_chunks
            return _moe_mlp_local_sonic_cute_chunked(
                x,
                selected_experts,
                combine_weights,
                moe_w13,
                moe_w2,
                num_experts=num_experts,
                chunk_sizes=(experts_per_chunk,) * expert_chunks,
                data_axis_name="data",
            )
        return _moe_mlp_local_sonic_cute(
            x,
            selected_experts,
            combine_weights,
            moe_w13,
            moe_w2,
            num_experts=num_experts,
        )
    if expert_chunks != 1:
        raise ValueError(f"expert_chunks requires implementation='sonic_cute', got {implementation!r}")
    local_key = implementation if implementation in _LOCAL_MOE_IMPLEMENTATIONS else "scatter"
    return _MOE_LOCAL_FNS[local_key](
        x,
        selected_experts,
        combine_weights,
        moe_w13,
        moe_w2,
        activation_fn=activation_fn,
        num_experts=num_experts,
    )
