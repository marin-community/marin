# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Local Grug MoE backend dispatch."""

from collections.abc import Callable

import jax
from jaxtyping import Array, Float, Int

from levanter.grug.moe_common import _LOCAL_MOE_IMPLEMENTATIONS, MoeImplementation
from levanter.grug.moe_scatter import _moe_mlp_local_scatter
from levanter.grug.moe_sonic_xla import (
    _moe_mlp_local_sonic_xla,
    _moe_mlp_local_sonic_xla_interleaved_w13,
    _moe_mlp_local_sonic_xla_interleaved_w13_custom_vjp_down,
)

_MOE_LOCAL_FNS = {
    "scatter": _moe_mlp_local_scatter,
    "sonic_xla": _moe_mlp_local_sonic_xla,
    "sonic_xla_interleaved_w13": _moe_mlp_local_sonic_xla_interleaved_w13,
    "sonic_xla_interleaved_w13_custom_vjp_down": _moe_mlp_local_sonic_xla_interleaved_w13_custom_vjp_down,
}


def _moe_mlp_local(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    moe_w13: Float[Array, "E D I2"],
    moe_w2: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    implementation: MoeImplementation,
) -> tuple[Float[Array, "T D"], Int[Array, ""]]:
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
