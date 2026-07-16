# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Local Grug MoE backend dispatch."""

from collections.abc import Callable

import jax
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import _LOCAL_MOE_IMPLEMENTATIONS, MoeImplementation
from levanter.grug._moe.scatter import _moe_mlp_local_scatter
from levanter.grug._moe.sonic import _moe_mlp_local_sonic

try:
    from levanter.grug._moe.sonic_cute import _moe_mlp_local_sonic_cute
except ModuleNotFoundError as _e:  # quack-kernels (and its torch dep) are optional
    _sonic_cute_error = _e

    def _moe_mlp_local_sonic_cute(*args, **kwargs):
        raise ModuleNotFoundError(
            f"moe_implementation='sonic_cute' requires quack-kernels and torch: {_sonic_cute_error}"
        ) from _sonic_cute_error

_MOE_LOCAL_FNS = {
    "scatter": _moe_mlp_local_scatter,
    "sonic": _moe_mlp_local_sonic,
    "sonic_cute": _moe_mlp_local_sonic_cute,
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
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
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
