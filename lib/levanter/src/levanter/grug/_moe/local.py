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
except ImportError:
    # sonic_cute uses the QuACK SM100 (Blackwell) kernel; optional on non-Blackwell GPUs (e.g. H100).
    _moe_mlp_local_sonic_cute = None

_MOE_LOCAL_FNS = {
    "scatter": _moe_mlp_local_scatter,
    "sonic": _moe_mlp_local_sonic,
}
if _moe_mlp_local_sonic_cute is not None:
    _MOE_LOCAL_FNS["sonic_cute"] = _moe_mlp_local_sonic_cute


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
    fn = _MOE_LOCAL_FNS.get(local_key)
    if fn is None:
        raise ValueError(
            f"MoE implementation '{implementation}' is unavailable in this build "
            f"(needs QuACK/SM100); available: {sorted(_MOE_LOCAL_FNS)}"
        )
    return fn(
        x,
        selected_experts,
        combine_weights,
        moe_w13,
        moe_w2,
        activation_fn=activation_fn,
        num_experts=num_experts,
    )
