# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public API for the MoE dispatch-up subkernel."""

from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias, cast

import jax

from levanter.kernels.pallas.moe_dispatch_up.errors import MosaicGpuUnsupportedError
from levanter.kernels.pallas.moe_dispatch_up.mosaic_gpu import dispatch_prepacked_moe_dispatch_up_mosaic_gpu
from levanter.kernels.pallas.moe_dispatch_up.reference import (
    MoeDispatchUpLayout,
    MoeDispatchUpPrepackedSend,
    compute_moe_up_from_layout_reference,
    compute_moe_up_from_layout_ragged_dot,
    dispatch_prepacked_moe_dispatch_up_reference,
    moe_dispatch_up_reference_bwd,
    moe_dispatch_up_layout_reference as _moe_dispatch_up_layout_reference,
    prepack_moe_dispatch_up_reference,
)

Implementation: TypeAlias = Literal["reference", "mosaic_gpu"]
W13Implementation: TypeAlias = Literal["reference", "ragged_dot"]
RaggedDotImplementation: TypeAlias = Literal["auto", "megablox", "triton", "xla"]


def _mosaic_gpu_dispatch(
    prepacked: MoeDispatchUpPrepackedSend,
    *,
    recv_capacity: int | None,
) -> MoeDispatchUpLayout:
    return dispatch_prepacked_moe_dispatch_up_mosaic_gpu(prepacked, recv_capacity=recv_capacity)


DISPATCH_IMPLEMENTATIONS: dict[str, Callable[..., MoeDispatchUpLayout]] = {
    "reference": dispatch_prepacked_moe_dispatch_up_reference,
    "mosaic_gpu": _mosaic_gpu_dispatch,
}
_DEFAULT_DISPATCH_IMPLEMENTATION: tuple[Implementation, ...] = ("reference",)


def prepack_moe_dispatch_up(
    x_by_rank: jax.Array,
    expert_ids_by_rank: jax.Array,
    router_weights_by_rank: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float = 1.25,
    recv_capacity: int | None = None,
    send_capacity: int | None = None,
) -> MoeDispatchUpPrepackedSend:
    return prepack_moe_dispatch_up_reference(
        x_by_rank,
        expert_ids_by_rank,
        router_weights_by_rank,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        recv_capacity=recv_capacity,
        send_capacity=send_capacity,
    )


def moe_dispatch_up_layout(
    prepacked: MoeDispatchUpPrepackedSend,
    *,
    recv_capacity: int | None = None,
    implementation: Implementation | Sequence[Implementation | Callable[..., MoeDispatchUpLayout]] | None = None,
) -> MoeDispatchUpLayout:
    if implementation is None:
        implementations: Sequence[Implementation | Callable[..., MoeDispatchUpLayout]] = (
            _DEFAULT_DISPATCH_IMPLEMENTATION
        )
    elif isinstance(implementation, Sequence) and not isinstance(implementation, (str, bytes)):
        implementations = cast(Sequence[Implementation | Callable[..., MoeDispatchUpLayout]], implementation)
    else:
        implementations = (cast(Implementation, implementation),)

    errors: list[Exception] = []
    explicit_single = implementation is not None and len(implementations) == 1
    for impl in implementations:
        if callable(impl):
            try:
                return impl(prepacked, recv_capacity=recv_capacity)
            except MosaicGpuUnsupportedError as exc:
                if explicit_single:
                    raise
                errors.append(exc)
                continue
        else:
            fn = DISPATCH_IMPLEMENTATIONS.get(impl)
            if fn is None:
                raise ValueError(f"Unsupported MoE dispatch-up Mosaic GPU dispatch implementation: {impl!r}")
            try:
                return fn(prepacked, recv_capacity=recv_capacity)
            except MosaicGpuUnsupportedError as exc:
                if explicit_single:
                    raise
                errors.append(exc)
                continue
    if errors:
        raise ExceptionGroup("all MoE dispatch-up Mosaic GPU dispatch implementations failed", errors)
    raise AssertionError("no MoE dispatch-up Mosaic GPU dispatch implementations were attempted")


def moe_dispatch_up(
    x_by_rank: jax.Array,
    expert_ids_by_rank: jax.Array,
    router_weights_by_rank: jax.Array,
    w_gate_up_by_rank: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float = 1.25,
    recv_capacity: int | None = None,
    implementation: Implementation | Sequence[Implementation | Callable[..., MoeDispatchUpLayout]] | None = None,
    w13_implementation: W13Implementation = "reference",
    ragged_dot_implementation: RaggedDotImplementation = "auto",
) -> tuple[jax.Array, MoeDispatchUpLayout]:
    prepacked = prepack_moe_dispatch_up(
        x_by_rank,
        expert_ids_by_rank,
        router_weights_by_rank,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        recv_capacity=recv_capacity,
    )
    layout = moe_dispatch_up_layout(prepacked, recv_capacity=recv_capacity, implementation=implementation)
    if w13_implementation == "reference":
        dispatch_up = compute_moe_up_from_layout_reference(layout, w_gate_up_by_rank)
    elif w13_implementation == "ragged_dot":
        dispatch_up = compute_moe_up_from_layout_ragged_dot(
            layout,
            w_gate_up_by_rank,
            implementation=ragged_dot_implementation,
        )
    else:
        raise ValueError(f"Unsupported MoE dispatch-up W13 implementation: {w13_implementation!r}")
    return dispatch_up, layout


def moe_dispatch_up_layout_reference(
    x_by_rank: jax.Array,
    expert_ids_by_rank: jax.Array,
    router_weights_by_rank: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float = 1.25,
    recv_capacity: int | None = None,
) -> MoeDispatchUpLayout:
    return _moe_dispatch_up_layout_reference(
        x_by_rank,
        expert_ids_by_rank,
        router_weights_by_rank,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        recv_capacity=recv_capacity,
    )


def moe_dispatch_up_bwd_reference(
    x_by_rank: jax.Array,
    expert_ids_by_rank: jax.Array,
    router_weights_by_rank: jax.Array,
    w_gate_up_by_rank: jax.Array,
    grad_dispatch_up: jax.Array,
    *,
    num_experts: int,
    capacity_factor: float = 1.25,
    recv_capacity: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    return moe_dispatch_up_reference_bwd(
        x_by_rank,
        expert_ids_by_rank,
        router_weights_by_rank,
        w_gate_up_by_rank,
        grad_dispatch_up,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        recv_capacity=recv_capacity,
    )
