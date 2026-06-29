# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public Grug MoE interface and implementation dispatcher.

Implementation overview:
- Routing keeps the argsort-grouped dispatch path that emerged as the stable
  default from https://github.com/marin-community/marin/issues/2704 and commit
  89318a910 (and its parent).
- Expert parallelism is selected through the implementation dispatcher. The
  older ring path follows https://github.com/marin-community/marin/issues/2710;
  newer paths include ragged all-to-all and the Hopper Pallas MGPU backend.
- Backend bodies live in the private `levanter.grug._moe` package; this module
  keeps the stable public API used by Grug model code and benchmarks.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.jax_utils import named_call
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _DEFAULT_EP_CAPACITY_FACTOR,
    _EP_MOE_IMPLEMENTATIONS,
    _LOCAL_MOE_IMPLEMENTATIONS,
    _init_weight,
    MOE_REMAT_SAVE_NAMES as MOE_REMAT_SAVE_NAMES,
    MoEExpertMlpPspecs,
    MoeActivation,
    MoeImplementation,
    MoeImplementationSpec,
    PspecAxis,
    resolve_moe_implementation,
    resolve_moe_implementations,
    split_moe_w13_output,
)
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes as _clip_receiver_group_sizes,
    _compact_by_keep_mask as _compact_by_keep_mask,
    _expand_from_keep_mask as _expand_from_keep_mask,
    _shard_a2a_params as _shard_a2a_params,
)
from levanter.grug._moe.ep_deepep import _moe_mlp_ep_deepep_local
from levanter.grug._moe.pallas_mgpu import (
    _MAX_PALLAS_MGPU_EP_SIZE,
    _moe_mlp_ep_pallas_mgpu_local,
    _validate_pallas_mgpu_dtype_and_tile_requirements,
    _validate_local_hopper_gpu_topology,
    infer_moe_mgpu_config,
)
from levanter.grug._moe.ep_ragged_all_to_all import _moe_mlp_ep_ragged_a2a_local
from levanter.grug._moe.ep_ring import _moe_mlp_ep_ring_local
from levanter.grug._moe.local import _moe_mlp_local
from levanter.grug.sharding import (
    _batch_spec_from_x,
    _current_mesh,
    _mesh_axis_size,
    _mesh_has_axis,
    _reshard_for_init,
    _reshard_for_shard_map,
    _value_spec_or_default,
)
from levanter.utils.activation import ActivationFunctionEnum


_FALLBACK_EXCEPTIONS = (NotImplementedError, jax.errors.JaxRuntimeError)
_LOCAL_EP_UNSUPPORTED_MESSAGE = (
    "Local MoE implementations do not yet support expert-parallel collectives; adding EP support "
    "requires a dispatch/combine schedule inside each expert shard plus cross-shard routing."
)


@dataclass(frozen=True)
class _MoEMlpStaticShapes:
    num_experts: int
    intermediate_dim: int


class MoEExpertMlp(eqx.Module):
    """Expert MLP weights plus backend-specific W13 layout for routed MoE calls."""

    w_gate_up: jax.Array
    w_down: jax.Array
    implementation: tuple[MoeImplementation, ...] = eqx.field(static=True)
    activation: MoeActivation = eqx.field(static=True)
    capacity_factor: float = eqx.field(static=True)

    @staticmethod
    def init(
        *,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        initializer_std: float,
        key: jax.Array,
        implementation: MoeImplementationSpec = None,
        activation: MoeActivation = ActivationFunctionEnum.silu,
        capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR,
        pspecs: MoEExpertMlpPspecs = MoEExpertMlpPspecs(),
    ) -> "MoEExpertMlp":
        resolved_implementations = resolve_moe_implementations(implementation)
        k_gate, k_up, k_down = jax.random.split(key, 3)
        w_gate = _init_weight(k_gate, (num_experts, hidden_dim, intermediate_dim), initializer_std)
        w_up = _init_weight(k_up, (num_experts, hidden_dim, intermediate_dim), initializer_std)
        w_gate_up = jnp.concatenate([w_gate, w_up], axis=-1)

        return MoEExpertMlp(
            w_gate_up=_reshard_for_init(w_gate_up, pspecs.w_gate_up),
            w_down=_reshard_for_init(
                _init_weight(k_down, (num_experts, intermediate_dim, hidden_dim), initializer_std),
                pspecs.w_down,
            ),
            implementation=resolved_implementations,
            activation=activation,
            capacity_factor=capacity_factor,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "T D"],
        selected_experts: Int[Array, "T K"],
        combine_weights: Float[Array, "T K"],
        *,
        mesh: jax.sharding.AbstractMesh | None = None,
        report_capacity_overflow: bool = False,
    ) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            self.w_gate_up,
            self.w_down,
            activation=self.activation,
            implementation=self.implementation,
            mesh=mesh,
            capacity_factor=self.capacity_factor,
            report_capacity_overflow=report_capacity_overflow,
        )


@named_call
def moe_mlp(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation: MoeActivation = ActivationFunctionEnum.silu,
    implementation: MoeImplementationSpec = None,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None = None,
    capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR,
    report_capacity_overflow: bool = False,
) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
    """Functional routed MoE MLP core used by Grug modules and benchmarks.

    This helper handles dispatch/permute/unpermute (+EP collectives) from
    precomputed token-to-expert assignments. Routing logits/top-k selection
    stays in the caller (e.g. model MLP block).

    Set `report_capacity_overflow=True` to also return a scalar count of
    dropped expert assignments from EP capacity clipping.
    """
    resolved_implementations = resolve_moe_implementations(implementation)
    if len(resolved_implementations) > 1:
        return _moe_mlp_with_fallbacks(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=activation,
            implementations=resolved_implementations,
            mesh=mesh,
            capacity_factor=capacity_factor,
            report_capacity_overflow=report_capacity_overflow,
        )

    return _moe_mlp_single_implementation(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation=activation,
        implementation=resolved_implementations[0],
        mesh=mesh,
        capacity_factor=capacity_factor,
        report_capacity_overflow=report_capacity_overflow,
    )


def _moe_mlp_with_fallbacks(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation: MoeActivation,
    implementations: tuple[MoeImplementation, ...],
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None,
    capacity_factor: float,
    report_capacity_overflow: bool,
) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
    failures: list[tuple[MoeImplementation, Exception]] = []
    for index, candidate in enumerate(implementations):
        try:
            return _moe_mlp_single_implementation(
                x,
                selected_experts,
                combine_weights,
                w_up_gate,
                w_down,
                activation=activation,
                implementation=candidate,
                mesh=mesh,
                capacity_factor=capacity_factor,
                report_capacity_overflow=report_capacity_overflow,
            )
        except _FALLBACK_EXCEPTIONS as exc:
            _record_moe_fallback(failures, candidate, exc, has_next=index + 1 < len(implementations))
        except ValueError as exc:
            if not _is_backend_unavailable_for_fallback(candidate, exc):
                raise
            _record_moe_fallback(failures, candidate, exc, has_next=index + 1 < len(implementations))
    details = "; ".join(f"{candidate!r}: {type(exc).__name__}: {exc}" for candidate, exc in failures)
    raise RuntimeError(f"No requested MoE implementation succeeded: {details}") from failures[-1][1]


def _record_moe_fallback(
    failures: list[tuple[MoeImplementation, Exception]],
    candidate: MoeImplementation,
    exc: Exception,
    *,
    has_next: bool,
) -> None:
    failures.append((candidate, exc))
    if has_next:
        warnings.warn(
            f"MoE implementation {candidate!r} failed; trying next fallback. Error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


def _is_backend_unavailable_for_fallback(candidate: MoeImplementation, exc: ValueError) -> bool:
    message = str(exc)
    if candidate == "pallas_mgpu":
        return "implementation='pallas_mgpu'" in message
    if candidate in _LOCAL_MOE_IMPLEMENTATIONS:
        return _LOCAL_EP_UNSUPPORTED_MESSAGE in message
    return False


def _validate_moe_mlp_static_inputs(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
) -> _MoEMlpStaticShapes:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [T, D], got shape={x.shape}")
    if selected_experts.ndim != 2:
        raise ValueError(f"selected_experts must be rank-2 [T, K], got shape={selected_experts.shape}")
    if selected_experts.shape != combine_weights.shape:
        raise ValueError(
            "selected_experts and combine_weights must have identical [T, K] shapes; "
            f"got {selected_experts.shape} vs {combine_weights.shape}"
        )
    if selected_experts.shape[0] != x.shape[0]:
        raise ValueError(
            f"selected_experts/combine_weights token dim ({selected_experts.shape[0]}) must match x token "
            f"dim ({x.shape[0]})"
        )

    if w_up_gate.ndim != 3:
        raise ValueError(f"w_up_gate must be rank-3 [E, D, 2I], got shape={w_up_gate.shape}")
    if w_down.ndim != 3:
        raise ValueError(f"w_down must be rank-3 [E, I, D], got shape={w_down.shape}")
    num_experts = int(w_up_gate.shape[0])
    if w_down.shape[0] != num_experts:
        raise ValueError(
            f"w_down expert dimension ({w_down.shape[0]}) must match w_up_gate expert dimension ({num_experts})"
        )
    if w_up_gate.shape[1] != x.shape[1]:
        raise ValueError(f"w_up_gate hidden dimension {w_up_gate.shape[1]} must match x dimension {x.shape[1]}")
    if w_up_gate.shape[2] % 2 != 0:
        raise ValueError(f"w_up_gate output dimension must be even [2I], got {w_up_gate.shape[2]}")
    intermediate_dim = w_up_gate.shape[2] // 2
    expected_w_down_shape = (num_experts, intermediate_dim, x.shape[1])
    if w_down.shape != expected_w_down_shape:
        raise ValueError(f"w_down must have shape {expected_w_down_shape}, got {w_down.shape}")
    return _MoEMlpStaticShapes(num_experts=num_experts, intermediate_dim=intermediate_dim)


def _validate_public_pallas_mgpu_request(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    shapes: _MoEMlpStaticShapes,
    has_expert_axis: bool,
    expert_axis_size: int,
    capacity_factor: float,
) -> None:
    if not has_expert_axis or expert_axis_size <= 1:
        raise ValueError("implementation='pallas_mgpu' requires an expert mesh axis with size > 1")
    if _mesh_axis_size(mesh, "data") != 1:
        raise ValueError("implementation='pallas_mgpu' currently requires data mesh axis size 1")
    if expert_axis_size > _MAX_PALLAS_MGPU_EP_SIZE:
        raise ValueError(
            "implementation='pallas_mgpu' supports expert axis size <= "
            f"{_MAX_PALLAS_MGPU_EP_SIZE}, got {expert_axis_size}"
        )

    pallas_config = infer_moe_mgpu_config(
        hidden_dim=x.shape[1],
        intermediate_dim=shapes.intermediate_dim,
        ep_size=expert_axis_size,
        dtype=x.dtype,
        capacity_factor=capacity_factor,
    )
    if x.shape[0] <= 0:
        raise ValueError(f"implementation='pallas_mgpu' requires a positive token dimension, got T={x.shape[0]}")
    if selected_experts.shape[1] <= 0:
        raise ValueError(
            "implementation='pallas_mgpu' requires a positive top-k route dimension, "
            f"got K={selected_experts.shape[1]}"
        )
    if shapes.num_experts <= 0:
        raise ValueError(
            f"implementation='pallas_mgpu' requires a positive expert dimension, got E={shapes.num_experts}"
        )
    if x.shape[1] <= 0:
        raise ValueError(f"implementation='pallas_mgpu' requires a positive hidden dimension, got D={x.shape[1]}")
    if shapes.intermediate_dim <= 0:
        raise ValueError(
            "implementation='pallas_mgpu' requires a positive intermediate dimension, "
            f"got I={shapes.intermediate_dim}"
        )
    _validate_pallas_mgpu_dtype_and_tile_requirements(
        x_dtype=x.dtype,
        selected_experts_dtype=selected_experts.dtype,
        combine_weights_dtype=combine_weights.dtype,
        w13_dtype=w_up_gate.dtype,
        w2_dtype=w_down.dtype,
        hidden_dim=x.shape[1],
        intermediate_dim=shapes.intermediate_dim,
        config=pallas_config,
        w13_name="w_up_gate",
        w2_name="w_down",
    )
    if shapes.num_experts % expert_axis_size != 0:
        raise ValueError(f"num_experts={shapes.num_experts} must be divisible by expert axis size={expert_axis_size}")
    _validate_local_hopper_gpu_topology(expert_axis_size)


def _moe_mlp_single_implementation(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation: MoeActivation,
    implementation: MoeImplementation,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh | None,
    capacity_factor: float,
    report_capacity_overflow: bool,
) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
    resolved_implementation = implementation

    if mesh is None:
        mesh = _current_mesh()

    activation_fn = _activation_fn(activation)
    shapes = _validate_moe_mlp_static_inputs(x, selected_experts, combine_weights, w_up_gate, w_down)

    has_expert_axis = _mesh_has_axis(mesh, "expert")
    expert_axis_size = _mesh_axis_size(mesh, "expert")
    if has_expert_axis and expert_axis_size > 1 and capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be positive for expert-parallel MoE, got {capacity_factor}")
    if resolved_implementation == "pallas_mgpu":
        _validate_public_pallas_mgpu_request(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            mesh=mesh,
            shapes=shapes,
            has_expert_axis=has_expert_axis,
            expert_axis_size=expert_axis_size,
            capacity_factor=capacity_factor,
        )

    if mesh is None or mesh.empty:
        out, dropped = _moe_mlp_local(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=activation_fn,
            num_experts=shapes.num_experts,
            implementation=resolved_implementation,
        )
        if report_capacity_overflow:
            return out, dropped
        return out

    batch_spec = _batch_spec_from_x(x, mesh)

    if has_expert_axis and expert_axis_size > 1:
        return _moe_mlp_expert_parallel(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation_fn=activation_fn,
            implementation=resolved_implementation,
            mesh=mesh,
            batch_spec=batch_spec,
            shapes=shapes,
            expert_axis_size=expert_axis_size,
            capacity_factor=capacity_factor,
            report_capacity_overflow=report_capacity_overflow,
        )

    return _moe_mlp_no_expert_parallel_shard_map(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation_fn=activation_fn,
        implementation=resolved_implementation,
        mesh=mesh,
        batch_spec=batch_spec,
        shapes=shapes,
        report_capacity_overflow=report_capacity_overflow,
    )


def _activation_fn(activation: MoeActivation) -> Callable[[jax.Array], jax.Array]:
    if isinstance(activation, ActivationFunctionEnum):
        return activation.to_jax_fn()
    return activation


def _moe_mlp_expert_parallel(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    implementation: MoeImplementation,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    batch_spec: P,
    shapes: _MoEMlpStaticShapes,
    expert_axis_size: int,
    capacity_factor: float,
    report_capacity_overflow: bool,
) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
    if implementation not in _EP_MOE_IMPLEMENTATIONS:
        raise ValueError(
            f"{_LOCAL_EP_UNSUPPORTED_MESSAGE}. got implementation={implementation!r} "
            f"with expert axis size={expert_axis_size}"
        )
    if shapes.num_experts % expert_axis_size != 0:
        raise ValueError(f"num_experts={shapes.num_experts} must be divisible by expert axis size={expert_axis_size}")

    shard_local_fn = _moe_mlp_ep_local_fn(implementation)
    if implementation == "pallas_mgpu":
        batch_spec = P("expert")
    w_up_gate_spec = P("expert", None, None)
    w_down_spec = P("expert", None, None)

    x = _reshard_for_shard_map(x, mesh, batch_spec)
    selected_experts = _reshard_for_shard_map(selected_experts, mesh, batch_spec)
    combine_weights = _reshard_for_shard_map(combine_weights, mesh, batch_spec)
    w_up_gate = _reshard_for_shard_map(w_up_gate, mesh, w_up_gate_spec)
    w_down = _reshard_for_shard_map(w_down, mesh, w_down_spec)

    shard_fn = shard_map(
        partial(
            shard_local_fn,
            activation_fn=activation_fn,
            num_experts=shapes.num_experts,
            capacity_factor=capacity_factor,
        ),
        mesh=mesh,
        in_specs=(
            batch_spec,
            batch_spec,
            batch_spec,
            w_up_gate_spec,
            w_down_spec,
        ),
        out_specs=(batch_spec, P()),
        check_vma=False,
    )
    out, dropped = shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
    if report_capacity_overflow:
        return out, dropped
    return out


def _moe_mlp_ep_local_fn(implementation: MoeImplementation):
    if implementation == "ring":
        return _moe_mlp_ep_ring_local
    if implementation == "ragged_all_to_all":
        return _moe_mlp_ep_ragged_a2a_local
    if implementation == "deepep":
        return _moe_mlp_ep_deepep_local
    if implementation == "pallas_mgpu":
        return _moe_mlp_ep_pallas_mgpu_local
    raise AssertionError(f"Unhandled MoE implementation {implementation!r}")


def _moe_mlp_no_expert_parallel_shard_map(
    x: Float[Array, "T D"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E D I2"],
    w_down: Float[Array, "E I D"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    implementation: MoeImplementation,
    mesh: jax.sharding.Mesh | jax.sharding.AbstractMesh,
    batch_spec: P,
    shapes: _MoEMlpStaticShapes,
    report_capacity_overflow: bool,
) -> Float[Array, "T D"] | tuple[Float[Array, "T D"], Int[Array, ""]]:
    # JAX 0.9 requires shard_map in_specs to match input sharding, so reshard
    # ordinary inputs to mesh specs that preserve data-axis parallelism.
    x_spec = _value_spec_or_default(x, batch_spec, replace_replicated=True)
    selected_experts_spec = _value_spec_or_default(selected_experts, batch_spec, replace_replicated=True)
    combine_weights_spec = _value_spec_or_default(combine_weights, batch_spec, replace_replicated=True)
    w_up_gate_spec = _value_spec_or_default(w_up_gate, P(*(None for _ in range(w_up_gate.ndim))))
    w_down_spec = _value_spec_or_default(w_down, P(*(None for _ in range(w_down.ndim))))

    x = _reshard_for_shard_map(x, mesh, x_spec)
    selected_experts = _reshard_for_shard_map(selected_experts, mesh, selected_experts_spec)
    combine_weights = _reshard_for_shard_map(combine_weights, mesh, combine_weights_spec)
    w_up_gate = _reshard_for_shard_map(w_up_gate, mesh, w_up_gate_spec)
    w_down = _reshard_for_shard_map(w_down, mesh, w_down_spec)

    shard_fn = shard_map(
        partial(
            _moe_mlp_local,
            activation_fn=activation_fn,
            num_experts=shapes.num_experts,
            implementation=implementation,
        ),
        mesh=mesh,
        in_specs=(
            x_spec,
            selected_experts_spec,
            combine_weights_spec,
            w_up_gate_spec,
            w_down_spec,
        ),
        out_specs=(x_spec, P()),
        check_vma=False,
    )
    out, dropped = shard_fn(x, selected_experts, combine_weights, w_up_gate, w_down)
    if report_capacity_overflow:
        return out, dropped
    return out


__all__ = [
    "MoeActivation",
    "MoEExpertMlp",
    "MoEExpertMlpPspecs",
    "MoeImplementation",
    "MoeImplementationSpec",
    "PspecAxis",
    "moe_mlp",
    "resolve_moe_implementation",
    "resolve_moe_implementations",
    "split_moe_w13_output",
]
