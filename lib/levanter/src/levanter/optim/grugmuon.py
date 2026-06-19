# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Muon optimizer for models using raw JAX arrays with (fan_in, fan_out) layout,
such as Grug models.

All 2D arrays are routed to Muon, except those whose path contains
'embed', 'lm_head', or 'output' (case-insensitive), which use AdamW.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from jax.sharding import PartitionSpec, reshard
from optax import tree_utils as otu

from levanter.optim.config import OptimizerConfig
from levanter.optim.muon import MuonConfig, ScaleByMuonState
from levanter.optim.util import NEWTON_SCHULZ_COEFFICIENTS, CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

VMAP_REPLICATED = "vmap_replicated"
STACK_BATCH_SHARDED = "stack_batch_sharded"
STACK_BATCH_4D_SHARDED = "stack_batch_4d_sharded"
ORTHOGONALIZATION_LAYOUTS = (VMAP_REPLICATED, STACK_BATCH_SHARDED, STACK_BATCH_4D_SHARDED)
DEFAULT_MAX_GROUPED_STACK_SIZE = 256
DEFAULT_MAX_GROUPED_STACK_LOCAL_PER_SHARD = 2
DEFAULT_GROUPED_4D_GROUP_SIZE = 2
STACK_PADDING_MAX_OVERHEAD = 1.25
STACK_BATCH_AXIS_CANDIDATES = ("replica_dcn", "data", "expert")
NS_COMPUTE_DTYPES = ("input", "bf16", "bfloat16", "fp32", "float32", "fp16", "float16")


def _ns_compute_dtype_from_name(name: str, input_dtype: jnp.dtype) -> jnp.dtype:
    normalized = name.lower()
    if normalized == "input":
        return jnp.dtype(input_dtype)
    if normalized in ("bf16", "bfloat16"):
        return jnp.dtype(jnp.bfloat16)
    if normalized in ("fp32", "float32"):
        return jnp.dtype(jnp.float32)
    if normalized in ("fp16", "float16"):
        return jnp.dtype(jnp.float16)
    valid = ", ".join(NS_COMPUTE_DTYPES)
    raise ValueError(f"ns_compute_dtype={name!r} must be one of {valid}")


def _cast_for_ns_compute(x: jax.Array, ns_compute_dtype: str) -> tuple[jax.Array, jnp.dtype]:
    original_dtype = jnp.dtype(x.dtype)
    compute_dtype = _ns_compute_dtype_from_name(ns_compute_dtype, original_dtype)
    if compute_dtype == original_dtype:
        return x, original_dtype
    with jax.named_scope("grug_muon/ns_compute_dtype/cast_input"):
        return x.astype(compute_dtype), original_dtype


def _restore_ns_compute_dtype(x: jax.Array, original_dtype: jnp.dtype) -> jax.Array:
    if jnp.dtype(x.dtype) == original_dtype:
        return x
    with jax.named_scope("grug_muon/ns_compute_dtype/cast_output"):
        return x.astype(original_dtype)


def _target_sharding(array) -> jax.sharding.NamedSharding | PartitionSpec | None:
    if array is None or not hasattr(array, "shape"):
        return None

    def valid_target(sharding):
        if isinstance(sharding, jax.sharding.NamedSharding):
            return None if sharding.mesh.empty else sharding
        if isinstance(sharding, PartitionSpec):
            return sharding
        return None

    sharding = getattr(array, "sharding", None)
    target = valid_target(sharding)
    if target is not None:
        return target

    aval = jax.typeof(array)
    sharding = getattr(aval, "sharding", None)
    return valid_target(sharding)


def _stack_axis_shards(target_pspec: jax.sharding.NamedSharding | PartitionSpec | None) -> int:
    if target_pspec is None:
        return 1
    if isinstance(target_pspec, jax.sharding.NamedSharding):
        mesh_shape = target_pspec.mesh.shape
        target_spec = target_pspec.spec
    else:
        mesh = jax.sharding.get_abstract_mesh()
        if mesh.empty:
            return 1
        mesh_shape = mesh.shape
        target_spec = target_pspec

    stack_axis = target_spec[0]
    if stack_axis is None:
        return 1
    stack_axis_names = stack_axis if isinstance(stack_axis, tuple) else (stack_axis,)
    return math.prod(int(mesh_shape[name]) for name in stack_axis_names)


def _padded_stack_size(stack_size: int, target_pspec: jax.sharding.NamedSharding | PartitionSpec | None) -> int:
    axis_shards = _stack_axis_shards(target_pspec)
    if axis_shards <= 1:
        return stack_size
    return math.ceil(stack_size / axis_shards) * axis_shards


def _padding_overhead(stack_size: int, axis_shards: int) -> float:
    if axis_shards <= 1 or stack_size % axis_shards == 0:
        return 1.0
    return math.ceil(stack_size / axis_shards) * axis_shards / stack_size


def _candidate_stack_axes(mesh) -> tuple[tuple[str, ...], ...]:
    candidate_axes = tuple(
        axis_name for axis_name in STACK_BATCH_AXIS_CANDIDATES if int(mesh.shape.get(axis_name, 1)) > 1
    )
    candidates = []
    for axis_mask in range(1, 1 << len(candidate_axes)):
        axes = tuple(axis for index, axis in enumerate(candidate_axes) if axis_mask & (1 << index))
        axis_size = math.prod(int(mesh.shape[axis]) for axis in axes)
        candidates.append((axis_size, axes))
    return tuple(axes for _, axes in sorted(candidates, key=lambda item: item[0], reverse=True))


def _batch_sharded_stack_target_pspec(array) -> PartitionSpec | None:
    if array is None or not hasattr(array, "shape") or array.ndim != 3:
        return None

    mesh = jax.sharding.get_abstract_mesh()
    if mesh.empty:
        return None

    for axes in _candidate_stack_axes(mesh):
        axis_size = math.prod(int(mesh.shape[axis]) for axis in axes)
        if _padding_overhead(array.shape[0], axis_size) <= STACK_PADDING_MAX_OVERHEAD:
            stack_axis = axes[0] if len(axes) == 1 else axes
            return PartitionSpec(stack_axis, None, None)

    sharding = _target_sharding(array)
    spec = getattr(sharding, "spec", None)
    if spec is not None and len(spec) > 0 and spec[0] is not None:
        stack_axis = spec[0]
        stack_axis_names = stack_axis if isinstance(stack_axis, tuple) else (stack_axis,)
        stack_axis_size = math.prod(int(mesh.shape[name]) for name in stack_axis_names if name in mesh.shape)
        if stack_axis_size > 1 and _padding_overhead(array.shape[0], stack_axis_size) <= STACK_PADDING_MAX_OVERHEAD:
            return PartitionSpec(stack_axis, None, None)

    mesh_shape = tuple((axis_name, axis_size) for axis_name, axis_size in mesh.shape.items() if axis_size > 1)
    if not mesh_shape:
        return None

    batch_axis = tuple(axis_name for axis_name, _ in mesh_shape)
    batch_shards = math.prod(axis_size for _, axis_size in mesh_shape)
    if array.shape[0] % batch_shards != 0:
        return None

    if len(batch_axis) == 1:
        return PartitionSpec(batch_axis[0], None, None)
    return PartitionSpec(batch_axis, None, None)


def _mesh_shape_from_abstract_or_array(array):
    mesh = jax.sharding.get_abstract_mesh()
    if not mesh.empty:
        return mesh.shape

    sharding = _target_sharding(array)
    if isinstance(sharding, jax.sharding.NamedSharding) and not sharding.mesh.empty:
        return sharding.mesh.shape
    return None


Grouped4DTarget = jax.sharding.NamedSharding | PartitionSpec | None


def _grouped_4d_stack_target_pspec(array) -> PartitionSpec | None:
    if array is None or not hasattr(array, "shape") or array.ndim != 4:
        return None

    mesh_shape = _mesh_shape_from_abstract_or_array(array)
    if mesh_shape is None:
        return None

    group_axis = _grouped_4d_group_axis(mesh_shape, array.shape[0])

    expert_axis = None
    if int(mesh_shape.get("expert", 1)) > 1 and array.shape[1] % int(mesh_shape["expert"]) == 0:
        expert_axis = "expert"

    if expert_axis is None:
        return None
    return PartitionSpec(group_axis, expert_axis, None, None)


def _grouped_4d_group_axis(mesh_shape, group_size: int) -> str | tuple[str, ...] | None:
    candidate_axes = tuple(axis_name for axis_name in ("replica_dcn", "data") if int(mesh_shape.get(axis_name, 1)) > 1)
    candidates = []
    for axis_mask in range(1, 1 << len(candidate_axes)):
        axes = tuple(axis for index, axis in enumerate(candidate_axes) if axis_mask & (1 << index))
        axis_size = math.prod(int(mesh_shape[axis]) for axis in axes)
        if group_size % axis_size == 0:
            candidates.append((axis_size, axes))
    if not candidates:
        return None

    _, axes = max(candidates, key=lambda item: item[0])
    if len(axes) == 1:
        return axes[0]
    return axes


def _grouped_4d_stack_target(array) -> Grouped4DTarget:
    target_pspec = _grouped_4d_stack_target_pspec(array)
    if target_pspec is None:
        return None

    sharding = _target_sharding(array)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return jax.sharding.NamedSharding(sharding.mesh, target_pspec)

    return target_pspec


def _sharding_spec(array) -> PartitionSpec | None:
    sharding = _target_sharding(array)
    if isinstance(sharding, PartitionSpec):
        return sharding
    return getattr(sharding, "spec", None)


def _target_spec(target: Grouped4DTarget) -> PartitionSpec | None:
    if isinstance(target, jax.sharding.NamedSharding):
        return target.spec
    return target


def _with_target_spec(target: Grouped4DTarget, spec: PartitionSpec) -> jax.sharding.NamedSharding | PartitionSpec:
    if isinstance(target, jax.sharding.NamedSharding):
        return jax.sharding.NamedSharding(target.mesh, spec)
    return spec


def _canonical_target_for_array(array, target: Grouped4DTarget) -> Grouped4DTarget:
    if not isinstance(target, PartitionSpec):
        return target
    sharding = _target_sharding(array)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return jax.sharding.NamedSharding(sharding.mesh, target)
    return target


def _assert_stack_batch_sharded(
    array,
    target_pspec: jax.sharding.NamedSharding | PartitionSpec | None,
    label: str,
) -> None:
    """Fail fast if a stacked Muon batch lost its intended stack-axis sharding."""
    if target_pspec is None:
        return
    if isinstance(target_pspec, jax.sharding.NamedSharding):
        target_pspec = target_pspec.spec
    elif jax.sharding.get_abstract_mesh().empty:
        return
    actual_pspec = _sharding_spec(array)
    if actual_pspec != target_pspec:
        raise AssertionError(f"{label} expected stacked Muon sharding {target_pspec}, got {actual_pspec}")
    stack_axis = target_pspec[0]
    if stack_axis is None:
        raise AssertionError(f"{label} expected non-replicated stacked Muon batch axis, got {target_pspec}")


def _assert_grouped_4d_batch_sharded(array, target_pspec: Grouped4DTarget, label: str) -> None:
    """Fail fast if a 4D grouped Muon batch lost its intended group/expert sharding."""
    if target_pspec is None:
        return
    target_pspec = _target_spec(target_pspec)
    assert target_pspec is not None
    actual_pspec = _sharding_spec(array)
    if actual_pspec != target_pspec:
        raise AssertionError(f"{label} expected 4D grouped Muon sharding {target_pspec}, got {actual_pspec}")
    if target_pspec[1] is None:
        raise AssertionError(f"{label} expected expert-axis sharding for 4D grouped Muon, got {target_pspec}")


@OptimizerConfig.register_subclass("grug_muon")
@dataclass(frozen=True)
class GrugMuonConfig(MuonConfig):
    """
    Muon optimizer for models that use raw JAX arrays in (fan_in, fan_out) layout.

    Routing rules:
    - 2D arrays whose path does NOT contain 'embed', 'lm_head', or 'output' -> Muon
    - Everything else -> AdamW
    """

    ns_compute_dtype: str = "input"

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muon_transform():
                components = []
                components.append(
                    _grug_scale_with_muon(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        self.use_kimi_scaling,
                        self.coefficient_type,
                        ns_compute_dtype=self.ns_compute_dtype,
                    )
                )
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                components.append(_match_update_sharding())
                return optax.chain(*components)

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                adam_weight_decay = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
                if adam_weight_decay > 0:
                    components.append(optax.add_decayed_weights(adam_weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                return optax.chain(*components)

            transformations = {
                "muon": muon_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling)
            )

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params, use_kimi_scaling=True):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if "embed" in path_lower or "lm_head" in path_lower or "output" in path_lower:
                return "adamw"
            elif hasattr(param, "ndim") and param.ndim == 2:
                return "muon"
            elif (
                hasattr(param, "ndim")
                and param.ndim == 3
                and ("w_up_gate" in path_lower or "w_gate_up" in path_lower or "w_down" in path_lower)
            ):
                return "muon"
            else:
                return "adamw"

        return jax.tree.map(mask_fn, params, paths)


def _grug_scale_with_muon(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    use_kimi_scaling=False,
    coefficient_type="quintic",
    orthogonalization_layout: str = STACK_BATCH_SHARDED,
    momentum_sharding_fn: Callable[[jax.Array], jax.sharding.Sharding | None] | None = None,
    max_grouped_stack_size: int = DEFAULT_MAX_GROUPED_STACK_SIZE,
    ns_compute_dtype: str = "input",
):
    """Muon gradient transformation for raw arrays with matrix-shaped trailing dimensions."""
    steps = int(steps)
    max_grouped_stack_size = int(max_grouped_stack_size)
    _ns_compute_dtype_from_name(ns_compute_dtype, jnp.float32)
    if max_grouped_stack_size < 1:
        raise ValueError(f"max_grouped_stack_size must be positive, got {max_grouped_stack_size}.")
    if orthogonalization_layout not in ORTHOGONALIZATION_LAYOUTS:
        raise ValueError(
            f"Unknown orthogonalization_layout={orthogonalization_layout!r}. "
            f"Expected one of {ORTHOGONALIZATION_LAYOUTS!r}."
        )

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        if momentum_sharding_fn is not None:

            def shard_momentum(momentum_leaf, param):
                if momentum_leaf is None:
                    return None
                target_sharding = momentum_sharding_fn(param)
                if target_sharding is None:
                    return momentum_leaf
                return reshard(momentum_leaf, target_sharding)

            momentum_buffer = jax.tree.map(
                shard_momentum,
                momentum_buffer,
                params,
                is_leaf=lambda x: x is None,
            )
        return ScaleByMuonState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer

        def match_state_sharding(update, momentum_leaf):
            if update is None:
                return None
            if momentum_sharding_fn is None:
                return update
            target_sharding = _target_sharding(momentum_leaf)
            if target_sharding is None or not isinstance(target_sharding, jax.sharding.NamedSharding):
                return update
            return reshard(update, target_sharding)

        with jax.named_scope("grug_muon/update_momentum_buffer"):
            buf = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + match_state_sharding(g, m),
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        if nesterov:
            with jax.named_scope("grug_muon/nesterov_update"):
                updates = jax.tree.map(
                    lambda m, g: None if g is None else momentum * m + match_state_sharding(g, m),
                    buf,
                    updates,
                    is_leaf=lambda x: x is None,
                )
        else:
            updates = buf

        has_mesh = not jax.sharding.get_abstract_mesh().empty

        def stack_target_pspec(x, param):
            target_pspec = _batch_sharded_stack_target_pspec(x)
            if target_pspec is None:
                target_pspec = _batch_sharded_stack_target_pspec(param)
            return target_pspec

        def grouped_4d_target(x):
            return _grouped_4d_stack_target(x)

        def grouped_4d_shape_dtype_struct(shape, sample_x):
            sample_sharding = _target_sharding(sample_x)
            if isinstance(sample_sharding, jax.sharding.NamedSharding) and len(sample_sharding.spec) == 3:
                sharding = jax.sharding.NamedSharding(
                    sample_sharding.mesh,
                    PartitionSpec(None, *sample_sharding.spec),
                )
                return jax.ShapeDtypeStruct(shape, sample_x.dtype, sharding=sharding)
            return jax.ShapeDtypeStruct(shape, sample_x.dtype)

        def stack_limit(target_pspec):
            axis_shards = _stack_axis_shards(target_pspec)
            if axis_shards <= 1:
                return max_grouped_stack_size
            local_stack_limit = axis_shards * DEFAULT_MAX_GROUPED_STACK_LOCAL_PER_SHARD
            global_stack_limit = max_grouped_stack_size // axis_shards * axis_shards
            return max(local_stack_limit, global_stack_limit)

        def scale_and_restore_param_sharding(updated, param):
            with jax.named_scope("grug_muon/scale_orthogonalized_update"):
                fan_in, fan_out = updated.shape[-2:]
                if not use_kimi_scaling:
                    scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
                else:
                    scale = 0.2 * jnp.sqrt(jnp.maximum(fan_in, fan_out))
                updated *= scale
            target_sharding = _target_sharding(param)
            if target_sharding is not None:
                with jax.named_scope("grug_muon/restore_param_sharding"):
                    updated = reshard(updated, target_sharding)
            return updated

        def orthogonalize_3d(x, param):
            x, original_dtype = _cast_for_ns_compute(x, ns_compute_dtype)
            if orthogonalization_layout == VMAP_REPLICATED:
                with jax.named_scope("grug_muon/orthogonalize_3d_vmap_replicated"):
                    updated = jax.vmap(
                        lambda matrix: _zeropower_via_newtonschulz_replicated(
                            matrix,
                            steps,
                            muon_eps,
                            coefficient_type,
                            None,
                        )
                    )(x)
                return _restore_ns_compute_dtype(updated, original_dtype)

            target_pspec = stack_target_pspec(x, param)
            if target_pspec is None and has_mesh:
                with jax.named_scope("grug_muon/orthogonalize_3d_vmap_fallback"):
                    updated = jax.vmap(
                        lambda matrix: _zeropower_via_newtonschulz_replicated(
                            matrix,
                            steps,
                            muon_eps,
                            coefficient_type,
                            None,
                        )
                    )(x)
                return _restore_ns_compute_dtype(updated, original_dtype)

            with jax.named_scope("grug_muon/orthogonalize_3d_stack_sharded"):
                updated = _zeropower_via_newtonschulz_batched_stack_sharded(
                    x,
                    steps,
                    muon_eps,
                    coefficient_type,
                    target_pspec,
                )
            return _restore_ns_compute_dtype(updated, original_dtype)

        def transform_chunked_3d_array(x, param):
            target_pspec = stack_target_pspec(x, param)
            max_stack_size = stack_limit(target_pspec)
            if _padded_stack_size(x.shape[0], target_pspec) <= max_stack_size:
                return scale_and_restore_param_sharding(orthogonalize_3d(x, param), param)

            if target_pspec is None and has_mesh:
                return scale_and_restore_param_sharding(orthogonalize_3d(x, param), param)

            updated_chunks = []
            split_indices = tuple(range(max_stack_size, x.shape[0], max_stack_size))
            for x_chunk in jnp.split(x, split_indices, axis=0):
                with jax.named_scope("grug_muon/orthogonalize_3d_chunked_stack"):
                    updated_chunks.append(orthogonalize_3d(x_chunk, None))
            updated = jnp.concatenate(updated_chunks, axis=0)
            return scale_and_restore_param_sharding(updated, param)

        def transform_array(x, param):
            if not hasattr(x, "ndim") or x.ndim not in (2, 3):
                return x
            if x.ndim == 2:
                x, original_dtype = _cast_for_ns_compute(x, ns_compute_dtype)
                with jax.named_scope("grug_muon/orthogonalize_2d_replicated"):
                    updated = _zeropower_via_newtonschulz_replicated(
                        x,
                        steps,
                        muon_eps,
                        coefficient_type,
                        None,
                    )
                updated = _restore_ns_compute_dtype(updated, original_dtype)
                return scale_and_restore_param_sharding(updated, param)

            return transform_chunked_3d_array(x, param)

        def transform_grouped_4d_arrays(updates, params):
            update_leaves, treedef = jax.tree.flatten(updates, is_leaf=lambda x: x is None)
            param_leaves, param_treedef = jax.tree.flatten(params, is_leaf=lambda x: x is None)
            if treedef != param_treedef:
                return jax.tree.map(transform_array, updates, params, is_leaf=lambda x: x is None)

            output_leaves = [None] * len(update_leaves)
            groups = {}

            for index, (x, param) in enumerate(zip(update_leaves, param_leaves, strict=True)):
                if hasattr(x, "ndim") and x.ndim == 3:
                    key = (tuple(x.shape), str(x.dtype))
                    groups.setdefault(key, []).append((index, x, param))
                else:
                    output_leaves[index] = transform_array(x, param)

            for entries in groups.values():
                for chunk_start in range(0, len(entries), DEFAULT_GROUPED_4D_GROUP_SIZE):
                    entry_chunk = entries[chunk_start : chunk_start + DEFAULT_GROUPED_4D_GROUP_SIZE]
                    if len(entry_chunk) < DEFAULT_GROUPED_4D_GROUP_SIZE:
                        for index, x, param in entry_chunk:
                            output_leaves[index] = transform_array(x, param)
                        continue

                    sample_x = entry_chunk[0][1]
                    stack_shape = (len(entry_chunk), *sample_x.shape)
                    target = grouped_4d_target(grouped_4d_shape_dtype_struct(stack_shape, sample_x))
                    target_spec = _target_spec(target)
                    if target is None and has_mesh:
                        for index, x, param in entry_chunk:
                            output_leaves[index] = transform_array(x, param)
                        continue

                    with jax.named_scope("grug_muon/orthogonalize_3d_grouped_4d_stack"):
                        stacked = jnp.stack([x for _, x, _ in entry_chunk], axis=0)
                        stacked, original_dtype = _cast_for_ns_compute(stacked, ns_compute_dtype)
                        updated_stacked = _zeropower_via_newtonschulz_grouped_4d_sharded(
                            stacked,
                            steps,
                            muon_eps,
                            coefficient_type,
                            target,
                        )
                        if target_spec is not None and target_spec[0] is not None:
                            updated_stacked = reshard(
                                updated_stacked,
                                _with_target_spec(target, PartitionSpec(None, target_spec[1], None, None)),
                            )
                        updated_stacked = _restore_ns_compute_dtype(updated_stacked, original_dtype)
                        updated_parts = [
                            jnp.squeeze(updated_part, axis=0)
                            for updated_part in jnp.split(updated_stacked, len(entry_chunk), axis=0)
                        ]

                    for (index, _, param), updated in zip(entry_chunk, updated_parts, strict=True):
                        output_leaves[index] = scale_and_restore_param_sharding(updated, param)

            return jax.tree.unflatten(treedef, output_leaves)

        def transform_grouped_3d_arrays(updates, params):
            update_leaves, treedef = jax.tree.flatten(updates, is_leaf=lambda x: x is None)
            param_leaves, param_treedef = jax.tree.flatten(params, is_leaf=lambda x: x is None)
            if treedef != param_treedef:
                return jax.tree.map(transform_array, updates, params, is_leaf=lambda x: x is None)

            output_leaves = [None] * len(update_leaves)
            groups = {}

            for index, (x, param) in enumerate(zip(update_leaves, param_leaves, strict=True)):
                if hasattr(x, "ndim") and x.ndim == 3:
                    key = (tuple(x.shape[-2:]), str(x.dtype))
                    groups.setdefault(key, []).append((index, x, param))
                else:
                    output_leaves[index] = transform_array(x, param)

            for entries in groups.values():
                chunked_entries = []
                chunk = []
                chunk_stack_size = 0
                for entry in entries:
                    _, x, _ = entry
                    stack_size = x.shape[0]
                    candidate_stack_size = chunk_stack_size + stack_size
                    sample_x = x
                    candidate_target_pspec = stack_target_pspec(
                        jax.ShapeDtypeStruct((candidate_stack_size, *sample_x.shape[-2:]), sample_x.dtype),
                        None,
                    )
                    candidate_stack_limit = stack_limit(candidate_target_pspec)
                    candidate_padded_size = _padded_stack_size(candidate_stack_size, candidate_target_pspec)
                    if chunk and candidate_padded_size > candidate_stack_limit:
                        chunked_entries.append(chunk)
                        chunk = []
                        chunk_stack_size = 0
                    chunk.append(entry)
                    chunk_stack_size += stack_size
                if chunk:
                    chunked_entries.append(chunk)

                for entry_chunk in chunked_entries:
                    if len(entry_chunk) == 1:
                        index, x, param = entry_chunk[0]
                        output_leaves[index] = transform_array(x, param)
                        continue

                    stack_size = sum(x.shape[0] for _, x, _ in entry_chunk)
                    sample_x = entry_chunk[0][1]
                    target_pspec = stack_target_pspec(
                        jax.ShapeDtypeStruct((stack_size, *sample_x.shape[-2:]), sample_x.dtype),
                        None,
                    )
                    if target_pspec is None and has_mesh:
                        for index, x, param in entry_chunk:
                            output_leaves[index] = transform_array(x, param)
                        continue
                    with jax.named_scope("grug_muon/orthogonalize_3d_grouped_stack"):
                        parts = []
                        for _, x, _ in entry_chunk:
                            parts.append(x)
                        stack_sizes = [x.shape[0] for x in parts]
                        stacked = jnp.concatenate(parts, axis=0)
                        stacked, original_dtype = _cast_for_ns_compute(stacked, ns_compute_dtype)
                        updated_stacked = _zeropower_via_newtonschulz_batched_stack_sharded(
                            stacked,
                            steps,
                            muon_eps,
                            coefficient_type,
                            target_pspec,
                        )
                        updated_stacked = _restore_ns_compute_dtype(updated_stacked, original_dtype)
                        split_indices = []
                        running_size = 0
                        for stack_size in stack_sizes[:-1]:
                            running_size += stack_size
                            split_indices.append(running_size)
                        updated_parts = jnp.split(updated_stacked, split_indices, axis=0)

                    for (index, _, param), updated in zip(entry_chunk, updated_parts, strict=True):
                        output_leaves[index] = scale_and_restore_param_sharding(updated, param)

            return jax.tree.unflatten(treedef, output_leaves)

        if params is None:
            with jax.named_scope("grug_muon/transform_updates"):
                updates = jax.tree.map(lambda x: transform_array(x, None), updates)
        else:
            with jax.named_scope("grug_muon/transform_updates"):
                if orthogonalization_layout == STACK_BATCH_4D_SHARDED:
                    updates = transform_grouped_4d_arrays(updates, params)
                else:
                    updates = transform_grouped_3d_arrays(updates, params)

        return updates, ScaleByMuonState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


def _match_update_sharding():
    """Ensure updates inherit the parameter sharding expected by apply_updates."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state

        def match_sharding(update, param):
            if update is None:
                return None
            target_sharding = _target_sharding(param)
            if target_sharding is None:
                return update
            return jax.sharding.reshard(update, target_sharding)

        updates = jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


@jax.named_call
def _zeropower_via_newtonschulz_replicated(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
    target_pspec: PartitionSpec | None = None,
) -> jax.Array:
    """Legacy Grug Muon orthogonalization that fully replicates each matrix.

    Replicates the array across devices before iterating to avoid sharding
    ambiguities in the X @ X.T contractions. The caller is responsible for
    restoring the final parameter layout. Kept for A/B benchmarking.
    """
    P = PartitionSpec
    assert X.ndim == 2
    del target_pspec  # Kept for signature parity with the other Newton-Schulz helpers.

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    target_sharding = _target_sharding(X)
    replicated_sharding = None
    if isinstance(target_sharding, jax.sharding.NamedSharding):
        replicated_sharding = jax.sharding.NamedSharding(target_sharding.mesh, P(None, None))
    elif isinstance(target_sharding, PartitionSpec) and not jax.sharding.get_abstract_mesh().empty:
        replicated_sharding = P(None, None)
    with jax.named_scope("newton_schulz_replicated/normalize_input"):
        if replicated_sharding is not None:
            X = reshard(X, replicated_sharding)
        X = X / (jnp.linalg.norm(X) + eps)

    transpose = False
    if X.shape[0] > X.shape[1]:
        with jax.named_scope("newton_schulz_replicated/transpose_tall_matrix"):
            X = X.T
        transpose = True

    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        out_sharding = replicated_sharding
        with jax.named_scope(f"newton_schulz_replicated/iter_{i}/gram"):
            A = jnp.einsum("ik,jk->ij", X, X, out_sharding=out_sharding)
        with jax.named_scope(f"newton_schulz_replicated/iter_{i}/polynomial"):
            B = b * A + c * jnp.einsum("ik,kj->ij", A, A, out_sharding=out_sharding)
        with jax.named_scope(f"newton_schulz_replicated/iter_{i}/apply"):
            X = a * X + jnp.einsum("ik,kj->ij", B, X, out_sharding=out_sharding)

    if transpose:
        with jax.named_scope("newton_schulz_replicated/restore_transpose"):
            X = X.T

    return X


@jax.named_call
def _zeropower_via_newtonschulz_batched_stack_sharded(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
    target_pspec: jax.sharding.NamedSharding | PartitionSpec | None = None,
) -> jax.Array:
    """Run Newton-Schulz on a stacked batch of matrices with only the batch axis sharded."""
    assert X.ndim == 3

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    has_mesh = not jax.sharding.get_abstract_mesh().empty
    has_explicit_sharding = isinstance(target_pspec, jax.sharding.NamedSharding)
    if target_pspec is None:
        target_pspec = _batch_sharded_stack_target_pspec(X)
        if target_pspec is not None:
            target_pspec = _canonical_target_for_array(X, target_pspec)
            has_explicit_sharding = isinstance(target_pspec, jax.sharding.NamedSharding)
    has_target_sharding = target_pspec is not None and (has_mesh or has_explicit_sharding)
    original_stack_size = X.shape[0]
    padded_stack_size = _padded_stack_size(original_stack_size, target_pspec)
    if padded_stack_size != original_stack_size:
        with jax.named_scope("newton_schulz_stack_sharded/pad_stack_axis"):
            X = jnp.pad(X, ((0, padded_stack_size - original_stack_size), (0, 0), (0, 0)))
    if has_target_sharding:
        with jax.named_scope("newton_schulz_stack_sharded/reshard_stack"):
            X = reshard(X, target_pspec)
            _assert_stack_batch_sharded(X, target_pspec, "Muon NS input")

    with jax.named_scope("newton_schulz_stack_sharded/normalize_input"):
        X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) + eps)

    transpose = False
    if X.shape[-2] > X.shape[-1]:
        with jax.named_scope("newton_schulz_stack_sharded/transpose_tall_matrices"):
            X = jnp.swapaxes(X, -1, -2)
        transpose = True

    if has_target_sharding:
        with jax.named_scope("newton_schulz_stack_sharded/assert_resharded_stack"):
            X = reshard(X, target_pspec)
            _assert_stack_batch_sharded(X, target_pspec, "Muon NS dot input")

    X_out_sharding = target_pspec if has_target_sharding else None
    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        with jax.named_scope(f"newton_schulz_stack_sharded/iter_{i}/gram"):
            A = jnp.einsum("...ik,...jk->...ij", X, X, out_sharding=X_out_sharding)
        with jax.named_scope(f"newton_schulz_stack_sharded/iter_{i}/polynomial"):
            B = b * A + c * jnp.einsum("...ik,...kj->...ij", A, A, out_sharding=X_out_sharding)
        with jax.named_scope(f"newton_schulz_stack_sharded/iter_{i}/apply"):
            X = a * X + jnp.einsum("...ik,...kj->...ij", B, X, out_sharding=X_out_sharding)

    if transpose:
        with jax.named_scope("newton_schulz_stack_sharded/restore_transpose"):
            X = jnp.swapaxes(X, -1, -2)

    if padded_stack_size != original_stack_size:
        with jax.named_scope("newton_schulz_stack_sharded/slice_padded_stack_axis"):
            if has_target_sharding:
                X = reshard(X, _with_target_spec(target_pspec, PartitionSpec(None, None, None)))
            X = X[:original_stack_size]

    return X


@jax.named_call
def _zeropower_via_newtonschulz_grouped_4d_sharded(
    X: jax.Array,
    steps: int = 5,
    eps: float = 1e-7,
    coefficient_type: CoefficientType = "quintic",
    target_pspec: Grouped4DTarget = None,
) -> jax.Array:
    """Run Newton-Schulz on `[group, expert, fan_in, fan_out]` without flattening batch axes."""
    assert X.ndim == 4

    coeffs = NEWTON_SCHULZ_COEFFICIENTS[coefficient_type]
    if target_pspec is None:
        target_pspec = _grouped_4d_stack_target(X)
    target_pspec = _canonical_target_for_array(X, target_pspec)
    has_target_sharding = target_pspec is not None
    if has_target_sharding:
        with jax.named_scope("newton_schulz_grouped_4d/reshard_stack"):
            X = reshard(X, target_pspec)
            _assert_grouped_4d_batch_sharded(X, target_pspec, "Muon 4D NS input")

    with jax.named_scope("newton_schulz_grouped_4d/normalize_input"):
        X = X / (jnp.linalg.norm(X, axis=(-2, -1), keepdims=True) + eps)

    transpose = False
    if X.shape[-2] > X.shape[-1]:
        with jax.named_scope("newton_schulz_grouped_4d/transpose_tall_matrices"):
            X = jnp.swapaxes(X, -1, -2)
        transpose = True

    if has_target_sharding:
        with jax.named_scope("newton_schulz_grouped_4d/assert_resharded_stack"):
            X = reshard(X, target_pspec)
            _assert_grouped_4d_batch_sharded(X, target_pspec, "Muon 4D NS dot input")

    X_out_sharding = target_pspec if has_target_sharding else None
    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        with jax.named_scope(f"newton_schulz_grouped_4d/iter_{i}/gram"):
            A = jnp.einsum("...ik,...jk->...ij", X, X, out_sharding=X_out_sharding)
        with jax.named_scope(f"newton_schulz_grouped_4d/iter_{i}/polynomial"):
            B = b * A + c * jnp.einsum("...ik,...kj->...ij", A, A, out_sharding=X_out_sharding)
        with jax.named_scope(f"newton_schulz_grouped_4d/iter_{i}/apply"):
            X = a * X + jnp.einsum("...ik,...kj->...ij", B, X, out_sharding=X_out_sharding)

    if transpose:
        with jax.named_scope("newton_schulz_grouped_4d/restore_transpose"):
            X = jnp.swapaxes(X, -1, -2)

    return X
