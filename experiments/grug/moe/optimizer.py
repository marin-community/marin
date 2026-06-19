# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
import os
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from jax import lax, shard_map
from levanter.optim import OptimizerConfig
from levanter.optim.grugmuon import (
    DEFAULT_GROUPED_4D_GROUP_SIZE,
    DEFAULT_MAX_GROUPED_STACK_SIZE,
    ORTHOGONALIZATION_LAYOUTS,
    STACK_BATCH_SHARDED,
    _cast_for_ns_compute,
    _grouped_4d_stack_target,
    _grug_scale_with_muon,
    _ns_compute_dtype_from_name,
    _restore_ns_compute_dtype,
    _target_sharding,
    _target_spec,
    _with_target_spec,
    _zeropower_via_newtonschulz_grouped_4d_sharded,
)
from levanter.optim.util import CoefficientType
from levanter.utils.jax_utils import leaf_key_paths

from experiments.grug.moe.adamh import scale_by_adamh
from experiments.grug.moe.optimizer_sharding import assert_update_sharding_matches_params, target_named_sharding

Expert3DOptimizer = Literal["muonh", "adamh", "grouped_muonh"]
VALID_EXPERT_3D_OPTIMIZERS: tuple[Expert3DOptimizer, ...] = ("muonh", "adamh", "grouped_muonh")
Ordinary2DOptimizer = Literal["muonh", "adamh", "adam", "sgd"]
VALID_ORDINARY_2D_OPTIMIZERS: tuple[Ordinary2DOptimizer, ...] = ("muonh", "adamh", "adam", "sgd")
MayOptimizer = Literal["muonh", "sgd"]
VALID_MAY_OPTIMIZERS: tuple[MayOptimizer, ...] = ("muonh", "sgd")
REPLICA_DCN_AXIS = "replica_dcn"
EXPERT_AXIS = "expert"
MATCH_OPTIMIZER_SHARDING_ENV = "MAY_MATCH_OPTIMIZER_SHARDING"
GROUPED_MUONH_EXPERT_PATH = ".mlp.expert_mlp.w_"


def _match_optimizer_sharding_enabled() -> bool:
    return os.environ.get(MATCH_OPTIMIZER_SHARDING_ENV, "true").lower() != "false"


def _uses_adamh_baseline_adam_group(path_lower: str) -> bool:
    # Use endswith for attn_gate so attn_gated_norm weights do not match.
    return (
        "token_embed" in path_lower
        or "router_bias" in path_lower
        or path_lower.endswith(".attn_gate")
        or ".router" in path_lower
    )


def _target_named_sharding(array) -> jax.sharding.NamedSharding | None:
    return target_named_sharding(array)


def _expert_momentum_sharding(array) -> jax.sharding.NamedSharding | None:
    if array is None or not hasattr(array, "shape") or array.ndim != 3:
        return None

    target_sharding = _target_named_sharding(array)
    if target_sharding is None:
        return None

    mesh = target_sharding.mesh
    if mesh.shape.get(REPLICA_DCN_AXIS, 1) <= 1:
        return None

    spec = target_sharding.spec
    if len(spec) != 3 or spec[0] is None:
        return None

    stack_axis = spec[0] if isinstance(spec[0], tuple) else (spec[0],)
    if EXPERT_AXIS not in stack_axis or REPLICA_DCN_AXIS in stack_axis:
        return None

    sharded_stack_axis = (REPLICA_DCN_AXIS, *stack_axis)
    stack_axis_size = math.prod(int(mesh.shape[name]) for name in sharded_stack_axis if name in mesh.shape)
    if array.shape[0] % stack_axis_size != 0:
        return None

    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharded_stack_axis, *spec[1:]))


def _match_named_update_sharding() -> optax.GradientTransformation:
    """Restore named mesh sharding on updates before applying them."""

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            return updates, state
        if not _match_optimizer_sharding_enabled():
            return updates, state

        def match_sharding(update, param):
            if update is None:
                return None
            target_sharding = _target_named_sharding(param)
            if target_sharding is None:
                return update
            return jax.sharding.reshard(update, target_sharding)

        updates = jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def _with_named_update_scope(name: str, transform: optax.GradientTransformation) -> optax.GradientTransformation:
    """Annotate an Optax transform update with a stable profiler region."""

    def init_fn(params):
        return transform.init(params)

    def update_fn(updates, state, params=None):
        with jax.named_scope(name):
            return transform.update(updates, state, params)

    return optax.GradientTransformation(init_fn, update_fn)


def _match_named_sharding_to_params(updates, params):
    if not _match_optimizer_sharding_enabled():
        return updates

    def match_sharding(update, param):
        if update is None:
            return None
        target_sharding = _target_named_sharding(param)
        if target_sharding is None:
            return update
        return jax.sharding.reshard(update, target_sharding)

    return jax.tree.map(match_sharding, updates, params, is_leaf=lambda x: x is None)


def _match_named_sharding_to_updates(params, updates):
    if not _match_optimizer_sharding_enabled():
        return params

    def match_sharding(param, update):
        if update is None or not hasattr(param, "shape"):
            return param
        target_sharding = _target_named_sharding(update)
        if target_sharding is None:
            return param
        return jax.sharding.reshard(param, target_sharding)

    return jax.tree.map(match_sharding, params, updates, is_leaf=lambda x: x is None)


def _scale_invariant_hyperball_updates(
    params,
    direction_updates,
    learning_rate: float,
    *,
    label: str = "scale-invariant hyperball",
):
    with jax.named_scope("muonh/hyperball/match_direction_sharding"):
        direction_updates = _match_named_sharding_to_params(direction_updates, params)
    assert_update_sharding_matches_params(direction_updates, params, f"{label} direction_updates after sharding match")

    def scale_invariant_update(param, update):
        if update is None:
            return None
        if not hasattr(param, "ndim"):
            return update
        if param.ndim == 2:
            with jax.named_scope("muonh/hyperball/matrix_norms"):
                param_norm = jnp.linalg.norm(param)
                update_norm = jnp.linalg.norm(update)
                step_scale = learning_rate * param_norm / jnp.maximum(update_norm, 1e-10)
                dot = jnp.sum(param * update)
            with jax.named_scope("muonh/hyperball/matrix_projection"):
                new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * update_norm**2
                new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
                rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
                return (rescale - 1) * param - rescale * step_scale * update

        axes = tuple(range(1, param.ndim))
        with jax.named_scope("muonh/hyperball/expert_stack_norms"):
            param_norm = jnp.sqrt(jnp.sum(jnp.square(param), axis=axes, keepdims=True))
            update_norm = jnp.sqrt(jnp.sum(jnp.square(update), axis=axes, keepdims=True))
            step_scale = learning_rate * param_norm / jnp.maximum(update_norm, 1e-10)
            dot = jnp.sum(param * update, axis=axes, keepdims=True)
        with jax.named_scope("muonh/hyperball/expert_stack_projection"):
            new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * update_norm**2
            new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
            rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
            return (rescale - 1) * param - rescale * step_scale * update

    with jax.named_scope("muonh/hyperball/project_updates"):
        hyperball_updates = jax.tree.map(
            scale_invariant_update,
            params,
            direction_updates,
            is_leaf=lambda x: x is None,
        )
    assert_update_sharding_matches_params(hyperball_updates, params, f"{label} updates after hyperball")
    return hyperball_updates


def _grouped_muonh_leaf_name(path) -> str:
    path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
    return path_str.rsplit(".", maxsplit=1)[-1]


def _is_grouped_muonh_expert_leaf(path, leaf) -> bool:
    if not hasattr(leaf, "ndim") or leaf.ndim != 3:
        return False
    path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
    return GROUPED_MUONH_EXPERT_PATH in path_str.lower()


def _grouped_muonh_stack_axis_size(sample_param) -> int:
    sharding = _target_sharding(sample_param)
    if isinstance(sharding, jax.sharding.NamedSharding):
        mesh_shape = sharding.mesh.shape
    else:
        mesh = jax.sharding.get_abstract_mesh()
        if mesh.empty:
            return 1
        mesh_shape = mesh.shape

    candidate_axes = tuple(axis for axis in (REPLICA_DCN_AXIS, "data") if int(mesh_shape.get(axis, 1)) > 1)
    if not candidate_axes:
        return 1
    return math.prod(int(mesh_shape[axis]) for axis in candidate_axes)


def _mesh_axis_size(mesh, axis_name: str) -> int:
    return int(mesh.shape.get(axis_name, 1))


def _live_group_axis(mesh, group_axis):
    if isinstance(group_axis, tuple):
        live_axes = tuple(axis for axis in group_axis if _mesh_axis_size(mesh, axis) > 1)
        if len(live_axes) > 1:
            return live_axes
        if live_axes:
            return live_axes[0]
        return None
    if group_axis is not None and _mesh_axis_size(mesh, group_axis) > 1:
        return group_axis
    return None


def _axis_spec_contains(axis_spec, axis_name: str) -> bool:
    if isinstance(axis_spec, tuple):
        return axis_name in axis_spec
    return axis_spec == axis_name


def _grouped_muonh_shape_dtype_struct(shape, dtype, sample_param):
    sharding = _target_sharding(sample_param)
    if isinstance(sharding, jax.sharding.NamedSharding) and len(sharding.spec) == 3:
        stack_sharding = jax.sharding.NamedSharding(
            sharding.mesh,
            jax.sharding.PartitionSpec(None, *sharding.spec),
        )
        return jax.ShapeDtypeStruct(shape, dtype, sharding=stack_sharding)
    return jax.ShapeDtypeStruct(shape, dtype)


def _grouped_muonh_chunk_size(sample_param, requested_group_size: int | None, max_grouped_stack_size: int) -> int:
    stack_axis_size = _grouped_muonh_stack_axis_size(sample_param)
    if requested_group_size is None:
        requested_group_size = stack_axis_size if stack_axis_size > 1 else DEFAULT_GROUPED_4D_GROUP_SIZE
    if requested_group_size < 1:
        raise ValueError(f"expert_grouped_muonh_group_size={requested_group_size} must be positive")
    return max(1, min(requested_group_size, max_grouped_stack_size))


def _pad_grouped_muonh_stack(stacked, padded_size: int):
    if padded_size == stacked.shape[0]:
        return stacked
    pad_width = ((0, padded_size - stacked.shape[0]), (0, 0), (0, 0), (0, 0))
    return jnp.pad(stacked, pad_width)


def _restore_grouped_muonh_for_split(grouped_update, target, valid_size: int):
    target_spec = _target_spec(target)
    if target_spec is not None and target_spec[0] is not None:
        grouped_update = jax.sharding.reshard(
            grouped_update,
            _with_target_spec(target, jax.sharding.PartitionSpec(None, target_spec[1], None, None)),
        )
    if grouped_update.shape[0] != valid_size:
        grouped_update = grouped_update[:valid_size]
    return grouped_update


def _restore_grouped_muonh_for_split_explicit(grouped_update, target, valid_size: int, sample_param):
    target_spec = _target_spec(target)
    param_sharding = _target_named_sharding(sample_param)
    if (
        grouped_update.ndim != 4
        or target_spec is None
        or target_spec[0] is None
        or not isinstance(param_sharding, jax.sharding.NamedSharding)
        or len(param_sharding.spec) != 3
    ):
        return _restore_grouped_muonh_for_split(grouped_update, target, valid_size)

    mesh = param_sharding.mesh
    group_axis = _live_group_axis(mesh, target_spec[0])
    input_spec = jax.sharding.PartitionSpec(group_axis, target_spec[1], None, None)
    output_spec = jax.sharding.PartitionSpec(None, *param_sharding.spec)
    data_axis = None
    for axis_index, axis_spec in enumerate(param_sharding.spec, start=1):
        if _axis_spec_contains(axis_spec, "data"):
            data_axis = axis_index
            break

    data_axis_size = _mesh_axis_size(mesh, "data")
    if data_axis is not None and grouped_update.shape[data_axis] % data_axis_size != 0:
        return _restore_grouped_muonh_for_split(grouped_update, target, valid_size)

    def restore_group(local_update):
        if group_axis is None:
            gathered = local_update
        else:
            gathered = lax.all_gather(local_update, axis_name=group_axis, axis=0, tiled=True)
        if data_axis is None or data_axis_size <= 1:
            return gathered
        data_index = lax.axis_index("data")
        local_axis_size = gathered.shape[data_axis] // data_axis_size
        start = data_index * local_axis_size
        return lax.dynamic_slice_in_dim(gathered, start, local_axis_size, axis=data_axis)

    grouped_update = shard_map(
        restore_group,
        mesh=mesh,
        in_specs=input_spec,
        out_specs=output_spec,
        check_vma=False,
    )(grouped_update)
    if grouped_update.shape[0] != valid_size:
        grouped_update = grouped_update[:valid_size]
    return grouped_update


def _restore_param_sharding(update, param):
    target_sharding = _target_named_sharding(param)
    if target_sharding is None:
        return update
    return jax.sharding.reshard(update, target_sharding)


def _scale_grouped_muonh_direction(direction):
    fan_in, fan_out = direction.shape[-2:]
    scale = jnp.sqrt(jnp.maximum(1, fan_out / fan_in))
    return direction * scale


def _grouped_muonh_hyperball_update(stacked_params, direction, learning_rate):
    axes = (-2, -1)
    with jax.named_scope("grouped_muonh/hyperball/norms"):
        param_norm = jnp.sqrt(jnp.sum(jnp.square(stacked_params), axis=axes, keepdims=True))
        update_norm = jnp.sqrt(jnp.sum(jnp.square(direction), axis=axes, keepdims=True))
        step_scale = learning_rate * param_norm / jnp.maximum(update_norm, 1e-10)
        dot = jnp.sum(stacked_params * direction, axis=axes, keepdims=True)
    with jax.named_scope("grouped_muonh/hyperball/project"):
        new_param_norm_sq = param_norm**2 - 2 * step_scale * dot + step_scale**2 * update_norm**2
        new_param_norm = jnp.sqrt(jnp.maximum(new_param_norm_sq, 1e-30))
        rescale = param_norm / jnp.maximum(new_param_norm, 1e-10)
        return (rescale - 1) * stacked_params - rescale * step_scale * direction


def _grouped_expert_muonh_updates(
    params,
    updates,
    learning_rate: float,
    *,
    steps: int,
    muon_eps: float,
    coefficient_type: CoefficientType,
    max_grouped_stack_size: int,
    ns_compute_dtype: str,
    expert_grouped_muonh_group_size: int | None,
):
    update_leaves, treedef = jax.tree.flatten(updates, is_leaf=lambda x: x is None)
    param_leaves, param_treedef = jax.tree.flatten(params, is_leaf=lambda x: x is None)
    path_leaves, path_treedef = jax.tree.flatten(leaf_key_paths(params), is_leaf=lambda x: x is None)
    if treedef != param_treedef or treedef != path_treedef:
        raise ValueError("Grouped MuonH requires updates, params, and paths to have matching tree structure")

    output_leaves = [None] * len(update_leaves)
    groups: dict[tuple[str, tuple[int, ...], str], list[tuple[int, object, object]]] = {}
    for index, (update, param, path) in enumerate(zip(update_leaves, param_leaves, path_leaves, strict=True)):
        if update is None:
            output_leaves[index] = None
            continue
        if _is_grouped_muonh_expert_leaf(path, param):
            key = (_grouped_muonh_leaf_name(path), tuple(param.shape), str(param.dtype))
            groups.setdefault(key, []).append((index, update, param))
            continue
        output_leaves[index] = update

    for entries in groups.values():
        chunk_size = _grouped_muonh_chunk_size(
            entries[0][2],
            expert_grouped_muonh_group_size,
            max_grouped_stack_size,
        )
        for chunk_start in range(0, len(entries), chunk_size):
            entry_chunk = entries[chunk_start : chunk_start + chunk_size]
            valid_size = len(entry_chunk)
            stack_axis_size = _grouped_muonh_stack_axis_size(entry_chunk[0][2])
            padded_size = math.ceil(valid_size / stack_axis_size) * stack_axis_size

            with jax.named_scope("grouped_muonh/stack_expert_updates"):
                stacked_updates = jnp.stack([update for _, update, _ in entry_chunk], axis=0)
                stacked_params = jnp.stack([param for _, _, param in entry_chunk], axis=0)
                stacked_updates = _pad_grouped_muonh_stack(stacked_updates, padded_size)
                stacked_params = _pad_grouped_muonh_stack(stacked_params, padded_size)

            target = _grouped_4d_stack_target(
                _grouped_muonh_shape_dtype_struct(stacked_updates.shape, stacked_updates.dtype, entry_chunk[0][2])
            )
            if target is not None:
                with jax.named_scope("grouped_muonh/reshard_grouped_stack"):
                    stacked_updates = jax.sharding.reshard(stacked_updates, target)
                    stacked_params = jax.sharding.reshard(stacked_params, target)

            with jax.named_scope("grouped_muonh/newton_schulz"):
                stacked_updates, original_dtype = _cast_for_ns_compute(stacked_updates, ns_compute_dtype)
                direction = _zeropower_via_newtonschulz_grouped_4d_sharded(
                    stacked_updates,
                    steps,
                    muon_eps,
                    coefficient_type,
                    target,
                )
                direction = _restore_ns_compute_dtype(direction, original_dtype)
                direction = _scale_grouped_muonh_direction(direction)

            grouped_update = _grouped_muonh_hyperball_update(stacked_params, direction, learning_rate)
            grouped_update = _restore_grouped_muonh_for_split_explicit(
                grouped_update,
                target,
                valid_size,
                entry_chunk[0][2],
            )
            update_parts = [
                jnp.squeeze(update_part, axis=0) for update_part in jnp.split(grouped_update, valid_size, axis=0)
            ]

            for (index, _, param), update_part in zip(entry_chunk, update_parts, strict=True):
                output_leaves[index] = _restore_param_sharding(update_part, param)

    return jax.tree.unflatten(treedef, output_leaves)


def scale_with_grouped_expert_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
    max_grouped_stack_size: int = DEFAULT_MAX_GROUPED_STACK_SIZE,
    ns_compute_dtype: str = "input",
    expert_grouped_muonh_group_size: int | None = None,
) -> optax.GradientTransformation:
    """Expert-only MuonH transform that computes NS on grouped 4D stacks and returns FSDP updates."""

    _ns_compute_dtype_from_name(ns_compute_dtype, jnp.float32)
    if max_grouped_stack_size < 1:
        raise ValueError(f"max_grouped_stack_size={max_grouped_stack_size} must be positive")
    if expert_grouped_muonh_group_size is not None and expert_grouped_muonh_group_size < 1:
        raise ValueError(f"expert_grouped_muonh_group_size={expert_grouped_muonh_group_size} must be positive")

    def init_fn(params):
        return optax.TraceState(trace=jax.tree.map(jnp.zeros_like, params))

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grouped_expert_muonh requires params")

        with jax.named_scope("grouped_muonh/update_momentum_buffer"):
            trace = jax.tree.map(
                lambda trace_leaf, update: None if update is None else momentum * trace_leaf + update,
                state.trace,
                updates,
                is_leaf=lambda x: x is None,
            )
        if nesterov:
            with jax.named_scope("grouped_muonh/nesterov_update"):
                direction_inputs = jax.tree.map(
                    lambda trace_leaf, update: None if update is None else momentum * trace_leaf + update,
                    trace,
                    updates,
                    is_leaf=lambda x: x is None,
                )
        else:
            direction_inputs = trace

        with jax.named_scope("grouped_muonh/transform_updates"):
            muonh_updates = _grouped_expert_muonh_updates(
                params,
                direction_inputs,
                learning_rate,
                steps=steps,
                muon_eps=muon_eps,
                coefficient_type=coefficient_type,
                max_grouped_stack_size=max_grouped_stack_size,
                ns_compute_dtype=ns_compute_dtype,
                expert_grouped_muonh_group_size=expert_grouped_muonh_group_size,
            )
        assert_update_sharding_matches_params(muonh_updates, params, "Grouped MuonH updates")
        return muonh_updates, optax.TraceState(trace=trace)

    return optax.GradientTransformation(init_fn, update_fn)


def scale_with_grug_muonh(
    momentum: float = 0.95,
    nesterov: bool = True,
    steps: int = 5,
    muon_eps: float = 1e-8,
    learning_rate: float = 0.02,
    coefficient_type: CoefficientType = "quintic",
    momentum_sharding_fn=None,
    orthogonalization_layout: str = STACK_BATCH_SHARDED,
    max_grouped_stack_size: int = DEFAULT_MAX_GROUPED_STACK_SIZE,
    ns_compute_dtype: str = "input",
) -> optax.GradientTransformation:
    """MuonH transform for raw Grug arrays with matrix-shaped trailing dims."""
    muon_transform = _grug_scale_with_muon(
        momentum=momentum,
        nesterov=nesterov,
        steps=steps,
        muon_eps=muon_eps,
        use_kimi_scaling=False,
        coefficient_type=coefficient_type,
        momentum_sharding_fn=momentum_sharding_fn,
        orthogonalization_layout=orthogonalization_layout,
        max_grouped_stack_size=max_grouped_stack_size,
        ns_compute_dtype=ns_compute_dtype,
    )

    def init_fn(params):
        return muon_transform.init(params)

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("scale_with_grug_muonh requires params for norm-preserving updates")

        with jax.named_scope("muonh/direction_update"):
            muon_updates, next_state = muon_transform.update(updates, state, params)
        assert_update_sharding_matches_params(muon_updates, params, "MuonH direction_updates before hyperball")
        with jax.named_scope("muonh/hyperball"):
            muonh_updates = _scale_invariant_hyperball_updates(params, muon_updates, learning_rate, label="MuonH")
        assert_update_sharding_matches_params(muonh_updates, params, "MuonH updates after hyperball")
        return muonh_updates, next_state

    return optax.GradientTransformation(init_fn, update_fn)


@OptimizerConfig.register_subclass("grug_moe_adamh_v2")
@dataclass(frozen=True)
class GrugMoeAdamHConfig(OptimizerConfig):
    """AdamH for Grug MoE. Four optimizer groups, no flags.

    - adamh: attention weights, dense MLP weights (2D matrices)
    - adamh_expert: expert MLP weights (mlp.expert_mlp.w_gate_up,
      mlp.expert_mlp.w_down, shared.w_*)
    - adam: norms, biases, router, embeddings, attention gates (1D / small params)
    """

    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    adam_lr: float = 6e-4
    expert_lr: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)
        expert_lr_val = self.expert_lr if self.expert_lr is not None else self.learning_rate
        expert_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=expert_lr_val)

        def optimizer(learning_rate, adam_lr, expert_lr):
            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_expert_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, expert_lr))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "adamh": _with_named_update_scope("optimizer/group/adamh", adamh_transform()),
                    "adamh_expert": _with_named_update_scope(
                        "optimizer/group/adamh_expert",
                        adamh_expert_transform(),
                    ),
                    "adam": _with_named_update_scope("optimizer/group/adam", adam_transform()),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
            expert_lr=expert_lr_schedule,
        )

    def create_mask(self, params):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if _uses_adamh_baseline_adam_group(path_lower):
                return "adam"
            if ".mlp.expert_mlp.w_" in path_lower or ".mlp.w_" in path_lower or ".shared.w_" in path_lower:
                return "adamh_expert"
            if hasattr(param, "ndim") and param.ndim >= 2:
                return "adamh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_muonh_v1")
@dataclass(frozen=True)
class GrugMoeMuonHConfig(OptimizerConfig):
    """May Recipe MuonH optimizer for raw Grug MoE arrays.

    - ``muonh``: attention, expert/shared MLP matrices, and GatedNorm matrices
    - ``adamh``: lm head / output projection
    - ``adam``: token embeddings, router leaves, attention gates, and vector norms
    """

    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float | None = None
    coefficient_type: CoefficientType = "quintic"
    expert_3d_optimizer: Expert3DOptimizer = "muonh"
    ordinary_2d_optimizer: Ordinary2DOptimizer = "muonh"
    orthogonalization_layout: str = STACK_BATCH_SHARDED
    max_grouped_stack_size: int = DEFAULT_MAX_GROUPED_STACK_SIZE
    ns_compute_dtype: str = "input"
    expert_grouped_muonh_group_size: int | None = None

    def build(self, num_train_steps):
        if self.expert_3d_optimizer not in VALID_EXPERT_3D_OPTIMIZERS:
            valid = ", ".join(VALID_EXPERT_3D_OPTIMIZERS)
            raise ValueError(f"expert_3d_optimizer={self.expert_3d_optimizer!r} must be one of {valid}")
        if self.ordinary_2d_optimizer not in VALID_ORDINARY_2D_OPTIMIZERS:
            valid = ", ".join(VALID_ORDINARY_2D_OPTIMIZERS)
            raise ValueError(f"ordinary_2d_optimizer={self.ordinary_2d_optimizer!r} must be one of {valid}")
        if self.orthogonalization_layout not in ORTHOGONALIZATION_LAYOUTS:
            valid = ", ".join(ORTHOGONALIZATION_LAYOUTS)
            raise ValueError(f"orthogonalization_layout={self.orthogonalization_layout!r} must be one of {valid}")
        if self.max_grouped_stack_size < 1:
            raise ValueError(f"max_grouped_stack_size={self.max_grouped_stack_size} must be positive")
        if self.expert_grouped_muonh_group_size is not None and self.expert_grouped_muonh_group_size < 1:
            raise ValueError(f"expert_grouped_muonh_group_size={self.expert_grouped_muonh_group_size} must be positive")
        _ns_compute_dtype_from_name(self.ns_compute_dtype, jnp.float32)

        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def muonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grug_muonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                        coefficient_type=self.coefficient_type,
                        momentum_sharding_fn=_expert_momentum_sharding,
                        orthogonalization_layout=self.orthogonalization_layout,
                        max_grouped_stack_size=self.max_grouped_stack_size,
                        ns_compute_dtype=self.ns_compute_dtype,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def grouped_muonh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_grouped_expert_muonh(
                        momentum=self.momentum,
                        nesterov=self.nesterov,
                        steps=self.backend_steps,
                        muon_eps=self.muon_epsilon,
                        learning_rate=learning_rate,
                        coefficient_type=self.coefficient_type,
                        max_grouped_stack_size=self.max_grouped_stack_size,
                        ns_compute_dtype=self.ns_compute_dtype,
                        expert_grouped_muonh_group_size=self.expert_grouped_muonh_group_size,
                    )
                )
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adamh_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, lr))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def adam_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-lr))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            def sgd_transform_at(lr):
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale(-lr))
                components.append(_match_named_update_sharding())
                return optax.chain(*components)

            return optax.multi_transform(
                {
                    "muonh": _with_named_update_scope("optimizer/group/muonh", muonh_transform()),
                    "grouped_muonh": _with_named_update_scope(
                        "optimizer/group/grouped_muonh",
                        grouped_muonh_transform(),
                    ),
                    "adamh": _with_named_update_scope("optimizer/group/adamh", adamh_transform_at(learning_rate)),
                    "adam": _with_named_update_scope("optimizer/group/adam", adam_transform_at(adam_lr)),
                    "sgd": _with_named_update_scope("optimizer/group/sgd", sgd_transform_at(learning_rate)),
                },
                self.create_mask,
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=learning_rate_schedule,
            adam_lr=adam_lr_schedule,
        )

    def create_mask(self, params):
        if self.expert_3d_optimizer not in VALID_EXPERT_3D_OPTIMIZERS:
            valid = ", ".join(VALID_EXPERT_3D_OPTIMIZERS)
            raise ValueError(f"expert_3d_optimizer={self.expert_3d_optimizer!r} must be one of {valid}")
        if self.ordinary_2d_optimizer not in VALID_ORDINARY_2D_OPTIMIZERS:
            valid = ", ".join(VALID_ORDINARY_2D_OPTIMIZERS)
            raise ValueError(f"ordinary_2d_optimizer={self.ordinary_2d_optimizer!r} must be one of {valid}")

        paths = leaf_key_paths(params)
        expert_3d_optimizer = self.expert_3d_optimizer
        ordinary_2d_optimizer = self.ordinary_2d_optimizer

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            path_lower = path_str.lower()
            if _uses_adamh_baseline_adam_group(path_lower):
                return "adam"
            if "output_proj" in path_lower or "lm_head" in path_lower:
                return "adamh"
            if ".mlp.expert_mlp.w_" in path_lower and hasattr(param, "ndim") and param.ndim == 3:
                return expert_3d_optimizer
            if hasattr(param, "ndim") and param.ndim == 2:
                return ordinary_2d_optimizer
            if hasattr(param, "ndim") and param.ndim == 3:
                return "muonh"
            return "adam"

        return jax.tree.map(mask_fn, params, paths)


@OptimizerConfig.register_subclass("grug_moe_sgd_v1")
@dataclass(frozen=True)
class GrugMoeSgdConfig(OptimizerConfig):
    """Stateless SGD for Grug MoE throughput diagnostics."""

    weight_decay: float = 0.0
    max_grad_norm: float | None = None

    def build(self, num_train_steps):
        learning_rate_schedule = self.lr_scheduler(num_train_steps)

        def optimizer(learning_rate):
            components = []
            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
            components.append(optax.scale(-learning_rate))
            components.append(_match_named_update_sharding())
            return optax.chain(*components)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule)


__all__ = [
    "VALID_EXPERT_3D_OPTIMIZERS",
    "VALID_MAY_OPTIMIZERS",
    "Expert3DOptimizer",
    "GrugMoeAdamHConfig",
    "GrugMoeMuonHConfig",
    "GrugMoeSgdConfig",
    "MayOptimizer",
    "scale_with_grouped_expert_muonh",
    "scale_with_grug_muonh",
]
