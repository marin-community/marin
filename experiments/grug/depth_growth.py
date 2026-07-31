# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-state transformation for shallow-to-deep Grug training."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol, TypeVar, cast

import jax
import jax.numpy as jnp
from jax.tree_util import DictKey, FlattenedIndexKey, GetAttrKey, SequenceKey
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint


class _DepthGrowthState(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[object]]]

    step: jax.Array
    params: object
    opt_state: object
    ema_params: object | None
    pending_qb_betas: jax.Array


StateT = TypeVar("StateT", bound=_DepthGrowthState)
PathPart = tuple[str, object]
TreePath = tuple[PathPart, ...]


@dataclass(frozen=True)
class DepthGrowthConfig:
    """Shape and resume invariants for a shallow-to-deep transition."""

    source_layers: int
    target_layers: int
    expected_step: int
    expected_data_offset: int

    def __post_init__(self) -> None:
        if self.source_layers < 1:
            raise ValueError(f"source_layers must be positive, got {self.source_layers}")
        if self.target_layers <= self.source_layers:
            raise ValueError(
                f"target_layers must exceed source_layers, got {self.source_layers} -> {self.target_layers}"
            )
        if self.target_layers % self.source_layers != 0:
            raise ValueError(
                f"target_layers must be divisible by source_layers, got {self.source_layers} -> {self.target_layers}"
            )
        if self.expected_step < 1:
            raise ValueError(f"expected_step must be positive, got {self.expected_step}")
        if self.expected_data_offset < 0:
            raise ValueError(f"expected_data_offset must be non-negative, got {self.expected_data_offset}")


@dataclass(frozen=True)
class DepthGrowthReport:
    """Structured summary of a completed state transformation."""

    step: int
    data_offset: int
    copied_parameter_leaves: int
    reset_optimizer_leaves: int
    preserved_optimizer_leaves: int


def grow_grug_depth_state(
    source_state: StateT,
    fresh_target_state: StateT,
    config: DepthGrowthConfig,
) -> tuple[StateT, DepthGrowthReport]:
    """Copy a shallow Grug state into a fresh deeper state.

    Model parameters, EMA parameters, and QB router state repeat the complete source
    stack. Optimizer schedule counters and non-block parameter state are inherited.
    Block optimizer buffers stay at their freshly initialized target values so every
    copied block begins without inherited momentum.
    """

    source_step = int(source_state.step)
    if source_step != config.expected_step:
        raise ValueError(f"source checkpoint is at step {source_step}, expected {config.expected_step}")
    if int(fresh_target_state.step) != 0:
        raise ValueError("depth growth requires a fresh target state at step 0")
    _validate_optimizer_schedule_count(source_state.opt_state, source_step)

    params, copied_parameter_leaves = _grow_model_tree(source_state.params, fresh_target_state.params, config)
    ema_params = _grow_ema_tree(source_state.ema_params, fresh_target_state.ema_params, config)
    pending_qb_betas = _repeat_layer_array(
        source_state.pending_qb_betas,
        fresh_target_state.pending_qb_betas,
        config,
        path="pending_qb_betas",
    )
    opt_state, reset_optimizer_leaves, preserved_optimizer_leaves = _grow_optimizer_state(
        source_state.opt_state,
        fresh_target_state.opt_state,
    )

    grown_state = cast(
        StateT,
        dataclasses.replace(
            fresh_target_state,
            step=_copy_array(source_state.step, fresh_target_state.step, path="step"),
            params=params,
            opt_state=opt_state,
            ema_params=ema_params,
            pending_qb_betas=pending_qb_betas,
        ),
    )
    return grown_state, DepthGrowthReport(
        step=source_step,
        data_offset=config.expected_data_offset,
        copied_parameter_leaves=copied_parameter_leaves,
        reset_optimizer_leaves=reset_optimizer_leaves,
        preserved_optimizer_leaves=preserved_optimizer_leaves,
    )


def load_and_grow_grug_depth_state(
    source_state_exemplar: StateT,
    fresh_target_state: StateT,
    checkpoint_path: str,
    *,
    config: DepthGrowthConfig,
    mesh: jax.sharding.Mesh | None,
    allow_partial: bool = False,
    _load_fn: Callable[..., StateT] = load_checkpoint,
    _latest_checkpoint_fn: Callable[[str], str] = latest_checkpoint_path,
) -> tuple[StateT, DepthGrowthReport]:
    """Load the latest source-shaped full checkpoint and transform it to target depth."""

    concrete_checkpoint_path = _latest_checkpoint_fn(checkpoint_path)
    source_state = _load_fn(
        source_state_exemplar,
        concrete_checkpoint_path,
        axis_mapping=None,
        mesh=mesh,
        allow_partial=allow_partial,
    )
    return grow_grug_depth_state(source_state, fresh_target_state, config)


def validate_depth_growth_data_offset(
    config: DepthGrowthConfig,
    *,
    actual_data_offset: int,
) -> None:
    """Fail when the target batch schedule would resume at a different data cursor."""

    if actual_data_offset != config.expected_data_offset:
        raise ValueError(
            "target batch schedule changes the depth-growth data cursor: "
            f"step {config.expected_step} maps to {actual_data_offset}, expected {config.expected_data_offset}"
        )


def _grow_ema_tree(source: object | None, target: object | None, config: DepthGrowthConfig) -> object | None:
    if source is None and target is None:
        return None
    if source is None or target is None:
        raise ValueError("source and target must use the same EMA configuration")
    grown, _ = _grow_model_tree(source, target, config)
    return grown


def _grow_model_tree(source: object, target: object, config: DepthGrowthConfig) -> tuple[object, int]:
    source_leaves = _leaves_by_path(source)
    source_segment_lengths = _stacked_segment_lengths(source)
    target_segment_lengths = _stacked_segment_lengths(target)
    if source_segment_lengths and sum(source_segment_lengths) != config.source_layers:
        raise ValueError(
            f"source model segments sum to {sum(source_segment_lengths)}, expected {config.source_layers} layers"
        )
    if target_segment_lengths and sum(target_segment_lengths) != config.target_layers:
        raise ValueError(
            f"target model segments sum to {sum(target_segment_lengths)}, expected {config.target_layers} layers"
        )
    target_path_leaves, target_treedef = jax.tree_util.tree_flatten_with_path(target)
    grown_leaves: list[object] = []
    copied_parameter_leaves = 0

    for raw_target_path, target_leaf in target_path_leaves:
        target_path = _path_identity(raw_target_path)
        if _is_array_stacked_block_path(target_path):
            source_leaf, target_layer_offset = _source_stacked_leaf(
                source_leaves,
                target_path,
                source_segment_lengths=source_segment_lengths,
                target_segment_lengths=target_segment_lengths,
            )
            grown_leaves.append(
                _repeat_layer_slice(
                    source_leaf,
                    target_leaf,
                    config,
                    target_layer_offset=target_layer_offset,
                    path=_display_path(raw_target_path),
                )
            )
            copied_parameter_leaves += 1
            continue

        source_path = _source_unrolled_block_path(target_path, config.source_layers)
        source_leaf = _required_leaf(source_leaves, source_path)
        grown_leaves.append(_copy_leaf(source_leaf, target_leaf, path=_display_path(raw_target_path)))
        copied_parameter_leaves += 1

    return target_treedef.unflatten(grown_leaves), copied_parameter_leaves


def _grow_optimizer_state(source: object, target: object) -> tuple[object, int, int]:
    source_leaves = _leaves_by_path(source)
    target_path_leaves, target_treedef = jax.tree_util.tree_flatten_with_path(target)
    grown_leaves: list[object] = []
    reset_optimizer_leaves = 0
    preserved_optimizer_leaves = 0

    for raw_target_path, target_leaf in target_path_leaves:
        target_path = _path_identity(raw_target_path)
        if _is_block_path(target_path):
            grown_leaves.append(target_leaf)
            if _is_array(target_leaf):
                reset_optimizer_leaves += 1
            continue

        source_leaf = _required_leaf(source_leaves, target_path)
        grown_leaves.append(_copy_leaf(source_leaf, target_leaf, path=_display_path(raw_target_path)))
        if _is_array(target_leaf):
            preserved_optimizer_leaves += 1

    return target_treedef.unflatten(grown_leaves), reset_optimizer_leaves, preserved_optimizer_leaves


def _validate_optimizer_schedule_count(opt_state: object, expected_step: int) -> None:
    count_path: TreePath = (("attr", "count"),)
    count = _required_leaf(_leaves_by_path(opt_state), count_path)
    if not _is_array(count) or count.shape != ():
        raise ValueError("optimizer schedule count must be a scalar array")
    if int(count) != expected_step:
        raise ValueError(f"optimizer schedule is at step {int(count)}, expected {expected_step}")


def _repeat_layer_array(source: object, target: object, config: DepthGrowthConfig, *, path: str) -> jax.Array:
    return _repeat_layer_slice(source, target, config, target_layer_offset=0, path=path)


def _repeat_layer_slice(
    source: object,
    target: object,
    config: DepthGrowthConfig,
    *,
    target_layer_offset: int,
    path: str,
) -> jax.Array:
    if not _is_array(source) or not _is_array(target):
        raise ValueError(f"{path} must be an array in both source and target")
    if source.ndim == 0 or target.ndim == 0:
        raise ValueError(f"{path} must carry a leading layer dimension")
    if source.shape[0] != config.source_layers:
        raise ValueError(f"{path} source shape {source.shape} does not start with {config.source_layers} layers")
    if source.shape[1:] != target.shape[1:]:
        raise ValueError(f"{path} changes non-layer shape from {source.shape} to {target.shape}")
    if source.dtype != target.dtype:
        raise ValueError(f"{path} changes dtype from {source.dtype} to {target.dtype}")

    layer_indices = (jnp.arange(target.shape[0]) + target_layer_offset) % config.source_layers
    repeated = jnp.take(source, layer_indices, axis=0)
    return _put_with_target_sharding(repeated, target)


def _copy_leaf(source: object, target: object, *, path: str) -> object:
    if _is_array(source) or _is_array(target):
        if not _is_array(source) or not _is_array(target):
            raise ValueError(f"{path} changes between array and non-array state")
        return _copy_array(source, target, path=path)
    if type(source) is not type(target):
        raise ValueError(f"{path} changes state type from {type(source).__name__} to {type(target).__name__}")
    return source


def _copy_array(source: jax.Array, target: jax.Array, *, path: str) -> jax.Array:
    if source.shape != target.shape:
        raise ValueError(f"{path} changes shape from {source.shape} to {target.shape}")
    if source.dtype != target.dtype:
        raise ValueError(f"{path} changes dtype from {source.dtype} to {target.dtype}")
    return _put_with_target_sharding(source, target)


def _put_with_target_sharding(value: jax.Array, target: jax.Array) -> jax.Array:
    return jax.device_put(value, target.sharding)


def _leaves_by_path(tree: object) -> dict[TreePath, object]:
    result: dict[TreePath, object] = {}
    for raw_path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        path = _path_identity(raw_path)
        if path in result:
            raise ValueError(f"duplicate PyTree path {_display_path(raw_path)}")
        result[path] = leaf
    return result


def _required_leaf(leaves: dict[TreePath, object], path: TreePath) -> object:
    try:
        return leaves[path]
    except KeyError as exc:
        raise ValueError(f"source state is missing target path {_format_identity(path)}") from exc


def _source_unrolled_block_path(path: TreePath, source_layers: int) -> TreePath:
    parts = list(path)
    for index, part in enumerate(parts[:-1]):
        if part == ("attr", "blocks") and parts[index + 1][0] == "index":
            parts[index + 1] = ("index", int(parts[index + 1][1]) % source_layers)
            return tuple(parts)
    return path


def _source_stacked_leaf(
    source_leaves: dict[TreePath, object],
    target_path: TreePath,
    *,
    source_segment_lengths: tuple[int, ...],
    target_segment_lengths: tuple[int, ...],
) -> tuple[object, int]:
    segment_position = _segment_index_position(target_path)
    if segment_position is None:
        return _required_leaf(source_leaves, target_path), 0
    if not source_segment_lengths or not target_segment_lengths:
        raise ValueError("segmented block paths require segment metadata on source and target models")

    target_segment_index = int(target_path[segment_position][1])
    target_layer_offset = sum(target_segment_lengths[:target_segment_index])
    source_segment_leaves: list[jax.Array] = []
    for source_segment_index in range(len(source_segment_lengths)):
        source_path = list(target_path)
        source_path[segment_position] = ("index", source_segment_index)
        source_leaf = _required_leaf(source_leaves, tuple(source_path))
        if not _is_array(source_leaf):
            raise ValueError(f"segmented block leaf {_format_identity(tuple(source_path))} must be an array")
        source_segment_leaves.append(source_leaf)
    return jnp.concatenate(source_segment_leaves, axis=0), target_layer_offset


def _stacked_segment_lengths(tree: object) -> tuple[int, ...]:
    segments = getattr(tree, "stacked_block_segments", None)
    if segments is not None:
        return tuple(int(segment.num_layers) for segment in segments)
    stack = getattr(tree, "stacked_blocks", None)
    if stack is not None:
        return (int(stack.num_layers),)
    return ()


def _segment_index_position(path: TreePath) -> int | None:
    for index, part in enumerate(path[:-1]):
        if part == ("attr", "stacked_block_segments") and path[index + 1][0] == "index":
            return index + 1
    return None


def _is_array_stacked_block_path(path: TreePath) -> bool:
    has_stack = ("attr", "stacked_blocks") in path or ("attr", "stacked_block_segments") in path
    return has_stack and ("attr", "stacked") in path


def _is_block_path(path: TreePath) -> bool:
    return ("attr", "blocks") in path or ("attr", "stacked_blocks") in path or ("attr", "stacked_block_segments") in path


def _path_identity(path: tuple[object, ...]) -> TreePath:
    return tuple(_path_part(key) for key in path)


def _path_part(key: object) -> PathPart:
    if isinstance(key, GetAttrKey):
        return ("attr", key.name)
    if isinstance(key, SequenceKey):
        return ("index", key.idx)
    if isinstance(key, DictKey):
        return ("key", key.key)
    if isinstance(key, FlattenedIndexKey):
        return ("flat", key.key)
    return ("other", str(key))


def _display_path(path: tuple[object, ...]) -> str:
    return jax.tree_util.keystr(path) or "<root>"


def _format_identity(path: TreePath) -> str:
    return "/".join(f"{kind}:{value}" for kind, value in path) or "<root>"


def _is_array(value: object) -> bool:
    return isinstance(value, jax.Array)


__all__ = [
    "DepthGrowthConfig",
    "DepthGrowthReport",
    "grow_grug_depth_state",
    "load_and_grow_grug_depth_state",
    "validate_depth_growth_data_offset",
]
