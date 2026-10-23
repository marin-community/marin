# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import functools
from typing import Any, Callable, Generic, Protocol, Self, Sequence, TypeVar, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name

from levanter.layers.scan_checkpoint import ScanCheckpointPolicy, ScanCheckpointSpec

M = TypeVar("M", bound=eqx.Module)
CarryT = TypeVar("CarryT")
OutputT_co = TypeVar("OutputT_co", covariant=True)
AxisSpec = int | None
StateDict = dict[str, Any]
InAxesSpec = AxisSpec | tuple[Any, Any]


class ModuleInit(Protocol[M]):
    def __call__(self, *args, **kwargs) -> M: ...


@runtime_checkable
class _StateDictWriter(Protocol):
    def to_state_dict(self, prefix: str | None = None) -> StateDict: ...


@runtime_checkable
class _StateDictReader(Protocol):
    def from_state_dict(self, state_dict: StateDict, prefix: str | None = None) -> Self: ...


@runtime_checkable
class _StateDictKeyMap(Protocol):
    def _state_dict_key_map(self) -> dict[str, str | None]: ...


def _is_array_like(value: Any) -> bool:
    return isinstance(value, (jax.Array, np.ndarray, jax.ShapeDtypeStruct))


def _normalize_unroll(unroll: int | bool | None, num_layers: int) -> int | bool:
    if unroll is None:
        return 1
    if isinstance(unroll, bool):
        return unroll
    resolved = int(unroll)
    if resolved < 1:
        raise ValueError(f"unroll must be >= 1; got {resolved}.")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1; got {num_layers}.")
    return resolved


def _is_layer_batched_leaf(x: Any, num_layers: int) -> bool:
    return _is_array_like(x) and x.ndim > 0 and x.shape[0] == num_layers


def _slice_layer(x: Any, index: int, num_layers: int) -> Any:
    if _is_layer_batched_leaf(x, num_layers):
        return x[index]
    return x


class ArrayStacked(eqx.Module, Generic[M]):
    """Equinox layers stored as one pytree with a leading layer axis on arrays.

    Initialization vmaps inputs whose leading dimension equals ``num_layers``;
    other inputs are shared. ``fold`` and ``scan`` share extra arguments unless
    ``in_axes`` explicitly marks them as layer-batched. Python and static leaves
    remain shared, and parameter paths retain the ``stacked`` field.
    """

    stacked: M
    num_layers: int = eqx.field(static=True)
    gradient_checkpointing: ScanCheckpointPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        num_layers: int,
        module: type[M],
        *,
        gradient_checkpointing: ScanCheckpointSpec = False,
    ) -> ModuleInit["ArrayStacked[M]"]:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1; got {num_layers}.")
        gradient_checkpointing = ScanCheckpointPolicy._mk(gradient_checkpointing)

        @functools.wraps(module)
        def init_fn(*args, **kwargs):
            # Tracing each layer separately makes the initializer's HLO grow with depth, which
            # dominates XLA compile time for a deep stack.
            batched, shared = eqx.partition(
                (args, kwargs),
                lambda x: _is_layer_batched_leaf(x, num_layers),
            )

            def init_layer(layer_batched):
                layer_args, layer_kwargs = eqx.combine(layer_batched, shared)
                return module.init(*layer_args, **layer_kwargs)

            # `filter_vmap` over `jax.vmap`: a non-array leaf has to come back as the value it
            # was. `axis_size` covers arguments that carry no layer axis.
            stacked = eqx.filter_vmap(init_layer, axis_size=num_layers)(batched)
            return cls(stacked=stacked, num_layers=num_layers, gradient_checkpointing=gradient_checkpointing)

        return init_fn

    @property
    def _carry_ckpt_name(self) -> str:
        return f"ArrayStacked[{self.num_layers}].carry"

    @property
    def _input_ckpt_name(self) -> str:
        return f"ArrayStacked[{self.num_layers}].inputs"

    def scan(
        self,
        init,
        *extra_args,
        unroll: int | bool | None = None,
        in_axes: InAxesSpec = None,
        **extra_kwargs,
    ):
        def call_layer(layer: M, carry, *args, **kwargs):
            # eqx.Module defines __call__ on subclasses but not on the base, so M is not seen as callable.
            return layer(carry, *args, **kwargs)  # pyrefly: ignore[not-callable]

        return self.scan_via(call_layer, unroll=unroll, in_axes=in_axes)(init, *extra_args, **extra_kwargs)

    def fold(
        self,
        init,
        *args,
        unroll: int | bool | None = None,
        in_axes: InAxesSpec = None,
        **kwargs,
    ):
        def call_layer(layer: M, carry, *layer_args, **layer_kwargs):
            # eqx.Module defines __call__ on subclasses but not on the base, so M is not seen as callable.
            return layer(carry, *layer_args, **layer_kwargs)  # pyrefly: ignore[not-callable]

        return self.fold_via(call_layer, unroll=unroll, in_axes=in_axes)(init, *args, **kwargs)

    def fold_via(
        self,
        fn: Callable[..., CarryT],
        *,
        unroll: int | bool | None = None,
        in_axes: InAxesSpec = None,
    ) -> Callable[..., CarryT]:
        resolved_unroll = _normalize_unroll(unroll, self.num_layers)

        def do_fold(init: CarryT, *args, **kwargs) -> CarryT:
            args_axes, kwargs_axes = _resolve_in_axes(in_axes, args, kwargs)

            def body(carry: CarryT, layer_index):
                layer = self.get_layer(layer_index)
                layer_args = tuple(
                    _slice_with_axis(arg, axis, layer_index, self.num_layers, context="fold_via")
                    for arg, axis in zip(args, args_axes, strict=True)
                )
                layer_kwargs = {
                    key: _slice_with_axis(kwargs[key], axis, layer_index, self.num_layers, context=f"fold_via[{key}]")
                    for key, axis in kwargs_axes.items()
                }
                carry = _tree_checkpoint_name(carry, self._carry_ckpt_name)
                layer_args, layer_kwargs = _tree_checkpoint_name((layer_args, layer_kwargs), self._input_ckpt_name)
                carry = fn(layer, carry, *layer_args, **layer_kwargs)
                return carry, None

            checkpointed_body = self.gradient_checkpointing.checkpoint(
                self._carry_ckpt_name,
                self._input_ckpt_name,
                body,
            )
            carry, _ = _scan_layers(
                checkpointed_body,
                init,
                num_layers=self.num_layers,
                unroll=resolved_unroll,
                nested=self.gradient_checkpointing.nested,
            )
            return carry

        return do_fold

    def scan_via(
        self,
        fn: Callable[..., tuple[CarryT, OutputT_co]],
        *,
        unroll: int | bool | None = None,
        in_axes: InAxesSpec = None,
    ) -> Callable[..., tuple[CarryT, OutputT_co]]:
        resolved_unroll = _normalize_unroll(unroll, self.num_layers)

        def do_scan(init: CarryT, *args, **kwargs) -> tuple[CarryT, OutputT_co]:
            args_axes, kwargs_axes = _resolve_in_axes(in_axes, args, kwargs)

            def body(carry: CarryT, layer_index):
                layer = self.get_layer(layer_index)
                layer_args = tuple(
                    _slice_with_axis(arg, axis, layer_index, self.num_layers, context="scan_via")
                    for arg, axis in zip(args, args_axes, strict=True)
                )
                layer_kwargs = {
                    key: _slice_with_axis(kwargs[key], axis, layer_index, self.num_layers, context=f"scan_via[{key}]")
                    for key, axis in kwargs_axes.items()
                }
                carry = _tree_checkpoint_name(carry, self._carry_ckpt_name)
                layer_args, layer_kwargs = _tree_checkpoint_name((layer_args, layer_kwargs), self._input_ckpt_name)
                carry, out = fn(layer, carry, *layer_args, **layer_kwargs)
                return carry, out

            checkpointed_body = self.gradient_checkpointing.checkpoint(
                self._carry_ckpt_name,
                self._input_ckpt_name,
                body,
            )
            carry, out = _scan_layers(
                checkpointed_body,
                init,
                num_layers=self.num_layers,
                unroll=resolved_unroll,
                nested=self.gradient_checkpointing.nested,
            )
            return carry, out

        return do_scan

    def vmap_via(
        self,
        fn: Callable[..., OutputT_co],
        *,
        in_axes: InAxesSpec = None,
        out_axes: AxisSpec = 0,
    ) -> Callable[..., OutputT_co]:
        def do_vmap(*args, **kwargs) -> OutputT_co:
            args_axes, kwargs_axes = _resolve_in_axes(in_axes, args, kwargs)
            if out_axes is not None and not isinstance(out_axes, int):
                raise TypeError(f"out_axes must be int or None, got {type(out_axes)}.")

            def mapped(layer, mapped_args, mapped_kwargs):
                return fn(layer, *mapped_args, **mapped_kwargs)

            return jax.vmap(
                mapped,
                in_axes=(_layer_in_axes(self.stacked, self.num_layers), args_axes, kwargs_axes),
                out_axes=out_axes,
            )(self.stacked, args, kwargs)

        return do_vmap

    def get_layer(self, index: int) -> M:
        return jax.tree.map(
            lambda x: _slice_layer(x, index, self.num_layers),
            self.stacked,
        )

    def unstacked(self) -> Sequence[M]:
        leaves, structure = jax.tree_util.tree_flatten(self.stacked)
        unstacked_leaves = []
        for leaf in leaves:
            if _is_layer_batched_leaf(leaf, self.num_layers):
                unstacked_leaves.append(tuple(leaf[i] for i in range(self.num_layers)))
            else:
                unstacked_leaves.append(tuple(leaf for _ in range(self.num_layers)))

        return tuple(
            jax.tree_util.tree_unflatten(structure, layer_leaves)
            for layer_leaves in zip(*unstacked_leaves, strict=True)
        )

    def to_state_dict(self, prefix: str | None = None) -> StateDict:
        """Export arrays under ``<prefix>.<layer>.<parameter>`` keys.

        Shared Python leaves retain their unindexed keys.
        """
        state: StateDict = {}
        path_prefix = "" if prefix is None else prefix + "."
        for key, value in _to_state_dict(self.stacked, prefix).items():
            if _is_array_like(value):
                for index, layer_value in enumerate(value):
                    state[f"{path_prefix}{index}.{key[len(path_prefix) :]}"] = layer_value
            else:
                state[key] = value
        return state

    def from_state_dict(self, state_dict: StateDict, prefix: str | None = None) -> Self:
        """Load a devectorized state dict into real arrays or a shape template."""
        state: StateDict = {}
        grouped: dict[str, dict[int, Any]] = {}
        path_prefix = "" if prefix is None else prefix + "."
        for key, value in state_dict.items():
            if key.startswith(path_prefix):
                index, separator, suffix = key[len(path_prefix) :].partition(".")
                if separator and index.isdigit():
                    grouped.setdefault(path_prefix + suffix, {})[int(index)] = value
                    continue
            state[key] = value
        for key, values in grouped.items():
            state[key] = jnp.stack([values[index] for index in range(self.num_layers)])
        return dataclasses.replace(self, stacked=_from_state_dict(self.stacked, state, prefix))


def _normalize_axis(axis: int, ndim: int) -> int:
    if axis < 0:
        axis = ndim + axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f"in_axes axis {axis} is out of bounds for ndim={ndim}.")
    return axis


def _slice_with_axis(x: Any, axis: AxisSpec, layer_index: int, num_layers: int, *, context: str) -> Any:
    if axis is None:
        return x
    if not isinstance(axis, int):
        raise TypeError(f"{context}: in_axes leaves must be int or None, got {type(axis)}.")
    if not _is_array_like(x):
        raise TypeError(f"{context}: in_axes={axis} requires an array-like input, got {type(x)}.")

    resolved_axis = _normalize_axis(axis, x.ndim)
    if x.shape[resolved_axis] != num_layers:
        raise ValueError(
            f"{context}: in_axes={axis} expects size {num_layers} on axis {resolved_axis}, got shape {x.shape}."
        )
    return jnp.take(x, layer_index, axis=resolved_axis)


def _layer_in_axes(tree: Any, num_layers: int) -> Any:
    return jax.tree.map(
        lambda leaf: 0 if _is_layer_batched_leaf(leaf, num_layers) else None,
        tree,
    )


def _scan_layers(
    fn: Callable[[CarryT, jax.Array], tuple[CarryT, OutputT_co]],
    init: CarryT,
    *,
    num_layers: int,
    unroll: int | bool,
    nested: bool | int,
) -> tuple[CarryT, OutputT_co]:
    indices = jnp.arange(num_layers)
    outer_block_size = _nested_scan_outer_block_size(nested, num_layers)
    if outer_block_size is None:
        return jax.lax.scan(fn, init, indices, unroll=unroll)
    return _multilevel_scan(fn, init, indices, outer_block_size, length=num_layers, unroll=unroll)


def _nested_scan_outer_block_size(nested: bool | int, num_layers: int) -> int | None:
    if nested is True:
        for size in range(int(num_layers**0.5), 0, -1):
            if num_layers % size == 0:
                return size
        return 1
    if nested is False:
        return None
    if nested < 1:
        raise ValueError(f"nested checkpointing block size must be >= 1, got {nested}.")
    return nested


def _resolve_in_axes(
    in_axes: InAxesSpec,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[AxisSpec, ...], dict[str, AxisSpec]]:
    if isinstance(in_axes, tuple) and len(in_axes) == 2 and isinstance(in_axes[1], dict):
        args_axes, kwargs_axes = in_axes
    else:
        args_axes, kwargs_axes = in_axes, None

    if isinstance(args_axes, (int, type(None))):
        resolved_args_axes = tuple(args_axes for _ in args)
    else:
        if not isinstance(args_axes, tuple):
            raise TypeError("in_axes positional spec must be int, None, or a tuple matching positional args.")
        if len(args_axes) != len(args):
            raise ValueError(f"in_axes positional spec length {len(args_axes)} != number of args {len(args)}.")
        resolved_args_axes = args_axes

    if isinstance(kwargs_axes, (int, type(None))):
        resolved_kwargs_axes = {k: kwargs_axes for k in kwargs}
    else:
        if not isinstance(kwargs_axes, dict):
            raise TypeError("in_axes kwargs spec must be int, None, or a dict matching kwargs.")
        missing = set(kwargs) - set(kwargs_axes)
        extra = set(kwargs_axes) - set(kwargs)
        if missing or extra:
            raise ValueError(f"in_axes kwargs keys must match kwargs exactly; missing={missing}, extra={extra}.")
        resolved_kwargs_axes = kwargs_axes

    return resolved_args_axes, resolved_kwargs_axes


def _tree_checkpoint_name(tree: Any, name: str) -> Any:
    return jax.tree.map(lambda leaf: checkpoint_name(leaf, name) if _is_array_like(leaf) else leaf, tree)


def _multilevel_scan(fn, carry, xs, outer_size: int, length: int, *, unroll: int | bool):
    """Checkpoint groups of scan steps while retaining the flat output layout."""
    if length % outer_size:
        raise ValueError(f"Length {length} must be divisible by outer_size {outer_size}")
    inner_size = length // outer_size
    grouped = xs.reshape(outer_size, inner_size)
    carry, outputs = jax.lax.scan(
        jax.checkpoint(functools.partial(jax.lax.scan, fn, unroll=unroll)), carry, grouped, unroll=True
    )
    return carry, jax.tree.map(lambda x: x.reshape(length, *x.shape[2:]), outputs)


def _with_prefix(prefix: str | None, key: str | None) -> str | None:
    if prefix is None:
        return key
    return prefix if key is None else f"{prefix}.{key}"


def _to_state_dict(tree: Any, prefix: str | None) -> StateDict:
    if tree is None:
        return {}
    if isinstance(tree, _StateDictWriter):
        return tree.to_state_dict(prefix)
    if isinstance(tree, eqx.Module):
        key_map = tree._state_dict_key_map() if isinstance(tree, _StateDictKeyMap) else {}
        state: StateDict = {}
        for field in dataclasses.fields(tree):
            if not field.metadata.get("static", False):
                state.update(
                    _to_state_dict(
                        getattr(tree, field.name), _with_prefix(prefix, key_map.get(field.name, field.name))
                    )
                )
        return state
    if isinstance(tree, (list, tuple, dict)):
        items = tree.items() if isinstance(tree, dict) else enumerate(tree)
        return {
            key: value
            for index, child in items
            for key, value in _to_state_dict(child, _with_prefix(prefix, str(index))).items()
        }
    if prefix is None:
        raise ValueError("A state-dict leaf requires a prefix")
    return {prefix: tree}


def _from_state_dict(tree: Any, state: StateDict, prefix: str | None) -> Any:
    if tree is None:
        return None
    if isinstance(tree, _StateDictReader):
        return tree.from_state_dict(state, prefix)
    if isinstance(tree, eqx.Module):
        key_map = tree._state_dict_key_map() if isinstance(tree, _StateDictKeyMap) else {}
        values = {
            field.name: _from_state_dict(
                getattr(tree, field.name), state, _with_prefix(prefix, key_map.get(field.name, field.name))
            )
            for field in dataclasses.fields(tree)
            if not field.metadata.get("static", False)
        }
        return eqx.tree_at(
            lambda module: tuple(getattr(module, name) for name in values),
            tree,
            tuple(values.values()),
            is_leaf=lambda value: value is None,
        )
    if isinstance(tree, dict):
        return {key: _from_state_dict(value, state, _with_prefix(prefix, str(key))) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(
            _from_state_dict(value, state, _with_prefix(prefix, str(index))) for index, value in enumerate(tree)
        )
    if _is_array_like(tree):
        if prefix is None:
            raise ValueError("A state-dict leaf requires a prefix")
        return jnp.asarray(state[prefix])
    return state.get(prefix, tree)
