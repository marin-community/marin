# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, Callable, Generic, Self, Sequence, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from haliax._src.scan import ScanCheckpointPolicy, ScanCheckpointSpec, find_closest_divisible_int_to_sqrt
from haliax._src.state_dict import ModuleWithStateDictSerialization, StateDict
from haliax.jax_utils import is_jax_array_like, multilevel_scan, tree_checkpoint_name
from haliax.util import is_named_array
from .scan import ModuleInit, _stack_state_dict, _unstack_state_dict

M = TypeVar("M", bound=eqx.Module)
CarryT = TypeVar("CarryT")
OutputT_co = TypeVar("OutputT_co", covariant=True)
AxisSpec = int | None
InAxesSpec = AxisSpec | tuple[Any, Any]


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
    return is_jax_array_like(x) and not is_named_array(x) and x.ndim > 0 and x.shape[0] == num_layers


def _slice_layer(x: Any, index: int, num_layers: int) -> Any:
    if _is_layer_batched_leaf(x, num_layers):
        return x[index]
    return x


def _reject_named_arrays(stacked: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(stacked, is_leaf=is_named_array):
        if is_named_array(leaf):
            raise TypeError(
                "ArrayStacked does not support NamedArray leaves. "
                "Use haliax.nn.Stacked for modules with named-array parameters."
            )


class ArrayStacked(ModuleWithStateDictSerialization, Generic[M]):
    """Array-native stack container with leading layer dimensions in array leaves."""

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
                is_leaf=is_named_array,
            )

            def init_layer(layer_batched):
                layer_args, layer_kwargs = eqx.combine(layer_batched, shared)
                return module.init(*layer_args, **layer_kwargs)

            # `axis_size` covers a module whose arguments carry no layer axis at all, where
            # every layer initializes from the same inputs.
            stacked = jax.vmap(init_layer, axis_size=num_layers)(batched)
            _reject_named_arrays(stacked)
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
                carry = tree_checkpoint_name(carry, self._carry_ckpt_name)
                layer_args, layer_kwargs = tree_checkpoint_name((layer_args, layer_kwargs), self._input_ckpt_name)
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
                carry = tree_checkpoint_name(carry, self._carry_ckpt_name)
                layer_args, layer_kwargs = tree_checkpoint_name((layer_args, layer_kwargs), self._input_ckpt_name)
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

    def _state_dict_key_map(self) -> dict[str, str | None]:
        return {"stacked": None}

    def to_state_dict(self, prefix: str | None = None) -> StateDict:
        state_dict: StateDict = super().to_state_dict(prefix)
        return _unstack_state_dict(state_dict, prefix)

    def from_state_dict(self, state_dict: StateDict, prefix: str | None = None) -> Self:
        stacked = _stack_state_dict(state_dict, prefix=prefix)
        return super().from_state_dict(stacked, prefix=prefix)  # type: ignore


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
    if not is_jax_array_like(x):
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
        is_leaf=is_named_array,
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
    return multilevel_scan(fn, init, indices, outer_block_size, length=num_layers, unroll=unroll)


def _nested_scan_outer_block_size(nested: bool | int, num_layers: int) -> int | None:
    if nested is True:
        return find_closest_divisible_int_to_sqrt(num_layers)
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
