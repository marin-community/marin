# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import dataclasses
import functools
import re
import warnings
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    cast,
    overload,
    ParamSpec,
)

import equinox as eqx
import jax
from jax import numpy as jnp

import haliax
import haliax.util
from haliax.jax_utils import tree_checkpoint_name
from haliax.util import is_jax_or_hax_array_like

from .._src.scan import ScanCheckpointPolicy, ScanCheckpointSpec
from .._src.state_dict import ModuleWithStateDictSerialization, StateDict, with_prefix
from ..axis import Axis

M = TypeVar("M", bound=eqx.Module)
M_co = TypeVar("M_co", bound=eqx.Module, covariant=True)
M_contra = TypeVar("M_contra", bound=eqx.Module, contravariant=True)
S = TypeVar("S", bound=eqx.Module)
T = TypeVar("T")
CarryT = TypeVar("CarryT")
OutputT_co = TypeVar("OutputT_co", covariant=True)
P = ParamSpec("P")


class FoldFunction(Protocol[M_contra, P, CarryT]):
    def __call__(self, module: M_contra, carry: CarryT, *args: P.args, **kwargs: P.kwargs) -> CarryT: ...


class ScanFunction(Protocol[M_contra, CarryT, P, OutputT_co]):
    def __call__(
        self, module: M_contra, carry: CarryT, *args: P.args, **kwargs: P.kwargs
    ) -> tuple[CarryT, OutputT_co]: ...


class VmapFunction(Protocol[M_contra, P, OutputT_co]):
    def __call__(self, module: M_contra, *args: P.args, **kwargs: P.kwargs) -> OutputT_co: ...


class ModuleInit(Protocol[M_co]):
    def __call__(self, *args, **kwargs) -> M_co: ...


class BlockFoldable(Protocol[M]):
    """Common interface for :class:`~haliax.nn.Stacked` and :class:`~haliax.nn.BlockSeq`.

    The interface exposes the :py:meth:`fold` and :py:meth:`scan` methods along with the helper
    methods :py:meth:`fold_via`, :py:meth:`scan_via`, and :py:meth:`vmap_via`.

    This is a protocol, so you can use it as a type hint for a function that takes a ``Stacked`` or ``BlockSeq``.
    Equinox modules can't directly inherit from ``Protocol`` classes, but you can use it as a type hint.
    """

    Block: Axis

    @classmethod
    def init(
        cls: Type[S],
        Block: Axis,
        module: Type[M],
        *,
        gradient_checkpointing: ScanCheckpointSpec = False,
        prevent_cse: bool = False,
    ) -> ModuleInit[S]: ...

    def scan(self, init: T, *extra_args, **extra_kwargs): ...

    def fold(self, init: T, *args, **kwargs) -> T: ...

    @overload
    def fold_via(self, fn: FoldFunction[M, P, CarryT]) -> Callable[Concatenate[CarryT, P], CarryT]: ...

    @overload
    def fold_via(self, fn: Callable[[M, CarryT], CarryT]) -> Callable[[CarryT], CarryT]: ...

    def fold_via(self, fn: Callable[..., CarryT]) -> Callable[Concatenate[CarryT, P], CarryT]: ...

    @overload
    def scan_via(
        self, fn: ScanFunction[M, CarryT, P, OutputT_co]
    ) -> Callable[Concatenate[CarryT, P], tuple[CarryT, OutputT_co]]: ...

    @overload
    def scan_via(
        self, fn: Callable[[M, CarryT], tuple[CarryT, OutputT_co]]
    ) -> Callable[[CarryT], tuple[CarryT, OutputT_co]]: ...

    def scan_via(self, fn: Callable[..., tuple[CarryT, OutputT_co]]) -> Callable[P, tuple[CarryT, OutputT_co]]: ...

    @overload
    def vmap_via(self, fn: VmapFunction[M, P, OutputT_co]) -> Callable[P, OutputT_co]: ...

    @overload
    def vmap_via(self, fn: Callable[[M], OutputT_co]) -> Callable[[], OutputT_co]: ...

    def vmap_via(self, fn: Callable[..., OutputT_co]) -> Callable[..., OutputT_co]: ...

    def unstacked(self) -> Sequence[M]:
        """
        Returns the unstacked version of this module. This is useful for logging or saving checkpoints.

        """
        ...


class BlockSeq(ModuleWithStateDictSerialization, Generic[M]):
    """
    A "BlockSeq" wraps another module and produces a "sequential" version of it, where an input is applied
    to each instance of the sequential module in sequence. This is useful for e.g. transformers
    where you have multiple instances of the same transformer block and the input is applied in a fold/for loop
    in sequence.

    It's similar in spirit to an [equinox.nn.Sequential][]. Unlike [equinox.nn.Sequential][], BlockSeq does not need to be
    homogeneous (though the init method assumes that it is).
    """

    blocks: Sequence[M]
    Block: Axis = eqx.field(static=True)
    gradient_checkpointing: ScanCheckpointPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls: Type[S],
        Block: Axis,
        module: Type[M],
        *,
        gradient_checkpointing: ScanCheckpointSpec = False,
        prevent_cse: bool | None = None,
    ) -> ModuleInit[S]:
        """
        This is a curried init method that takes the Block and module and returns a function that takes
        the arguments to the module's init method. Any NamedArrays in the arguments will be sliced along the
        Block axis (if it exists). JAX arrays will be sliced along the first axis.
        """

        gradient_checkpointing = ScanCheckpointPolicy._mk(gradient_checkpointing)

        if prevent_cse is not None:
            warnings.warn(
                "The prevent_cse argument is deprecated and will be removed in a future version of Haliax. Use the"
                " ScanCheckpointPolicy instead.",
                DeprecationWarning,
            )
            gradient_checkpointing = dataclasses.replace(gradient_checkpointing, prevent_cse=prevent_cse)

        @functools.wraps(module)
        def fn(*args, **kwargs):
            # The only complexity here is that the args and kwargs might have a Block axis in them,
            # in which case we need to loop over them to slice them out.

            def init_block(i):
                (block_args, block_kwargs) = haliax.tree_util.tree_map(
                    functools.partial(BlockSeq._slice_out, Block, i), (args, kwargs)
                )
                return module.init(*block_args, **block_kwargs)

            seq = [init_block(i) for i in range(Block.size)]

            return BlockSeq(seq, Block, gradient_checkpointing)

        return fn

    def scan(self, init: T, *extra_args, **extra_kwargs):
        def do_scan(init, *extra_args, **extra_kwargs):
            out = []
            carry = init

            for i, block in enumerate(self.blocks):

                (block_args, block_kwargs) = haliax.tree_util.tree_map(
                    functools.partial(BlockSeq._slice_out, self.Block, i), (extra_args, extra_kwargs)
                )

                block_result = block(carry, *block_args, **block_kwargs)

                if not isinstance(block_result, (tuple, list)) or len(block_result) != 2:
                    raise ValueError(
                        f"BlockSeq.scan expects the block to return a pair of (carry, extra), got {block_result}"
                    )

                carry, extra = block_result

                carry = tree_checkpoint_name(carry, self._carry_ckpt_name)
                extra = tree_checkpoint_name(extra, self._output_ckpt_name)

                out.append(extra)

            return carry, haliax.tree_util.tree_map(lambda *x: haliax.stack(self.Block, x), *out)

        return do_scan(init, *extra_args, **extra_kwargs)

    def fold(self, init: T, *args, **kwargs) -> T:
        def do_fold(init, *args, **kwargs):
            carry = init
            for i, block in enumerate(self.blocks):
                (block_args, block_kwargs) = haliax.tree_util.tree_map(
                    functools.partial(BlockSeq._slice_out, self.Block, i), (args, kwargs)
                )
                carry = block(carry, *block_args, **block_kwargs)
                carry = tree_checkpoint_name(carry, self._carry_ckpt_name)
            return carry

        return do_fold(init, *args, **kwargs)

    @overload
    def fold_via(self, fn: FoldFunction[M, P, CarryT]) -> Callable[Concatenate[CarryT, P], CarryT]: ...

    @overload
    def fold_via(self, fn: Callable[[M, CarryT], CarryT]) -> Callable[[CarryT], CarryT]: ...

    def fold_via(self, fn: Callable[..., CarryT]):
        """Return a function that folds over the sequence using ``fn``.

        ``fn`` should take a block and a carry and return a new carry. The
        returned function mirrors :func:`haliax.fold` over the block axis.
        """

        def do_fold(init: CarryT, *args, **kwargs) -> CarryT:
            carry = init
            for block in self.blocks:
                carry = fn(block, carry, *args, **kwargs)
                carry = tree_checkpoint_name(carry, self._carry_ckpt_name)
            return carry

        return do_fold

    @overload
    def scan_via(
        self, fn: ScanFunction[M, CarryT, P, OutputT_co]
    ) -> Callable[Concatenate[CarryT, P], tuple[CarryT, OutputT_co]]: ...

    @overload
    def scan_via(
        self, fn: Callable[[M, CarryT], tuple[CarryT, OutputT_co]]
    ) -> Callable[[CarryT], tuple[CarryT, OutputT_co]]: ...

    def scan_via(self, fn: Callable[..., tuple[CarryT, OutputT_co]]):
        """Return a function that scans over the sequence using ``fn``.

        ``fn`` should take a block and a carry and return ``(carry, output)``.
        Semantics match :func:`haliax.scan` over the block axis.
        """

        def do_scan(init: CarryT, *args, **kwargs) -> tuple[CarryT, OutputT_co]:
            out = []
            carry = init
            for block in self.blocks:
                carry, extra = fn(block, carry, *args, **kwargs)
                carry = tree_checkpoint_name(carry, self._carry_ckpt_name)
                extra = tree_checkpoint_name(extra, self._output_ckpt_name)
                out.append(extra)

            stacked_out = haliax.tree_util.tree_map(lambda *x: haliax.stack(self.Block, x), *out)
            return carry, stacked_out

        return do_scan

    @overload
    def vmap_via(self, fn: VmapFunction[M, P, OutputT_co]) -> Callable[P, OutputT_co]: ...

    @overload
    def vmap_via(self, fn: Callable[[M], OutputT_co]) -> Callable[[], OutputT_co]: ...

    def vmap_via(self, fn: Callable[..., OutputT_co]) -> Callable[..., OutputT_co]:
        """Return a function that applies each block independently using ``fn``.

        ``fn`` should take a block and a carry and return an output. The
        returned function mirrors :func:`haliax.vmap` over the block axis.
        """

        def do_vmap(init: CarryT, *args, **kwargs) -> OutputT_co:
            # Apply fn to each block independently
            outputs = []
            for block in self.blocks:
                output = fn(block, init, *args, **kwargs)
                outputs.append(output)

            # Stack the outputs
            stacked_out = haliax.tree_util.tree_map(lambda *x: haliax.stack(self.Block, x), *outputs)
            return stacked_out

        return do_vmap

    def unstacked(self) -> Sequence[M]:
        return self.blocks

    @staticmethod
    def _slice_out(Block, i, x):
        if isinstance(x, haliax.core.NamedArray):
            if haliax.selects_axis(x.axes, Block):
                return x[Block, i]
            else:
                return x
        elif haliax.jax_utils.is_jax_array_like(x):
            return x[i]
        else:
            return x

    def _state_dict_key_map(self) -> dict[str, str | None]:
        return {"blocks": None}

    def from_state_dict(self: M, state_dict: StateDict, prefix: str | None = None) -> M:
        out_blocks = []
        for i, block in enumerate(self.blocks):
            my_prefix = with_prefix(prefix, str(i))
            block = block.from_state_dict(state_dict, my_prefix)
            out_blocks.append(block)

        return eqx.tree_at(lambda m: m.blocks, self, out_blocks)

    def to_state_dict(self, prefix: str | None = None) -> StateDict:
        """
        Returns the unstacked format of the module, which is compatible with torch.nn.Sequential, with keys of the form (...). The stacked/vectorized format is required for haliax.nn.Stacked and vectorizes all such tensors into a single shared key.".
        """
        state_dict: StateDict = {}
        for i, block in enumerate(self.blocks):
            my_prefix = with_prefix(prefix, str(i))
            # we can't assume to_state_dict is implemented, so we have to do it manually
            block_dict = haliax.state_dict.to_state_dict(block, my_prefix)
            state_dict.update(block_dict)

        return state_dict

    @property
    def _output_ckpt_name(self):
        return f"BlockSeq[{self.Block}, {self.blocks[0].__class__.__name__}].outputs"

    @property
    def _carry_ckpt_name(self):
        return f"BlockSeq[{self.Block}, {self.blocks[0].__class__.__name__}].carry"


class Stacked(ModuleWithStateDictSerialization, Generic[M]):
    """
    A "Stacked" wraps another module and produces a "stacked" version of it, where an input is applied
    to each instance of the stacked module in sequence. This is useful for e.g. transformers
    where you have multiple instances of the same transformer block and the input is applied in a fold/for loop
    in sequence.

    It's similar in spirit to an [equinox.nn.Sequential], but it must be homogeneous. In Jax, this is much cheaper to
    compile than a sequential (or moral equivalent), because Jax compiles the module's method once, instead of unrolling
    the sequential and compiling everything as a giant graph. In Jax, this pattern is often called "scan layers" or
    "scan over layers".

    A further constraint is that the elements of the stack must have the same Python control flow. This is because
    Jax's scan primitive requires that the function you pass to it is pure, and the only way to do that is to ensure
    that the function has the same control flow for every element of the stack.

    Stacked supports both "fold" and "scan" semantics. "fold" is the same as a for loop that accumulates a single
    output, while "scan" is the same as a for loop that accumulates a list of outputs as well as the final output.

    Stacked also supports gradient checkpointing, which is useful for very large models that don't fit in memory.
    If your blocks are independent of each other you can instead use :py:meth:`Stacked.vmap`
    to apply every block in parallel.

    Typically only one of "fold" or "scan" can be used with a given Stacked module, depending on the what the module
    returns: if the module returns a single output, use "fold"; if the module returns a sequence of outputs and
    an output to be passed to the next layer, use "scan". More concretely, for a transformer, you would use "scan" if
    you wanted to return a kv cache (or the attention matrix) as well as the output of the transformer. If you just
    wanted the output of the transformer, you would use "fold".


    Example:
        ```python
        >>> import equinox as eqx
        >>> import haliax as hax
        >>> import haliax.nn as hnn
        >>> class MyModule(eqx.Module):
        ...     def __init__(self, num_layers: int, hidden: hax.Axis, *, key):
        ...         self.axis = hax.Axis("layer", num_layers)
        ...         split_key = jax.random.split(key, num_layers)
        ...         self.layers = Stacked.init(self.axis, hnn.Linear)(In=hidden, Out=hidden, key=split_key)
        ...
        ...     def __call__(self, x):
        ...         return self.layers.fold(x)  # applies each layer in sequence
        ...
        >>> Hidden = hax.Axis("hidden", 10)
        >>> mod = MyModule(5, Hidden)
        >>> mod(hax.ones(Hidden))
        ```
    """

    # TODO: we can probably make this module support pipeline parallelism, but that's a whole project in itself

    stacked: M
    Block: Axis = eqx.field(static=True)
    gradient_checkpointing: ScanCheckpointPolicy = eqx.field(static=True)

    @classmethod
    def init(
        cls,
        Block: Axis,
        module: Type[M],
        *,
        gradient_checkpointing: ScanCheckpointSpec = False,
        prevent_cse: bool | None = None,
    ) -> ModuleInit["Stacked[M]"]:
        """
        Initialize a Stacked module. This method is curried: you can pass in the Block and module, and it will return
        a function that takes (batched) arguments to the vmapped module's init method.

        Args:
            Block: The axis that will be stacked over. This is typically a "layer" axis, but could be any axis.
            module: The module that will be stacked. This module must take a batched input and return a batched output.
            gradient_checkpointing: Whether to use gradient checkpointing. If True, uses the default policy. If a string,
                uses the policy specified by the string. If a ScanCheckpointPolicy, uses that policy.
            prevent_cse: Whether to prevent common subexpression elimination in the checkpointed function. This is useful
                for debugging, but may slow down the function.
        """

        gradient_checkpointing = ScanCheckpointPolicy._mk(gradient_checkpointing)

        if prevent_cse is not None:
            warnings.warn(
                "The prevent_cse argument is deprecated and will be removed in a future version of Haliax. Use the"
                " ScanCheckpointPolicy instead.",
                DeprecationWarning,
            )

            gradient_checkpointing = dataclasses.replace(gradient_checkpointing, prevent_cse=prevent_cse)

        @functools.wraps(module)
        def fn(*args, **kwargs):
            stacked = haliax.vmap(module.init, Block)(*args, **kwargs)
            return Stacked(stacked, Block, gradient_checkpointing)

        return fn

    def scan(self, init, *extra_args, **extra_kwargs):
        """
        Scan over the stacked module. This is the same as a for loop that applies each instance of the module in sequence
        to the input, passing the output of one instance to the next instance. It returns a stack of outputs as
        well as the final output.

        That is, it behaves similarly to the following Python code:

        ```python
        carry = init
        outputs = []

        for block in self.stacked:
            carry, extra = block(carry)
            outputs.append(extra)

        return carry, hax.stack(Block, outputs)
        ```

        Args:
            init:
            *extra_args:
            **extra_kwargs:

        Returns:

        """

        def do_block(carry, block, *args, **kwargs):
            carry, out = block(carry, *args, **kwargs)
            return carry, out

        def do_scan(init, *extra_args, **extra_kwargs):
            carry, out = haliax.scan(do_block, self.Block, remat=self.gradient_checkpointing)(
                init, self.stacked, *extra_args, **extra_kwargs
            )
            return carry, out

        return do_scan(init, *extra_args, **extra_kwargs)

    def fold(self, init, *args, **kwargs):
        """
        Fold over the stacked module. This is the same as a for loop that applies each instance of the module in sequence
        to the input, passing the output of one instance to the next instance.
        That is, it behaves similarly to the following Python code:

        ```python
        carry = init
        for block in self.stacked:
            carry = block(carry)

        return carry
        ```

        Args:
            init: The initial value of carry to pass to the first block
            *args: Extra arguments to pass to the blocks. These are passed directly to the blocks
            **kwargs: Extra keyword arguments to pass to the blocks. These are passed directly to the blocks

        Returns:

        """

        def do_block(carry, block, *args, **kwargs):
            carry = block(carry, *args, **kwargs)
            return carry

        def do_fold(init, *extra_args, **extra_kwargs):
            carry = haliax.fold(do_block, self.Block, remat=self.gradient_checkpointing)(
                init, self.stacked, *extra_args, **extra_kwargs
            )
            return carry

        return do_fold(init, *args, **kwargs)

    @overload
    def fold_via(self, fn: FoldFunction[M, P, CarryT]) -> Callable[Concatenate[CarryT, P], CarryT]: ...

    @overload
    def fold_via(self, fn: Callable[[M, CarryT], CarryT]) -> Callable[[CarryT], CarryT]: ...

    def fold_via(self, fn: Callable[..., CarryT]):
        """Return a function that folds over the stack using ``fn``.

        ``fn`` should take a block and a carry and return a new carry.  The
        returned function mirrors :func:`haliax.fold` over the block axis.
        """

        def do_block(carry: CarryT, block: M, *args, **kwargs) -> CarryT:
            return fn(block, carry, *args, **kwargs)

        def do_fold(init: CarryT, *args, **kwargs) -> CarryT:
            return haliax.fold(do_block, self.Block, remat=self.gradient_checkpointing)(
                init, self.stacked, *args, **kwargs
            )

        return do_fold

    @overload
    def scan_via(
        self, fn: ScanFunction[M, CarryT, P, OutputT_co]
    ) -> Callable[Concatenate[CarryT, P], tuple[CarryT, OutputT_co]]: ...

    @overload
    def scan_via(
        self, fn: Callable[[M, CarryT], tuple[CarryT, OutputT_co]]
    ) -> Callable[[CarryT], tuple[CarryT, OutputT_co]]: ...

    def scan_via(self, fn: Callable[..., tuple[CarryT, OutputT_co]]):
        """Return a function that scans over the stack using ``fn``.

        ``fn`` should take a block and a carry and return ``(carry, output)``.
        Semantics match :func:`haliax.scan` over the block axis.
        """

        def do_block(carry: CarryT, block: M, *args, **kwargs) -> tuple[CarryT, OutputT_co]:
            carry, output = fn(block, carry, *args, **kwargs)
            return carry, output

        def do_scan(init: CarryT, *args, **kwargs) -> tuple[CarryT, OutputT_co]:
            return haliax.scan(do_block, self.Block, remat=self.gradient_checkpointing)(
                init, self.stacked, *args, **kwargs
            )

        return do_scan

    @overload
    def vmap_via(self, fn: VmapFunction[M, P, OutputT_co]) -> Callable[P, OutputT_co]: ...

    @overload
    def vmap_via(self, fn: Callable[[M], OutputT_co]) -> Callable[[], OutputT_co]: ...

    def vmap_via(self, fn: Callable[..., OutputT_co]) -> Callable[..., OutputT_co]:
        """Return a function that applies each block independently using ``fn``.

        ``fn`` should take a block and a carry and return an output. The
        returned function mirrors :func:`haliax.vmap` over the block axis.
        """

        def do_vmap(*args, **kwargs) -> OutputT_co:
            # Create a function that captures the additional arguments
            def do_block_with_args(block: M, *args, **kwargs) -> OutputT_co:
                return fn(block, *args, **kwargs)

            return haliax.vmap(do_block_with_args, self.Block)(self.stacked, *args, **kwargs)

        return do_vmap

    def vmap(self, *extra_args, **extra_kwargs):
        """Apply each block independently using :func:`haliax.vmap`.

        Returns the stacked outputs of each block.
        """

        return haliax.vmap(type(self.stacked).__call__, self.Block)(self.stacked, *extra_args, **extra_kwargs)

    @staticmethod
    def _do_block(carry, block, *extra_args, **extra_kwargs):
        return block(carry, *extra_args, **extra_kwargs)

    # TODO: this is for logic that's in levanter. We should move that logic to haliax I guess?
    def _state_dict_key_map(self) -> dict[str, str | None]:
        return {"stacked": None}

    def unstacked(self) -> Sequence[M]:
        """
        Returns the unstacked version of this module. This is useful for logging or saving checkpoints.
        Returns:
            A sequence of modules, one for each element of the stack
        """

        def unbatch_leaf(x):
            if isinstance(x, haliax.core.NamedArray):
                if haliax.selects_axis(x.axes, self.Block):
                    return haliax.unbind(x, self.Block)
                else:
                    return tuple(x for _ in range(self.Block.size))
            elif haliax.jax_utils.is_jax_array_like(x):
                assert (
                    x.shape[0] == self.Block.size
                ), f"Expected first dimension to be {self.Block.size}, got {x.shape[0]}"
                return tuple(x[i] for i in range(self.Block.size))
            else:
                return tuple(x for _ in range(self.Block.size))

        leaves, structure = jax.tree_util.tree_flatten(self.stacked, is_leaf=haliax.util.is_named_array)
        unstacked_leaves = tuple(map(unbatch_leaf, leaves))
        # now we need to transpose the leaves
        unstacked_leaves = tuple(zip(*unstacked_leaves))
        return tuple(map(lambda x: jax.tree_util.tree_unflatten(structure, x), unstacked_leaves))

    def to_state_dict(self, prefix: str | None = None) -> StateDict:
        # this method needs to "devectorize" the blocks, so that we have a list of blocks h.0.FOO, h.1.FOO, etc.
        # first just do the normal thing with our own dict, which we'll post-process
        state_dict: StateDict = super().to_state_dict(prefix)

        return _unstack_state_dict(state_dict, prefix)

    def from_state_dict(self: M, state_dict: StateDict, prefix: str | None = None) -> M:
        # this method needs to "vectorize" the blocks, so that we have a single block h.FOO
        # first just do the normal thing with our own dict, which we'll post-process
        stacked = _stack_state_dict(state_dict, prefix=prefix)
        out = super().from_state_dict(stacked, prefix=prefix)  # type: ignore
        return out


def _stack_state_dict(state_dict: StateDict, prefix: str | None = None) -> StateDict:
    """
    Stack all keys matching prefix in a new state dict, returning a state dict that has all keys matching
    prefix stacked, but otherwise the same.

    Stacked in this case means roughly "compatible with a torch.nn.Sequential", which means that the
    keys are of the form "<prefix>.0.<key>", "<prefix>.1.<key>", etc.

    Mostly for use with [haliax.nn.Stacked][].
    """
    vectorized_dict: StateDict = {}

    tensors_to_vectorize: dict[str, list[Any | None]] = {}
    if prefix is not None:
        prefix_for_pat = re.escape(prefix + ".")
    else:
        prefix_for_pat = ""
    pattern = re.compile(rf"{prefix_for_pat}(\d+)\.(.*)")

    for k, v in state_dict.items():
        match = pattern.match(k)
        if match:
            block_idx = int(match.group(1))
            block_key = match.group(2)
            tensors = tensors_to_vectorize.setdefault(block_key, [])
            if len(tensors) <= block_idx:
                tensors.extend([None] * (block_idx - len(tensors) + 1))
            assert tensors[block_idx] is None, f"Duplicate key {k}"
            tensors[block_idx] = v
        else:
            vectorized_dict[k] = v

    # now we have to vectorize the tensors
    for k, tensors in tensors_to_vectorize.items():
        vectorized_dict[cast(str, with_prefix(prefix, k))] = jnp.stack(tensors, axis=0)

    return vectorized_dict


def _unstack_state_dict(state_dict: StateDict, prefix: str | None = None) -> StateDict:
    """
    Unstack all keys matching prefix in a new state dict, returning a state dict that has all keys matching
    prefix unstacked, but otherwise the same. Mostly for use with [haliax.nn.Stacked][].

    Unstacked in this case means roughly "compatible with a torch.nn.Sequential", which means that the
    keys are of the form "<prefix>.0.<key>", "<prefix>.1.<key>", etc.
    """
    new_dict: StateDict = {}
    prefix = with_prefix(prefix, "")
    assert prefix is not None

    for k, v in state_dict.items():
        if k.startswith(prefix) and is_jax_or_hax_array_like(v):
            for i, v_i in enumerate(v):
                new_dict[f"{prefix}{i}.{k[len(prefix):]}"] = v_i
        else:
            new_dict[k] = v

    return new_dict
