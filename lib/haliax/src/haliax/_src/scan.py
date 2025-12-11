# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import functools as ft
import inspect
from typing import Any, Callable, Literal, ParamSpec, Protocol, Sequence, TypeVar, overload

import equinox as eqx
import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

import haliax
import haliax.tree_util as htu
from haliax._src.util import index_where
from haliax.axis import Axis, AxisSelector, selects_axis
from haliax.core import NamedArray
from haliax.jax_utils import is_jax_array_like, multilevel_scan, tree_checkpoint_name
from haliax.util import is_jax_or_hax_array_like, is_named_array

BoolAxisSpec = bool | Callable[[Any], bool]
Carry = TypeVar("Carry")
X = TypeVar("X", contravariant=True)
Y = TypeVar("Y", covariant=True)
Args = ParamSpec("Args")


def is_named_or_shaped_array_like(x):
    return (is_jax_array_like(x) and x.ndim >= 1) or is_named_array(x)


class ScanFn(Protocol[Carry, Args, Y]):
    """ """

    def __call__(self, carry: Carry, *args: Args.args, **kwargs: Args.kwargs) -> tuple[Carry, Y]: ...


@dataclasses.dataclass(frozen=True)
class ScanCheckpointPolicy:
    """
    A class that represents a gradient checkpoint policy for blocks in a Stacked module. This is used to control
    gradient checkpointing in [haliax.scan][], [haliax.fold][], [haliax.nn.Stacked][] and [haliax.nn.BlockSeq][].

    Gradient checkpointing is a technique for reducing memory usage in training large models. It works by saving only a
    subset of the forward pass and recomputing the rest in the backward pass. (By doing parts of the forward pass again)
    JAX suggests that this usually isn't necessary when not using scan-over-layers (i.e. Stacked), so this is mostly
    useful for Stacked modules.

    A scan block takes a "carry" and some extra arguments, and returns a "carry" and an "output". The "carry" is passed
    to the next block, and the "output" is concatenated into a final result (sort of like an RNN).

    Schematically it might look like this:

    ```
          I       I       I       I
          |       |       |       |
    C ->  B -C->  B -C->  B -C->  B --> C
          |       |       |       |
          O       O       O       O
    ```

    where "C" is the carry and "O" is the output. A block will typically do some computation (e.g. a Transformer block)
    as well, which might require saving or recomputing in the backward pass.

    Without checkpointing, we will save all block inputs and outputs, and use them to compute the gradient. This requires
    memory, of course. With checkpointing, we can save only some of the computation. Typically, to compute the gradient
    of the scan, we need the inputs to each block, the intermediates within each block, as well as the carries (which
    are inputs to the next block). We can save all of these, or we can save only some of them.

    With this class, you can specify if you want to save the carries, or the internals/inputs. You can also offload
    the carries to the host, which can reduce memory usage on the device.

    #### Nested Remat

    Alternatively, we can do something a bit more clever. We can break the computation into "blocks" of size "B", and
    save the carries and outputs of each block. Then, during the backward pass, we can recompute the outputs of each
    block using the carries and inputs, and then compute the gradient as usual. This requires O(B) memory and O(N)
    time. When B = sqrt(N), this is O(sqrt(N)) memory and O(N) time. This is the "nested scan" policy.
    In practice, this is about 20% slower than the O(N) memory policy, but it can be worth it for large models.

    #### Offloading

    Another choice is to "offload" carries and outputs to the host, which can reduce memory usage on the device.
    We support offloading carries and outputs to the host, but not internals.

    See Also:
        * [JAX docs on gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
    """

    save_carries: bool | Literal["offload"] = True
    """
    Whether to save all carries in the forward pass. If True, (input) carries are saved in the forward pass and used in the
    backward pass. If "offload", carries are saved in the forward pass and offloaded to the host

    If False, carries are recomputed in the backward pass.
    """

    save_inputs: bool | Literal["offload"] = False
    """
    Whether to save all non-carry inputs in the forward pass. If True, inputs are saved in the forward pass and used in
    the backward pass. If "offload", inputs are saved in the forward pass and offloaded to the host.
    """

    save_block_internals: bool | list[str] = False
    """
    Whether to save internal state of blocks. If a list, only the listed names are saved, as
    with [jax.checkpoint_policies.save_only_these_names][].

    See Also: https://docs.jax.dev/en/latest/gradient-checkpointing.html#custom-policies-for-offload
    """

    offload_block_internals: list[str] = dataclasses.field(default_factory=list)
    """
    List of named block internals to offload to the host. This is useful for reducing memory usage on the device
    while still avoiding rematerialization.
    """

    prevent_cse: bool = False
    """
    Whether to prevent common subexpression elimination in the checkpointed function.
    """

    disable: bool = False
    """
    Whether to disable gradient checkpointing entirely. This is useful for debugging.
    """

    simple: bool = False
    """
    Whether to use the simple gradient checkpointing policy. This is useful for debugging.
    """

    nested: bool | int = False
    """
    Allows for nested remat with a double scan. We reshape the stack into [nested_remat, -1] and then scan over both
    in sequence. If True, we find the closest int to sqrt(len(stack)) such that len(stack) % int == 0.
    If False, we don't do anything.
    """

    @staticmethod
    def from_bool_or_str(remat_policy: bool | str):
        """
        Convert a boolean or string into a BlockCheckpointPolicy. This is useful for converting user input
        into a BlockCheckpointPolicy.

        Choices:
            * True: save outputs, don't save block internals. This is the classic Haliax behavior.
            * False: save everything.
            * "offload": offload outputs to the host, don't save block internals.
            * "recompute" or "full": don't save outputs or block internals.
            * "save_all": save outputs and block internals. Equivalent to False
            * "nested": use nested remat. Equivalent to True
        """
        if remat_policy == "offload":
            return ScanCheckpointPolicy(save_carries="offload", save_inputs="offload", save_block_internals=False)
        elif remat_policy == "recompute" or remat_policy == "full":
            return ScanCheckpointPolicy(save_carries=False, save_inputs=False, save_block_internals=False)
        elif remat_policy == "save_all":
            return ScanCheckpointPolicy(save_carries=True, save_inputs=True, save_block_internals=True)
        elif remat_policy == "nested":
            return ScanCheckpointPolicy(nested=True)
        elif remat_policy is True:
            return ScanCheckpointPolicy(simple=True)
        elif remat_policy is False:
            return ScanCheckpointPolicy(save_carries=True, save_inputs=True, save_block_internals=True, disable=True)
        else:
            raise ValueError(f"Invalid checkpoint policy {remat_policy}")

    @staticmethod
    def _mk(remat_policy: bool | str | "ScanCheckpointPolicy") -> "ScanCheckpointPolicy":
        if isinstance(remat_policy, ScanCheckpointPolicy):
            return remat_policy
        else:
            return ScanCheckpointPolicy.from_bool_or_str(remat_policy)

    def checkpoint(self, carry_name: str, input_name: str, callable):
        if self.disable:
            return callable
        elif self.simple:
            return eqx.filter_checkpoint(callable, prevent_cse=self.prevent_cse)
        else:
            policy = self._to_jax_policy(carry_name, input_name)
            return eqx.filter_checkpoint(callable, policy=policy, prevent_cse=self.prevent_cse)

    def _to_jax_policy(self, carry_name: str, input_name: str):
        assert isinstance(carry_name, str)
        assert isinstance(input_name, str)
        our_names_to_save = []
        our_names_to_offload = []
        our_names_to_remat = []

        # return jax.checkpoint_policies.save_only_these_names(carry_name, output_name)

        if self.save_inputs is True:
            our_names_to_save.append(input_name)
        elif self.save_inputs == "offload":
            our_names_to_offload.append(input_name)
        else:
            assert self.save_inputs is False, f"Invalid save_inputs {self.save_inputs}"
            our_names_to_remat.append(input_name)

        if self.save_carries is True:
            our_names_to_save.append(carry_name)
        elif self.save_carries == "offload":
            our_names_to_offload.append(carry_name)
        else:
            assert self.save_carries is False, f"Invalid save_carries {self.save_carries}"
            our_names_to_remat.append(carry_name)

        if isinstance(self.save_block_internals, Sequence):
            our_names_to_save.extend(self.save_block_internals)

        if self.offload_block_internals:
            our_names_to_offload.extend(self.offload_block_internals)

        if not our_names_to_save and not our_names_to_offload and not self.save_block_internals:
            return None

        if len(our_names_to_offload) > 0:
            if self.save_block_internals is True:
                raise ValueError("Can't save all block internals and offload too. Use a list of names instead.")

            return jax.checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=our_names_to_save,
                names_which_can_be_offloaded=our_names_to_offload,
                offload_src="device",
                offload_dst="pinned_host",
            )
        else:
            if len(our_names_to_remat) > 0:
                if self.save_block_internals is True:
                    p1 = jax.checkpoint_policies.save_anything_except_these_names(*our_names_to_remat)
                    if len(our_names_to_save) > 0:
                        p2 = jax.checkpoint_policies.save_only_these_names(*our_names_to_save)
                        return jax.checkpoint_policies.save_from_both_policies(p1, p2)
                    else:
                        return p1
                else:
                    return jax.checkpoint_policies.save_only_these_names(*our_names_to_save)
            elif len(our_names_to_save) > 0:
                p1 = jax.checkpoint_policies.save_only_these_names(*our_names_to_save)
                if self.save_block_internals is True:
                    p2 = jax.checkpoint_policies.save_anything_except_these_names(*our_names_to_remat)
                    return jax.checkpoint_policies.save_from_both_policies(p1, p2)
                else:
                    return p1
            elif self.save_block_internals is True:
                return jax.checkpoint_policies.save_anything_except_these_names(*our_names_to_remat)
            else:
                return None


ScanCheckpointSpec = ScanCheckpointPolicy | bool | Literal["offload", "recompute", "full", "save_all", "nested"]


@overload
def scan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    axis: AxisSelector,
    *,
    remat: ScanCheckpointSpec = False,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
) -> Callable[[Carry, PyTree[X]], tuple[Carry, PyTree[Y]]]: ...


@overload
def scan(
    f: Callable,
    axis: AxisSelector,
    *,
    remat: ScanCheckpointSpec = False,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
) -> Callable: ...


def scan(
    f: Callable,  # : ScanFn[Carry, Args, Y],  This confuses mypy too much
    axis: AxisSelector,
    *,
    remat: ScanCheckpointSpec = False,
    reverse=False,
    unroll=1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
):
    """
    Scan over a named axis. Non-scalar unnamed arrays will have their first axis scanned over.

    Unlike [jax.lax.scan][], this function is curried: it takes the function, axis, and configuration arguments first, and
    then the initial carry and then any arguments to scan over as a separate curried function call.

    That is, `scan(f, axis)(init, xs)` is equivalent to `jax.lax.scan(f, init, xs)`

    Args:
        f (Callable): function to scan over
        axis (AxisSelector): axis to scan over
        reverse (bool): if True, scan in reverse
        unroll (int): unroll the loop by this amount
        remat: a ScanCheckpointPolicy or a boolean. If True, rematerialize block internals during
            gradient computation. If False, rematerialize nothing. If "offload", offload all carries and inputs.
            See [haliax.ScanCheckpointPolicy][] for more information.
        is_scanned (BoolAxisSpec): a function that takes a leaf of the tree and returns True if it should be scanned
            over, False otherwise. Behaves similarly to the `default` argument in filter_jit
    """
    checkpoint = ScanCheckpointPolicy._mk(remat)

    if isinstance(is_scanned, bool):
        q = is_scanned  # this is to make mypy happy
        is_scanned = lambda _: q

    def is_scanned_with_axis(leaf):
        if is_named_array(leaf):
            return selects_axis(leaf.axes, axis) and is_scanned(leaf)
        else:
            return is_scanned(leaf)

    def scanned_f(init, *args, **kwargs):
        # This implementation is a bit tricky.

        # first we want to partition the arguments into scanned and unscanned
        # unscanned arguments are just passed through, essentially captured as part of a lambda
        # scanned arguments are passed through the scan, which means we need to hoist the axis to the front
        xs = (args, kwargs)
        scanned_xs, unscanned_xs = eqx.partition(xs, is_scanned_with_axis, is_leaf=is_named_array)

        carry_name = f"scan({haliax.axis_name(axis)})__carry"
        input_name = f"scan({haliax.axis_name(axis)})__input"

        # Next we have to hoist the axis we're scanning over to the front of the array, because that's what scan
        # expects. Then we have to scan over the 0th dim of the arrays (as flattened non-pytrees)
        # We have to be careful that we don't try to create NamedArrays that have the shape of the scanned result
        # but don't yet have the scanned axis as ones of `axes`, so we use _ScannedArrayResult that doesn't check
        # invariants until we're ready to create the result.
        axis_first_xs = htu.tree_map(_ensure_first(axis), scanned_xs)

        # now get a template of an element of "X"
        x_elem = htu.tree_map(_select_0th(axis), axis_first_xs)
        # NB: we don't want to use htu.tree_structure here because we want to eliminate the leading axis
        x_elem_structure = jax.tree_util.tree_structure(x_elem)

        # now we can fold over the axis
        @ft.wraps(f)
        def wrapped_fn(carry, scanned_x_leaves):
            scanned_x = jax.tree_util.tree_unflatten(x_elem_structure, scanned_x_leaves)
            scanned_x = tree_checkpoint_name(scanned_x, input_name)
            carry = tree_checkpoint_name(carry, carry_name)

            # this part is the most delicate: combining the scanned x with the unscanned x
            scanned_x = eqx.combine(scanned_x, unscanned_xs, is_leaf=is_named_array)
            args, kwargs = scanned_x
            carry, y = f(carry, *args, **kwargs)
            y = htu.tree_map(_pacify_named_arrays, y)

            return carry, y

        true_axis = _infer_axis_size_from_tree(axis_first_xs, axis)
        axis_size = true_axis.size

        # build a mapping from positional argument indices to their names for friendlier error messages
        sig = inspect.signature(f)
        arg_pos_names: dict[int, str] = {}
        params = list(sig.parameters.values())[1:]  # skip carry
        pos_count = 0
        var_pos_name: str | None = None
        for param in params:
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                arg_pos_names[pos_count] = param.name
                pos_count += 1
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                var_pos_name = param.name
                break
        if var_pos_name is not None:
            for i in range(pos_count, len(args)):
                arg_pos_names[i] = f"{var_pos_name}[{i - pos_count}]"

        path_leaves, _ = jtu.tree_flatten_with_path(axis_first_xs, is_leaf=is_named_array)
        mismatched = []
        for path, leaf in path_leaves:
            if isinstance(leaf, NamedArray):
                lead_size = leaf.array.shape[0]
            elif is_jax_array_like(leaf):
                lead_size = leaf.shape[0]
            else:
                continue
            if lead_size != axis_size:
                mismatched.append((path, lead_size))
        if mismatched:
            details = ", ".join(
                f"{_format_tree_path(p, arg_pos_names)} has leading dimension {s}" for p, s in mismatched
            )
            raise ValueError(
                f"scan got `length` argument of {axis_size} but some inputs had different leading axis sizes: {details}"
            )

        nested_scan = checkpoint.nested
        outer_block_size = nested_scan_outer_block(nested_scan, axis_size)

        checkpointed_fn = checkpoint.checkpoint(carry_name, input_name, wrapped_fn)

        # as above, we don't want to use htu.tree_leaves here because we want to eliminate the leading axis
        leaves = jax.tree_util.tree_leaves(axis_first_xs)

        with jax.named_scope(f"scan({haliax.axis_name(axis)})"):
            if outer_block_size is not None:
                carry, ys = multilevel_scan(
                    checkpointed_fn, init, leaves, outer_block_size, reverse=reverse, unroll=unroll, length=axis_size
                )
            else:
                carry, ys = jax.lax.scan(
                    checkpointed_fn, init, leaves, reverse=reverse, unroll=unroll, length=axis_size
                )

        ys = jax.tree_util.tree_map(_prepend_named_batch_axis(true_axis), ys, is_leaf=_is_passive_array)

        return carry, ys

    def nested_scan_outer_block(nested_remat, axis_size):
        outer_block_size: int | None
        if nested_remat is True:
            outer_block_size = find_closest_divisible_int_to_sqrt(axis_size)
        elif nested_remat is False:
            outer_block_size = None
        else:
            outer_block_size = nested_remat
        return outer_block_size

    return scanned_f


@overload
def fold(
    fn: Callable[[Carry, X], Carry],
    axis: AxisSelector,
    *,
    remat: ScanCheckpointSpec = False,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
) -> Callable[[Carry, PyTree[X]], Carry]: ...


@overload
def fold(
    fn: Callable,
    axis: AxisSelector,
    *,
    remat: ScanCheckpointSpec = False,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
) -> Callable: ...


def fold(
    fn: Callable,
    axis: AxisSelector,
    *,
    remat: ScanCheckpointSpec = False,
    reverse: bool = False,
    unroll: int = 1,
    is_scanned: BoolAxisSpec = is_named_or_shaped_array_like,
) -> Callable:
    """
    Slightly simpler implementation of scan that folds over the named axis of the array, not returning intermediates.

    As with scan, this function is curried: it takes the function, axis, and configuration arguments first, and
    then the initial carry and then any arguments to scan over as a separate curried function call.

    Unnamed arrays will have their first axis scanned over, unless they are scalars, in which case they will be passed
    through unchanged.

    Args:
        fn: function to reduce over
        axis: axis to reduce over
        reverse: if True, reduce in reverse
        unroll: unroll the loop by this amount
        nested_scan: Use nested scans to reduce memory usage. If an integer, use that many nested scans.
                If true, use the closest int to sqrt(axis.size) that divides axis.size, which gives
                O(sqrt(N)) memory and O(N) time.
        is_scanned: a function that takes a leaf of the tree and returns True if it should be scanned over,
                    False otherwise. Behaves similarly to the `default` argument in filter_jit

    Returns:
        A function that takes the initial carry and then the arguments to reduce over, and returns the final carry
    """

    def scan_compatible_fn(carry, *args, **kwargs):
        return fn(carry, *args, **kwargs), None

    scan_preconfig = scan(scan_compatible_fn, axis, reverse=reverse, unroll=unroll, is_scanned=is_scanned, remat=remat)

    def scanned_f(init, *args, **kwargs):
        return scan_preconfig(init, *args, **kwargs)[0]

    return scanned_f


def map(
    fn: Callable[[X], Y],
    axis: Axis,
    *,
    remat: ScanCheckpointSpec = False,
    reverse: bool = False,
    unroll: int = 1,
    is_mapped: BoolAxisSpec = is_jax_or_hax_array_like,
) -> Callable[[PyTree[X]], PyTree[Y]]:
    """
    NamedArray aware version of jax.lax.map. Normal arrays are mapped according to the specs as in equinox.filter_map,
    except that the output axis is always 0 b/c it's annoying to make anything else work.

    You'll typically want to use map (instead of a vmap or just vectorized code) when you want to encourage XLA to
    loop over the axis to control memory.

    Args:
        fn (Callable):  function to map over
        axis (Axis): axis to map over
        reverse (bool): if True, map in reverse
        unroll (int): unroll the loop by this amount

    """

    def scan_compatible_fn(_, x):
        del _
        return None, fn(x)

    scan_preconfig = scan(scan_compatible_fn, axis, reverse=reverse, unroll=unroll, is_scanned=is_mapped, remat=remat)

    def scanned_f(*args, **kwargs):
        return scan_preconfig(None, *args, **kwargs)[1]

    return scanned_f


ResolvedUnnamedAxisSpec = int | None
UnnamedAxisSpec = ResolvedUnnamedAxisSpec | Callable[[Any], ResolvedUnnamedAxisSpec]


def _zero_if_array_else_none(x: Any) -> ResolvedUnnamedAxisSpec:
    return 0 if is_jax_array_like(x) else None


def _format_tree_path(path: tuple[jtu.KeyEntry, ...], arg_pos_names: dict[int, str] | None = None) -> str:
    parts: list[str] = []
    i = 0
    if len(path) >= 2 and isinstance(path[0], jtu.SequenceKey):
        if path[0].idx == 0 and isinstance(path[1], jtu.SequenceKey):
            name = (arg_pos_names or {}).get(path[1].idx)
            if name is not None:
                parts.append(name)
            else:
                parts.append(f"[{path[1].idx}]")
            i = 2
        elif path[0].idx == 1 and isinstance(path[1], jtu.DictKey):
            parts.append(str(path[1].key))
            i = 2
    for p in path[i:]:
        if isinstance(p, jtu.GetAttrKey):
            parts.append("." + p.name)
        elif isinstance(p, jtu.DictKey):
            parts.append(f"[{p.key!r}]")
        elif isinstance(p, jtu.SequenceKey):
            parts.append(f"[{p.idx}]")
        else:  # pragma: no cover - future-proofing
            parts.append(str(p))
    if parts and parts[0].startswith("."):
        parts[0] = parts[0][1:]
    return "".join(parts) or "<root>"


def _infer_axis_size_from_tree(result, axis):
    if isinstance(axis, str):
        result_leaves = jax.tree_util.tree_leaves(result, is_leaf=_is_passive_array)
        if len(result_leaves) == 0:
            # this really shouldn't happen
            return None
        if isinstance(result_leaves[0], _PassiveNamedArray):
            true_axis_size = result_leaves[0].array.shape[0]  # batch axis is defined to be 0 above
            true_axis = Axis(axis, true_axis_size)
        else:
            true_axis_size = result_leaves[0].shape[0]  # batch axis is defined to be 0 above
            true_axis = Axis(axis, true_axis_size)
    else:
        true_axis = axis
    return true_axis


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class _PassiveNamedArray:
    """For some higher-order jax manipulations, jax will insert/remove an axis at the beginning of the array (or
    sometimes elsewhere). Jax doesn't know about our NamedArray wrapper, so we need a variant of NamedArray
    that doesn't care about the axis names. This is that variant.This class is a 'chill' version of NamedArray
    that doesn't check invariants until we're ready to create the result

    For example, with scan in NamedArray, we can't just have the scan tree prepend the scanned axis to the result,
    because we don't have a way to feed it the name of the scanned axis.
    """

    array: jax.numpy.ndarray
    main_axes: tuple[Axis, ...]

    def as_scanned_result(self, scan_axis: Axis):
        return NamedArray(self.array, (scan_axis,) + self.main_axes)

    def strip_axis(self, axis: AxisSelector):
        if isinstance(axis, Axis):
            index = self.main_axes.index(axis)
        else:
            index = index_where(lambda a: a.name == axis, self.main_axes)
        return NamedArray(self.array, self.main_axes[:index] + self.main_axes[index + 1 :])

    def to_named_array(self):
        return NamedArray(self.array, self.main_axes)

    def tree_flatten(self) -> Any:
        return ((self.array,), self.main_axes)

    @classmethod
    def tree_unflatten(cls, aux, tree: Any) -> Any:
        assert len(tree) == 1
        return cls(tree[0], main_axes=aux)


def _pacify_named_arrays(leaf):
    if isinstance(leaf, NamedArray):
        return _PassiveNamedArray(leaf.array, leaf.axes)
    elif isinstance(leaf, _PassiveNamedArray):
        assert False, "PassiveNamedArray should not be present in the tree"
    else:
        return leaf


def _select_0th(axis):
    def select_0th(leaf):
        if isinstance(leaf, NamedArray):
            return leaf.take(axis, 0)
        elif isinstance(leaf, _PassiveNamedArray):
            assert False, "PassiveNamedArray should not be present in the tree"
        else:
            # other leaves don't matter
            return leaf

    return select_0th


def _ensure_first(axis):
    def ensure_first(leaf):
        if isinstance(leaf, NamedArray):
            return leaf.rearrange((axis, ...))
        elif isinstance(leaf, _PassiveNamedArray):
            assert False, "PassiveNamedArray should not be present in the tree"
        else:
            return leaf

    return ensure_first


def find_closest_divisible_int_to_sqrt(n: int) -> int:
    """
    Find the closest integer to the square root of n (less than or equal to sqrt(n)) that divides n.
    """
    assert n > 0, f"Expected n > 0, got {n}"
    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return i

    return 1


def _is_passive_array(arr):
    return isinstance(arr, _PassiveNamedArray)


def _prepend_named_batch_axis(leading_axis: Axis):
    def to_active_named_array(leaf):
        if isinstance(leaf, _PassiveNamedArray):
            return leaf.as_scanned_result(leading_axis)
        else:
            return leaf

    return to_active_named_array
