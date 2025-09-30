# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


import contextlib
import dataclasses
import functools
import threading
import typing
import warnings
from math import prod
from typing import Callable, ContextManager, Mapping, Optional, ParamSpec, Sequence, TypeVar, Union, cast

import equinox as eqx
import jax
from equinox import is_array, module_update_wrapper
from jax.lax import with_sharding_constraint
from jax.sharding import AbstractMesh, NamedSharding, Mesh, PartitionSpec, SingleDeviceSharding, get_abstract_mesh


from jaxtyping import PyTree

import haliax.tree_util as htu
from haliax._src.compile_utils import compile_cache

from .axis import Axis, AxisSelection, AxisSelector, axis_spec_to_shape_dict
from .core import NamedArray
from .jax_utils import Static, is_in_jit, is_jax_array_like, is_on_mac_metal
from .tree_util import hashable_combine, hashable_partition
from .util import StringHolderEnum

PhysicalAxisSpec = Union[(str), Sequence[str]]
ResourceMapping = Mapping[(str), PhysicalAxisSpec]
MeshLike = Union[Mesh, AbstractMesh]
"""Mapping from logical axis names to physical axis names"""

F = typing.TypeVar("F", bound=typing.Callable)
Args = ParamSpec("Args")
R = typing.TypeVar("R", covariant=True)
T = TypeVar("T", bound=PyTree)


class ResourceAxis(StringHolderEnum):
    """Standard names for physical axes"""

    MODEL = "model"
    DATA = "data"
    REPLICA = "replica"


class _ResourceMappingHolder:
    """Global resource mapping, used with a context manager to give dynamic scoping to resource mappings"""

    def __init__(self):
        self.thread_data = threading.local()
        self.thread_data.resource_mapping = None


_mapping_holder = _ResourceMappingHolder()


@contextlib.contextmanager
def axis_mapping(mapping: ResourceMapping, *, merge: bool = False, **kwargs):
    """Context manager for setting the global resource mapping"""
    mapping = dict(mapping)

    old_mapping = current_thread_local_mapping()
    if merge:
        mapping.update(old_mapping or {})

    if len(kwargs):
        mapping.update(kwargs)

    _mapping_holder.thread_data.resource_mapping = mapping
    try:
        yield
    finally:
        _mapping_holder.thread_data.resource_mapping = old_mapping


def current_thread_local_mapping():
    """
    Get the current thread-local resource mapping, or None if there is no resource mapping set.
    :return:
    """
    if _mapping_holder.thread_data is None:
        return None
    if not hasattr(_mapping_holder.thread_data, "resource_mapping"):
        return None

    return _mapping_holder.thread_data.resource_mapping


def _resolve_mesh(mesh: Optional[MeshLike] = None) -> Optional[MeshLike]:
    """Inside jit, prefer an abstract mesh, outside jit prefer a concrete mesh."""

    from jax._src.mesh import get_concrete_mesh

    if mesh is not None:
        if is_in_jit() and isinstance(mesh, Mesh):
            return mesh.abstract_mesh
        return mesh

    if is_in_jit():
        abstract = get_abstract_mesh()
        if not abstract or abstract.empty:
            concrete = get_concrete_mesh()
            if concrete is not None and not concrete.empty:
                return concrete.abstract_mesh

        from jax.interpreters.pxla import thread_resources

        old_mesh = thread_resources.env.physical_mesh
        if old_mesh is not None and not old_mesh.empty:
            return old_mesh.abstract_mesh

        return abstract
    else:
        mesh = get_concrete_mesh() or get_abstract_mesh()
        if mesh is not None and not mesh.empty:
            return mesh

        from jax.interpreters.pxla import thread_resources

        old_mesh = thread_resources.env.physical_mesh
        if old_mesh is not None and not old_mesh.empty:
            return old_mesh

    return None


def mesh_context(mesh: MeshLike) -> ContextManager[None]:
    """Context manager that normalizes mesh handling across JAX versions."""

    set_mesh_fn = getattr(jax, "set_mesh", None)
    use_mesh_fn = getattr(jax.sharding, "use_mesh", None)

    manager_factory: Optional[Callable[[MeshLike], ContextManager[None]]] = None
    if set_mesh_fn is not None:
        manager_factory = cast(Callable[[MeshLike], ContextManager[None]], set_mesh_fn)
    elif use_mesh_fn is not None:
        manager_factory = cast(Callable[[MeshLike], ContextManager[None]], use_mesh_fn)

    if manager_factory is None:
        msg = "Haliax requires a version of JAX that provides either `jax.set_mesh` or `jax.sharding.use_mesh`."
        raise RuntimeError(msg)

    context_manager = manager_factory(mesh)

    return context_manager


def set_mesh(mesh: MeshLike) -> ContextManager[None]:
    """Compatibility wrapper around `mesh_context` matching the JAX 0.7 API."""

    return mesh_context(mesh)


def auto_sharded(x: T, mesh: Optional[Mesh] = None) -> T:
    """
    Shard a PyTree using the global axis mapping. NamedArrays in the PyTree are sharded using the axis mapping
     and the names in the tree.

    If there is no axis mapping, the global axis mapping, this function is a no-op.
    """
    mapping = current_thread_local_mapping()

    if mapping is None:
        return x

    return shard(x, mapping=mapping, mesh=mesh)


def shard(x: T, mapping: Optional[ResourceMapping] = None, mesh: Optional[Mesh] = None) -> T:
    """
    Shard a PyTree using the provided axis mapping. NamedArrays in the PyTree are sharded using the axis mapping.
    Other arrays (i.e. plain JAX arrays) are left alone.

    This is basically a fancy wrapper around `with_sharding_constraint` that uses the axis mapping to determine
    the sharding.
    """

    if mapping is None:
        mapping = current_thread_local_mapping()

    if mapping is None:
        if not is_in_jit():
            warnings.warn("No resource mapping found. Not sharding.", RuntimeWarning)
        return x

    assert not isinstance(mesh, dict)

    resolved_mesh = _resolve_mesh(mesh)

    if resolved_mesh is None:
        if not is_in_jit():
            warnings.warn("No mesh found. Not sharding.", RuntimeWarning)
        return x

    if isinstance(resolved_mesh, AbstractMesh) and resolved_mesh.empty:
        return x

    if is_in_jit() and is_on_mac_metal():
        warnings.warn("Sharding constraints are not supported in jit on metal", RuntimeWarning)
        return x

    def _do_device_put(named):
        if not isinstance(named, NamedArray):
            return named

        if not is_jax_array_like(named.array):
            # this happens when we filter out params for things like lora.
            # could use eqx.partition to avoid this, but eh
            return named

        pspec = pspec_for(named, mapping, preserve_existing_shardings=False)
        assert isinstance(pspec, PartitionSpec)
        sharding = NamedSharding(resolved_mesh, pspec)
        if is_in_jit():
            return with_sharding_constraint(named, sharding)
        else:
            ret = jax.device_put(named, sharding)
            return ret

    return htu.tree_map(_do_device_put, x)


@functools.wraps(shard)
def shard_with_axis_mapping(x: T, mapping: ResourceMapping, mesh: Optional[Mesh] = None) -> T:
    # warnings.warn("`shard_with_axis_mapping` is deprecated. Use `shard` instead", DeprecationWarning)
    return shard(x, mapping, mesh)


def pspec_for(
    tree: PyTree,
    resource_mapping: Optional[ResourceMapping] = None,
    preserve_existing_shardings: bool = True,
) -> PyTree:
    """Infer the :class:`PartitionSpec` for a module.

    This behaves like :func:`infer_resource_partitions` but returns ``PartitionSpec``
    objects instead of :class:`~jax.sharding.NamedSharding`. It is primarily a helper
    for :func:`infer_resource_partitions` but may be useful when only the partition
    specification is required.

    If ``preserve_existing_shardings`` is ``True``, then arrays that already have a
    sharding are left untouched and ``None`` is returned for those leaves.
    """
    if resource_mapping is None:
        resource_mapping = current_thread_local_mapping()

    if resource_mapping is None:
        raise ValueError("No resource mapping found")

    def partition_spec(node: typing.Any):
        if isinstance(node, NamedArray):
            # If our NamedArray doesn't have an array (or a shapedtypestruct), we can't shard it
            if not is_jax_array_like(node.array):
                return None

            current_sharding = getattr(node.array, "sharding", None) if preserve_existing_shardings else None
            if current_sharding is not None:
                return None
            else:
                return pspec_for_axis(node.axes, resource_mapping)
        elif isinstance(node, eqx.Module):
            # handle eqx.Module explicitly so that we can look at axis_names metadata
            updates: dict[str, typing.Any] = {}
            for field in dataclasses.fields(node):
                if field.metadata.get("static", False):
                    continue

                value = getattr(node, field.name)
                axis_names = field.metadata.get("axis_names") if field.metadata is not None else None
                if axis_names is not None and is_jax_array_like(value):
                    current_sharding = getattr(value, "sharding", None) if preserve_existing_shardings else None
                    if current_sharding is not None:
                        updates[field.name] = None
                    else:
                        updates[field.name] = pspec_for_axis(axis_names, resource_mapping)
                else:
                    updates[field.name] = htu.tree_map(
                        partition_spec, value, is_leaf=lambda x: isinstance(x, eqx.Module)
                    )

            new_node = object.__new__(type(node))
            for field in dataclasses.fields(node):
                object.__setattr__(
                    new_node,
                    field.name,
                    updates.get(field.name, getattr(node, field.name)),
                )

            return new_node
        elif is_jax_array_like(node):
            sharding = getattr(node, "sharding", None)
            # TODO: these are usually replicated. Is there a better way to tell?
            if node.shape == ():
                return PartitionSpec()
            elif isinstance(sharding, SingleDeviceSharding):
                return PartitionSpec(None)
            elif sharding is not None and preserve_existing_shardings:
                return None
            # elif use_auto_sharding:
            # TODO: auto doesn't seem to really work reliably yet
            #     compat between 0.4.10 and 0.4.11
            # if isinstance(AUTO, typing.Callable):  # type: ignore
            #     return AUTO(mesh)
            # else:
            #     return AUTO
            return PartitionSpec(None)
        elif isinstance(node, (bool, float, complex, int)):
            return PartitionSpec()
        else:
            return None

    return htu.tree_map(partition_spec, tree, is_leaf=lambda x: isinstance(x, eqx.Module))


def infer_resource_partitions(
    tree: PyTree,
    resource_mapping: Optional[ResourceMapping] = None,
    preserve_existing_shardings: bool = True,
    mesh: Optional[Mesh] = None,
) -> PyTree:
    """
    Infer the sharding for a module, to be used with ``named_jit``.

    This first calls :func:`pspec_for` to compute ``PartitionSpec`` objects and then
    wraps them in :class:`~jax.sharding.NamedSharding` using the provided mesh. If
    ``preserve_existing_shardings`` is ``True``, then arrays that are already sharded
    retain their current sharding.
    """
    pspecs = pspec_for(
        tree,
        resource_mapping=resource_mapping,
        preserve_existing_shardings=preserve_existing_shardings,
    )

    resolved_mesh = _resolve_mesh(mesh)
    if resolved_mesh is None:
        raise ValueError("No mesh found")
    assert not isinstance(resolved_mesh, dict)

    def to_sharding(node: typing.Any, spec: typing.Any):
        if spec is None:
            if isinstance(node, NamedArray):
                return getattr(node.array, "sharding", None)
            elif is_jax_array_like(node):
                return getattr(node, "sharding", None)
            else:
                return None
        else:
            return NamedSharding(resolved_mesh, spec)

    return htu.tree_map(to_sharding, tree, pspecs)


class WrappedCallable(typing.Protocol[Args, R]):
    """
    A wrapper for a callable that preserves the original function's name and qualname.
    """

    def __call__(self, *args: Args.args, **kwargs: Args.kwargs) -> R:
        raise NotImplementedError

    def lower(self, *args: Args.args, **kwargs: Args.kwargs) -> jax.stages.Lowered:
        raise NotImplementedError


class _NamedJitWrapper(eqx.Module):
    _fn: Callable  # [Args, R]
    _dynamic_fun: PyTree
    _static_fun: typing.Any
    _axis_resources: Optional[ResourceMapping]
    _in_axis_resources: Optional[ResourceMapping]
    _out_axis_resources: Optional[ResourceMapping]
    _donate_args: Optional[PyTree]
    _donate_kwargs: Optional[PyTree]
    _pjit_args: Mapping[str, typing.Any]

    @property
    def __wrapped__(self):
        return self._fn

    def __call__(self, *args, **kwargs):
        return self._call(False, *args, **kwargs)

    def lower(self, *args, **kwargs) -> jax.stages.Lowered:
        return self._call(True, *args, **kwargs)

    def _call(self, is_lower, *args, **kwargs):
        axis_resources = self._axis_resources
        if axis_resources is None:
            axis_resources = current_thread_local_mapping()

        in_axis_resources = self._in_axis_resources
        out_axis_resources = self._out_axis_resources

        if out_axis_resources is None:
            out_axis_resources = axis_resources

        dynamic_argspec, static_argspec = hashable_partition((args, kwargs), is_array)
        dynamic = (self._dynamic_fun, dynamic_argspec)

        donate_args = self._donate_args
        donate_kwargs = self._donate_kwargs

        if donate_args is not None or donate_kwargs is not None:
            if donate_args is None:
                dargs = (False,) * len(args)
            elif isinstance(donate_args, bool):
                dargs = (donate_args,) * len(args)
            elif not isinstance(donate_args, tuple):
                dargs = tuple(donate_args)
            else:
                dargs = donate_args

            if len(dargs) < len(args):
                dargs = dargs + (False,) * (len(args) - len(dargs))

            if len(dargs) != len(args):
                raise ValueError(f"Expected {len(args)} donate_args, got {len(dargs)}")

            dkwargs = donate_kwargs or {k: False for k in kwargs}
            dkwargs = {k: dkwargs.get(k, False) for k in kwargs}
            dynamic_donated, dynamic_reserved = eqx.partition(dynamic, (False, (dargs, dkwargs)))
        else:
            dynamic_donated = jax.tree_util.tree_map(lambda _: None, dynamic)
            dynamic_reserved = dynamic

        static = (self._static_fun, static_argspec)

        cmanager: ContextManager
        if axis_resources is not None:
            cmanager = axis_mapping(axis_resources)
        else:
            cmanager = contextlib.nullcontext()

        with cmanager:
            output_shape = _cached_filter_eval_shape(self._fn, *args, **kwargs)
            my_pjit_args = dict(**self._pjit_args)

            if in_axis_resources is not None:
                in_resources = infer_resource_partitions(
                    (dynamic_donated, dynamic_reserved),
                    in_axis_resources,
                    preserve_existing_shardings=in_axis_resources is None,
                )
                my_pjit_args["in_shardings"] = in_resources

            if out_axis_resources is not None:
                # TODO: when AUTO is fixed (or eval_shape can give shardings), use it here
                out_resources = infer_resource_partitions(
                    output_shape, out_axis_resources, preserve_existing_shardings=False
                )
                my_pjit_args["out_shardings"] = out_resources

            cached_pjitted_fun = _named_pjit_cache(self._fn, **my_pjit_args)
            if is_lower:
                return cached_pjitted_fun.lower(dynamic_donated, dynamic_reserved, static)
            else:
                out, out_static = cached_pjitted_fun(dynamic_donated, dynamic_reserved, static)
                out = hashable_combine(out, out_static.value)

                return out


@typing.overload
def named_jit(
    fn: Callable[Args, R],
    axis_resources: Optional[ResourceMapping] = None,
    *,
    in_axis_resources: Optional[ResourceMapping] = None,
    out_axis_resources: Optional[ResourceMapping] = None,
    donate_args: Optional[PyTree] = None,
    donate_kwargs: Optional[PyTree] = None,
    # args from jit
    keep_unused: bool = False,
    backend: Optional[str] = None,
    inline: Optional[bool] = None,
) -> WrappedCallable[Args, R]: ...


@typing.overload
def named_jit(
    *,
    axis_resources: Optional[ResourceMapping] = None,
    in_axis_resources: Optional[ResourceMapping] = None,
    out_axis_resources: Optional[ResourceMapping] = None,
    donate_args: Optional[PyTree] = None,
    donate_kwargs: Optional[PyTree] = None,
    # args from jit
    keep_unused: bool = False,
    backend: Optional[str] = None,
    inline: Optional[bool] = None,
) -> typing.Callable[[Callable[Args, R]], WrappedCallable[Args, R]]: ...


def named_jit(
    fn: Optional[Callable[Args, R]] = None,
    axis_resources: Optional[ResourceMapping] = None,
    *,
    in_axis_resources: Optional[ResourceMapping] = None,
    out_axis_resources: Optional[ResourceMapping] = None,
    donate_args: Optional[PyTree] = None,
    donate_kwargs: Optional[PyTree] = None,
    **pjit_args,
) -> typing.Union[WrappedCallable[Args, R], typing.Callable[[Callable[Args, R]], WrappedCallable[Args, R]]]:
    """
    A version of pjit that uses NamedArrays and the provided resource mapping to infer resource partitions for
    sharded computation for.

    `axis_resources` will be used for a context-specific resource mapping when the function is invoked.
    In addition, if in_axis_resources is not provided, the arguments' own (pre-existing) shardings will be used as the in_axis_resources.
    If out_axis_resources is not provided, axis_resources will be used as the out_axis_resources.

    If no resource mapping is provided, this function attempts to use the context resource mapping.

    Functionally this is very similar to something like:

    This function can be used as a decorator or as a function.

    ```python
     def wrapped_fn(arg):
        result = fn(arg)
        return hax.shard(result, out_axis_resources)

     arg = hax.shard(arg, in_axis_resources)
     with hax.axis_mapping(axis_resources):
        result = jax.jit(wrapped_fn, **pjit_args)(arg)
    return result
    ```

    Args:
        fn (Callable, optional): The function to be jit'd.
        axis_resources (ResourceMapping, optional): A mapping from logical axis names to physical axis names use for
                the context-specific resource mapping.
        in_axis_resources (ResourceMapping, optional): A mapping from logical axis names to physical axis names for
                arguments. If not passed, it uses the argument's own shardings.
        out_axis_resources (ResourceMapping, optional): A mapping from logical axis names to physical axis names for the
                result, defaults to axis_resources.
        donate_args (PyTree, optional): A PyTree of booleans or function leaf->bool, indicating if the arguments should
                be donated to the computation.
        donate_kwargs (PyTree, optional): A PyTree of booleans or function leaf->bool, indication if the keyword
                arguments should be donated to the computation.

    Returns:
        A jit'd version of the function.
    """

    if fn is None:
        return functools.partial(  # type: ignore
            named_jit,  # type: ignore
            axis_resources=axis_resources,
            in_axis_resources=in_axis_resources,
            out_axis_resources=out_axis_resources,
            donate_args=donate_args,
            donate_kwargs=donate_kwargs,
            **pjit_args,
        )

    dynamic_fun, static_fun = hashable_partition(fn, is_array)

    wrapper = _NamedJitWrapper(
        fn,
        dynamic_fun,
        static_fun,
        axis_resources,
        in_axis_resources,
        out_axis_resources,
        donate_args,
        donate_kwargs,
        pjit_args,
    )

    return module_update_wrapper(wrapper, fn)  # type: ignore


@typing.overload
def fsdp(fn: F, parameter_mapping: ResourceMapping, compute_mapping: ResourceMapping) -> F: ...


@typing.overload
def fsdp(parameter_mapping: ResourceMapping, compute_mapping: ResourceMapping) -> typing.Callable[[F], F]: ...


def fsdp(*args, **kwargs):
    """
    A convenience wrapper around named_jit / pjit to encode the FSDP pattern. It's basically equivalent to this:

    ```python
    @named_jit(in_axis_resources=parameter_mapping, out_axis_resources=parameter_mapping, axis_resources=compute_mapping)
    def f(*args, **kwargs):
        return fn(*args, **kwargs)
    ```

    This function can be used as a decorator or as a function.
    """
    if "fn" in kwargs:
        return _fsdp_impl(*args, **kwargs)
    elif len(args) > 1 and callable(args[0]):
        return _fsdp_impl(*args, **kwargs)
    else:
        return lambda fn: _fsdp_impl(fn, *args, **kwargs)


def _fsdp_impl(fn: F, parameter_mapping, compute_mapping):
    return named_jit(
        fn, in_axis_resources=parameter_mapping, out_axis_resources=parameter_mapping, axis_resources=compute_mapping
    )


# This is more or less copy-pasted from Equinox's similar functions (pmap, vmap, etc), but
# it's not really explained there so we'll explain it here.
# Many jax functions work by compiling functions to XLA. The compilation process is expensive,
# so we want to cache the compiled functions. However, the compiled functions are tied to the
# "static" arguments to the functions. This is particularly important for a library like Equinox,
# which Haliax is built on top of, because Equinox uses pytrees extensively for modules, and mixes "static"
# configuration with "dynamic" data.
# Thus we need to carefully partition the arguments to the function into "static" and "dynamic" arguments,
# and cache our compiled functions based on the static arguments.
# In Equinox conceptually there are three types of "arguments": positional, named, and the function itself.
# All of these are pytrees, and we need to partition them into static and dynamic arguments.
# Inside the function, we then combine the arguments into a single pytree, and pass that to the original function.
# With pjit we also have "donated" arguments, which are arguments that we promise not to use after the function
# returns. This is useful for conserving memory, but we also have to splice them back in.
# Also recall that a "pytree" can split into leaves and a "treedef", which can then be reconstructed.
@compile_cache
def _named_pjit_cache(fun_names, **jitkwargs) -> WrappedCallable:
    def fun_wrapped(dynamic_donated, dynamic_reserved, static):
        dynamic = eqx.combine(dynamic_donated, dynamic_reserved)
        dynamic_fun, dynamic_spec = dynamic
        static_fun, static_spec = static

        fun = hashable_combine(dynamic_fun, static_fun)
        args, kwargs = hashable_combine(dynamic_spec, static_spec)
        out = fun(*args, **kwargs)
        out_dynamic, out_static = hashable_partition(out, is_array)
        return out_dynamic, Static(out_static)

    fun_name, fun_qualname = fun_names
    fun_wrapped.__name__ = fun_name
    fun_wrapped.__qualname__ = fun_qualname

    jitkwargs = dict(jitkwargs)
    if "out_shardings" in jitkwargs:
        out_shardings = jitkwargs["out_shardings"]
        # None for the static
        jitkwargs["out_shardings"] = (out_shardings, None)

    return jax.jit(
        fun_wrapped,
        donate_argnums=0,
        static_argnums=2,
        **jitkwargs,
    )


_eval_shape_cache = {}


def _cached_filter_eval_shape(fun, *args, **kwargs):
    """
    eval_shape is surprisingly expensive, so we cache it. We use this for named_pjit for evaluating resource partitions
    of the output.
    """
    dynamic, static = hashable_partition((fun, args, kwargs), is_array)
    if static not in _eval_shape_cache:
        _eval_shape_cache[static] = eqx.filter_eval_shape(fun, *args, **kwargs)

    return _eval_shape_cache[static]


def physical_axis_name(axis: AxisSelector, mapping: Optional[ResourceMapping] = None) -> Optional[PhysicalAxisSpec]:
    """Get the physical axis name for a logical axis from the mapping. Returns none if the axis is not mapped."""
    if mapping is None:
        mapping = current_thread_local_mapping()
    if mapping is None:
        return None
    elif isinstance(axis, str):
        return mapping.get(axis, None)
    else:
        return mapping.get(axis.name, None)


def physical_axis_size(axis: AxisSelector, mapping: Optional[ResourceMapping] = None) -> Optional[int]:
    """Get the physical axis size for a logical axis. This is the product of the size of all physical axes
    that this logical axis is mapped to."""
    mesh = _resolve_mesh()

    if mesh is None:
        raise ValueError("No mesh found")

    mesh_shape = mesh.shape

    name: Union[None, str, Sequence[str]] = physical_axis_name(axis, mapping)
    if name is None:
        return None
    elif isinstance(name, str):
        name = (name,)

    return prod([mesh_shape[n] for n in name])


def sharding_for_axis(
    axis: AxisSelection, mapping: Optional[ResourceMapping] = None, mesh: Optional[MeshLike] = None
) -> NamedSharding:
    """Get the sharding for a single axis"""
    resolved_mesh = _resolve_mesh(mesh)
    if resolved_mesh is None:
        raise ValueError("No mesh found")

    return NamedSharding(resolved_mesh, pspec_for_axis(axis, mapping))


def pspec_for_axis(axis: AxisSelection, mapping: Optional[ResourceMapping] = None) -> PartitionSpec:
    """Get the PartitionSpec for a single axis"""
    axis = axis_spec_to_shape_dict(axis)
    return PartitionSpec(*(physical_axis_name(a, mapping) for a in axis))


def round_axis_for_partitioning(axis: Axis, mapping: Optional[ResourceMapping] = None) -> Axis:
    """Round an axis so that it's divisible by the size of the partition it's on"""
    size = physical_axis_size(axis, mapping)
    if size is None:
        return axis
    else:
        new_size = (axis.size + size - 1) // size * size
        return Axis(axis.name, new_size)


def _get_mesh() -> Mesh | None:
    """Deprecated helper that simply proxies to :func:`get_abstract_mesh`."""

    warnings.warn(
        "`_get_mesh` is deprecated; use `jax's get_abstract_mesh or get_concrete_mesh` instead",
        DeprecationWarning,
        stacklevel=2,
    )

    mesh = _resolve_mesh()
    return mesh


def _is_jit_tracer(x) -> bool:
    if isinstance(x, NamedArray):
        x = x.array
    return isinstance(x, jax.core.Tracer)


__all__ = [
    "PhysicalAxisSpec",
    "ResourceAxis",
    "ResourceMapping",
    "axis_mapping",
    "auto_sharded",
    "shard",
    "shard_with_axis_mapping",
    "pspec_for",
    "infer_resource_partitions",
    "named_jit",
    "fsdp",
    "physical_axis_name",
    "pspec_for_axis",
    "round_axis_for_partitioning",
    "current_thread_local_mapping",
]
