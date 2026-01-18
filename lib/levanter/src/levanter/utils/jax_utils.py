# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import functools
import json
import warnings
import zlib
from dataclasses import fields
from typing import Any, Callable, Optional, TypeVar

import equinox as eqx
import haliax as hax
import haliax.partitioning
import jax
import numpy as np
from haliax import is_named_array
from haliax.jax_utils import is_jax_array_like
from haliax.partitioning import ResourceAxis, ResourceMapping
from jax import numpy as jnp
from jax._src.mesh import get_concrete_mesh
from jax.experimental.multihost_utils import host_local_array_to_global_array, process_allgather
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jaxtyping import PRNGKeyArray, PyTree

from levanter.utils.mesh import create_mesh_from_axis_specs
from levanter.utils.tree_utils import key_path_to_str, tree_flatten_one_level_with_keys

X = TypeVar("X")
T = TypeVar("T", bound=PyTree)
L = TypeVar("L")


def jnp_to_python(a: jnp.ndarray):
    if isinstance(a, (float, int)):
        return float(a)
    elif a.shape == () or a.shape == (1,):
        return a.item()
    else:
        return a.tolist()


@contextlib.contextmanager
def use_cpu_device():
    """Temporarily sets the default device to CPU"""
    cpu = jax.local_devices(backend="cpu")[0]
    with jax.default_device(cpu):
        yield cpu


@contextlib.contextmanager
def local_cpu_mesh():
    """Temporarily sets the default device to CPU and creates a mesh with a single CPU device"""
    cpu = jax.local_devices(backend="cpu")[0]
    mesh = create_mesh_from_axis_specs(
        ici_axes={
            ResourceAxis.REPLICA: 1,
            ResourceAxis.DATA: 1,
            ResourceAxis.MODEL: 1,
            ResourceAxis.CONTEXT: 1,
        },
        dcn_axes={},
        devices=[cpu],
    )
    with use_cpu_device(), haliax.partitioning.set_mesh(mesh):
        yield mesh


def is_inside_jit():
    """Returns True if we're currently inside a jit"""
    return isinstance(jnp.zeros(()), jax.core.Tracer)


def parameter_count(model: PyTree):
    # especially with jax.vjp, we get duplicate arrays and want to uniq them
    # NB we need to use object identity here, mostly because of ShapedDtypeStruct
    leaves = {id(x): x for x in jax.tree_util.tree_leaves(model) if is_jax_array_like(x)}
    return sum(x.size for x in leaves.values())


_sync_counter = 0


def multihost_broadcast_sync(obj: X, is_source: Optional[bool] = None, timeout: float = 200.0) -> X:
    """
    Uses jax's unpublished distributed api to sync a value across hosts using json dump. If is_source is None, then
    process_index 0 is the source.
    """
    global _sync_counter
    key = f"LEVANTER_MULTIHOST_BROADCAST_SYNC{_sync_counter}"
    if is_source is None:
        is_source = jax.process_index() == 0

    if jax.process_count() == 1:
        return obj

    import jax._src.distributed as distributed

    client = distributed.global_state.client

    if client is None:
        raise RuntimeError("multihost_broadcast_sync requires jax distributed client to be initialized")

    if is_source:
        # serialized = pickle.dumps(obj, 0)  # 0 is pickle protocol. jax only accepts utf-8, and 0 gives us ascii
        # client.key_value_set(key, serialized.decode("ascii"))
        serialized = json.dumps(obj)
        client.key_value_set(key, serialized)

    client.wait_at_barrier(f"multihost_broadcast_sync{_sync_counter}", timeout_in_ms=int(timeout * 1000.0))

    if not is_source:
        serialized = client.blocking_key_value_get(key, timeout_in_ms=int(timeout * 1000.0))
        obj = json.loads(serialized)

    _sync_counter += 1
    return obj


def barrier_sync(timeout: float = 200):
    """
    Uses jax's unpublished distributed api to wait for all processes to reach a barrier. This is useful for ensuring
    that all processes have reached a certain point in the code before continuing.
    """
    global _sync_counter
    if jax.process_count() == 1:
        return
    import jax._src.distributed as distributed

    try:
        from jaxlib.xla_extension import DistributedRuntimeClient
    except ModuleNotFoundError:  # jaxlib>=0.6.2
        from jax._src.lib import _jax as _jax_lib

        DistributedRuntimeClient = _jax_lib.DistributedRuntimeClient

    client: Optional[DistributedRuntimeClient] = distributed.global_state.client

    if client is None:
        raise RuntimeError("barrier_sync requires jax distributed client to be initialized")

    _sync_counter += 1
    client.wait_at_barrier(f"levanter_barrier_sync_{_sync_counter}", timeout_in_ms=int(timeout * 1000.0))


# from https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple
# python is a disgusting language
def _isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] is not tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(isinstance(n, str) for n in f)


def leaf_key_paths(
    pytree,
    prefix: Optional[str] = "",
    *,
    is_leaf: Optional[Callable[[Any], bool]] = None,
    use_state_dict_keys: bool = False,
):
    """Creates unique, hopefully meaningful key paths for each leaf in a pytree. This is useful for
    serialization mostly. This functions knows about dicts, lists, NamedTuples, tuples, and equinox-style modules"""
    # TODO: jax now has a tree_flatten_with_path function. We should use that instead
    rec = lambda x, p: leaf_key_paths(  # noqa: E731
        x, prefix=join_key(prefix, p), is_leaf=is_leaf, use_state_dict_keys=use_state_dict_keys
    )

    out: PyTree[str]

    if is_leaf is not None and is_leaf(pytree):
        out = prefix
    elif pytree is None:
        out = None
    elif isinstance(pytree, dict):
        out = {k: rec(v, k) for k, v in pytree.items()}
    elif _isnamedtupleinstance(pytree):
        d = {k: rec(v, k) for k, v in pytree._asdict().items()}
        out = pytree.__class__(**d)
    elif isinstance(pytree, list):
        out = [rec(v, str(i)) for i, v in enumerate(pytree)]
    elif isinstance(pytree, tuple):
        out = tuple(rec(v, str(i)) for i, v in enumerate(pytree))
    elif isinstance(pytree, eqx.Module):
        names = []
        rec_values = []
        for field in fields(pytree):
            if field.metadata.get("static", False):
                continue
            field_name = field.name
            field_value = getattr(pytree, field_name)
            names.append(field_name)

            if use_state_dict_keys and hasattr(pytree, "_state_dict_key_map"):
                field_name = pytree._state_dict_key_map().get(field_name, field_name)

            rec_value = rec(field_value, field_name)
            rec_values.append(rec_value)

        _, tree_def = eqx.tree_flatten_one_level(pytree)
        out = jax.tree_util.tree_unflatten(tree_def, rec_values)
    elif isinstance(pytree, hax.NamedArray):
        leaves, treedef = jax.tree_util.tree_flatten(pytree, is_leaf=is_leaf)
        out = jax.tree_util.tree_unflatten(treedef, [f"{prefix}"])
    else:
        leaves, treedef = jax.tree_util.tree_flatten(pytree)
        if len(leaves) == 0:
            out = None
        elif len(leaves) == 1:
            out = jax.tree_util.tree_unflatten(treedef, [f"{prefix}"])
        else:
            # new behavior: use registered keys
            leaves_with_keys, treedef = tree_flatten_one_level_with_keys(pytree)
            out_leaves = []
            for key, leaf in leaves_with_keys:
                if key is None:
                    out_leaves.append(join_key(prefix, ""))
                else:
                    key_str = key_path_to_str([key])
                    # out_leaves.append(join_key(prefix, key_str))
                    rec_pref = join_key(prefix, key_str)
                    out_leaves.append(
                        leaf_key_paths(leaf, rec_pref, is_leaf=is_leaf, use_state_dict_keys=use_state_dict_keys)
                    )
            out = jax.tree_util.tree_unflatten(treedef, out_leaves)

    # assert len(jax.tree.leaves(out, is_leaf=is_leaf)) == len(jax.tree.leaves(pytree, is_leaf=is_leaf)), (out, pytree)
    return out


def join_key(prefix, k):
    if k is None:
        return prefix
    return f"{prefix}.{k}" if prefix else k


def key_iterator(key: PRNGKeyArray | int):
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def is_inexact_arrayish(x):
    """
    Similar to [equinox.is_inexact_array][] but works on anything that has a shape and dtype
    and the dtype is inexact.

    Specifically, we want to work with [jax.ShapeDtypeStruct][]s, which are not arrays.
    """
    if hasattr(x, "shape") and hasattr(x, "dtype"):
        return jnp.issubdtype(x.dtype, jnp.inexact)
    else:
        return False


def tree_filter_like(template: X, tree: X) -> X:
    """
    Filters a tree to only include the leaves that are not None in the template.

    This is useful for filtering out nontrainable parameters from a tree.
    """

    def match_like(templ_leaf, tree_leaf):
        if templ_leaf is None:
            return None
        else:
            if tree_leaf is None:
                warnings.warn(f"Template has a non-None value where tree is None. Template value: {templ_leaf}")
            return tree_leaf

    return jax.tree_util.tree_map(match_like, template, tree, is_leaf=lambda x: x is None)


def best_effort_sharding(shape, *, devices=None, mesh=None):
    if hasattr(shape, "shape"):
        shape = shape.shape

    if devices is None:
        devices = jax.devices()

    if mesh is None:
        # TODO: we shouldn't be getting a concrete mesh here. Need to fix/remove this whole function
        mesh = get_concrete_mesh()
        if mesh is not None and mesh.shape == ():
            mesh = None

    if mesh is None:
        device_shape = (len(devices),)
        # we want to shard an array with shape shape across len(devices)
        # each axis in the array has to be divisible by the corresponding axis in device_shape, so
        # we iterate from the right, taking the gcd of the shape and the left-most axis of device_shape
        num_devices = device_shape[0]

        for i in range(len(shape) - 1, -1, -1):
            shape_i = shape[i]
            gcd = np.gcd(shape_i, num_devices)
            num_devices //= gcd
            device_shape = (num_devices, gcd) + device_shape[1:]

        device_mesh = np.array(devices).reshape(list(device_shape))
        axis_names = [f"d{i}" for i in range(len(shape))]
        mesh = Mesh(device_mesh, ["b"] + axis_names)
        sharding = NamedSharding(mesh, PartitionSpec(*axis_names))
        return sharding
    else:
        # get the existing mesh and find the FSDP axis
        num_devices = mesh.shape[hax.partitioning.ResourceAxis.DATA]

        for i in range(len(shape) - 1, -1, -1):
            shape_i = shape[i]
            if shape_i % num_devices == 0:
                sharded_axis = i
                break
        else:
            return NamedSharding(mesh, PartitionSpec(None))

        axis_sharding: list[str | None] = [None] * len(shape)
        axis_sharding[sharded_axis] = hax.partitioning.ResourceAxis.DATA
        sharding = NamedSharding(mesh, PartitionSpec(*axis_sharding))

        return sharding


def estimated_free_device_memory(device=None) -> Optional[float]:
    """
    Returns free memory in GiB. If the device doesn't support memory stats, returns None. If no device is provided,
    sums across all devices.
    Args:
        device: if None, sums all devices

    Returns:

    """
    if device is not None:
        devices = [device]
    else:
        devices = jax.devices()

    total = 0.0
    for device in devices:
        stats = device.memory_stats()
        if stats is None:
            return None
        else:
            limit = stats.get("bytes_limit", None)
            if limit is None:
                return None

            in_use = stats.get("bytes_in_use", 0)

            total += (limit - in_use) // (1024.0**3)

    return total


def zeros_like_tree(tree: T, axis_mapping: Optional[ResourceMapping] = None, dtype: Optional[jnp.dtype] = None) -> T:
    """
    Creates a tree of zeros with the same structure as the input tree. If the input tree contains NamedArrays, then
    those will be sharded according to the axis_mapping (or the context axis mapping if not provided).
    """
    _zeros = functools.partial(_zeros_like, axis_mapping, dtype)
    acc = jax.tree_util.tree_map(_zeros, tree, is_leaf=is_named_array)
    return acc


def _zeros_like(mapping, dtype, n):
    if isinstance(n, hax.NamedArray):
        return hax.shard(hax.zeros_like(n, dtype=dtype), mapping)
    elif is_jax_array_like(n):
        return jnp.zeros_like(n, dtype)
    else:
        assert jnp.isscalar(n)
        if dtype is None:
            # if it's a nan, we want to go to 0
            if n != n:
                return 0
            return n - n
        else:
            return jnp.zeros((), dtype=dtype)


def broadcast_shard(x: T, out_axis_specs: Any, source: int = 0) -> T:
    """
    Given a tree of arrays that are on a single source host, and other data (e.g. zeros) with
    the same structure, broadcast and shard the data to all hosts, using the axis mapping provided.

    For some reason, I had a ton of trouble figuring this out.

    Our strategy is, for each leaf:
     1. create a host_local_array_to_global_array with the data if we're the source, or zeros if we're not.
        This gives us an array [num_devices, ...]
     2. Then, inside jit, we select the source'th element of the array, then reshard with the out_axis_specs

    """
    # NOTE: Prior implementations attempted to use `jax.make_array_from_callback` with a `NamedSharding` constructed
    # from the active training mesh. On multi-host TPU, that can crash with "not fully addressable" / "not addressable
    # sharding" errors.
    #
    # We broadcast using a temporary mesh over ("processes", "local_devices") and a reduction across the sharded
    # "processes" axis. Some callers (notably eval-harness control-plane messages) use `PartitionSpec()` (replicated)
    # outputs; in that case we intentionally return host-local arrays to avoid device-order mismatches between the
    # temporary mesh and the active training mesh.

    def _maybe_constrain(arr: jax.Array, spec: Any) -> jax.Array:
        # `jax.jit` (and some multihost contexts) can error if we try to apply a sharding constraint with a sharding
        # that isn't fully addressable from the current host. In practice, callers like eval-harness pass
        # `NamedSharding` objects produced from a global mesh; for robustness we avoid constraining in that case and
        # let downstream `named_jit` / pjitted functions reshard as needed.
        if spec is None:
            return arr
        if isinstance(spec, PartitionSpec):
            resolved_mesh = haliax.partitioning._get_mesh()
            return jax.lax.with_sharding_constraint(arr, NamedSharding(resolved_mesh, spec))
        return arr

    def _replicated_out_specs(specs: Any) -> bool:
        if specs is None:
            return True
        if isinstance(specs, PartitionSpec):
            return specs == PartitionSpec()
        leaves = jax.tree_util.tree_leaves(specs, is_leaf=lambda s: isinstance(s, PartitionSpec))
        if not leaves:
            return False
        for leaf in leaves:
            if leaf is None:
                continue
            if not isinstance(leaf, PartitionSpec) or leaf != PartitionSpec():
                return False
        return True

    if jax.process_count() == 1:

        def in_jit_single(x_leaf: Any, spec: Any) -> Any:
            arr = x_leaf.array if isinstance(x_leaf, hax.NamedArray) else x_leaf
            arr = _maybe_constrain(arr, spec)
            return hax.named(arr, x_leaf.axis_names) if isinstance(x_leaf, hax.NamedArray) else arr

        return eqx.filter_jit(jax.tree.map)(in_jit_single, x, out_axis_specs, is_leaf=is_named_array)

    active_mesh = haliax.partitioning._get_mesh()
    devices: np.ndarray = np.asarray(active_mesh.devices).reshape(jax.process_count(), jax.local_device_count())
    global_mesh = Mesh(devices, ("processes", "local_devices"))
    in_pspec = PartitionSpec("processes")

    def pre_jit(x_leaf: Any) -> jax.Array:
        arr = x_leaf.array if isinstance(x_leaf, hax.NamedArray) else x_leaf
        if jax.process_index() == source:
            host = np.asarray(arr)
        else:
            host = np.zeros(arr.shape, dtype=arr.dtype)
        host = np.expand_dims(host, axis=0)
        return host_local_array_to_global_array(host, global_mesh, in_pspec)

    if _replicated_out_specs(out_axis_specs):
        # Broadcast to all hosts, then materialize host-local arrays. This avoids JAX errors when the active training
        # mesh uses a device order that differs from `jax.devices()` order (common on multi-host TPU).
        x_global = jax.tree.map(pre_jit, x, is_leaf=is_named_array)
        with haliax.partitioning.set_mesh(global_mesh):
            reduced = jax.jit(_psum, out_shardings=NamedSharding(global_mesh, PartitionSpec()))(x_global)

        def post_jit(x_leaf: Any, x_leaf_orig: Any) -> Any:
            host_arr = jax.device_get(x_leaf.addressable_data(0))
            arr = jnp.asarray(host_arr)
            return hax.named(arr, x_leaf_orig.axis_names) if isinstance(x_leaf_orig, hax.NamedArray) else arr

        return jax.tree.map(post_jit, reduced, x, is_leaf=is_named_array)

    def in_jit(x_global: jax.Array, spec: Any, x_leaf: Any) -> Any:
        arr = jnp.sum(x_global, axis=0)
        arr = _maybe_constrain(arr, spec)
        return hax.named(arr, x_leaf.axis_names) if isinstance(x_leaf, hax.NamedArray) else arr

    x_global = jax.tree.map(pre_jit, x, is_leaf=is_named_array)
    return eqx.filter_jit(jax.tree.map)(in_jit, x_global, out_axis_specs, x, is_leaf=is_named_array)


def tree_broadcast_to(prefix: PyTree[L], t: T, *, is_leaf: Optional[Callable[[Any], bool]] = None) -> T:
    """
    Broadcasts a prefix tree to match the structure of a full tree. This is useful when you need to
    tree_map over t and prefix (using t as the leaves) but prefix is a tree prefix of t.
    """
    return jax.tree.map(
        # note the swap
        lambda pref, xtree: jax.tree.map(lambda x: pref, xtree, is_leaf=is_leaf),
        prefix,
        t,
        is_leaf=is_leaf,
    )


# Non-busted version of broadcast_one_to_all from jax.multihost_utils. (The issue is that  if you use a non-contiguous
# mesh, their utility blows up because it makes a contiguous mesh.)


def _psum(xs: Any) -> Any:
    return jax.tree.map(lambda x: jnp.sum(x, dtype=x.dtype, axis=0), xs)


def broadcast_one_to_all(in_tree: Any, is_source: bool | None = None) -> Any:
    """Broadcast data from a source host (host 0 by default) to all other hosts.

    Args:
      in_tree: pytree of arrays - each array *must* have the same shape across the
        hosts.
      is_source: optional bool denoting whether the caller is the source. Only
        'source host' will contribute the data for the broadcast. If None, then
        host 0 is used.

    Returns:
      A pytree matching in_tree where the leaves now all contain the data from the
      first host.
    """
    if jax.process_count() == 1:
        return jax.tree.map(np.asarray, in_tree)

    if is_source is None:
        source_process = 0
    else:
        flags = process_allgather(np.asarray(1 if is_source else 0, dtype=np.int32))
        flags_sum = int(np.sum(flags))
        if flags_sum != 1:
            raise ValueError(f"broadcast_one_to_all expected exactly one source process, got sum={flags_sum}.")
        source_process = int(np.argmax(flags))

    gathered = process_allgather(jax.tree.map(np.asarray, in_tree))

    def _select_source(x: np.ndarray) -> np.ndarray:
        # `process_allgather` stacks over a new leading axis (processes).
        return np.asarray(x[source_process])

    return jax.tree.map(_select_source, gathered)


def assert_equal(in_tree, fail_message: str = ""):
    """Verifies that all the hosts have the same tree of values."""
    expected = broadcast_one_to_all(in_tree)
    if not jax.tree_util.tree_all(jax.tree_util.tree_map(lambda *x: np.all(np.equal(*x)), in_tree, expected)):
        raise AssertionError(f"{fail_message} Expected: {expected}; got: {in_tree}.")


def sync_global_devices(name: str):
    """Creates a barrier across all hosts/devices."""
    h = np.uint32(zlib.crc32(name.encode()))
    assert_equal(h, f"sync_global_devices name mismatch ('{name}')")


def sharded_tree_size(
    tree, mesh: Optional[haliax.partitioning.MeshLike] | None = None, mapping: ResourceMapping | None = None
) -> int:
    """
    Returns the size of a sharded tree, in bytes. If the tree is sharded, this returns the size of a per-device shard.

    If mesh is None, uses the current mesh.
    If mapping is None, uses the current mapping.

    For named arrays, this uses the provided mesh and mapping to determine the sharding.
    For real jax.Arrays, uses their existing sharding.
    Inside jit or with ShapeDTypeStruct, see if there is a sharding. If not, assumes unsharded.
    For other arrays, assumes unsharded.

    Args:
        tree: the tree to measure
        mesh: the mesh to use for sharding. If None, uses the current mesh.
    """
    if mesh is None:
        mesh = jax.sharding.get_abstract_mesh()

    if mapping is None:
        mapping = haliax.partitioning.current_thread_local_mapping()

    def _mesh_axis_size(axis_name) -> int:
        if mesh is None:
            return 1
        return mesh.shape[axis_name]

    def _shards_for_pspec(pspec):
        if mesh is None or pspec is None:
            return 1

        count = 1
        for axis in pspec:
            if axis is None:
                continue
            if isinstance(axis, tuple):
                for sub_axis in axis:
                    count *= _mesh_axis_size(sub_axis)
            else:
                count *= _mesh_axis_size(axis)
        return count

    def _size(x):
        if isinstance(x, hax.NamedArray):
            pspec = haliax.partitioning.pspec_for(x, mapping)
            num_shards = _shards_for_pspec(pspec)
            x_a = x.array
            if hasattr(x_a, "nbytes"):
                return x_a.nbytes // num_shards
            else:
                return x_a.size * x_a.dtype.itemsize // num_shards
        elif is_jax_array_like(x):
            sharding = getattr(x, "sharding", None)
            pspec = getattr(sharding, "spec", None) if sharding is not None else None
            if pspec is None and sharding is not None:
                warnings.warn(
                    f"{x} has sharding {sharding} but no spec. Assuming unsharded. If you see this, please report a bug."
                )
            num_shards = _shards_for_pspec(pspec)

            # ShapeDtypeStruct doesn't have nbytes
            if hasattr(x, "nbytes"):
                total = x.nbytes
            else:
                shape = getattr(x, "shape", None)
                dtype = getattr(x, "dtype", None)
                if shape is None or dtype is None:
                    raise ValueError("Unable to determine byte size for JAX array-like leaf without shape/dtype")
                total = int(np.prod(shape)) * np.dtype(dtype).itemsize

            return total // num_shards if num_shards > 0 else total
        else:
            assert jnp.isscalar(x)
            return jnp.dtype(type(x)).itemsize

    return sum(jax.tree.leaves(jax.tree.map(_size, tree, is_leaf=is_named_array)))
