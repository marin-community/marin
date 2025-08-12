# References:
# * Orbax: https://github.com/google/orbax/blob/11d2934ecfff77e86b5e07d0fef02b67eff4511b/orbax/checkpoint/pytree_checkpoint_handler.py#L312
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import equinox
import jax
import jax.experimental.array_serialization.serialization as array_ser
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.sharding import Mesh, Sharding
from jaxtyping import PyTree

import haliax as hax
from haliax.jax_utils import is_jax_array_like
from haliax.partitioning import ResourceMapping
from haliax.util import is_named_array

from levanter.utils import fsspec_utils, jax_utils


logger = logging.getLogger(__name__)


def _is_named_or_none(x):
    return x is None or is_named_array(x)


def tree_serialize_leaves_tensorstore(
    checkpoint_dir,
    pytree,
    manager: Optional[array_ser.GlobalAsyncCheckpointManager] = None,
    *,
    commit_callback: Optional[Callable] = None,
):
    if manager is None:
        manager = array_ser.GlobalAsyncCheckpointManager()
        manager_was_none = True
    else:
        manager_was_none = False

    leaf_key_paths = jax_utils.leaf_key_paths(pytree, is_leaf=is_named_array)
    assert len(jax.tree.leaves(leaf_key_paths, is_leaf=is_named_array)) == len(
        jax.tree.leaves(pytree, is_leaf=is_named_array)
    )

    paths = _fs_paths_from_key_paths(checkpoint_dir, leaf_key_paths)

    # make a dataclass since tuples are pytrees
    @dataclass
    class Pair:
        path: str
        leaf: Any

    zipped = jax.tree.map(lambda x, y: Pair(x, y), paths, pytree, is_leaf=lambda x: x is None)
    paired_leaves = jax.tree.leaves(zipped)
    paths = [p.path for p in paired_leaves]
    leaves = [p.leaf.array if is_named_array(p.leaf) else p.leaf for p in paired_leaves]

    # ok, not all of these are arrays, but we'll deal with that in the async function
    def _ensure_is_array(x):
        if isinstance(x, (int, float, bool, complex)):
            return jnp.array(x)
        else:
            return x

    arrays = [_ensure_is_array(x) for x in leaves]

    # filter out the None leaves and paths (must be zip)
    arrays, paths = zip(*[(a, p) for a, p in zip(arrays, paths) if equinox.is_array_like(a)])

    arrays = list(arrays)
    paths = list(paths)

    if commit_callback is None:
        commit_callback = lambda: logger.info("Committed checkpoint to Tensorstore")  # noqa

    manager.serialize_with_paths(arrays, paths, on_commit_callback=commit_callback)

    if manager_was_none:
        manager.wait_until_finished()


def _fs_paths_from_key_paths(checkpoint_dir, leaf_key_paths):
    def path_from_key_path(key_path):
        return os.path.join(checkpoint_dir, *key_path.split("."))

    paths = jtu.tree_map(path_from_key_path, leaf_key_paths)
    return paths


def _sharding_from_leaf(leaf, axis_mapping, mesh) -> Optional[jax.sharding.Sharding]:
    if is_named_array(leaf):
        if not is_jax_array_like(leaf.array):
            return None
        return hax.partitioning.sharding_for_axis(leaf.axes, axis_mapping, mesh)
    elif hasattr(leaf, "sharding") and getattr(leaf, "sharding") is not None:
        return leaf.sharding
    elif is_jax_array_like(leaf):
        return _fully_replicated_sharding(mesh)
    elif isinstance(leaf, (bool, float, complex, int, np.ndarray)):
        return _fully_replicated_sharding(mesh)
    else:
        logger.warning(f"Unknown leaf type {type(leaf)}")
        return None


def _fully_replicated_sharding(mesh):
    return hax.partitioning.sharding_for_axis((), {}, mesh)


def tree_deserialize_leaves_tensorstore(
    checkpoint_dir,
    pytree,
    axis_mapping: Optional[ResourceMapping] = None,
    mesh: Optional[Mesh] = None,
    manager: Optional[array_ser.GlobalAsyncCheckpointManager] = None,
    *,
    allow_missing: bool = False,
):
    """
    Deserializes a PyTree of Arrays and NamedArrays from a Tensorstore checkpoint, returning a pytree with the same shape
    as the one provided. This method is capable of deserializing NamedArrays that are the result of an eval_shape call
    (i.e. they are not yet arrays but are ShapedDtypeStructs), provided you pass in the axis_mapping and mesh (or
    they are available by context)

    Args:
        checkpoint_dir: the directory containing the tensorstore checkpoint, can be a local path or a GCS path
        pytree: the exemplar pytree
        axis_mapping: optional, the axis mapping for the NamedArrays (if they are not yet arrays)
        mesh: optional, the mesh for the NamedArrays (if they are not yet arrays)
        manager: optional, the checkpoint manager to use. If not provided, a new one will be created
        allow_missing: if True, missing leaves will be allowed and kept as-is

    Returns:
        A pytree with the same shape as the exemplar pytree, but with the arrays deserialized from the checkpoint
    """
    if manager is None:
        manager = array_ser.GlobalAsyncCheckpointManager()

    shardings: PyTree[Optional[Sharding]] = jtu.tree_map(
        partial(_sharding_from_leaf, axis_mapping=axis_mapping, mesh=mesh), pytree, is_leaf=_is_named_or_none
    )

    # TODO: support ShapeDtypeStructs that are not NamedArrays
    leaf_key_paths = jax_utils.leaf_key_paths(shardings, is_leaf=_is_named_or_none)
    paths = _fs_paths_from_key_paths(checkpoint_dir, leaf_key_paths)
    paths = jtu.tree_leaves(paths, is_leaf=lambda x: x is None)

    shardings_leaves, shardings_structure = jtu.tree_flatten(shardings, is_leaf=_is_named_or_none)

    assert len(shardings_leaves) == len(paths)
    # ok, so, jax really doesn't want any Nones in the leaves here, so we need to temporarily partition the pytree
    real_indices = [i for i, x in enumerate(shardings_leaves) if x is not None]
    paths_to_load = []
    indices_to_load = []
    shardings_to_load = []

    missing_paths = []
    missing_indices = []

    for i in real_indices:
        path = paths[i]

        if not fsspec_utils.exists(path):
            missing_paths.append(path)
            missing_indices.append(i)
            continue

        paths_to_load.append(path)
        indices_to_load.append(i)
        shardings_to_load.append(shardings_leaves[i])

    # ok now check for missing paths
    if missing_paths:
        if not allow_missing:
            raise FileNotFoundError(f"Missing paths: {missing_paths}")
        else:
            to_log = f"Several keys were missing from the checkpoint directory {checkpoint_dir}:"
            leaf_paths = jtu.tree_leaves(leaf_key_paths, is_leaf=_is_named_or_none)
            for i in missing_indices:
                to_log += f"\n  - {leaf_paths[i]}"
            logger.warning(to_log)

    deser_leaves = manager.deserialize_with_paths(shardings=shardings_to_load, paths=paths_to_load)

    # now we need to recreate the original structure

    out_leaves = jax.tree.leaves(pytree, is_leaf=_is_named_or_none)
    assert len(out_leaves) == len(shardings_leaves)
    # out_leaves = [None] * len(shardings_leaves)
    for i, x in zip(indices_to_load, deser_leaves):
        out_leaves[i] = x

    deser_arrays = jtu.tree_unflatten(shardings_structure, out_leaves)

    # deser_arrays only has arrays for the deserialized arrays, but we need named arrays for at least some.
    # The original pytree has the structure we want, so we'll use that to rebuild the named arrays
    def _rebuild_named_array(like, array):
        if is_named_array(array):
            return array

        if is_named_array(like):
            return hax.NamedArray(array, like.axes)
        else:
            return array

    return jtu.tree_map(_rebuild_named_array, pytree, deser_arrays, is_leaf=_is_named_or_none)
