# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# References:
# * Orbax: https://github.com/google/orbax/blob/11d2934ecfff77e86b5e07d0fef02b67eff4511b/orbax/checkpoint/pytree_checkpoint_handler.py#L312
import asyncio
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import equinox
import haliax as hax
import jax
import jax.experimental.array_serialization.serialization as array_ser
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import tensorstore as ts
from haliax.jax_utils import is_jax_array_like
from haliax.partitioning import ResourceMapping
from haliax.util import is_named_array
from jax.sharding import Mesh, Sharding
from jaxtyping import PyTree

from levanter.utils import fsspec_utils, jax_utils

logger = logging.getLogger(__name__)


def _create_ocdbt_spec(checkpoint_base_path: str, array_path: str) -> dict:
    """
    Create a TensorStore spec with OCDBT (Optionally-Cooperative Distributed B-Tree) enabled.

    All arrays in a checkpoint share the same OCDBT base (checkpoint directory), with each array
    having a unique path within that OCDBT database.

    Args:
        checkpoint_base_path: Base path for the checkpoint (e.g., "/checkpoints/step-100")
        array_path: Relative path for this specific array (e.g., "model/layer0/weight")

    Returns:
        TensorStore spec dict with OCDBT kvstore driver
    """
    import urllib.parse

    parsed = urllib.parse.urlparse(checkpoint_base_path)

    spec: dict[str, Any] = {"driver": "zarr3", "kvstore": {"driver": "ocdbt", "base": {}, "path": ""}}

    # Configure the base kvstore based on the path protocol
    if parsed.scheme in ("", "file"):
        # Local filesystem: all arrays share the same OCDBT base (checkpoint directory)
        spec["kvstore"]["base"] = {"driver": "file", "path": checkpoint_base_path}
        spec["kvstore"]["path"] = array_path

    elif parsed.scheme == "gs":
        # GCS path: gs://bucket/path/to/checkpoint
        bucket = parsed.netloc
        gcs_path = parsed.path.lstrip("/")

        spec["kvstore"]["base"] = {"driver": "gcs", "bucket": bucket}
        if gcs_path:
            spec["kvstore"]["base"]["path"] = gcs_path

        spec["kvstore"]["path"] = array_path

    else:
        raise ValueError(f"Unsupported protocol: {parsed.scheme}. Supported: file, gs")

    return spec


async def _list_ocdbt_keys(checkpoint_base_path: str) -> list[str]:
    """
    Open the OCDBT KVStore for a checkpoint and list all keys.

    Args:
        checkpoint_base_path: Base path for the checkpoint (e.g., "/checkpoints/step-100")

    Returns:
        List of key paths (as strings) in the OCDBT kvstore
    """
    import urllib.parse

    parsed = urllib.parse.urlparse(checkpoint_base_path)

    kvstore_spec: dict[str, Any] = {"driver": "ocdbt"}

    # Configure the base kvstore based on the path protocol
    if parsed.scheme in ("", "file"):
        kvstore_spec["base"] = {"driver": "file", "path": checkpoint_base_path}
    elif parsed.scheme == "gs":
        bucket = parsed.netloc
        gcs_path = parsed.path.lstrip("/")
        kvstore_spec["base"] = {"driver": "gcs", "bucket": bucket}
        if gcs_path:
            kvstore_spec["base"]["path"] = gcs_path
    else:
        raise ValueError(f"Unsupported protocol: {parsed.scheme}. Supported: file, gs")

    # Open the kvstore and list keys
    kvstore = await ts.KvStore.open(kvstore_spec)
    keys_bytes = await kvstore.list()
    # Convert bytes to strings
    return [key.decode("utf-8") for key in keys_bytes]


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
    filtered = [(a, p) for a, p in zip(arrays, paths) if equinox.is_array_like(a)]
    arrays = [a for a, _ in filtered]
    paths = [p for _, p in filtered]

    if commit_callback is None:
        commit_callback = lambda: logger.info("Committed checkpoint to Tensorstore")  # noqa

    # Create specs for each array
    # All arrays share the same OCDBT base (checkpoint directory)
    tspecs = []
    for path in paths:
        # Compute relative path from checkpoint_dir
        assert path.startswith(checkpoint_dir)
        relative_path = path[len(checkpoint_dir) :].lstrip(os.sep)

        spec = _create_ocdbt_spec(checkpoint_dir, relative_path)
        tspecs.append(spec)

    manager.serialize(arrays, tspecs, on_commit_callback=commit_callback)

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


def _restore_ocdbt(
    checkpoint_root: str,
    paths: list[str],
    real_indices: list[int],
    shardings_leaves: list,
    leaf_key_paths,
    manager: array_ser.GlobalAsyncCheckpointManager,
    allow_missing: bool,
) -> tuple[list, list[int]]:
    """
    Restore arrays from an OCDBT checkpoint.

    Args:
        checkpoint_root: Root path of the OCDBT checkpoint (where manifest.ocdbt is located)
        paths: Full paths for all arrays
        real_indices: Indices of non-None shardings
        shardings_leaves: Flattened list of shardings
        leaf_key_paths: Key paths for logging
        manager: Checkpoint manager
        allow_missing: Whether to allow missing arrays

    Returns:
        Tuple of (deserialized_leaves, indices_to_load)
    """
    # List all keys in the OCDBT kvstore to check existence
    existing_keys = asyncio.run(_list_ocdbt_keys(checkpoint_root))
    existing_keys_set = set(existing_keys)

    paths_to_load = []
    indices_to_load = []
    shardings_to_load = []
    missing_paths = []
    missing_indices = []

    for i in real_indices:
        path = paths[i]
        # Compute relative path from checkpoint_root (where OCDBT kvstore is based)
        assert path.startswith(checkpoint_root)
        relative_path = path[len(checkpoint_root) :].lstrip(os.sep)

        # Check if this relative path exists in the kvstore
        zarr_metadata_key = f"{relative_path}/zarr.json"

        if zarr_metadata_key not in existing_keys_set:
            missing_paths.append(path)
            missing_indices.append(i)
            continue

        paths_to_load.append(path)
        indices_to_load.append(i)
        shardings_to_load.append(shardings_leaves[i])

    # Check for missing paths
    if missing_paths:
        if not allow_missing:
            raise FileNotFoundError(f"Missing {len(missing_paths)} arrays in OCDBT checkpoint: {missing_paths}")
        else:
            to_log = f"Several keys were missing from the OCDBT checkpoint {checkpoint_root}:"
            leaf_paths = jtu.tree_leaves(leaf_key_paths, is_leaf=_is_named_or_none)
            for i in missing_indices:
                to_log += f"\n  - {leaf_paths[i]}"
            logger.warning(to_log)

    # Create OCDBT TensorStore specs for deserialization
    tspecs_to_load = []
    for path in paths_to_load:
        assert path.startswith(checkpoint_root)
        relative_path = path[len(checkpoint_root) :].lstrip(os.sep)
        spec = _create_ocdbt_spec(checkpoint_root, relative_path)
        tspecs_to_load.append(spec)

    # Deserialize using OCDBT specs
    deser_leaves = manager.deserialize(shardings=shardings_to_load, tensorstore_specs=tspecs_to_load)
    return deser_leaves, indices_to_load


def _restore_old_ts(
    checkpoint_dir: str,
    paths: list[str],
    real_indices: list[int],
    shardings_leaves: list,
    leaf_key_paths,
    manager: array_ser.GlobalAsyncCheckpointManager,
    allow_missing: bool,
) -> tuple[list, list[int]]:
    """
    Restore arrays from an old (non-OCDBT) tensorstore checkpoint.

    Args:
        checkpoint_dir: Directory containing the checkpoint
        paths: Full paths for all arrays
        real_indices: Indices of non-None shardings
        shardings_leaves: Flattened list of shardings
        leaf_key_paths: Key paths for logging
        manager: Checkpoint manager
        allow_missing: Whether to allow missing arrays

    Returns:
        Tuple of (deserialized_leaves, indices_to_load)
    """
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

    # Check for missing paths
    if missing_paths:
        if not allow_missing:
            raise FileNotFoundError(f"Missing paths: {missing_paths}")
        else:
            to_log = f"Several keys were missing from the checkpoint directory {checkpoint_dir}:"
            leaf_paths = jtu.tree_leaves(leaf_key_paths, is_leaf=_is_named_or_none)
            for i in missing_indices:
                to_log += f"\n  - {leaf_paths[i]}"
            logger.warning(to_log)

    # Use old deserialize_with_paths for backward compatibility
    deser_leaves = manager.deserialize_with_paths(shardings=shardings_to_load, paths=paths_to_load)
    return deser_leaves, indices_to_load


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

    # Check if this is an OCDBT checkpoint by looking for manifest.ocdbt (sentinel file)
    # OCDBT manifests are at the checkpoint root, not in subpaths
    # So we need to check upwards from checkpoint_dir to find the manifest
    # The checkpoint root should have a metadata.json file
    def find_checkpoint_root(path):
        """Find the checkpoint root by looking for metadata.json"""
        current = path
        while current and current != os.path.dirname(current):
            metadata_path = os.path.join(current, "metadata.json")
            if fsspec_utils.exists(metadata_path):
                return current
            current = os.path.dirname(current)
        return path  # fallback to original path

    checkpoint_root = find_checkpoint_root(checkpoint_dir)
    ocdbt_manifest_path = os.path.join(checkpoint_root, "manifest.ocdbt")
    is_ocdbt_checkpoint = fsspec_utils.exists(ocdbt_manifest_path)

    if is_ocdbt_checkpoint:
        deser_leaves, indices_to_load = _restore_ocdbt(
            checkpoint_root, paths, real_indices, shardings_leaves, leaf_key_paths, manager, allow_missing
        )
    else:
        deser_leaves, indices_to_load = _restore_old_ts(
            checkpoint_dir, paths, real_indices, shardings_leaves, leaf_key_paths, manager, allow_missing
        )

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
