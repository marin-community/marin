# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# This module provides checkpoint serialization using Orbax with OCDBT support.
# OCDBT (Ordered Concurrent Distributed B-Tree) consolidates checkpoint files,
# reducing file count and improving save/restore performance.
import logging
import os
from functools import partial
from typing import Callable, Optional

import equinox
import haliax as hax
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import orbax.checkpoint as ocp
from haliax.jax_utils import is_jax_array_like
from haliax.partitioning import ResourceMapping
from haliax.util import is_named_array
from jax.sharding import Mesh, Sharding
from jaxtyping import PyTree

logger = logging.getLogger(__name__)


def _is_named_or_none(x):
    return x is None or is_named_array(x)


def tree_serialize_leaves_tensorstore(
    checkpoint_dir,
    pytree,
    *,
    commit_callback: Optional[Callable] = None,
):
    """
    Serialize a PyTree to a checkpoint directory using Orbax with OCDBT.

    Args:
        checkpoint_dir: Directory path to save the checkpoint
        pytree: The PyTree to serialize
        commit_callback: Optional callback to call after checkpoint is committed
    """

    # Convert NamedArrays to regular arrays for serialization
    def _unwrap_named_array(x):
        if is_named_array(x):
            return x.array
        return x

    unwrapped_tree = jtu.tree_map(_unwrap_named_array, pytree, is_leaf=is_named_array)

    # Filter to only array-like leaves
    def _is_saveable(x):
        if isinstance(x, (int, float, bool, complex)):
            return jnp.array(x)
        elif equinox.is_array_like(x):
            return x
        else:
            return None

    saveable_tree = jtu.tree_map(_is_saveable, unwrapped_tree)

    # Create async checkpointer
    handler = ocp.PyTreeCheckpointHandler(use_ocdbt=True)
    async_checkpointer = ocp.AsyncCheckpointer(handler, timeout_secs=60 * 30)

    # Create save args for async checkpointing
    save_args = jtu.tree_map(lambda x: ocp.SaveArgs() if x is not None else None, saveable_tree)

    # Save asynchronously with force=True to allow overwriting existing checkpoints
    async_checkpointer.save(checkpoint_dir, saveable_tree, save_args=save_args, force=True)

    # Wait for completion
    async_checkpointer.wait_until_finished()
    logger.info(f"Committed checkpoint to {checkpoint_dir} using OCDBT")
    if commit_callback is not None:
        commit_callback()


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
    *,
    allow_missing: bool = False,
):
    """
    Deserialize a PyTree from a checkpoint directory using Orbax.

    This function supports loading both old TensorStore checkpoints and new OCDBT checkpoints.
    It will automatically detect the format and load accordingly.

    Args:
        checkpoint_dir: the directory containing the checkpoint, can be a local path or a GCS path
        pytree: the exemplar pytree structure to restore into
        axis_mapping: optional, the axis mapping for the NamedArrays
        mesh: optional, the mesh for the NamedArrays
        manager: Deprecated parameter, kept for backward compatibility
        allow_missing: if True, missing leaves will be allowed and kept as-is

    Returns:
        A pytree with the same structure as the exemplar pytree, with arrays loaded from the checkpoint
    """
    # Compute shardings for all leaves
    shardings: PyTree[Optional[Sharding]] = jtu.tree_map(
        partial(_sharding_from_leaf, axis_mapping=axis_mapping, mesh=mesh), pytree, is_leaf=_is_named_or_none
    )

    # Convert NamedArrays to regular arrays for the exemplar tree
    def _unwrap_named_array(x):
        if is_named_array(x):
            return x.array
        return x

    unwrapped_tree = jtu.tree_map(_unwrap_named_array, pytree, is_leaf=is_named_array)

    # Create restore args with shardings
    def _create_restore_arg(sharding, value):
        if sharding is not None and value is not None:
            return ocp.ArrayRestoreArgs(sharding=sharding)
        return None

    restore_args = jtu.tree_map(_create_restore_arg, shardings, unwrapped_tree)

    # Create Orbax checkpointer with OCDBT enabled
    # Note: use_ocdbt=True also allows reading old non-OCDBT checkpoints
    handler = ocp.PyTreeCheckpointHandler(use_ocdbt=True)
    checkpointer = ocp.Checkpointer(handler)

    # Restore the checkpoint
    try:
        restored_tree = checkpointer.restore(checkpoint_dir, item=unwrapped_tree, restore_args=restore_args)
    except (FileNotFoundError, ValueError) as e:
        if allow_missing:
            logger.warning(f"Could not fully restore checkpoint from {checkpoint_dir}: {e}")
            restored_tree = unwrapped_tree
        else:
            raise

    # Check for None values that were in the checkpoint but not in the exemplar
    # This handles the case where a field was None when saved but has a value in the exemplar
    def _check_missing_values(exemplar_val, restored_val):
        # If checkpoint has None but exemplar has a value, this is a missing field
        if restored_val is None and exemplar_val is not None:
            if not allow_missing:
                raise FileNotFoundError(
                    "Checkpoint is missing a required field (saved as None, but exemplar has a value)"
                )
            # If allow_missing, keep the exemplar value
            return exemplar_val
        return restored_val

    restored_tree = jtu.tree_map(_check_missing_values, unwrapped_tree, restored_tree)

    # Rebuild NamedArrays from the restored arrays
    def _rebuild_named_array(like, array):
        if is_named_array(array):
            return array
        if is_named_array(like):
            return hax.NamedArray(array, like.axes)
        return array

    return jtu.tree_map(_rebuild_named_array, pytree, restored_tree, is_leaf=_is_named_or_none)
