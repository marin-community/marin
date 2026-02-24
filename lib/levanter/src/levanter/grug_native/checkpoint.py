# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from haliax.partitioning import ResourceMapping
from jax.sharding import Mesh
from jaxtyping import PyTree

from levanter.checkpoint import Checkpointer, load_checkpoint


def maybe_restore_checkpoint(
    state: PyTree,
    *,
    checkpoint_path: str,
    axis_mapping: ResourceMapping | None,
    mesh: Mesh | None,
    allow_partial: bool = False,
) -> PyTree:
    return load_checkpoint(
        state,
        checkpoint_path,
        discover_latest=True,
        axis_mapping=axis_mapping,
        mesh=mesh,
        allow_partial=allow_partial,
    )


def save_checkpoint_on_step(checkpointer: Checkpointer | None, train_state: PyTree, *, force: bool = False) -> None:
    if checkpointer is None:
        return

    step = int(train_state.step)  # type: ignore[attr-defined]
    checkpointer.on_step(tree={"train_state": train_state}, step=step, force=force)


def wait_for_checkpoints(checkpointer: Checkpointer | None) -> None:
    if checkpointer is None:
        return
    checkpointer.wait_until_finished()
