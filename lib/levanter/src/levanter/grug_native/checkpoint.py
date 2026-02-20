# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class _CheckpointState:
    train_state: PyTree

    @property
    def saveable_state(self) -> dict[str, Any]:
        return {"train_state": self.train_state}


@dataclass(frozen=True)
class _CheckpointStepInfo:
    step: int
    state: _CheckpointState


def _checkpoint_step_info_from_train_state(train_state: PyTree) -> _CheckpointStepInfo:
    try:
        raw_step = train_state.step  # type: ignore[attr-defined]
    except AttributeError as e:
        raise ValueError("train_state must define a 'step' attribute for checkpointing.") from e

    try:
        step = int(raw_step)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"train_state.step must be convertible to int, got value {raw_step!r} of type {type(raw_step)!r}."
        ) from e

    return _CheckpointStepInfo(step=step, state=_CheckpointState(train_state=train_state))


def save_checkpoint_on_step(checkpointer: Checkpointer | None, train_state: PyTree, *, force: bool = False) -> None:
    if checkpointer is None:
        return

    checkpointer.on_step(_checkpoint_step_info_from_train_state(train_state), force=force)


def wait_for_checkpoints(checkpointer: Checkpointer | None) -> None:
    if checkpointer is None:
        return
    checkpointer.wait_until_finished()
