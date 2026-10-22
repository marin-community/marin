"""Explicit checkpoint schedules: which steps save weights and which include the optimizer.

Training runs state the schedule as explicit step lists: fp32 master weights every 10
steps to 50 and every 25 after, with the full optimizer state at a short list of those
steps. Checkpoints outside the optimizer list keep the model's BF16 records and the
FP32 master params but drop the moment states, and carry an ``optimizer_state.json``
marker so a later resume fails loudly instead of silently restarting the optimizer.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import json
import os
from typing import Any

from marinskyrl.checkpoint_paths import POLICY_CHECKPOINT_SUBDIRECTORY
from omegaconf import OmegaConf

from skyrl_train.distributed.megatron.checkpoint_writer import model_key_for_master
from skyrl_train.io import io

OPTIMIZER_STATE_MARKER = "optimizer_state.json"


class OptimizerStateMode(StrEnum):
    """How much optimizer state one checkpoint carries."""

    FULL = "full"
    MASTERS_ONLY = "masters"


@dataclass(frozen=True)
class CheckpointSchedule:
    """The checkpoint decision for one run.

    ``weight_steps`` replaces the interval when non-empty. ``optimizer_steps`` restricts
    full optimizer saves to the listed checkpoint steps; empty means every checkpoint
    carries the optimizer (the historical behavior, and the resume-safe default).
    """

    interval: int
    weight_steps: tuple[int, ...]
    optimizer_steps: tuple[int, ...]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CheckpointSchedule":
        trainer = config["trainer"]
        weight_steps = _step_list(trainer, "checkpoint_steps")
        optimizer_steps = _step_list(trainer, "optimizer_checkpoint_steps")
        if weight_steps and not optimizer_steps:
            raise ValueError(
                "trainer.optimizer_checkpoint_steps is required when trainer.checkpoint_steps is set; "
                "list the steps whose checkpoints must resume training"
            )
        _require_savable_optimizer_steps(weight_steps, optimizer_steps, checkpoint_interval(trainer))
        _require_fully_reshardable(trainer, optimizer_steps)
        return cls(
            interval=checkpoint_interval(trainer),
            weight_steps=weight_steps,
            optimizer_steps=optimizer_steps,
        )

    @property
    def saves_checkpoints(self) -> bool:
        """Whether the schedule saves any checkpoint at all."""
        return bool(self.weight_steps) or self.interval > 0

    @property
    def max_gap(self) -> int:
        """The largest gap between consecutive checkpoint steps, for resume validation."""
        if self.weight_steps:
            return max((later - earlier for earlier, later in zip(self.weight_steps, self.weight_steps[1:])), default=1)
        return self.interval

    def should_save_weights(self, step: int) -> bool:
        if self.weight_steps:
            return step in self.weight_steps
        return self.interval > 0 and step % self.interval == 0

    def should_save_optimizer(self, step: int) -> bool:
        if self.optimizer_steps:
            return step in self.optimizer_steps
        return True

    def optimizer_mode(self, step: int) -> OptimizerStateMode:
        """The optimizer records a checkpoint at ``step`` carries."""
        return OptimizerStateMode.FULL if self.should_save_optimizer(step) else OptimizerStateMode.MASTERS_ONLY


def checkpoint_interval(trainer: Mapping[str, Any]) -> int:
    """The checkpoint interval, or 0 when the run saves only listed steps."""
    return int(trainer.get("ckpt_interval", 0) or 0)


def _require_savable_optimizer_steps(weight_steps: tuple[int, ...], optimizer_steps: tuple[int, ...], interval: int):
    """Reject optimizer steps that would silently never save."""
    if not optimizer_steps:
        return
    if weight_steps:
        outside = [step for step in optimizer_steps if step not in weight_steps]
        cadence = "trainer.checkpoint_steps"
    elif interval > 0:
        outside = [step for step in optimizer_steps if step % interval != 0]
        cadence = f"trainer.ckpt_interval ({interval})"
    else:
        raise ValueError(
            "trainer.optimizer_checkpoint_steps is set but no checkpoint ever saves; "
            "set trainer.checkpoint_steps or trainer.ckpt_interval"
        )
    if outside:
        raise ValueError(
            f"trainer.optimizer_checkpoint_steps lists steps that never save a checkpoint: {outside}; "
            f"every optimizer checkpoint step must also be saved under {cadence}"
        )


def _require_fully_reshardable(trainer: Mapping[str, Any], optimizer_steps: tuple[int, ...]):
    """Master-only checkpoints need the DP-gathered optimizer format.

    The ``dp_reshardable`` format stores each bucket's param, master, and moment shards
    as records named by bucket position, so a masters-only filter cannot separate them.
    """
    if not optimizer_steps:
        return
    policy = trainer.get("policy")
    megatron_config = policy.get("megatron_config") if policy is not None else None
    sharding_type = megatron_config.get("optimizer_checkpoint_sharding_type") if megatron_config is not None else None
    if sharding_type is not None and sharding_type != "fully_reshardable":
        raise ValueError(
            "trainer.policy.megatron_config.optimizer_checkpoint_sharding_type must be 'fully_reshardable' "
            f"when trainer.optimizer_checkpoint_steps is set; got {sharding_type!r}, whose optimizer "
            "records cannot be filtered down to masters only"
        )


def _step_list(trainer: Mapping[str, Any], field: str) -> tuple[int, ...]:
    value = OmegaConf.to_container(trainer[field], resolve=True) if field in trainer else []
    if not isinstance(value, list) or any(
        not isinstance(step, int) or isinstance(step, bool) or step <= 0 for step in value
    ):
        raise ValueError(f"trainer.{field} must be a list of positive step numbers")
    return tuple(sorted(set(value)))


def assert_optimizer_resumable(checkpoint_path: str) -> None:
    """Refuse a masters-only checkpoint; resuming from one silently restarts the optimizer."""
    marker_path = os.path.join(checkpoint_path, POLICY_CHECKPOINT_SUBDIRECTORY, OPTIMIZER_STATE_MARKER)
    if not io.exists(marker_path):
        return
    with io.open_file(marker_path, "r") as stream:
        marker = json.load(stream)
    if marker.get("optimizer_state") == OptimizerStateMode.MASTERS_ONLY.value:
        raise ValueError(
            f"the checkpoint at {checkpoint_path} carries no optimizer state; resume from a checkpoint "
            "whose step is listed in trainer.optimizer_checkpoint_steps"
        )


def master_only_optimizer_state(optimizer_state: Mapping[str, Any]) -> dict[str, Any]:
    """Drop moment and scalar records, keeping the param and FP32 master records.

    DCP names records by their ShardedTensor key, which differs from the dict path that
    carries it, so the filter matches ``leaf.key``. The dropped moment states are what
    a resume needs, so master-only checkpoints are not optimizer-resumable.
    """

    def keep_leaf(value: Any) -> bool:
        key = getattr(value, "key", None)
        return isinstance(key, str) and model_key_for_master(key) is not None

    def filter_subtree(value: Any) -> tuple[Any, bool]:
        if not isinstance(value, Mapping):
            return (value, keep_leaf(value))
        kept: dict[str, Any] = {}
        for key, child in value.items():
            subtree, has_master = filter_subtree(child)
            if has_master:
                kept[key] = subtree
        return (kept, bool(kept))

    filtered, _ = filter_subtree(optimizer_state)
    if not filtered:
        raise ValueError(
            "optimizer state carries no param or FP32 master records; cannot write a master-only checkpoint"
        )
    return filtered
