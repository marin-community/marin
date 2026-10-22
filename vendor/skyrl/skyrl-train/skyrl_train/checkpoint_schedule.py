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
from typing import Any

from omegaconf import DictConfig, OmegaConf

from skyrl_train.distributed.megatron.checkpoint_writer import model_key_for_master

OPTIMIZER_STATE_MARKER = "optimizer_state.json"


class OptimizerStateMode(StrEnum):
    """How much optimizer state one checkpoint carries."""

    FULL = "full"
    MASTERS_ONLY = "masters"


FULL_OPTIMIZER_STATE = OptimizerStateMode.FULL
MASTER_ONLY_OPTIMIZER_STATE = OptimizerStateMode.MASTERS_ONLY


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
        trainer = config["trainer"] if "trainer" in config else config
        weight_steps = _step_list(trainer, "checkpoint_steps")
        optimizer_steps = _step_list(trainer, "optimizer_checkpoint_steps")
        if weight_steps and not optimizer_steps:
            raise ValueError(
                "trainer.optimizer_checkpoint_steps is required when trainer.checkpoint_steps is set; "
                "list the steps whose checkpoints must resume training"
            )
        return cls(
            interval=int(trainer.get("ckpt_interval", 0) or 0),
            weight_steps=weight_steps,
            optimizer_steps=optimizer_steps,
        )

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
        return FULL_OPTIMIZER_STATE if self.should_save_optimizer(step) else MASTER_ONLY_OPTIMIZER_STATE


def _step_list(trainer: Mapping[str, Any], field: str) -> tuple[int, ...]:
    value = OmegaConf.to_container(trainer[field], resolve=True) if field in trainer else []
    if not isinstance(value, list) or any(
        not isinstance(step, int) or isinstance(step, bool) or step <= 0 for step in value
    ):
        raise ValueError(f"trainer.{field} must be a list of positive step numbers")
    return tuple(sorted(set(value)))


def checkpoint_schedule(cfg: DictConfig) -> CheckpointSchedule:
    """Resolve the run's checkpoint schedule from the trainer config."""
    return CheckpointSchedule.from_config(cfg)


def master_only_optimizer_state(optimizer_state: Mapping[str, Any]) -> dict[str, Any]:
    """Drop moment and scalar records, keeping the param and FP32 master records.

    Walks the sharded-state nesting and keeps every subtree whose leaf keys name a
    param or FP32 master record — the records weight analysis reads. The dropped
    moment states are what a resume needs, so master-only checkpoints are not
    optimizer-resumable.
    """

    def keep_leaf(path: str, value: Any) -> bool:
        return not isinstance(value, Mapping) and model_key_for_master(path) is not None

    def filter_subtree(path: str, value: Any) -> tuple[dict[str, Any], bool]:
        if not isinstance(value, Mapping):
            return ({**{path: value}}, keep_leaf(path, value))
        kept: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            subtree, has_master = filter_subtree(child_path, child)
            if has_master:
                kept[key] = subtree
        return (kept, bool(kept))

    filtered, _ = filter_subtree("", optimizer_state)
    if not filtered:
        raise ValueError(
            "optimizer state carries no param or FP32 master records; cannot write a master-only checkpoint"
        )
    return filtered
