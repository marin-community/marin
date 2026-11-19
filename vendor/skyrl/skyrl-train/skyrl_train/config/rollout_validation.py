"""Validate launch requirements for the shared task rollout engine."""

from enum import StrEnum

from omegaconf import DictConfig

from marinskyrl.distillation import compile_distillation_plan_from_config
from marinskyrl.runtime_options import WeightSyncTransport


class EntrypointOperation(StrEnum):
    TRAIN = "train"
    GENERATE = "generate"


def validate_rollout_launch(
    cfg: DictConfig,
    operation: EntrypointOperation = EntrypointOperation.TRAIN,
) -> None:
    """Validate exact-token transport and training-only distillation requirements."""
    if cfg.generator.backend != "vllm":
        raise ValueError("Task rollouts require generator.backend=vllm for exact structured-chat transport")
    if (
        cfg.trainer.placement.colocate_all
        and cfg.generator.weight_sync_transport == WeightSyncTransport.EXPERT_BLOCK
        and operation is not EntrypointOperation.TRAIN
    ):
        raise ValueError("colocated expert_block requires a training entrypoint")
    if operation is EntrypointOperation.GENERATE and compile_distillation_plan_from_config(cfg) is not None:
        raise ValueError("teacher-scored distillation is training-only and cannot be configured for generation")
