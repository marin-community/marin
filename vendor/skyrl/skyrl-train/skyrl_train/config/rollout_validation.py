"""Validate launch requirements for the shared task rollout engine."""

from collections.abc import Mapping
from enum import StrEnum

from omegaconf import DictConfig, OmegaConf

from marinskyrl.distillation import compile_distillation_plan_from_config
from marinskyrl.runtime_options import WeightSyncTransport


class EntrypointOperation(StrEnum):
    TRAIN = "train"
    GENERATE = "generate"


def validate_family_quota_config(cfg: DictConfig) -> dict[str, int]:
    """Read the batch quotas and check their interaction with training settings."""
    key = "trainer.rollout_buffer.family_quotas"
    quotas = OmegaConf.select(cfg, key, default={})
    if not isinstance(quotas, Mapping) or any(
        not isinstance(family, str) or not family or type(count) is not int or count <= 0
        for family, count in quotas.items()
    ):
        raise ValueError(f"{key} requires family names and positive integer counts")
    if quotas:
        if cfg.trainer.rollout_buffer.batch_policy != "full_batch":
            raise ValueError(f"{key} requires batch_policy=full_batch")
        batch_size = cfg.trainer.train_batch_size
        if sum(quotas.values()) != batch_size:
            raise ValueError(f"{key} sums to {sum(quotas.values())}, expected train_batch_size={batch_size}")
        if cfg.data.sampling.kind is not None:
            raise ValueError(f"{key} requires data.sampling.kind to be null")
    return dict(quotas)


def validate_family_pools(
    quotas: Mapping[str, int], family_counts: Mapping[str | None, int], max_staleness_steps: int
) -> None:
    """Check that every family can supply its full window of distinct groups."""
    key = "trainer.rollout_buffer.family_quotas"
    if family_counts.get(None, 0) or family_counts.get("", 0):
        raise ValueError(f"{key} requires a family on every training row")
    if family_counts.keys() != quotas.keys():
        raise ValueError(f"{key} names {sorted(quotas)}, but training rows have families {sorted(map(str, family_counts))}")
    for family, quota in quotas.items():
        available = family_counts[family]
        needed = (max_staleness_steps + 1) * quota
        if available < needed:
            raise ValueError(f"{key} requires {needed} rows for family {family!r}, found {available}")


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
