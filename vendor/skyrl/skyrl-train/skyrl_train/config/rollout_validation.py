"""Validate launch requirements for the shared task rollout engine."""

from collections.abc import Mapping
from enum import StrEnum

from omegaconf import DictConfig, OmegaConf

from marinskyrl.distillation import compile_distillation_plan_from_config
from marinskyrl.runtime_options import WeightSyncTransport

FAMILY_QUOTAS_KEY = "trainer.rollout_buffer.family_quotas"
FAMILY_COLUMN = "family"


class BatchPolicy(StrEnum):
    """How committed groups are assigned to training batches."""

    FULL_BATCH = "full_batch"
    ROLLING = "rolling"


class EntrypointOperation(StrEnum):
    TRAIN = "train"
    GENERATE = "generate"


def validate_family_quota_config(cfg: DictConfig) -> dict[str, int]:
    """Read the batch quotas and check their interaction with training settings."""
    quotas = OmegaConf.select(cfg, FAMILY_QUOTAS_KEY, default={})
    if not isinstance(quotas, Mapping) or any(
        not isinstance(family, str) or not family or type(count) is not int or count <= 0
        for family, count in quotas.items()
    ):
        raise ValueError(f"{FAMILY_QUOTAS_KEY} requires family names and positive integer counts")
    if quotas:
        if cfg.trainer.rollout_buffer.batch_policy != BatchPolicy.FULL_BATCH:
            raise ValueError(f"{FAMILY_QUOTAS_KEY} requires batch_policy={BatchPolicy.FULL_BATCH}")
        batch_size = cfg.trainer.train_batch_size
        if sum(quotas.values()) != batch_size:
            raise ValueError(f"{FAMILY_QUOTAS_KEY} sums to {sum(quotas.values())}, expected train_batch_size={batch_size}")
        if cfg.data.sampling.kind is not None:
            raise ValueError(f"{FAMILY_QUOTAS_KEY} requires data.sampling.kind to be null")
    return dict(quotas)


def validate_family_pools(
    quotas: Mapping[str, int], family_counts: Mapping[str | None, int], max_staleness_steps: int
) -> None:
    """Check that every family can supply its full window of distinct groups."""
    if family_counts.get(None, 0) or family_counts.get("", 0):
        raise ValueError(f"{FAMILY_QUOTAS_KEY} requires a family on every training row")
    if family_counts.keys() != quotas.keys():
        raise ValueError(
            f"{FAMILY_QUOTAS_KEY} names {sorted(quotas)}, but training rows have families {sorted(map(str, family_counts))}"
        )
    for family, quota in quotas.items():
        available = family_counts[family]
        needed = (max_staleness_steps + 1) * quota
        if available < needed:
            raise ValueError(f"{FAMILY_QUOTAS_KEY} requires {needed} rows for family {family!r}, found {available}")


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
