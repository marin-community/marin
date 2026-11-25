"""Validate launch requirements for the shared task rollout engine."""

from collections.abc import Mapping
from enum import StrEnum

from omegaconf import DictConfig, OmegaConf

from marinskyrl.distillation import compile_distillation_plan_from_config
from marinskyrl.runtime_options import WeightSyncTransport

GROUPS_PER_FAMILY_KEY = "trainer.rollout_buffer.groups_per_family"
FAMILY_COLUMN = "family"


class BatchPolicy(StrEnum):
    """How committed groups are batched. Both policies coincide at max_staleness_steps=0."""

    FULL_BATCH = "full_batch"
    ROLLING = "rolling"


class EntrypointOperation(StrEnum):
    TRAIN = "train"
    GENERATE = "generate"


def validate_groups_per_family(cfg: DictConfig) -> dict[str, int]:
    """Read the groups per family in each batch and validate the training settings."""
    groups_per_family = OmegaConf.select(cfg, GROUPS_PER_FAMILY_KEY, default={})
    if not isinstance(groups_per_family, Mapping) or any(
        not isinstance(family, str) or not family or type(count) is not int or count <= 0
        for family, count in groups_per_family.items()
    ):
        raise ValueError(f"{GROUPS_PER_FAMILY_KEY} requires family names and positive integer counts")
    if groups_per_family:
        if cfg.trainer.rollout_buffer.batch_policy != BatchPolicy.FULL_BATCH:
            raise ValueError(f"{GROUPS_PER_FAMILY_KEY} requires batch_policy={BatchPolicy.FULL_BATCH}")
        batch_size = cfg.trainer.train_batch_size
        if sum(groups_per_family.values()) != batch_size:
            raise ValueError(
                f"{GROUPS_PER_FAMILY_KEY} sums to {sum(groups_per_family.values())}, "
                f"expected train_batch_size={batch_size}"
            )
        if cfg.data.sampling.kind is not None:
            raise ValueError(f"{GROUPS_PER_FAMILY_KEY} requires data.sampling.kind to be null")
    return dict(groups_per_family)


def minimum_family_pool_sizes(groups_per_family: Mapping[str, int], max_staleness_steps: int) -> dict[str, int]:
    """Reserve a distinct prompt for each open slot and one spare per family."""
    return {family: (max_staleness_steps + 1) * count + 1 for family, count in groups_per_family.items()}


def validate_family_pools(
    groups_per_family: Mapping[str, int], family_counts: Mapping[str | None, int], max_staleness_steps: int
) -> None:
    """Check that every family can fill its open batches and keep one spare."""
    if family_counts.get(None, 0) or family_counts.get("", 0):
        raise ValueError(f"{GROUPS_PER_FAMILY_KEY} requires a family on every training row")
    if family_counts.keys() != groups_per_family.keys():
        raise ValueError(
            f"{GROUPS_PER_FAMILY_KEY} names {sorted(groups_per_family)}, "
            f"but training rows have families {sorted(map(str, family_counts))}"
        )
    for family, needed in minimum_family_pool_sizes(groups_per_family, max_staleness_steps).items():
        available = family_counts[family]
        if available < needed:
            raise ValueError(
                f"{GROUPS_PER_FAMILY_KEY} requires {needed} distinct prompts for family {family!r} "
                f"to fill its open batches and keep one spare, found {available}"
            )


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
