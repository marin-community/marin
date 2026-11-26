"""Validate launch requirements for the shared task rollout engine."""

from collections.abc import Mapping
from enum import StrEnum

from omegaconf import DictConfig, OmegaConf

from marinskyrl.distillation import compile_distillation_plan_from_config
from marinskyrl.runtime_options import WeightSyncTransport
from skyrl_train.config.sampling import PromptAcquisition

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
        kind = cfg.data.sampling.kind
        if kind == PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE:
            raise ValueError(f"{GROUPS_PER_FAMILY_KEY} cannot partition a per-source prompt pass")
        if kind is not None and kind not in PromptAcquisition:
            raise ValueError(f"{GROUPS_PER_FAMILY_KEY} requires data.sampling.kind to be null or a prompt-Thompson kind")
    return dict(groups_per_family)


def minimum_family_pool_sizes(groups_per_family: Mapping[str, int], max_staleness_steps: int) -> dict[str, int]:
    """Return the number of distinct prompts each family needs across its open batches."""
    return {family: (max_staleness_steps + 1) * count for family, count in groups_per_family.items()}


def validate_family_pools(
    groups_per_family: Mapping[str, int], family_counts: Mapping[str | None, int], max_staleness_steps: int
) -> None:
    """Check that every family can supply its full window of distinct groups."""
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
            raise ValueError(f"{GROUPS_PER_FAMILY_KEY} requires {needed} rows for family {family!r}, found {available}")


def validate_rollout_launch(
    cfg: DictConfig,
    operation: EntrypointOperation = EntrypointOperation.TRAIN,
) -> None:
    """Validate rollout transport and training-only configuration before starting workers."""
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
    budget = cfg.trainer.get("rollout_token_budget")
    if budget is not None:
        if type(budget) is not int or budget <= 0:
            raise ValueError("trainer.rollout_token_budget must be a positive integer")
        if operation is not EntrypointOperation.TRAIN:
            raise ValueError("trainer.rollout_token_budget requires a training entrypoint")
        if cfg.trainer.rollout_buffer.max_staleness_steps != 0:
            raise ValueError("trainer.rollout_token_budget requires synchronous generation (max_staleness_steps=0)")
        if cfg.trainer.resume_mode != "none":
            raise ValueError("trainer.rollout_token_budget requires a fresh run (resume_mode=none)")
        if not cfg.data.val_data:
            raise ValueError("trainer.rollout_token_budget requires a primary evaluation panel in data.val_data")
