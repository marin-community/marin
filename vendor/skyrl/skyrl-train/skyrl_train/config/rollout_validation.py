"""Validate launch requirements for the shared task rollout engine."""

from collections.abc import Mapping
from enum import StrEnum

from omegaconf import DictConfig, OmegaConf

from marinskyrl.distillation import compile_distillation_plan_from_config
from marinskyrl.runtime_options import WeightSyncTransport
from skyrlrecipe.declared import PromptAcquisition
from skyrl_train.callbacks.types import FORK_SOURCE_QUOTA_TYPE
from skyrl_train.probe_mix import CALLBACK_TYPE as DEVELOPMENT_MIX_TYPE, WINDOW as PROBE_WINDOW

GROUPS_PER_FAMILY_KEY = "trainer.rollout_buffer.groups_per_family"
FAMILY_COLUMN = "family"
SPARE_PROMPTS_PER_POOL = 1


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
    """Return the minimum distinct prompt count for each family, including one spare."""
    return {
        family: (max_staleness_steps + 1) * count + SPARE_PROMPTS_PER_POOL for family, count in groups_per_family.items()
    }


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
    milestones = tuple(cfg.trainer.get("rollout_token_eval_milestones", ()))
    if milestones and (
        type(budget) is not int
        or any(type(value) is not int for value in milestones)
        or not all(left < right for left, right in zip((0, *milestones), (*milestones, budget), strict=True))
    ):
        raise ValueError("trainer.rollout_token_eval_milestones must increase strictly between zero and the budget")
    if budget is not None:
        if type(budget) is not int or budget <= 0:
            raise ValueError("trainer.rollout_token_budget must be a positive integer")
        if operation is not EntrypointOperation.TRAIN:
            raise ValueError("trainer.rollout_token_budget requires a training entrypoint")
        if cfg.trainer.rollout_buffer.max_staleness_steps != 0:
            raise ValueError("trainer.rollout_token_budget requires synchronous generation (max_staleness_steps=0)")
        if not cfg.data.val_data:
            raise ValueError("trainer.rollout_token_budget requires a primary evaluation panel in data.val_data")


    callbacks = cfg.trainer.get("callbacks") or ()
    forks = [item for item in callbacks if item.type == FORK_SOURCE_QUOTA_TYPE]
    if forks and (
        len(forks) != 1
        or forks[0].easy_quota not in (6, 10)
        or cfg.trainer.resume_mode != "latest"
        or not cfg.trainer.get("resume_path")
        or not cfg.trainer.restore_dataloader_state
        or cfg.trainer.get("reset_global_step_on_resume", False)
        or budget is None
        or milestones
        or cfg.trainer.train_batch_size != PROBE_WINDOW
        or cfg.data.sampling.kind != PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE
        or cfg.data.sampling.epsilon != 1
        or any(item.type == DEVELOPMENT_MIX_TYPE for item in callbacks)
    ):
        raise ValueError("A source allocation fork requires a native checkpoint, a gross budget and uniform source passes")
    mixes = [item for item in callbacks if item.type == DEVELOPMENT_MIX_TYPE]
    if mixes:
        evaluation = [item for item in callbacks if item.type == "evaluation"]
        profiles = evaluation[0].get("additional_evaluations") or {} if len(evaluation) == 1 else {}
        if (
            len(mixes) != 1
            or not milestones
            or cfg.trainer.train_batch_size != PROBE_WINDOW
            or cfg.data.sampling.kind != PromptAcquisition.WITHIN_SOURCE_PASS_COST_AWARE
            or cfg.data.sampling.epsilon != 1
            or cfg.generator.eval_n_samples_per_prompt != 1
            or len(evaluation) != 1
            or 0 < evaluation[0].get("eval_steps", 1) <= cfg.trainer.max_steps
            or not evaluation[0].get("eval_before_train")
            or any(profile.get("schedule") != "final" for profile in profiles.values())
            or set(profiles) != ({"confirmation"} if mixes[0].get("confirmation_sha256") else set())
            or bool(mixes[0].get("confirmation_tasks")) != bool(mixes[0].get("confirmation_sha256"))
        ):
            raise ValueError(
                "Development probe mix needs uniform source passes, token milestones, a primary one-sample "
                "development evaluation, and an optional pinned terminal-only confirmation"
            )
