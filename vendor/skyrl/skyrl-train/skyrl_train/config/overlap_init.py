"""Eligibility for overlapping policy and local inference startup."""

from omegaconf import DictConfig
from marinskyrl.distillation import compile_distillation_plan_from_config
from marinskyrl.runtime_options import reference_model_required


def validate_overlap_init(cfg: DictConfig) -> None:
    if not cfg.trainer.placement.overlap_init:
        return
    conditions = {
        "trainer.placement.colocate_all must be false": not cfg.trainer.placement.colocate_all,
        "trainer.placement.policy_strict_spread_pg must be true": cfg.trainer.placement.policy_strict_spread_pg,
        "the objective must not require a reference model": not reference_model_required(cfg.trainer.algorithm),
        "generator.run_engines_locally must be true": cfg.generator.run_engines_locally,
        "distillation must be disabled": compile_distillation_plan_from_config(cfg) is None,
    }
    blockers = [reason for reason, satisfied in conditions.items() if not satisfied]
    if blockers:
        raise ValueError("Invalid trainer.placement.overlap_init: " + "; ".join(blockers))
