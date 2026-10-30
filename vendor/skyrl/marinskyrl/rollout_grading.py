"""Config rules for training without per-rollout verification."""

from omegaconf import DictConfig, OmegaConf

from marinskyrl.distillation import DistillationPlan, DistillationRewardMode
from marinskyrl.runtime_options import TRAJECTORY_SELECTOR_TYPE_PATH, AdvantageEstimator, RolloutGrading


def validate_nemotron_ultra_grading(cfg: DictConfig, distillation_plan: DistillationPlan | None) -> None:
    """Allow skipped Nemotron Ultra grading only when nothing in training or eval reads the reward."""
    grading = RolloutGrading(cfg.environment.task_sessions.nemotron_ultra.grading)
    if grading is RolloutGrading.VERIFY:
        return
    prefix = "environment.task_sessions.nemotron_ultra.grading=skip requires"
    if distillation_plan is None or distillation_plan.reward_mode is not DistillationRewardMode.REPLACE:
        raise ValueError(f"{prefix} trainer.algorithm.distillation.reward_mode=replace")
    if cfg.trainer.algorithm.advantage_estimator != AdvantageEstimator.UNIFORM:
        raise ValueError(
            f"{prefix} trainer.algorithm.advantage_estimator=uniform; got {cfg.trainer.algorithm.advantage_estimator}"
        )
    if cfg.trainer.algorithm.dynamic_sampling.type is not None:
        raise ValueError(f"{prefix} trainer.algorithm.dynamic_sampling.type=null")
    if OmegaConf.select(cfg, TRAJECTORY_SELECTOR_TYPE_PATH, default=None) is not None:
        raise ValueError(f"{prefix} no trajectory selector")
    if cfg.trainer.eval_before_train or cfg.trainer.eval_interval > 0:
        raise ValueError(f"{prefix} trainer.eval_before_train=false and trainer.eval_interval<=0")
