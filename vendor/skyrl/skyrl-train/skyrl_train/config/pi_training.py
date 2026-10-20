"""Fail-closed Pi preset, installed-agent and training-objective binding."""

from collections.abc import Mapping

from omegaconf import DictConfig, OmegaConf

from marinskyrl.distillation import (
    DistillationObjectiveKind,
    DistillationRewardMode,
    TeacherEvidenceKind,
    compile_distillation_plan_from_config,
    validate_distillation_runtime_support,
)


def validate_pi_objective_config(cfg: DictConfig) -> None:
    """Validate a reviewed Pi preset against its mode and training objective."""
    scenario = OmegaConf.select(cfg, "scenario")
    if scenario is None:
        if OmegaConf.select(cfg, "terminal_bench_config.installed_agent.agent.pi.mode") == "teacher_trace":
            raise ValueError("Pi teacher trace requires a reviewed scenario.preset")
        return
    if not isinstance(scenario, DictConfig) or set(scenario) != {"preset"}:
        raise ValueError("Pi training requires exactly scenario.preset")
    if not OmegaConf.select(cfg, "data.train_data"):
        raise ValueError("Pi training requires direct Harbor task sources in data.train_data")
    if OmegaConf.select(cfg, "data.terminal_bench_data"):
        raise ValueError("Pi training does not use Nemotron terminal-bench reference routing")
    preset = scenario.preset
    modes = {
        "verifier_grpo_base": "literal",
        "teacher_distribution_pi_base": "literal",
        "teacher_demo_pi_base": "teacher_trace",
        "teacher_demo_pi_verified_base": "teacher_trace",
    }
    if preset not in modes:
        raise ValueError(f"Pi training preset {preset!r} is not wired to a production objective")
    agent = OmegaConf.select(cfg, "terminal_bench_config.installed_agent.agent")
    if agent is None or agent.get("name") != "pi" or OmegaConf.select(agent, "pi.mode") != modes[preset]:
        raise ValueError(f"{preset} requires an installed Pi agent with pi.mode={modes[preset]}")
    if not OmegaConf.select(agent, "kwargs.thinking_format"):
        raise ValueError(f"{preset} requires an explicit Pi thinking_format for the served model")
    if modes[preset] == "literal":
        served_model = cfg.generator.engine_init_kwargs.get("served_model_name") or cfg.trainer.policy.model.path
        if agent.get("model_name") not in (None, f"hosted_vllm/{served_model}"):
            raise ValueError("Pi student execution must identify the served student model")
        if cfg.generator.sampling_params.logprobs != 0:
            raise ValueError("Pi student execution requires sampled-token logprobs from the local inference engine")
    if cfg.trainer.eval_interval > 0 and cfg.generator.eval_sampling_params.logprobs != 0:
        raise ValueError("Pi student evaluation requires sampled-token logprobs from the local inference engine")
    step_wise = preset in ("verifier_grpo_base", "teacher_distribution_pi_base")
    if bool(cfg.trainer.step_wise_training) != step_wise:
        raise ValueError("Pi verifier GRPO and distribution require exact per-turn step-wise training rows")
    algorithm = cfg.trainer.algorithm
    if preset == "teacher_distribution_pi_base":
        plan = compile_distillation_plan_from_config(cfg)
        if (
            plan is None
            or plan.objective is not DistillationObjectiveKind.SPARSE_FORWARD_KL
            or plan.reward_mode is not DistillationRewardMode.REPLACE
            or len(plan.teachers) != 1
            or len(plan.routing.routes) != 1
            or plan.teachers[0].evidence is not TeacherEvidenceKind.TOPK_DISTRIBUTION
            or plan.teachers[0].model.path == cfg.trainer.policy.model.path
        ):
            raise ValueError(
                "Pi distribution requires one independent top-K teacher and sparse forward KL replacing rewards"
            )
        validate_distillation_runtime_support(plan)
        if (
            algorithm.policy_loss_type != "importance_sampling"
            or algorithm.advantage_estimator != "uniform"
            or cfg.generator.n_samples_per_prompt != 1
        ):
            raise ValueError("Pi distribution requires the shared top-K objective and one student trial per task")
    elif algorithm.get("distillation") is not None:
        raise ValueError("Only the Pi distribution preset may attach teacher-scoring evidence")
    if preset == "verifier_grpo_base":
        if algorithm.advantage_estimator != "grpo" or algorithm.policy_loss_type != "regular":
            raise ValueError("verifier_grpo_base requires GRPO advantages and regular clipped policy loss")
        if cfg.generator.n_samples_per_prompt < 2:
            raise ValueError("verifier_grpo_base requires at least two physical trials per prompt")
    elif preset != "teacher_distribution_pi_base" and (
        algorithm.policy_loss_type != "sft"
        or algorithm.advantage_estimator != "uniform"
        or cfg.generator.n_samples_per_prompt != 1
    ):
        raise ValueError("Pi teacher trace requires SFT, uniform advantages and one trial per prompt")
    if preset == "teacher_distribution_pi_base":
        if OmegaConf.select(cfg, "terminal_bench_config.harbor.verifier_disable") is True:
            raise ValueError("Pi student distillation requires a separate graded task-verifier outcome")
    if algorithm.dynamic_sampling.type is not None:
        raise ValueError("Pi scenarios do not support dynamic reward-based sampling")


def validate_pi_evaluation_results(cfg: DictConfig, metrics: Mapping[str, float]) -> None:
    """Reject a Pi evaluation when any physical task lacks a verifier result."""
    if OmegaConf.select(cfg, "scenario.preset") is None:
        return
    attempted = metrics.get("eval/all/num_attempted")
    if not attempted or metrics.get("eval/all/num_scored") != attempted:
        raise ValueError("Pi student evaluation requires a task-verifier score for every attempted trial")


def validate_pi_training_config(cfg: DictConfig) -> None:
    """Reject unavailable sources before allocating rollout or policy workers."""
    validate_pi_objective_config(cfg)
    if OmegaConf.select(cfg, "scenario.preset") not in ("teacher_demo_pi_base", "teacher_demo_pi_verified_base"):
        return
    from skyrl_train.rollouts.pi_teacher_binding import PiTeacherBinding

    binding = OmegaConf.select(cfg, "terminal_bench_config.installed_agent.teacher")
    if binding is None:
        raise ValueError("Pi teacher trace needs a distinct teacher model source")
    teacher = PiTeacherBinding.model_validate(OmegaConf.to_container(binding, resolve=True))
    if (
        OmegaConf.select(cfg, "terminal_bench_config.installed_agent.agent.model_name")
        != f"hosted_vllm/{teacher.model}"
    ):
        raise ValueError("Pi teacher agent must address the bound teacher alias")
    if teacher.student_model != cfg.trainer.policy.model.path:
        raise ValueError("Pi teacher binding must identify the configured student policy")
    preset = OmegaConf.select(cfg, "scenario.preset")
    verifier_disabled = OmegaConf.select(cfg, "terminal_bench_config.harbor.verifier_disable") is True
    if verifier_disabled != (preset == "teacher_demo_pi_base"):
        raise ValueError("Pi teacher preset must select its reviewed Harbor verification policy")
    if cfg.trainer.eval_interval > 0:
        if preset == "teacher_demo_pi_base":
            raise ValueError("Ungraded Pi teacher SFT requires a separate student evaluation run")
        if cfg.trainer.evaluation_runner != "harbor" or not cfg.data.val_data:
            raise ValueError("Scheduled Pi student evaluation requires Harbor validation tasks")
    if teacher.source == "skyrl_managed" and teacher.resources is None:
        raise ValueError("SkyRL-managed Pi teacher requires independent GPU resource allocation")
