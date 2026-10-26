"""Generation-only types, open mappings, code-default keys and public class names."""

from enum import Enum
from pathlib import Path
from types import MappingProxyType

from skyrl_train.config.ftpo import FTPOBoundary
from skyrl_train.config.objective_spec import KLEstimator, LossReduction
from skyrl_train.config.query_bias import GrugQueryBiasUpdateMode
from skyrl_train.config.remote_read import RemoteReadMode
from skyrl_train.config.reasoning_span_penalty import ReasoningSpanPenaltyTarget
from skyrl_train.config.weight_sync_pause import WeightSyncPauseMode
from skyrl_train.curriculum import SamplingKind, WeightingKind
from skyrl_train.dataset.preference_pairs import PreferencePairFormat
from skyrl_train.dynamic_sampling import DynamicSamplingRewardSource, DynamicSamplingType
from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.group_admission import GroupAdvantageKind
from skyrl_train.hf_export_schema import HFUploadMode
from skyrl_train.rollouts.buffer import BatchPolicy
from skyrl_train.trajectory_selection import TrajectorySelectorKind
from skyrl_train.utils.advantage_estimators import AdvantageEstimatorRegistry
from skyrl_train.utils.trainer_utils import ResumeMode

from marinskyrl import runtime_options
from marinskyrl.distillation import DistillationObjectiveKind, DistillationRewardMode
from marinskyrl.environment_contract import DebugMode
from marinskyrl.resource_locator import RolloutModelLoading
from marinskyrl.speculative_decoding import SpeculativeDecodingMethod


def _literal(choices) -> str:
    return "Literal[" + ", ".join(repr(choice.value if isinstance(choice, Enum) else choice) for choice in choices) + "]"


CORRECTION_PRESETS = tuple(
    sorted(
        path.stem
        for path in (Path(__file__).resolve().parents[1] / "skyrl-train/skyrl_train/config/off_policy_correction").glob(
            "*.yaml"
        )
    )
)

OPTIMIZER_CHECKPOINT_SHARDING = 'Literal["fully_reshardable", "dp_reshardable"]'

TYPES = MappingProxyType(
    {
        "terminal_bench": "OpenMap | None",
        "trainer.strategy": 'Literal["megatron"]',
        "trainer.policy.model.remote_read_mode": _literal(RemoteReadMode),
        "trainer.ref.model.remote_read_mode": _literal(RemoteReadMode),
        "trainer.hf_export_checkpoint": 'Literal["latest", "best"]',
        "trainer.policy.optimizer_config.lr_decay_steps": "int | None",
        "trainer.critic.optimizer_config.lr_decay_steps": "int | None",
        "trainer.policy.megatron_config.nccl_max_channels": "int | None",
        "trainer.ref.megatron_config.nccl_max_channels": "int | None",
        "generator.trajectory_reward_shaping.reasoning_span.target": _literal(ReasoningSpanPenaltyTarget),
        "trainer.resume_mode": _literal(ResumeMode),
        "trainer.progress.mode": 'Literal["auto", "tqdm", "logging"]',
        "trainer.debug_mode": _literal(DebugMode),
        "trainer.preflight_gate.on_failure": 'Literal["abort", "warn"]',
        "trainer.logger": "LoggingBackend | tuple[LoggingBackend, ...]",
        "trainer.policy.optimizer_config.scheduler": 'Literal["constant_with_warmup", "cosine"]',
        "trainer.critic.optimizer_config.scheduler": 'Literal["constant_with_warmup", "cosine"]',
        "trainer.policy.megatron_config.optimizer_checkpoint_sharding_type": OPTIMIZER_CHECKPOINT_SHARDING,
        "trainer.ref.megatron_config.optimizer_checkpoint_sharding_type": OPTIMIZER_CHECKPOINT_SHARDING,
        "trainer.rollout_buffer.max_staleness_steps": "NonNegativeInt",
        "trainer.rollout_buffer.batch_policy": _literal(BatchPolicy),
        "trainer.hf_upload_mode": _literal(HFUploadMode),
        "trainer.policy.grug_query_bias_update_mode": _literal(GrugQueryBiasUpdateMode),
        "trainer.algorithm.advantage_estimator": _literal(AdvantageEstimatorRegistry.list_available()),
        "trainer.algorithm.resolved_group_advantage.kind": _literal(GroupAdvantageKind),
        "trainer.algorithm.policy_loss_type": _literal(runtime_options.PolicyLossType),
        "trainer.algorithm.loss_reduction": _literal(LossReduction),
        "trainer.algorithm.kl_estimator_type": _literal(KLEstimator),
        "trainer.algorithm.kl_ctrl.type": 'Literal["fixed", "adaptive"]',
        "trainer.algorithm.off_policy_correction": _literal(("none", "custom", *CORRECTION_PRESETS)) + " | None",
        "trainer.algorithm.off_policy_correction_rules": "tuple[CorrectionRule, ...] | None",
        "trainer.algorithm.dynamic_sampling.type": _literal(DynamicSamplingType) + " | None",
        "trainer.algorithm.dynamic_sampling.informative_on": _literal(DynamicSamplingRewardSource),
        "trainer.algorithm.distillation": "Distillation | None",
        "trainer.algorithm.distillation.objective": _literal(DistillationObjectiveKind),
        "trainer.algorithm.distillation.reward_mode": _literal(DistillationRewardMode),
        "trainer.algorithm.distillation.advantage_clip": "int | float | None",
        "trainer.mismatch_probe.rescore_prefix_cache": 'Literal["off", "on", "both"]',
        "trainer.mismatch_probe.extra_trainer_modes": 'tuple[Literal["router_replay", "router_replay_filtered"], ...]',
        "generator.backend": 'Literal["vllm"]',
        "generator.weight_sync_backend": 'Literal["nccl", "gloo"]',
        "generator.weight_sync_transport": _literal(runtime_options.WeightSyncTransport),
        "generator.override_existing_update_group": 'Literal["auto", "enable", "disable"]',
        "generator.weight_sync_pause.mode": _literal(WeightSyncPauseMode),
        "generator.r3_transport": _literal(runtime_options.R3Transport),
        "generator.gdn_backend": _literal(runtime_options.GDNBackend),
        "generator.chat_template": "ChatTemplate | None",
        "generator.chat_template.source": 'Literal["name", "file"]',
        "generator.error_handling.default_error_treatment": _literal(ErrorTreatment),
        "generator.trajectory_retention.phases": 'tuple[Literal["train", "eval"], ...]',
        "trainer.policy.model.lora.dropout": "Annotated[int | float, Field(ge=0, le=1)]",
        "trainer.critic.model.lora.dropout": "Annotated[int | float, Field(ge=0, le=1)]",
        "trainer.step_phase_budgets.group_admission": "Annotated[int | float, Field(ge=0)]",
        "trainer.step_phase_budgets.batch_assembly": "Annotated[int | float, Field(ge=0)]",
        "trainer.step_phase_budgets.training_preparation": "Annotated[int | float, Field(ge=0)]",
        "trainer.step_phase_budgets.advantages": "Annotated[int | float, Field(ge=0)]",
        "trainer.step_phase_budgets.policy_training": "Annotated[int | float, Field(ge=0)]",
        "trainer.step_phase_budgets.group_bookkeeping": "Annotated[int | float, Field(ge=0)]",
        "trainer.step_phase_budgets.weight_sync": "Annotated[int | float, Field(ge=0)]",
        "trainer.step_phase_budgets.step_end_bookkeeping": "Annotated[int | float, Field(ge=0)]",
        "trainer.step_phase_budgets.checkpoint_work": "Annotated[int | float, Field(ge=0)]",
        "trainer.step_phase_budgets.evaluation": "Annotated[int | float, Field(ge=0)]",
        "trainer.step_phase_budgets.unaccounted": "Annotated[int | float, Field(ge=0)]",
        "trainer.train_batch_size": "PositiveInt",
        "trainer.policy_mini_batch_size": "PositiveInt",
        "trainer.critic_mini_batch_size": "PositiveInt",
        "trainer.micro_train_batch_size_per_gpu": "PositiveInt",
        "trainer.eval_batch_size": "PositiveInt",
        "trainer.update_epochs_per_batch": "PositiveInt",
        "trainer.epochs": "PositiveInt",
        "generator.n_samples_per_prompt": "PositiveInt",
        "generator.inference_engine_tensor_parallel_size": "PositiveInt",
        "generator.inference_engine_pipeline_parallel_size": "PositiveInt",
        "generator.inference_engine_data_parallel_size": "PositiveInt",
        "generator.inference_engine_expert_parallel_size": "PositiveInt",
        "trainer.mismatch_probe.prompts.count": "PositiveInt",
        "trainer.mismatch_probe.prompts.samples_per_prompt": "PositiveInt",
        "trainer.placement.policy_num_nodes": "PositiveInt | None",
        "trainer.placement.policy_num_gpus_per_node": "PositiveInt | None",
        "trainer.placement.ref_num_nodes": "PositiveInt | None",
        "trainer.placement.ref_num_gpus_per_node": "PositiveInt | None",
        "generator.num_inference_engines": "PositiveInt | None",
        "trainer.run_name": "str | None",
        "trainer.policy.max_consecutive_nonfinite_steps": "int | None",
        "trainer.policy.nccl_buffer_size_bytes": "PositiveInt | None",
        "trainer.ref.nccl_buffer_size_bytes": "PositiveInt | None",
        "data.train_data": "tuple[str, ...] | None",
        "data.val_data": "tuple[str, ...] | None",
        "trainer.max_steps": "NonNegativeInt | None",
        "trainer.resume_path": "str | None",
        "trainer.eval_num_prompts": "PositiveInt | None",
        "trainer.callbacks": "tuple[Callback, ...] | None",
        "trainer.rollout_buffer.max_in_flight": "PositiveInt | None",
        "trainer.rollout_buffer.object_store_root": "str | None",
        "trainer.policy.megatron_config.expert_tensor_parallel_size": "PositiveInt | None",
        "trainer.critic.model.path": "str | None",
        "trainer.mismatch_probe.seed": "NonNegativeInt | None",
        "trainer.mismatch_probe.archive_uri": "str | None",
        "trainer.mismatch_probe.reuse_probe": "str | None",
        "trainer.mismatch_probe.filtered_replay.keep_fraction": "Annotated[int | float, Field(ge=0, le=1)] | None",
        "generator.vllm_attention_backend": "str | None",
        "generator.chat_template.name_or_path": "str | None",
        "generator.sampling_params.logprobs": "NonNegativeInt | None",
        "generator.eval_sampling_params.logprobs": "NonNegativeInt | None",
        "generator.sampling_params.stop": "tuple[str, ...] | None",
        "generator.eval_sampling_params.stop": "tuple[str, ...] | None",
        "generator.sampling_params.stop_token_ids": "tuple[int, ...] | None",
        "generator.eval_sampling_params.stop_token_ids": "tuple[int, ...] | None",
        "data.sampling.kind": _literal(SamplingKind) + " | None",
        "data.sampling.weighting": _literal(WeightingKind),
        "data.sampling.domain_weights": "NumberMap",
        "data.sampling.seed": "NonNegativeInt | None",
        "trainer.collective_phase_diagnostics": "bool | None",
        "generator.error_handling.passthrough_exceptions": "tuple[str, ...]",
        "generator.error_handling.mask_exceptions": "tuple[str, ...]",
        "generator.error_handling.zero_exceptions": "tuple[str, ...]",
        "generator.trajectory_retention.redact_fields": "tuple[str, ...]",
        "data.terminal_bench_data": "tuple[str, ...]",
        "trainer.trajectory_selector.type": _literal(TrajectorySelectorKind) + " | None",
        "trainer.policy.grug_query_bias_interpolation_weight": "int | float | None",
        "trainer.policy.grug_query_bias_update_rate": "int | float | None",
        "trainer.policy.model.revision": "str | None",
        "trainer.policy.model.source_uri": "str | None",
        "trainer.policy.model.source_identity": "str | None",
        "trainer.policy.model.lora.adapter_path": "str | None",
        "trainer.policy.model.lora.adapter_revision": "str | None",
        "trainer.policy.model.lora.exclude_modules": "tuple[str, ...] | None",
        "trainer.critic.model.lora.exclude_modules": "tuple[str, ...] | None",
        "trainer.policy.megatron_config.torch_profiler_config.ranks": "tuple[int, ...]",
        "trainer.policy.megatron_config.torch_profiler_config.save_path": "str | None",
        "trainer.ref.megatron_config.expert_tensor_parallel_size": "PositiveInt | None",
        "trainer.ref.megatron_config.torch_profiler_config.ranks": "tuple[int, ...]",
        "trainer.ref.megatron_config.torch_profiler_config.save_path": "str | None",
        "trainer.algorithm.group_advantage_min_size": "PositiveInt | None",
        "trainer.algorithm.resolved_group_advantage.minimum_group_size": "PositiveInt | None",
        "trainer.algorithm.dynamic_sampling.max_mean_reward": "int | float | None",
        "trainer.algorithm.group_admission.stall_timeout": "int | float | None",
        "trainer.distillation_token_budget": "PositiveInt | None",
        "trainer.rope_theta": "int | float | None",
        "generator.trajectory_retention.reward_below": "int | float | None",
        "generator.trajectory_retention.reward_above": "int | float | None",
        "trainer.evaluation_runner": _literal(runtime_options.EvaluationRunner),
        "generator.model_loading": _literal(RolloutModelLoading),
        "data.preference_pair_format": _literal(PreferencePairFormat),
    }
)

OPEN = frozenset(
    {
        "environment.task_sessions",
        "generator.chat_template_kwargs",
        "generator.engine_init_kwargs",
        "generator.rope_scaling",
        "trainer.critic.optimizer_config.optimizer_kwargs",
        "trainer.policy.megatron_config.model_config_kwargs",
        "trainer.policy.megatron_config.transformer_config_kwargs",
        "trainer.policy.optimizer_config.optimizer_kwargs",
        "trainer.ref.megatron_config.model_config_kwargs",
        "trainer.ref.megatron_config.transformer_config_kwargs",
        "trainer.rope_scaling",
    }
)

UNDECLARED = MappingProxyType(
    {
        "trainer.policy.megatron_config.optimizer_config_kwargs.store_param_remainders": ("bool", Ellipsis),
        "teachers": ("SectionMap[Teacher]", Ellipsis),
        "teacher_routing": ("SectionMap[TeacherRouting]", Ellipsis),
        "data.kind": ('Literal["tasks", "parquet"]', "tasks"),
        "trainer.enable_db_registration": ("bool", True),
        "trainer.hf_hub_repo_id": ("str | None", None),
        "trainer.hf_hub_private": ("bool", False),
        "trainer.hf_hub_revision": ("str", "main"),
        "generator.sampling_params.min_tokens": ("NonNegativeInt", Ellipsis),
        "generator.eval_sampling_params.min_tokens": ("NonNegativeInt", Ellipsis),
        "trainer.algorithm.ftpo.margin": ("int | float", Ellipsis),
        "trainer.algorithm.ftpo.lambda_mse": ("int | float", Ellipsis),
        "trainer.algorithm.ftpo.lambda_mse_target": ("int | float", Ellipsis),
        "trainer.algorithm.ftpo.tau_mse_target": ("int | float", Ellipsis),
        "trainer.algorithm.ftpo.min_p": ("int | float", Ellipsis),
        "trainer.algorithm.ftpo.max_chosen_tokens": ("PositiveInt", Ellipsis),
        "trainer.algorithm.ftpo.min_decoded_chars": ("NonNegativeInt", Ellipsis),
        "trainer.algorithm.ftpo.require_alnum": ("bool", Ellipsis),
        "trainer.algorithm.ftpo.rejected_balance_strength": ("int | float", Ellipsis),
        "trainer.algorithm.ftpo.chosen_balance_strength": ("int | float", Ellipsis),
        "trainer.algorithm.ftpo.early_stopping_chosen_win": ("int | float | None", Ellipsis),
        "trainer.algorithm.distillation.routing_plan": ("str", Ellipsis),
        "trainer.algorithm.distillation.coefficient": ("int | float", Ellipsis),
        "trainer.algorithm.distillation.jsd_beta": ("int | float | None", Ellipsis),
        "trainer.algorithm.distillation.entry_clip": ("int | float | None", Ellipsis),
        "trainer.algorithm.distillation.residency": ("Residency | None", None),
        "trainer.algorithm.distillation.residency.max_resident": ("PositiveInt", Ellipsis),
        "trainer.algorithm.distillation.residency.minimum_residency_seconds": ("int | float", Ellipsis),
        "trainer.algorithm.distillation.domain_gradient_balance.target_shares": ("NumberMap", Ellipsis),
        "trainer.algorithm.distillation.domain_gradient_balance.gap_scale_alpha": ("int | float", Ellipsis),
        "generator.speculative_decoding.method": (_literal(SpeculativeDecodingMethod), Ellipsis),
        "generator.speculative_decoding.model.source_uri": ("str", Ellipsis),
        "generator.speculative_decoding.model.source_identity": ("str", Ellipsis),
        "generator.speculative_decoding.num_speculative_tokens": ("PositiveInt", Ellipsis),
        "generator.speculative_decoding.training": ("SpeculatorTraining | None", None),
        "generator.speculative_decoding.training.interval_steps": ("PositiveInt", Ellipsis),
        "generator.speculative_decoding.training.max_tokens_per_update": ("PositiveInt", Ellipsis),
        "generator.speculative_decoding.training.max_window_tokens": ("PositiveInt", Ellipsis),
        "generator.speculative_decoding.training.max_tokens_per_micro_batch": ("PositiveInt", Ellipsis),
        "generator.speculative_decoding.training.max_sequences_per_prompt_group": ("PositiveInt", Ellipsis),
        "generator.speculative_decoding.training.min_train_sequences": ("PositiveInt", Ellipsis),
        "generator.speculative_decoding.training.holdout_fraction": ("int | float", Ellipsis),
        "generator.speculative_decoding.training.min_holdout_sequences": ("PositiveInt", Ellipsis),
        "generator.speculative_decoding.training.epochs_per_update": ("PositiveInt", Ellipsis),
        "generator.speculative_decoding.training.learning_rate": ("int | float", Ellipsis),
        "generator.speculative_decoding.training.max_validation_loss_increase": ("int | float", Ellipsis),
        "generator.speculative_decoding.training.max_validation_agreement_decrease": ("int | float", Ellipsis),
        "generator.speculative_decoding.training.reserved_gpu_memory_gib": ("int | float", Ellipsis),
        "trainer.algorithm.ftpo.boundary": (_literal(FTPOBoundary), Ellipsis),
        "trainer.algorithm.ftpo.repair_sft_weight": ("int | float | None", Ellipsis),
        "trainer.algorithm.ftpo.auxiliary_weight": ("int | float | None", Ellipsis),
    }
)

NAMES = MappingProxyType(
    {
        "trainer.policy.megatron_config": "PolicyMegatronConfig",
        "trainer.ref.megatron_config": "RefMegatronConfig",
        "trainer.policy.model": "PolicyModel",
        "trainer.ref.model": "RefModel",
        "trainer.critic.model": "CriticModel",
        "trainer.policy.optimizer_config": "PolicyOptimizerConfig",
        "trainer.critic.optimizer_config": "CriticOptimizerConfig",
        "trainer.policy.model.lora": "PolicyLora",
        "trainer.critic.model.lora": "CriticLora",
        "trainer.policy.megatron_config.ddp_config": "DdpConfig",
        "trainer.ref.megatron_config.ddp_config": "RefDdpConfig",
        "trainer.policy.megatron_config.torch_profiler_config": "TorchProfilerConfig",
        "trainer.ref.megatron_config.torch_profiler_config": "RefTorchProfilerConfig",
        "trainer.policy.megatron_config.optimizer_config_kwargs": "OptimizerConfigKwargs",
        "trainer.ref.megatron_config.optimizer_config_kwargs": "RefOptimizerConfigKwargs",
        "generator.speculative_decoding.model": "SpeculatorModel",
        "generator.speculative_decoding.training": "SpeculatorTraining",
    }
)
