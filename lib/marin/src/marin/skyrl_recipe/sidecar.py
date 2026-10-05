"""Schema annotations for generation and launch-time key admission."""

from types import MappingProxyType

POSITIVE = "PositiveInt"
POSITIVE_OR_NONE = "PositiveInt | None"

_TYPES = {
    "terminal_bench": "OpenMap | None",
    # Choices (values from MarinSkyRL's own enums and checks).
    "trainer.strategy": 'Literal["megatron"]',
    "trainer.resume_mode": 'Literal["none", "latest", "from_path"]',
    "trainer.progress.mode": 'Literal["auto", "tqdm", "logging"]',
    "trainer.debug_mode": 'Literal["off", "light", "distributed"]',
    "trainer.preflight_gate.on_failure": 'Literal["abort", "warn"]',
    "trainer.logger": "LoggingBackend | tuple[LoggingBackend, ...]",
    "trainer.policy.optimizer_config.scheduler": 'Literal["constant_with_warmup"]',
    "trainer.critic.optimizer_config.scheduler": 'Literal["constant_with_warmup"]',
    "trainer.policy.megatron_config.optimizer_checkpoint_sharding_type": 'Literal["fully_reshardable", "dp_reshardable"]',
    "trainer.ref.megatron_config.optimizer_checkpoint_sharding_type": 'Literal["fully_reshardable", "dp_reshardable"]',
    "trainer.rollout_buffer.batch_policy": 'Literal["full_batch", "rolling"]',
    "trainer.hf_upload_mode": 'Literal["latest", "all"]',
    "checkpoint_export.hf_upload_mode": 'Literal["latest", "all"]',
    "trainer.policy.grug_query_bias_update_mode": 'Literal["frozen", "interpolate", "loss_free", "replace"]',
    "trainer.algorithm.advantage_estimator": 'Literal["gae", "grpo", "rloo", "rloo_n", "rloo_n_pbs", "reinforce++", "uniform", "reward"]',
    "trainer.algorithm.resolved_group_advantage.kind": 'Literal["exact_physical", "minimum_baseline_eligible", "none"]',
    "trainer.algorithm.policy_loss_type": 'Literal["regular", "dual_clip", "behavior_clip", "gspo", "cispo", "clip_cov", "kl_cov", "sapo", "sft", "ftpo", "importance_sampling"]',
    "trainer.algorithm.loss_reduction": 'Literal["token_mean", "sequence_mean", "seq_mean_token_sum_norm", "seq_mean_token_sum_norm_global"]',
    "trainer.algorithm.kl_estimator_type": 'Literal["k1", "abs", "k2", "k3", "k3_unbiased_gradient"]',
    "trainer.algorithm.kl_ctrl.type": 'Literal["fixed", "adaptive"]',
    "trainer.algorithm.off_policy_correction": 'Literal["none", "custom", "icepop", "outlier_mask", "seq_mask_tis", "tis"] | None',
    "trainer.algorithm.off_policy_correction_rules": "tuple[CorrectionRule, ...] | None",
    "trainer.algorithm.dynamic_sampling.type": 'Literal["filter"] | None',
    "trainer.algorithm.dynamic_sampling.informative_on": 'Literal["shaped", "unshaped"]',
    "trainer.algorithm.distillation": "Distillation | None",
    "trainer.algorithm.distillation.objective": 'Literal["sampled_reverse_kl", "sparse_forward_kl", "sparse_reverse_kl", "sparse_jsd", "student_topk_policy_surrogate"]',
    "trainer.algorithm.distillation.reward_mode": 'Literal["add", "replace"]',
    "trainer.algorithm.distillation.advantage_clip": "int | float | None",
    "trainer.mismatch_probe.rescore_prefix_cache": 'Literal["off", "on", "both"]',
    "trainer.mismatch_probe.extra_trainer_modes": 'tuple[Literal["router_replay", "router_replay_filtered"], ...]',
    "generator.backend": 'Literal["vllm", "sglang"]',
    "generator.weight_sync_backend": 'Literal["nccl", "gloo"]',
    "generator.weight_sync_transport": 'Literal["broadcast", "expert_block"]',
    "generator.override_existing_update_group": 'Literal["auto", "enable", "disable"]',
    "generator.weight_sync_pause.mode": 'Literal["abort", "wait", "keep"]',
    "generator.r3_transport": 'Literal["by_value", "resident", "decentral"]',
    "generator.gdn_backend": 'Literal["torch", "flashqla"]',
    "generator.chat_template.source": 'Literal["name", "file"]',
    "generator.error_handling.default_error_treatment": 'Literal["mask", "zero", "passthrough"]',
    "generator.trajectory_retention.phases": 'tuple[Literal["train", "eval"], ...]',
    # Preserve the numeric forms accepted by Hydra and shipped presets.
    "trainer.algorithm.cispo.cispo_eps_clip_high": "int | float",
    "trainer.policy.model.lora.dropout": "Annotated[int | float, Field(ge=0, le=1)]",
    "trainer.critic.model.lora.dropout": "Annotated[int | float, Field(ge=0, le=1)]",
    **{
        f"trainer.step_phase_budgets.{phase}": "Annotated[int | float, Field(ge=0)]"
        for phase in (
            "group_admission",
            "batch_assembly",
            "training_preparation",
            "advantages",
            "policy_training",
            "group_bookkeeping",
            "weight_sync",
            "step_end_bookkeeping",
            "checkpoint_work",
            "evaluation",
            "unaccounted",
        )
    },
    # Positive counts.
    **dict.fromkeys(
        (
            "trainer.train_batch_size",
            "trainer.policy_mini_batch_size",
            "trainer.critic_mini_batch_size",
            "trainer.micro_train_batch_size_per_gpu",
            "trainer.eval_batch_size",
            "trainer.update_epochs_per_batch",
            "trainer.epochs",
            "generator.n_samples_per_prompt",
            "generator.inference_engine_tensor_parallel_size",
            "generator.inference_engine_pipeline_parallel_size",
            "generator.inference_engine_data_parallel_size",
            "generator.inference_engine_expert_parallel_size",
            "trainer.mismatch_probe.prompts.count",
            "trainer.mismatch_probe.prompts.samples_per_prompt",
        ),
        POSITIVE,
    ),
    # Filled by the launch document when null.
    **dict.fromkeys(
        (
            "trainer.placement.policy_num_nodes",
            "trainer.placement.policy_num_gpus_per_node",
            "trainer.placement.ref_num_nodes",
            "trainer.placement.ref_num_gpus_per_node",
            "generator.num_inference_engines",
        ),
        POSITIVE_OR_NONE,
    ),
    "trainer.run_name": "str | None",
    "trainer.policy.max_consecutive_nonfinite_steps": "int | None",
    "trainer.policy.nccl_buffer_size_bytes": "PositiveInt | None",
    "trainer.ref.nccl_buffer_size_bytes": "PositiveInt | None",
    "data.train_data": "tuple[str, ...] | None",
    "data.val_data": "tuple[str, ...] | None",
    # Null defaults.
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
    "data.sampling.kind": 'Literal["domain-weighted", "naive", "thompson", "learnability", "grade-uniform", "grade-adaptive", "grade-prior"] | None',
    "data.sampling.weighting": 'Literal["pass-variance", "group-informative"]',
    "data.sampling.domain_weights": "NumberMap",
    "data.sampling.seed": "NonNegativeInt | None",
    "environment.skyrl_gym.gsm8k.reward_method": 'Literal["strict", "flexible", "final_line"]',
    "environment.skyrl_gym.llm_as_a_judge.base_url": "str | None",
    "environment.skyrl_gym.nemotron_ultra.grading": 'Literal["verify", "skip"]',
    "environment.skyrl_gym.nemotron_ultra.code_verifier.max_memory_bytes": "PositiveInt | None",
    "environment.skyrl_gym.nemotron_ultra.code_verifier.total_timeout_seconds": "int | float | None",
    **{
        f"environment.skyrl_gym.nemotron_ultra.{judge}.{key}": annotation
        for judge in ("judges.general", "judges.safety", "genrm.judge")
        for key, annotation in (
            ("api_key_env", "str | None"),
            ("reasoning_effort", "str | None"),
            ("response_transport", 'Literal["responses_metadata", "chat_completions"]'),
        )
    },
}

OPEN = frozenset(
    {
        "trainer.policy.optimizer_config.optimizer_kwargs",
        "trainer.critic.optimizer_config.optimizer_kwargs",
        "trainer.policy.megatron_config.model_config_kwargs",
        "trainer.policy.megatron_config.transformer_config_kwargs",
        "trainer.ref.megatron_config.model_config_kwargs",
        "trainer.ref.megatron_config.transformer_config_kwargs",
        "generator.engine_init_kwargs",
        "generator.chat_template_kwargs",
        "trainer.rope_scaling",
        "generator.rope_scaling",
    }
)

# Declared here because code reads them with a default but ppo_base_config.yaml does not list them.
# Ellipsis keeps code-default fields unset in authored documents.
_UNDECLARED = {
    "teachers": ("SectionMap[Teacher]", ...),
    "teacher_routing": ("SectionMap[TeacherRouting]", ...),
    "data.kind": ('Literal["tasks", "parquet"]', "tasks"),  # launcher-only: how the launch host stages data
    "trainer.enable_db_registration": ("bool", True),
    "trainer.hf_hub_repo_id": ("str | None", None),
    "trainer.hf_hub_private": ("bool", False),
    "trainer.hf_hub_revision": ("str", "main"),
    "generator.sampling_params.min_tokens": ("NonNegativeInt", ...),
    "generator.eval_sampling_params.min_tokens": ("NonNegativeInt", ...),
    **{
        f"environment.skyrl_gym.{environment}.verifyit_enabled": ("bool", ...)
        for environment in ("reasoning_gym", "ifeval", "text_to_sql", "text2sql", "lcb", "nemotron_ultra")
    },
    "environment.skyrl_gym.lcb.reward_mode": ('Literal["binary", "fractional"]', ...),
    "environment.skyrl_gym.lcb.sandbox.host": ("str", ...),
    "environment.skyrl_gym.lcb.sandbox.port": ("PositiveInt", ...),
    "environment.skyrl_gym.nemotron_ultra.verifyit_math_total_timeout_seconds": ("int | float", ...),
    "environment.skyrl_gym.nemotron_ultra.verifyit_judge_total_timeout_seconds": ("int | float", ...),
    "environment.skyrl_gym.nemotron_ultra.genrm.verifyit_enabled": ("bool", ...),
    "environment.skyrl_gym.nemotron_ultra.genrm.verifyit_strict_json": ("bool", ...),
    "environment.skyrl_gym.nemotron_ultra.genrm.verifyit_timeout_seconds": ("int | float", ...),
    **{
        f"environment.skyrl_gym.nemotron_ultra.{judge}.strict_completion": ("bool", ...)
        for judge in ("judges.general", "judges.safety", "genrm.judge")
    },
    # Sparse FTPO fields; skyrl_train.config.ftpo supplies execution defaults.
    "trainer.algorithm.ftpo.margin": ("int | float", ...),
    "trainer.algorithm.ftpo.lambda_mse": ("int | float", ...),
    "trainer.algorithm.ftpo.lambda_mse_target": ("int | float", ...),
    "trainer.algorithm.ftpo.tau_mse_target": ("int | float", ...),
    "trainer.algorithm.ftpo.min_p": ("int | float", ...),
    "trainer.algorithm.ftpo.max_chosen_tokens": ("PositiveInt", ...),
    "trainer.algorithm.ftpo.min_decoded_chars": ("NonNegativeInt", ...),
    "trainer.algorithm.ftpo.require_alnum": ("bool", ...),
    "trainer.algorithm.ftpo.rejected_balance_strength": ("int | float", ...),
    "trainer.algorithm.ftpo.chosen_balance_strength": ("int | float", ...),
    "trainer.algorithm.ftpo.early_stopping_chosen_win": ("int | float | None", ...),
    "trainer.algorithm.distillation.routing_plan": ("str", ...),
    "trainer.algorithm.distillation.coefficient": ("int | float", ...),
    "trainer.algorithm.distillation.jsd_beta": ("int | float | None", ...),
    "trainer.algorithm.distillation.entry_clip": ("int | float | None", ...),
    "trainer.algorithm.distillation.residency": ("Residency | None", None),
    "trainer.algorithm.distillation.residency.max_resident": ("PositiveInt", ...),
    "trainer.algorithm.distillation.residency.minimum_residency_seconds": ("int | float", ...),
    "trainer.algorithm.distillation.domain_gradient_balance.target_shares": ("NumberMap", ...),
    "trainer.algorithm.distillation.domain_gradient_balance.gap_scale_alpha": ("int | float", ...),
    "generator.speculative_decoding.method": ('Literal["eagle3"]', ...),
    "generator.speculative_decoding.model.source_uri": ("str", ...),
    "generator.speculative_decoding.model.source_identity": ("str", ...),
    "generator.speculative_decoding.num_speculative_tokens": ("PositiveInt", ...),
    "generator.speculative_decoding.training": ("Training | None", None),
    "generator.speculative_decoding.training.interval_steps": ("PositiveInt", ...),
    "generator.speculative_decoding.training.max_tokens_per_update": ("PositiveInt", ...),
    "generator.speculative_decoding.training.max_window_tokens": ("PositiveInt", ...),
    "generator.speculative_decoding.training.max_tokens_per_micro_batch": ("PositiveInt", ...),
    "generator.speculative_decoding.training.max_sequences_per_prompt_group": ("PositiveInt", ...),
    "generator.speculative_decoding.training.min_train_sequences": ("PositiveInt", ...),
    "generator.speculative_decoding.training.holdout_fraction": ("int | float", ...),
    "generator.speculative_decoding.training.min_holdout_sequences": ("PositiveInt", ...),
    "generator.speculative_decoding.training.epochs_per_update": ("PositiveInt", ...),
    "generator.speculative_decoding.training.learning_rate": ("int | float", ...),
    "generator.speculative_decoding.training.max_validation_loss_increase": ("int | float", ...),
    "generator.speculative_decoding.training.max_validation_agreement_decrease": ("int | float", ...),
    "generator.speculative_decoding.training.reserved_gpu_memory_gib": ("int | float", ...),
}

_NAMES = {
    "trainer.policy.megatron_config": "PolicyMegatronConfig",
    "trainer.ref.megatron_config": "RefMegatronConfig",
    "trainer.policy.model": "PolicyModel",
    "trainer.ref.model": "RefModel",
    "trainer.critic.model": "CriticModel",
    "trainer.policy.optimizer_config": "PolicyOptimizerConfig",
    "trainer.critic.optimizer_config": "CriticOptimizerConfig",
    "generator.sampling_params": "SamplingParams",
    "generator.eval_sampling_params": "EvalSamplingParams",
    "trainer.policy.model.lora": "PolicyLora",
    "trainer.critic.model.lora": "CriticLora",
}

# CLASSES fields pair an annotation with whether the field is required.
_CLASSES = {
    "TeacherModel": {"path": ("str", True), "revision": ("str", True)},
    "TeacherEndpoint": {
        "url": ("str", True),
        "auth": ("str | None", False),
        "max_concurrency": ("PositiveInt", True),
    },
    "TeacherResources": {
        "num_nodes": ("PositiveInt", True),
        "gpus_per_node": ("PositiveInt", True),
        "tensor_parallel_size": ("PositiveInt", True),
        "colocation_group": ("str", True),
        "max_num_batched_tokens": ("PositiveInt | None", False),
        "gpu_memory_utilization": ("int | float | None", False),
        "data_parallel_size": ("PositiveInt | None", False),
        "expert_parallel_size": ("PositiveInt | None", False),
    },
    "Teacher": {
        "source": ('Literal["openai_compatible", "local_inference", "frozen_worker", "resident"]', True),
        "placement": ('Literal["external", "pinned", "rotating", "co_resident"] | None', False),
        "model": ("TeacherModel", True),
        "evidence": ('Literal["chosen_token", "topk_distribution", "student_selected_topk"]', True),
        "top_k": ("PositiveInt | None", False),
        "endpoints": ("tuple[TeacherEndpoint, ...]", False),
        "backend": ('Literal["vllm"] | None', False),
        "resources": ("TeacherResources | None", False),
        "tokenizer_fingerprint": ("str | None", False),
        "max_sequence_length": ("PositiveInt | None", False),
        "request_timeout_seconds": ("int | float | None", False),
    },
    "TeacherRoute": {"teacher": ("str", True), "weight": ("int | float", True)},
    "TeacherRouting": {"revision": ("str", True), "routes": ("SectionMap[TeacherRoute]", True)},
    "TokenRule": {
        "kind": ('Literal["token"]', True),
        "action": ('Literal["truncate", "mask"]', True),
        "low": ("int | float | None", False),
        "high": ("int | float | None", False),
    },
    "SequenceRule": {
        "kind": ('Literal["sequence"]', True),
        "aggregate": ('Literal["geometric", "product", "extreme_token"]', True),
        "action": ('Literal["truncate", "mask"]', True),
        "low": ("int | float | None", False),
        "high": ("int | float | None", False),
    },
    "EvaluationSampling": {
        "sampling_params": ("EvalSamplingParams | None", False),
        "n_samples_per_prompt": ("PositiveInt | None", False),
    },
    "EvaluationMinimum": {"minimum": ("int | float", True), "min_improvement": ("None", False)},
    "EvaluationImprovement": {"min_improvement": ("int | float", True), "minimum": ("None", False)},
    "CheckpointCallback": {
        "type": ('Literal["checkpoint"]', True),
        "save_steps": ("int", False),
        "save_on_train_end": ("bool", False),
    },
    "DistillationTokenBudgetCallback": {
        "type": ('Literal["distillation_token_budget"]', True),
        "token_budget": ("PositiveInt", True),
    },
    "EvaluationCallback": {
        "type": ('Literal["evaluation"]', True),
        "eval_steps": ("int", False),
        "eval_on_train_end": ("bool", False),
        "eval_before_train": ("bool", False),
        "additional_evaluations": ("SectionMap[EvaluationSampling] | None", False),
        "metric_groups": (
            "Annotated[Mapping[str, tuple[str, ...]], AfterValidator(FrozenMap), PlainSerializer(thaw)] | None",
            False,
        ),
        "stop_when": ("SectionMap[EvaluationMinimum | EvaluationImprovement] | None", False),
    },
    "HFModelSaveCallback": {
        "type": ('Literal["hf_model_save"]', True),
        "save_steps": ("int", False),
        "save_on_train_end": ("bool", False),
    },
    "DatabaseRegistrationCallback": {
        "type": ('Literal["database_registration"]', True),
        "agent_name": ("str | None", False),
        "enabled": ("bool", False),
    },
    "RefModelUpdateCallback": {
        "type": ('Literal["ref_model_update"]', True),
        "update_every_epoch": ("bool", False),
    },
    "ProgressCallback": {"type": ('Literal["progress"]', True), "log_interval": ("int", False)},
    "LoggingCallback": {"type": ('Literal["logging"]', True), "log_every_step": ("bool", False)},
    "PreflightGateCallback": {
        "type": ('Literal["preflight_gate"]', True),
        "enabled": ("bool", False),
        "min_reward": ("int | float", False),
        "max_reward": ("int | float", False),
        "on_failure": ('Literal["abort", "warn"]', False),
        "num_trials": ("int", False),
    },
    "InferenceStatsCallback": {
        "type": ('Literal["inference_stats"]', True),
        "log_every_steps": ("int", False),
        "log_to_console": ("bool", False),
        "log_to_tracker": ("bool", False),
        "console_log_level": ("str", False),
        "poll_interval_seconds": ("int | float", False),
    },
}

_ALIASES = {
    "LoggingBackend": 'Literal["wandb", "mlflow", "swanlab", "tensorboard", "console"]',
    "CorrectionRule": 'Annotated[TokenRule | SequenceRule, Field(discriminator="kind")]',
    "Callback": 'Annotated[CheckpointCallback | DistillationTokenBudgetCallback | EvaluationCallback | HFModelSaveCallback | DatabaseRegistrationCallback | RefModelUpdateCallback | ProgressCallback | LoggingCallback | PreflightGateCallback | InferenceStatsCallback, Field(discriminator="type")]',
}

_TYPES.update(
    {
        "trainer.collective_phase_diagnostics": "bool | None",
        "generator.error_handling.passthrough_exceptions": "tuple[str, ...]",
        "generator.error_handling.mask_exceptions": "tuple[str, ...]",
        "generator.error_handling.zero_exceptions": "tuple[str, ...]",
        "generator.trajectory_retention.redact_fields": "tuple[str, ...]",
        "data.terminal_bench_data": "tuple[str, ...]",
        "trainer.trajectory_selector.type": 'Literal["best_of_n"] | None',
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
        "terminal_bench.prm.name": "str | None",
        "terminal_bench.agent_api_base": "str | None",
        "terminal_bench.literal_log_path": "str | None",
        "trainer.algorithm.group_advantage_min_size": "PositiveInt | None",
        "trainer.algorithm.resolved_group_advantage.minimum_group_size": "PositiveInt | None",
        "trainer.algorithm.tito_full": "bool | None",
        "trainer.algorithm.dynamic_sampling.max_mean_reward": "int | float | None",
        "trainer.algorithm.group_admission.stall_timeout": "int | float | None",
        "trainer.distillation_token_budget": "PositiveInt | None",
        "trainer.rope_theta": "int | float | None",
        "generator.trajectory_retention.reward_below": "int | float | None",
        "generator.trajectory_retention.reward_above": "int | float | None",
        "checkpoint_export.step": "NonNegativeInt | None",
        "checkpoint_export.checkpoint_path": "str | None",
        "checkpoint_export.export_root": "str | None",
        "checkpoint_export.hf_hub_repo_id": "str | None",
    }
)

# Explicit third-party OpenMap fields are intentional; generated scalar fields have concrete types.
ANY_ALLOWED = frozenset()

TYPES = MappingProxyType(_TYPES)
UNDECLARED = MappingProxyType(_UNDECLARED)
NAMES = MappingProxyType(_NAMES)
CLASSES = MappingProxyType(_CLASSES)
ALIASES = MappingProxyType(_ALIASES)
