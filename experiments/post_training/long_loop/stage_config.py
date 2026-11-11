# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require the run spec to supply execution semantics before SkyRL applies defaults."""


REQUIRED_FIELDS = (
    "entrypoint",
    "context_budget.request_window_tokens",
    "context_budget.max_new_tokens_per_turn",
    "context_budget.max_turns",
    "trainer.eval_interval",
    "trainer.eval_batch_size",
    "trainer.train_batch_size",
    "trainer.policy_mini_batch_size",
    "trainer.micro_train_batch_size_per_gpu",
    "trainer.seed",
    "trainer.resume_mode",
    "generator.backend",
    "generator.model_dtype",
    "generator.max_num_seqs",
    "generator.gpu_memory_utilization",
    "generator.enforce_eager",
    "generator.engine_init_timeout_seconds",
    "generator.sampling_params.temperature",
    "generator.sampling_params.top_p",
    "generator.sampling_params.stop_token_ids",
    "generator.sampling_params.skip_special_tokens",
    "generator.eval_sampling_params.temperature",
    "generator.eval_sampling_params.top_p",
    "generator.eval_sampling_params.stop_token_ids",
    "generator.eval_sampling_params.skip_special_tokens",
    "generator.n_samples_per_prompt",
    "terminal_bench.harbor.override_timeout_sec",
    "terminal_bench.harbor.eval_timeout_override_sec",
    "terminal_bench.harbor.trial_attempt_timeout_sec",
    "terminal_bench.harbor.max_retries",
    "terminal_bench.harbor.n_concurrent_trials",
    "terminal_bench.installed_agent.agent.name",
    "terminal_bench.installed_agent.agent.override_setup_timeout_sec",
    "terminal_bench.installed_agent.agent.kwargs.version",
    "terminal_bench.installed_agent.agent.kwargs.mcp_adapter_version",
    "terminal_bench.installed_agent.agent.kwargs.thinking",
    "terminal_bench.installed_agent.agent.kwargs.model_info.max_input_tokens",
    "terminal_bench.installed_agent.agent.kwargs.model_info.max_output_tokens",
    "terminal_bench.installed_agent.agent.kwargs.compaction_reserve_tokens",
    "terminal_bench.installed_agent.setup_timeout",
    "terminal_bench.installed_agent.run_timeout",
    "terminal_bench.installed_agent.cleanup_timeout",
    "terminal_bench.installed_agent.service_startup_timeout",
    "terminal_bench.installed_agent.service_shutdown_timeout",
)

RL_REQUIRED_FIELDS = (
    "trainer.max_steps",
    "trainer.epochs",
    "trainer.update_epochs_per_batch",
    "trainer.ckpt_interval",
    "trainer.max_ckpts_to_keep",
    "trainer.hf_export_checkpoint",
    "trainer.callbacks",
    "trainer.algorithm.advantage_estimator",
    "trainer.algorithm.loss_reduction",
    "trainer.algorithm.off_policy_correction",
    "trainer.algorithm.tis_splice",
    "trainer.policy.optimizer_config.lr",
    "trainer.policy.optimizer_config.scheduler",
    "trainer.policy.optimizer_config.weight_decay",
    "trainer.policy.optimizer_config.max_grad_norm",
    "trainer.policy.megatron_config.context_parallel_size",
    "trainer.policy.megatron_config.moe_router_replay",
    "trainer.policy.megatron_config.transformer_config_kwargs.cp_comm_type",
    "trainer.rollout_buffer.max_staleness_steps",
    "trainer.rollout_buffer.max_in_flight",
    "trainer.rollout_buffer.consumed_route_quotas",
    "trainer.rollout_buffer.source_routes",
    "environment.task_sessions.task_spec.session.max_turns",
    "environment.task_sessions.task_spec.session.attempt_timeout",
    "environment.task_sessions.task_spec.session.total_turn_timeout",
    "environment.task_sessions.task_spec.session.model_turn_timeout",
)


def config_value(config: dict, path: str):
    value = config
    for name in path.split("."):
        value = value[name]
    return value


def validate_stage_config(config: dict, stage: str) -> None:
    """Reject omitted load-bearing settings without filling them in."""
    fields = REQUIRED_FIELDS + (RL_REQUIRED_FIELDS if stage == "rl" else ())
    for path in fields:
        try:
            config_value(config, path)
        except KeyError as error:
            raise ValueError(f"{stage} run spec must explicitly set {path}") from error
    budget = config["context_budget"]
    agent = config["terminal_bench"]["installed_agent"]["agent"]["kwargs"]["model_info"]
    if agent["max_input_tokens"] != budget["request_window_tokens"]:
        raise ValueError(f"{stage} Pi input window differs from context_budget")
    if agent["max_output_tokens"] != budget["max_new_tokens_per_turn"]:
        raise ValueError(f"{stage} Pi output allowance differs from context_budget")
