"""Recipe keys supplied by context derivation or launch materialization."""

from types import MappingProxyType


DERIVED_PATHS = frozenset(
    {
        "trainer.max_prompt_length",
        "generator.max_input_length",
        "generator.max_turns",
        "generator.sampling_params.max_generate_length",
        "generator.engine_init_kwargs.max_model_len",
        "generator.trajectory_reward_shaping.overlong.l_max",
        "generator.trajectory_reward_shaping.overlong.l_cache",
        "terminal_bench.harbor.max_turns",
        "terminal_bench.harbor.llm_call_kwargs.max_tokens",
        "terminal_bench.model_info.max_input_tokens",
        "terminal_bench.model_info.max_output_tokens",
    }
)

LAUNCH_PATHS = frozenset(
    {
        "trainer.ckpt_path",
        "trainer.export_path",
        "trainer.seed",
        "trainer.max_ckpts_to_keep",
        "trainer.export_hf_artifact",
        "trainer.policy.model.path",
        "trainer.policy.model.revision",
        "trainer.policy.model.source_uri",
        "trainer.policy.model.source_identity",
        "trainer.policy.model.tokenizer_path",
        "trainer.policy.model.tokenizer_revision",
        "trainer.ref.model.path",
        "trainer.ref.model.tokenizer_path",
        "trainer.ref.model.tokenizer_revision",
        "generator.engine_init_kwargs.served_model_name",
        "generator.trajectory_retention.output_path",
        "terminal_bench.trials_dir",
        "terminal_bench.agent_api_base",
        "terminal_bench.literal_log_path",
    }
)

OWNER_MESSAGES = MappingProxyType(
    {
        **dict.fromkeys(DERIVED_PATHS, "derived from context_budget; set context_budget instead"),
        **dict.fromkeys(LAUNCH_PATHS, "the launch document sets it; it is not a recipe setting"),
    }
)

REMOVED = MappingProxyType(
    {
        "trainer.fully_async": "use trainer.rollout_buffer (MarinSkyRL #774)",
        "trainer.async_spans": "use trainer.rollout_spans (MarinSkyRL #774)",
        "trainer.generate_spans": "use trainer.rollout_spans (MarinSkyRL #774)",
        "generator.async_engine": "every engine is asynchronous (MarinSkyRL #774)",
        "generator.batched": "configure generator sampling and batch limits (MarinSkyRL #774)",
        "trainer.algorithm.use_tis": "configure trainer.algorithm.off_policy_correction (MarinSkyRL #858)",
        "trainer.policy.fsdp_config": "use trainer.policy.megatron_config (MarinSkyRL #776)",
        "terminal_bench.harbor.max_episodes": "use context_budget.max_turns; Harbor reads max_turns",
    }
)
