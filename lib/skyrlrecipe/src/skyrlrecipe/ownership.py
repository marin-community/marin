# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recipe keys supplied by context derivation or launch materialization."""

from types import MappingProxyType

from .budget import ContextBudget

DERIVED_PATHS = frozenset(
    ContextBudget(request_window_tokens=2, max_new_tokens_per_turn=1, max_turns=1).derived_values()
)

LAUNCH_PATHS = frozenset(
    {
        "trainer.ckpt_path",
        "trainer.export_path",
        "trainer.grader_report_dir",
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
    }
)

OWNER_MESSAGES = MappingProxyType(
    {
        **dict.fromkeys(DERIVED_PATHS, "derived from context_budget; set context_budget instead"),
        **dict.fromkeys(LAUNCH_PATHS, "the launch document sets it; it is not a recipe setting"),
        "checkpoint_export": "the launch document owns export settings; it is not a recipe section",
    }
)

REMOVED = MappingProxyType(
    {
        "generator.require_exact_chat_transport": "task sessions use the Rollout Engine transport",
        "generator.use_conversation_multi_turn": "use environment.task_sessions",
        "generator.append_eos_token_after_stop_str_in_multi_turn": "use environment.task_sessions",
        "trainer.algorithm.tito_full": "configure trainer.algorithm.off_policy_correction",
        "environment.skyrl_gym": "use environment.task_sessions",
        "trainer.fully_async": "use trainer.rollout_buffer (MarinSkyRL #774)",
        "trainer.async_spans": "use trainer.rollout_spans (MarinSkyRL #774)",
        "trainer.generate_spans": "use trainer.rollout_spans (MarinSkyRL #774)",
        "generator.async_engine": "every engine is asynchronous (MarinSkyRL #774)",
        "generator.batched": "configure generator sampling and batch limits (MarinSkyRL #774)",
        "trainer.algorithm.use_tis": "configure trainer.algorithm.off_policy_correction (MarinSkyRL #858)",
        "trainer.algorithm.rloo_n_filter_zero_reward_groups": "flat groups always give zero advantage",
        "trainer.algorithm.think_token_weight": "all eligible response tokens have equal policy weight",
        "trainer.policy.fsdp_config": "use trainer.policy.megatron_config (MarinSkyRL #776)",
    }
)
