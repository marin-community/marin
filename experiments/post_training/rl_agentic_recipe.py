# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared settings for RL batches that include Pi and one-turn tasks."""

from fractions import Fraction

from skyrlrecipe import RecipePatch, sections

from experiments.post_training.rl_presets import BatchMix

AGENTIC_BATCH_MIX = BatchMix({"agentic": Fraction(1, 2), "nonagentic": Fraction(1, 2)})

AGENTIC_RECIPE = RecipePatch(
    environment=sections.Environment(
        task_sessions={
            "local": {},
            "daytona": {"api_key_env": "DAYTONA_RL_API_KEY", "eval_api_key_env": "DAYTONA_EVAL_API_KEY"},
            "task_spec": {"harness_by_dataset": {"long-loop-smartie-swesmith": "harbor_trial"}},
        }
    ),
    trainer=sections.Trainer(use_sample_packing=True),
    generator=sections.Generator(
        enable_http_endpoint=True,
        chat_template_kwargs={"enable_thinking": True},
        engine_init_kwargs={"enable_auto_tools": True, "tool_parser": "hermes"},
    ),
    terminal_bench={
        "harbor": {
            "environment_type": "daytona",
            "daytona_api_key_env": "DAYTONA_RL_API_KEY",
            "eval_daytona_api_key_env": "DAYTONA_EVAL_API_KEY",
            "verifier_override_timeout_sec": 900,
            "n_concurrent_trials": 256,
            "max_retries": 1,
            "enable_reward_shaping": False,
            "exclude_exceptions": (
                "AgentTimeoutError",
                "VerifierTimeoutError",
                "RewardFileNotFoundError",
                "RewardFileEmptyError",
                "VerifierOutputParseError",
                "AddTestsDirError",
                "DownloadVerifierDirError",
                "VerifierRuntimeError",
            ),
        },
        "installed_agent": {
            "agent": {
                "name": "pi",
                "override_setup_timeout_sec": 360,
                "kwargs": {
                    "version": "0.87.0",
                    "mcp_adapter_version": "7.0.0",
                    "thinking": "high",
                    "thinking_format": "chat-template",
                    "compaction_reserve_tokens": 16_384,
                },
            },
            "setup_timeout": 360,
            "cleanup_timeout": 120,
            "service_startup_timeout": 60,
            "service_shutdown_timeout": 60,
        },
    },
)


def agentic_limits(
    *, attempt_timeout: int, model_turn_timeout: int, ttl_minutes: int, stall_timeout: int
) -> RecipePatch:
    return RecipePatch(
        environment=sections.Environment(
            task_sessions={
                "daytona": {"ttl_minutes": ttl_minutes},
                "task_spec": {
                    "session": {
                        "attempt_timeout": attempt_timeout,
                        "total_turn_timeout": model_turn_timeout,
                        "model_turn_timeout": model_turn_timeout,
                    }
                },
            }
        ),
        trainer=sections.Trainer(
            algorithm=sections.Algorithm(group_admission=sections.GroupAdmission(stall_timeout=stall_timeout))
        ),
        terminal_bench={
            "harbor": {
                "trial_attempt_timeout_sec": attempt_timeout,
                "override_timeout_sec": model_turn_timeout,
                "eval_timeout_override_sec": model_turn_timeout,
            },
            "installed_agent": {
                "agent": {"kwargs": {"http_idle_timeout": model_turn_timeout}},
                "run_timeout": model_turn_timeout,
            },
        },
    )
