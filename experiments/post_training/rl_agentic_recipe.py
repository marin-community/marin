# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared settings for RL batches that include Pi and one-turn tasks."""

from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from skyrlrecipe import RecipePatch, sections
from skyrlrecipe.declared import DEFAULT_RETRY_EXCLUSIONS

from experiments.post_training.rl_presets import BatchMix, Layout, Preset, harbor_evaluation

_HARBOR_EVALUATION = harbor_evaluation(Path(__file__).with_name("rl_harbor.yaml"))
_BATCH_MIX = BatchMix({"agentic": Fraction(1, 2), "nonagentic": Fraction(1, 2)})

_AGENTIC_RECIPE = RecipePatch(
    environment=sections.Environment(
        task_sessions={
            "daytona": {"api_key_env": "DAYTONA_RL_API_KEY", "eval_api_key_env": "DAYTONA_EVAL_API_KEY"},
            "task_spec": {"harness_by_dataset": {"long-loop-smartie-swesmith": "harbor_trial"}},
        }
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
            "exclude_exceptions": sorted(
                DEFAULT_RETRY_EXCLUSIONS | {"AddTestsDirError", "DownloadVerifierDirError", "VerifierRuntimeError"}
            ),
        },
        "installed_agent": {
            "agent": {
                "name": "pi",
                "override_setup_timeout_sec": 360,
                "kwargs": {
                    "version": "0.87.0",
                    "thinking": "high",
                    "thinking_format": "chat-template",
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
    *,
    attempt_timeout: int,
    model_turn_timeout: int,
    ttl_minutes: int,
    stall_timeout: int,
    session_attempt_timeout: int | None = None,
) -> RecipePatch:
    return RecipePatch(
        environment=sections.Environment(
            task_sessions={
                "daytona": {"ttl_minutes": ttl_minutes},
                "task_spec": {
                    "session": {
                        "attempt_timeout": (
                            attempt_timeout if session_attempt_timeout is None else session_attempt_timeout
                        ),
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


def agentic(preset: Preset, *, name: str, limits: RecipePatch, layout: Layout | None = None) -> Preset:
    """Apply the mixed workload to a model's training preset."""
    config = preset.recipe.config.merge(_AGENTIC_RECIPE).merge(limits)
    budget = config.context_budget
    assert budget is not None
    return replace(
        preset,
        name=name,
        tasks_input="required",
        batch_mix=_BATCH_MIX,
        harbor=_HARBOR_EVALUATION,
        recipe=replace(
            preset.recipe,
            layout=preset.recipe.layout if layout is None else layout,
            config=config.replace(context_budget=budget.replace(max_turns=128)),
        ),
    )
