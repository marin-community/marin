# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared settings for RL batches that include Pi and one-turn tasks."""

from dataclasses import replace
from fractions import Fraction

from skyrlrecipe import RecipePatch, sections
from skyrlrecipe.declared import DEFAULT_RETRY_EXCLUSIONS

from experiments.post_training.rl_presets import HARBOR_EVALUATION, BatchMix, Preset
from experiments.post_training.task_curation.task_health import task_health_step

_BATCH_MIX = BatchMix({"agentic": Fraction(1, 2), "nonagentic": Fraction(1, 2)})
_DAYTONA_RL_KEY = "DAYTONA_RL_API_KEY"
_DAYTONA_EVAL_KEY = "DAYTONA_EVAL_API_KEY"

_AGENTIC_RECIPE = RecipePatch(
    environment=sections.Environment(
        task_sessions={
            "daytona": {"api_key_env": _DAYTONA_RL_KEY, "eval_api_key_env": _DAYTONA_EVAL_KEY},
            "task_spec": {"harness_by_dataset": {"long-loop-smartie-swesmith": "harbor_trial"}},
        }
    ),
    terminal_bench={
        "harbor": {
            "environment_type": "daytona",
            "daytona_api_key_env": _DAYTONA_RL_KEY,
            "eval_daytona_api_key_env": _DAYTONA_EVAL_KEY,
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
            "setup_timeout": 1_800,
            "cleanup_timeout": 120,
            "service_startup_timeout": 60,
            "service_shutdown_timeout": 60,
        },
    },
)


def agentic_limits(*, trial_timeout: int, agent_timeout: int, stall_timeout: int) -> RecipePatch:
    return RecipePatch(
        environment=sections.Environment(
            task_sessions={
                "daytona": {"ttl_minutes": (trial_timeout + 59) // 60},
                "task_spec": {
                    "session": {
                        "attempt_timeout": trial_timeout,
                        "total_turn_timeout": agent_timeout,
                        "model_turn_timeout": agent_timeout,
                    }
                },
            }
        ),
        trainer=sections.Trainer(
            algorithm=sections.Algorithm(group_admission=sections.GroupAdmission(stall_timeout=stall_timeout))
        ),
        terminal_bench={
            "harbor": {
                "trial_attempt_timeout_sec": trial_timeout,
                "override_timeout_sec": agent_timeout,
                "eval_timeout_override_sec": agent_timeout,
            },
            "installed_agent": {
                "agent": {"kwargs": {"http_idle_timeout": agent_timeout}},
                "run_timeout": agent_timeout,
            },
        },
    )


def agentic_preset(preset: Preset, *, name: str, limits: RecipePatch) -> Preset:
    """Apply the task mix to a model's training preset."""
    config = preset.recipe.config.merge(_AGENTIC_RECIPE).merge(limits)
    budget = config.context_budget
    assert budget is not None
    return replace(
        preset,
        name=name,
        tasks_input="required",
        batch_mix=_BATCH_MIX,
        harbor=HARBOR_EVALUATION if preset.evaluations else None,
        then=task_health_step,
        recipe=replace(preset.recipe, config=config.replace(context_budget=budget.replace(max_turns=128))),
    )
