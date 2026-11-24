# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared settings for RL batches that include agent and shell-tool tasks."""

from dataclasses import replace
from fractions import Fraction

from rigging.task_errors import DEFAULT_RETRY_EXCLUSIONS
from rolloutengine.models import MachineRuntimeSpec
from shellbox.machine import Backend, CommandSemantics, NetworkPolicy
from skyrlrecipe import RecipePatch, sections

from experiments.post_training.rl_presets import HARBOR_EVALUATION, BatchMix, Preset, Recipe
from experiments.post_training.task_curation.task_health import task_health_step

# Gaps that a preset reports until it trains on the named tasks.
NO_AGENTIC_TASKS = "no-agentic-tasks"
NO_IMAGE_SANDBOX = "no-image-sandbox"

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


def _task_recipe(recipe: Recipe, *patches: RecipePatch, max_turns: int) -> Recipe:
    """Merge task settings into a recipe and set the turn limit of its context budget."""
    config = recipe.config
    for patch in patches:
        config = config.merge(patch)
    budget = config.context_budget
    assert budget is not None
    return replace(recipe, config=config.replace(context_budget=budget.replace(max_turns=max_turns)))


def agentic_preset(preset: Preset, *, name: str, limits: RecipePatch) -> Preset:
    """Apply the task mix to a model's training preset."""
    return replace(
        preset,
        name=name,
        tasks_input="required",
        batch_mix=_BATCH_MIX,
        harbor=HARBOR_EVALUATION if preset.evaluations else None,
        then=task_health_step,
        recipe=_task_recipe(preset.recipe, _AGENTIC_RECIPE, limits, max_turns=128),
    )


TOOL_TURNS = 16
TASK_MACHINE_STARTUP_TIMEOUT = 600
TASK_MACHINE_CLEANUP_TIMEOUT = 60
# Each shell-tool task runs on a machine that matches its command semantics.
TASK_MACHINES = {
    # ShellSim has no guest network.
    CommandSemantics.SHELL_SIMULATOR: MachineRuntimeSpec(
        backend=Backend.SHELLSIM,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=TASK_MACHINE_STARTUP_TIMEOUT,
        cleanup_timeout=TASK_MACHINE_CLEANUP_TIMEOUT,
    ),
    CommandSemantics.LINUX_PROCESS: MachineRuntimeSpec(
        backend=Backend.DAYTONA,
        network=NetworkPolicy.ALLOW,
        cpus=1,
        memory_mb=2048,
        storage_mb=10240,
        gpus=0,
        user=None,
        startup_timeout=TASK_MACHINE_STARTUP_TIMEOUT,
        cleanup_timeout=TASK_MACHINE_CLEANUP_TIMEOUT,
    ),
}
_SHELL_TOOL_RECIPE = RecipePatch(
    environment=sections.Environment(
        task_sessions={
            # Linux task machines and image graders are training work, so they use the RL organization.
            "daytona": {"api_key_env": _DAYTONA_RL_KEY, "ttl_minutes": 60},
            "task_spec": {
                "session": {"max_turns": TOOL_TURNS},
                "task_machine_by_command_semantics": {
                    semantics.value: machine.model_dump(mode="json") for semantics, machine in TASK_MACHINES.items()
                },
            },
        }
    ),
)


def shell_tool_preset(preset: Preset, *, name: str) -> Preset:
    """Train a model's preset on a selection of shell-tool tasks."""
    return replace(
        preset,
        name=name,
        tasks_input="required",
        recipe=_task_recipe(preset.recipe, _SHELL_TOOL_RECIPE, max_turns=TOOL_TURNS),
        gaps=tuple(gap for gap in preset.gaps if gap not in (NO_AGENTIC_TASKS, NO_IMAGE_SANDBOX)),
    )
