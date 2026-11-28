"""Launch-time harbor lowering honors a task's declared deadline (the .722 fix).

The .722 launch of v3-anchor-r41-compact died in prepare_task: the lowered row's
total_turn_timeout was None while the SWE-rebench task declared agent_timeout_sec 3000.
The fixture is the literal TaskRolloutSpec of one c55add0e pool row.
"""

import json
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from rolloutengine.models import RolloutLimits, TaskRolloutSpec

from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings

FIXTURE = Path(__file__).parent / "fixtures" / "swe_rebench_c55add0e_rollout.json"
COMPILED = Path(
    "/Users/benfeuer/Documents/experiments/active/rl-data-campaign/"
    "validation/v3-anchor-r41-compact/2026.11.27.722/native-controls/compiled.yaml"
)


def real_spec(**limit_updates) -> TaskRolloutSpec:
    spec = TaskRolloutSpec.model_validate_json(FIXTURE.read_text())
    if limit_updates:
        spec = spec.model_copy(update={"limits": spec.limits.model_copy(update=limit_updates)})
    return spec


def harbor_settings() -> HarborTaskSettings:
    config = yaml.safe_load(COMPILED.read_text())
    return HarborTaskSettings.from_config(OmegaConf.create(config["skyrl"]["terminal_bench_config"]))


def test_unset_total_turn_timeout_without_override_is_derived_from_the_declared_deadline():
    """A dropped total_turn_timeout is restored from the task's declared agent deadline.

    The .722 in-region lowering produced total_turn_timeout None for a task that declares
    agent_timeout_sec 3000; the deadline fallback derives it from the declaration instead.
    """
    settings = harbor_settings()
    settings = settings.model_copy() if hasattr(settings, "model_copy") else settings
    # Simulate the in-region path with no harbor override configured.
    without_override = HarborTaskSettings(
        environment=settings.environment,
        verifier=settings.verifier,
        agent_timeout=None,
        max_agent_timeout=None,
        max_turns=settings.max_turns,
        timeout_multiplier=settings.timeout_multiplier,
        eval_timeout=settings.eval_timeout,
        concurrent_trials=settings.concurrent_trials,
        error_handling=settings.error_handling,
        reward_shaping=settings.reward_shaping,
        retry=settings.retry,
        attempt_timeout=settings.attempt_timeout,
        declared_error_handling_fields=settings.declared_error_handling_fields,
    )
    spec = real_spec(total_turn_timeout=None)
    configured = without_override.configured_spec(spec, phase="train")
    deadline = spec.task.execution_requirements.agent_timeout_sec
    assert configured.limits.total_turn_timeout == deadline
    # The validation rule (total <= declared agent deadline) is satisfiable by construction.
    assert configured.limits.total_turn_timeout <= deadline


def test_override_total_turn_timeout_is_kept_and_honors_the_declared_deadline():
    """A configured override stays in force when it honors the task's declared deadline."""
    settings = harbor_settings()
    spec = real_spec()
    configured = settings.configured_spec(spec, phase="train")
    deadline = spec.task.execution_requirements.agent_timeout_sec
    assert configured.limits.total_turn_timeout == 1800.0
    assert configured.limits.total_turn_timeout <= deadline


def test_the_fix_does_not_touch_tasks_without_declared_requirements():
    settings = harbor_settings()
    spec = real_spec(total_turn_timeout=None)
    bare = spec.model_copy(
        update={"task": spec.task.model_copy(update={"execution_requirements": None})}
    )
    configured = settings.configured_spec(bare, phase="train")
    assert configured.limits.total_turn_timeout == 1800.0
