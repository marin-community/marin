"""Declared Harbor deadlines survive the complete launch-time worker binding."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.trial.config import AgentConfig
from omegaconf import OmegaConf
from rolloutengine.models import TaskRolloutSpec
from shellbox.backends.daytona.connection import DaytonaConnection

from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker

FIXTURE = Path(__file__).parent / "fixtures" / "swe_rebench_c55add0e_rollout.json"


def real_spec(**limit_updates) -> TaskRolloutSpec:
    spec = TaskRolloutSpec.model_validate_json(FIXTURE.read_text())
    return spec.model_copy(update={"limits": spec.limits.model_copy(update=limit_updates)})


def harbor_settings() -> HarborTaskSettings:
    return HarborTaskSettings.from_config(
        OmegaConf.create({"harbor": {"override_timeout_sec": 1800, "max_timeout_sec": 1800}})
    )


def test_unset_actor_limit_uses_the_declared_deadline():
    settings = replace(harbor_settings(), agent_timeout=None, max_agent_timeout=None)
    spec = real_spec(total_turn_timeout=None)
    configured = settings.configured_spec(spec, phase="train")
    assert configured.limits.total_turn_timeout == spec.task.execution_requirements.agent_timeout_sec


def test_configured_actor_limit_keeps_the_stricter_override():
    configured = harbor_settings().configured_spec(real_spec(), phase="train")
    assert configured.limits.total_turn_timeout == 1800


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
@pytest.mark.parametrize("actor_timeout", [1800, 4000])
async def test_real_harbor_trial_preparation_keeps_and_checks_the_actor_limit(tmp_path, phase, actor_timeout):
    async def unexpected_request(request):
        pytest.fail("Preparation must not call inference")

    tokenizer = SimpleNamespace(eos_token_id=0)
    config = OmegaConf.create({})
    runtime = InstalledAgentRuntime(
        InstalledAgentSettings(
            agent=AgentConfig(name="pi"), output_path=str(tmp_path),
            setup_timeout=5, run_timeout=1800, cleanup_timeout=5,
            service_startup_timeout=5, service_shutdown_timeout=5,
        ),
        AgentInferenceBackend("fixture", 131072, unexpected_request, tokenizer),
        98304,
        daytona_connection=DaytonaConnection(api_key="unused-test-key"),
        eval_daytona_connection=DaytonaConnection(api_key="unused-test-key"),
    )
    settings = replace(harbor_settings(), agent_timeout=None, max_agent_timeout=None, eval_timeout=None)
    worker = TaskRolloutWorker(
        config, WholeTaskProjection(config, tokenizer), None, {}, harbor=settings,
        max_verifier_workers=0, inference_queue_timeout=2280, shutdown_timeout=5, installed_agents=runtime,
    )
    spec = real_spec(total_turn_timeout=actor_timeout)
    try:
        if actor_timeout > spec.task.execution_requirements.agent_timeout_sec:
            with pytest.raises(ValueError, match="actor limit must honor"):
                worker.prepare_task(spec, phase=phase, session_id="deadline")
            return
        prepared = worker.prepare_task(spec, phase=phase, session_id="deadline")
        assert prepared.spec.limits.total_turn_timeout == actor_timeout
        assert prepared.spec.task.execution_requirements == spec.task.execution_requirements
        assert prepared.spec.limits.attempt_timeout == spec.limits.attempt_timeout
        assert prepared.spec.limits.verifier_timeout == spec.limits.verifier_timeout
        assert prepared.spec.limits.cleanup_timeout == spec.limits.cleanup_timeout
    finally:
        await worker.shutdown()
