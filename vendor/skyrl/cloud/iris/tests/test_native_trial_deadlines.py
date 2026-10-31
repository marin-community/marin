"""Native Harbor verification must outlive its agent phase's deadline."""

from functools import partial

import pytest
from omegaconf import OmegaConf
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from tests.cpu.task_specs import lowered_task, text_task


@pytest.mark.asyncio
async def test_native_trial_keeps_phase_timeout_without_outer_agent_cancellation():
    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "override_timeout_sec": 1800,
                    "eval_timeout_override_sec": 1800,
                    "verifier_override_timeout_sec": 300,
                    "trial_attempt_timeout_sec": 2460,
                }
            }
        )
    )
    task = text_task().model_copy(update={"tags": ("harbor",)})
    spec = settings.configured_spec(lowered_task(task, "harbor_trial"), phase="eval")
    installed = InstalledAgentRuntime(
        InstalledAgentSettings(
            agent={"name": "pi"},
            output_path="unused",
            setup_timeout=360,
            run_timeout=1800,
            cleanup_timeout=60,
            service_startup_timeout=60,
            service_shutdown_timeout=60,
        ),
        AgentInferenceBackend("fixture", 65536, None, None),
        16384,
    )
    config = OmegaConf.create({"sampling_params": {}, "apply_overlong_filtering": False})
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, None),
        None,
        {},
        shutdown_timeout=30,
        installed_agents=installed,
    )
    try:
        outer, harnesses = worker._configured_task_and_harnesses(spec, settings, "trial", "eval")
    finally:
        await worker.shutdown()
    run = harnesses["harbor_trial"].run_trial
    assert isinstance(run, partial)
    trial_config = run.keywords["config"]
    assert trial_config.agent.override_timeout_sec == 1800
    assert trial_config.verifier.override_timeout_sec == 300
    assert trial_config.trial_attempt_timeout_sec == outer.limits.attempt_timeout == 2460
    assert outer.limits.total_turn_timeout is None
    assert outer.limits.model_turn_timeout is None
    assert outer.limits.tool_turn_timeout is None
