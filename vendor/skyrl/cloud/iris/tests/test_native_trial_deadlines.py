"""Harbor-owned phase deadlines must outlive the outer rollout turn."""

import asyncio

import pytest
from omegaconf import OmegaConf
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.models import Prompt, RolloutLimits
from rolloutengine.test_support import ReplayModel
from taskcompendium.models import TaskSpec

from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from tests.cpu.task_specs import lowered_task, text_task


@pytest.mark.asyncio
async def test_native_trial_verifier_finishes_after_agent_deadline(monkeypatch):
    verifier_finished = asyncio.Event()
    loop = asyncio.get_running_loop()
    clock = [loop.time()]
    monkeypatch.setattr(loop, "time", lambda: clock[0])

    class NativeTrialHarness:
        def __init__(self, agent_timeout: float):
            self.agent_timeout = agent_timeout

        def check(self, _task: TaskSpec, _limits: RolloutLimits) -> None:
            pass

        async def run(self, _context: HarnessContext, session: ModelSession) -> None:
            async with asyncio.timeout(self.agent_timeout):
                await session.start(Prompt(({"role": "user", "content": "Answer."},), {}))
            clock[0] += self.agent_timeout * 2
            deadline_crossed = asyncio.Event()
            loop.call_soon(deadline_crossed.set)
            await deadline_crossed.wait()
            verifier_finished.set()

    class InstalledAgents:
        async def startup(self) -> None:
            pass

        async def shutdown(self) -> None:
            pass

        def original_harness(self, lowered, _session_id, _settings, _phase):
            assert lowered.limits.total_turn_timeout is not None
            return NativeTrialHarness(lowered.limits.total_turn_timeout)

        async def run(self, _machine, _complete, *, lowered, session_id) -> None:
            raise AssertionError("The full Harbor trial path must not use the direct agent runner")

    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "eval_timeout_override_sec": 0.05,
                    "verifier_override_timeout_sec": 0.2,
                    "trial_attempt_timeout_sec": 1,
                }
            }
        )
    )
    spec = settings.configured_spec(lowered_task(text_task(), "harbor_trial"), phase="eval")
    config = OmegaConf.create({"sampling_params": {}, "apply_overlong_filtering": False})
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, None),
        None,
        {},
        shutdown_timeout=1,
        installed_agents=InstalledAgents(),
    )
    try:
        outer, harnesses = worker._configured_task_and_harnesses(spec, settings, "trial", "eval")
        model = ReplayModel([{"role": "assistant", "content": "Done."}])
        await RolloutEngine({}, harnesses).run_task(outer, model.complete)
    finally:
        await worker.shutdown()

    assert verifier_finished.is_set()
