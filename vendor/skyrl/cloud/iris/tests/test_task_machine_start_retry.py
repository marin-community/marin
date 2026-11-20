"""A rollout whose task machine loses its connection while starting restarts with a fresh machine."""

from dataclasses import dataclass

import pytest
from daytona import DaytonaConnectionError
from omegaconf import OmegaConf
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, ExecutionRequirements, MachineSpec
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import MACHINE_START_RETRIES, TaskRolloutWorker
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TokenProvenance
from tests.cpu.task_specs import SHELL_ENVIRONMENT, lowered_task, task_input, text_task


class WorkerTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, **kwargs):
        return "Done"


class ReplyClient:
    async def generate(self, _request):
        return {
            "token_provenance": TokenProvenance.ENGINE,
            "assistant_messages": [{"role": "assistant", "content": "Done"}],
            "prompt_ids": [[1, 2]],
            "response_ids": [[3]],
            "response_logprobs": [[-0.1]],
            "stop_reasons": ["stop"],
        }


@dataclass
class ResettingFactory:
    """Fail the first ``failures`` creations as a reset Daytona connection does."""

    failures: int
    backend: Backend = Backend.SHELLSIM
    creations: int = 0

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        ShellSimMachineFactory().check_environment(requirements, spec)

    async def create(self, spec: MachineSpec):
        self.creations += 1
        if self.creations <= self.failures:
            raise DaytonaConnectionError("Failed to upload file: [Errno 104] Connection reset by peer")
        return await ShellSimMachineFactory().create(spec)


async def rollout(factory: ResettingFactory, *, harbor: HarborTaskSettings | None = None):
    """Run one shell task and return its result with the waits between restarts."""
    waits = []

    async def wait(delay):
        waits.append(delay)

    config = OmegaConf.create(
        {
            "backend": "vllm",
            "max_input_length": 64,
            "max_model_len": 128,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": 0,
            },
        }
    )
    task = text_task().model_copy(
        update={"environment_requirements": SHELL_ENVIRONMENT, "tags": () if harbor is None else ("harbor",)}
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, WorkerTokenizer()),
        ReplyClient(),
        {"shellsim": factory},
        harbor=harbor,
        retry_wait=wait,
        shutdown_timeout=5,
    )
    try:
        (result,) = await worker.generate(
            TaskRequestBatch(rows=(task_input(lowered_task(task, max_turns=1, backend="shellsim")),))
        )
    finally:
        await worker.shutdown()
    return result, waits


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failures", "recovers"),
    [(MACHINE_START_RETRIES, True), (MACHINE_START_RETRIES + 1, False)],
    ids=["recovers", "gives-up"],
)
async def test_lost_connection_at_machine_start_restarts_the_rollout(failures, recovers):
    factory = ResettingFactory(failures)
    result, waits = await rollout(factory)
    assert factory.creations == MACHINE_START_RETRIES + 1
    assert (result.failure is None) == recovers
    # Each restart waits first, because the resets arrive in bursts.
    assert len(waits) == MACHINE_START_RETRIES and all(wait > 0 for wait in waits)
    assert result.metrics["rollout_retries"] == MACHINE_START_RETRIES


@pytest.mark.asyncio
async def test_harbor_task_restarts_only_under_its_own_retry_policy():
    factory = ResettingFactory(failures=1)
    harbor = HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"max_retries": 0}}))
    result, waits = await rollout(factory, harbor=harbor)
    assert factory.creations == 1
    assert waits == []
    assert result.failure.exception_type == "TaskMachineError"
