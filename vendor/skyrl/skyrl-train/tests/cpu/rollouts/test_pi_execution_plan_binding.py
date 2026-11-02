"""The shared installed-agent service sends reviewed trial plans to Harbor Pi."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings


class Decoder:
    def decode(self, ids):
        return str(ids[0])


async def tokenize(_request):
    return {"tokens": [1]}


@asynccontextmanager
async def local_endpoint(port, *, request_timeout):
    async def api_base(*, required_lifetime):
        return f"http://127.0.0.1:{port}/v1"

    yield SimpleNamespace(api_base=api_base)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("control", "value", "expected_mode"),
    [
        (
            "turn_mixture_schedule",
            {
                "teacher_execution_probability": {
                    "scheduler": "linear",
                    "start": 1.0,
                    "end": 0.6,
                    "duration_iterations": 2,
                }
            },
            "dagger",
        ),
        ("critique_principle", "Review the completed attempt", "critique_revision"),
    ],
)
async def test_trial_binds_independent_teacher_plan_before_harbor_starts(tmp_path, control, value, expected_mode):
    settings = InstalledAgentSettings.model_validate(
        {
            "agent": {"name": "pi", "model_name": "hosted_vllm/student", "pi": {"mode": "literal"}},
            "teacher": {
                "source": "skyrl_managed",
                "model": "teacher",
                "model_path": "teacher-path",
                "revision": "a" * 40,
                "student_model": "student-path",
                "max_context_tokens": 4096,
                "resources": {"num_engines": 1, "tensor_parallel_size": 1},
            },
            control: value,
            "output_path": str(tmp_path / "attempts"),
            "setup_timeout": 2,
            "run_timeout": 2,
            "cleanup_timeout": 2,
            "service_startup_timeout": 5,
            "service_shutdown_timeout": 5,
        }
    )
    backend = AgentInferenceBackend(
        "student", 4096, tokenize, Decoder(), teacher_model_name="teacher", teacher_decoder=Decoder()
    )
    runtime = InstalledAgentRuntime(settings, backend, 32, endpoint_factory=local_endpoint, max_turns=4)
    await runtime.startup()
    try:
        async with runtime._served_agent(
            tmp_path, 2, capability_lifetime=10, trial_id="physical-0", policy_step=3
        ) as bound:
            assert bound.pi.mode == expected_mode
            assert bound.pi.trial_id == "physical-0" and bound.pi.policy_step == 3
            assert bound.model_name == "hosted_vllm/student"
            if expected_mode == "dagger":
                assert bound.pi.plan.protocol == "turn_mixture"
                assert bound.pi.plan.teacher_model == "teacher"
                assert len(bound.pi.plan.executors) == 4
                assert bound.pi.plan.executors == settings.turn_mixture_schedule.executors(
                    trial_id="physical-0", policy_step=3, max_turns=4
                )
                assert bound.pi.plan.teacher_api_base == bound.kwargs["api_base"]
            else:
                assert bound.pi.critique.teacher_model == "teacher"
                assert bound.pi.critique.trial_id == "physical-0"
                assert bound.pi.critique.teacher_api_base == bound.kwargs["api_base"]
                assert bound.pi.critique.principle == value
    finally:
        await runtime.shutdown()
