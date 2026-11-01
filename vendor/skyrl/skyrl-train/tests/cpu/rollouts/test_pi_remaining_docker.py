"""Opt-in real Pi/Docker execution with scripted, separately attributed model replies.

The scripted replies test routing, grading and admission, not model quality or a
physical sampled-model training gate. No failed or ungraded attempt is admitted.
"""

import os
import shutil
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.machine import Backend, NetworkPolicy
from skyrl_train.trajectory_runners.types import TokenProvenance
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from transformers import AutoTokenizer
from verifyit.spec import ExactSpec

from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.pi_scenario_etl import TokenSource
from skyrl_train.rollouts.pi_turn_schedule import PiTurnMixtureSchedule
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.types import BatchMetadata, TaskRequestBatch, TrajectoryID
from tests.cpu.task_specs import lowered_task, task_input
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


class ScriptedModelClient:
    """Return exact wire tokens for a bounded routing test, not a policy sample."""

    def __init__(self, tokenizer, actor: str, scenario: str):
        self.tokenizer = tokenizer
        self.actor = actor
        self.scenario = scenario
        self.requests = []
        self.completions = []

    async def generate(self, request):
        self.requests.append(request)
        if self.actor == "teacher":
            if self.scenario == "critique_revision_pi_base":
                answer = "The answer should be 30. Revise your response."
            else:
                answer = "30"
        elif self.scenario == "critique_revision_pi_base":
            answer = "29" if len(self.requests) == 1 else "30"
        else:
            answer = ""
        if self.actor == "student" and self.scenario == "dagger_pi_turn_mixture_base":
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-read",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"cat /workspace/settings.json"}'},
                    }
                ],
            }
            response = "cat /workspace/settings.json"
        else:
            message = {"role": "assistant", "content": answer}
            response = answer
        ids = self.tokenizer.encode(response, add_special_tokens=False)
        self.completions.append(tuple(ids))
        return {
            "assistant_messages": [message],
            "responses": [response],
            "prompt_ids": [[1, 2, 3]],
            "response_ids": [ids],
            "response_logprobs": [[-0.2] * len(ids)],
            "stop_reasons": ["tool_calls" if message.get("tool_calls") else "stop"],
            "token_provenance": TokenProvenance.ENGINE,
        }


@asynccontextmanager
async def local_endpoint(port, *, request_timeout):
    async def api_base(*, required_lifetime):
        return f"http://{socket.gethostbyname(socket.gethostname())}:{port}/v1"

    yield SimpleNamespace(api_base=api_base)


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize("preset", ["dagger_pi_turn_mixture_base", "critique_revision_pi_base"])
async def test_pi_remaining_scenarios_grade_scripted_physical_cpu_trial(tmp_path, preset):
    if os.environ.get("RUN_HARBOR_PI_REMAINING_DOCKER") != "1":
        pytest.skip("set RUN_HARBOR_PI_REMAINING_DOCKER=1 for the physical Pi/Docker routing test")
    if shutil.which("docker") is None:
        pytest.skip("Docker is required")
    dev_image = os.environ["PI_RL_DEV_IMAGE"]
    student = build_tiny_policy(tmp_path / "student", max_position_embeddings=8192)
    teacher = build_tiny_policy(tmp_path / "teacher", max_position_embeddings=8192)
    tokenizer = AutoTokenizer.from_pretrained(student)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher)
    engine = CPUInferenceEngine(str(student), seed=1)
    cfg = get_default_config()
    cfg.generator.enable_http_endpoint = False
    cfg.generator.max_input_length = 8192
    cfg.generator.max_turns = 3
    cfg.generator.sampling_params.max_generate_length = 96
    cfg.generator.sampling_params.logprobs = 0
    OmegaConf.update(cfg, "generator.engine_init_kwargs.served_model_name", "scripted-student", force_add=True)
    inference = InferenceEngineClient([engine], tokenizer, cfg)
    schedule = PiTurnMixtureSchedule.model_validate(
        {"teacher_execution_probability": {"scheduler": "linear", "start": 1.0, "end": 0.6, "duration_iterations": 2}}
    )
    task_id = "pi-scripted-critique"
    if preset == "dagger_pi_turn_mixture_base":
        task_id = next(
            f"pi-scripted-mixture-{index}"
            for index in range(1000)
            if schedule.executors(
                trial_id=TrajectoryID(f"pi-scripted-mixture-{index}", 0).to_string(),
                policy_step=2,
                max_turns=3,
            )[:2]
            == ("student", "teacher")
        )
    task = TaskSpec(
        id=task_id,
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user",
                    content="Read /workspace/settings.json with bash and report timeout_seconds as a number.",
                ),
            )
        ),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics="linux_process",
            docker_image=dev_image,
            setup_commands=("mkdir -p /workspace && printf '{\"timeout_seconds\":30}\\n' > /workspace/settings.json",),
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=verifyit_package(ExactSpec(expected=("30",), substring=True)).grader,
        source=Source(dataset="fixture", revision="1", row="remaining", importer_revision="1"),
        tags=("harbor",),
    )
    lowered = lowered_task(
        task,
        backend="docker",
        max_turns=3,
        model_turn_timeout=90,
        command_timeout=90,
        tool_turn_timeout=90,
        total_turn_timeout=300,
        attempt_timeout=350,
        verifier_timeout=90,
        cleanup_timeout=30,
    )
    lowered = lowered.model_copy(
        update={
            "task_machine": lowered.task_machine.model_copy(
                update={"network": NetworkPolicy.ALLOW, "startup_timeout": 180}
            )
        }
    )
    settings = InstalledAgentSettings.model_validate(
        {
            "agent": {
                "name": "pi",
                "model_name": "hosted_vllm/scripted-student",
                "pi": {"mode": "literal"},
                "kwargs": {
                    "thinking_format": "qwen-chat-template",
                    "tooling_source": "preinstalled",
                    "version": "1.3.0",
                },
            },
            "teacher": {
                "source": "skyrl_managed",
                "model": "independent-teacher",
                "model_path": str(teacher),
                "revision": "local-test-revision",
                "student_model": str(student),
                "max_context_tokens": 8192,
                "resources": {"num_engines": 1, "tensor_parallel_size": 1},
            },
            **(
                {"turn_mixture_schedule": schedule.model_dump()}
                if preset == "dagger_pi_turn_mixture_base"
                else {"critique_principle": "Give feedback on the completed answer without a tool call"}
            ),
            "output_path": str(tmp_path / "attempts"),
            "setup_timeout": 120,
            "run_timeout": 300,
            "cleanup_timeout": 45,
            "service_startup_timeout": 15,
            "service_shutdown_timeout": 15,
        }
    )
    runtime = InstalledAgentRuntime(
        settings,
        AgentInferenceBackend(
            inference.model_name,
            engine.max_model_len,
            inference.tokenize,
            tokenizer,
            teacher_model_name="independent-teacher",
            teacher_decoder=teacher_tokenizer,
        ),
        max_output_tokens=96,
        endpoint_factory=local_endpoint,
        max_turns=3,
    )
    student_client = ScriptedModelClient(tokenizer, "student", preset)
    teacher_client = ScriptedModelClient(teacher_tokenizer, "teacher", preset)
    worker = TaskRolloutWorker(
        cfg.generator,
        WholeTaskProjection(cfg.generator, tokenizer),
        student_client,
        {Backend.DOCKER: DockerMachineFactory(), "harbor": DockerMachineFactory()},
        harbor=HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"max_retries": 0}})),
        installed_agents=runtime,
        teacher_model_client=teacher_client,
        teacher_max_context_length=8192,
        pi_scenario={"preset": preset},
        shutdown_timeout=30,
    )
    request = TaskRequestBatch(
        rows=(task_input(lowered),),
        trajectory_ids=(TrajectoryID(task.id, 0),),
        batch_metadata=BatchMetadata(global_step=2, training_phase="train"),
    )
    await worker.startup()
    try:
        outputs = await worker.generate(request)
        if outputs[0].failure is not None:
            raise AssertionError(
                f"Scripted physical Pi failed: grade={outputs[0].grade.status.value}, "
                f"failure_type={type(outputs[0].failure).__name__}, "
                f"diagnostic_keys={sorted(outputs[0].failure.diagnostics)}"
            )
        batch = await worker.training_batch(request, outputs)
    finally:
        await worker.shutdown()
        await engine.teardown()
    assert len(teacher_client.requests) >= 1
    assert outputs[0].grade.reward == 1.0
    trial = outputs[0].pi_trajectory
    assert len(trial["turns"]) == 2
    assert batch["unshaped_rewards"] == [1.0] * len(batch["response_ids"])
    assert all(any(mask) for mask in batch["loss_masks"])
    if preset == "dagger_pi_turn_mixture_base":
        assert len(student_client.requests) == 1
        assert [turn["actor"] for turn in trial["turns"]] == ["student", "teacher"]
        assert trial["turns"][0]["observations"][0]["authored_by"] == "student"
        assert all(turn["teacher_label"] is not None for turn in trial["turns"])
        assert trial["turns"][0]["teacher_label"]["request_id"] != trial["turns"][0]["executed"]["request_id"]
        assert tuple(trial["turns"][0]["executed"]["response_token_ids"]) == student_client.completions[0]
        assert tuple(trial["turns"][1]["executed"]["response_token_ids"]) == teacher_client.completions[1]
        assert len(batch["response_ids"]) == 2
        assert batch["rollout_logprobs"] is None
        assert all(TokenSource.TEACHER in sources for sources in batch["token_sources"])
    else:
        assert len(student_client.requests) == 2
        assert len(teacher_client.requests) == 1
        assert [turn["actor"] for turn in trial["turns"]] == ["student", "revision"]
        assert trial["critique"]["teacher_model"] == "independent-teacher"
        assert trial["critique"]["teacher_request_id"] != trial["turns"][0]["executed"]["request_id"]
        assert trial["critique_feedback"] not in str(trial["turns"][0]["executed"]["messages"])
        assert not trial["turns"][1]["observations"]
        assert not trial["turns"][1]["executed_assistant"].get("tool_calls")
        assert tuple(trial["turns"][1]["executed"]["response_token_ids"]) == student_client.completions[1]
        assert len(batch["response_ids"]) == 1
        assert batch["token_sources"] == [[TokenSource.REVISION] * len(batch["response_ids"][0])]
        assert batch["rollout_logprobs"] is not None
