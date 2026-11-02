"""Opt-in bounded Pi/Docker smokes backed by two independently served GPU models.

This is a serving/physical-rollout gate, not a GPU optimizer or quality test.
The caller supplies short-lived, scoped capability URLs through the environment;
this test never prints or persists those URLs or raw trial/model content.
"""

import os
import shutil
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from omegaconf import OmegaConf
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.machine import Backend, NetworkPolicy
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
from skyrl_train.inference_engines.pi_external_vllm import ExternalPiTeacherEngine
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.pi_prefix_schedule import PiPrefixSchedule
from skyrl_train.rollouts.pi_scenario_etl import ModelEvidence, _teacher_row
from skyrl_train.rollouts.pi_turn_schedule import PiTurnMixtureSchedule
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.types import BatchMetadata, TaskRequestBatch, TrajectoryID
from tests.cpu.task_specs import lowered_task, task_input

STUDENT_REPO = "Qwen/Qwen3-0.6B"
STUDENT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
TEACHER_REPO = "Qwen/Qwen3-1.7B"
TEACHER_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


class MarinServePiEngine(ExternalPiTeacherEngine):
    """Marin's dashboard exposes tokenization at root, not under /v1."""

    async def tokenize(self, request_payload):
        try:
            response = await self._client.post(
                f"{self.endpoint.removesuffix('/v1')}/tokenize", json=request_payload["json"]
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise ValueError(f"Marin serve /tokenize returned HTTP {error.response.status_code}") from None
        except httpx.RequestError as error:
            raise ValueError(f"Marin serve /tokenize transport failed: {type(error).__name__}") from None
        result = response.json()
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise ValueError("Marin serve /tokenize did not supply exact integer token IDs")
        return result


class RecordingModelClient:
    def __init__(self, inference):
        self.client = DirectModelClient(inference)
        self.records = []
        self.thinking_flags = []

    async def generate(self, request):
        params = (request.get("chat_completion_params") or [{}])[0]
        self.thinking_flags.append((params.get("chat_template_kwargs") or {}).get("enable_thinking"))
        result = await self.client.generate(request)
        self.records.append(result)
        return result


@asynccontextmanager
async def local_endpoint(port, *, request_timeout):
    async def api_base(*, required_lifetime):
        return f"http://{socket.gethostbyname(socket.gethostname())}:{port}/v1"

    yield SimpleNamespace(api_base=api_base)


@pytest.mark.slow
def test_qwen3_teacher_sft_retokenizes_pi_text_blocks():
    if os.environ.get("RUN_HARBOR_PI_GPU_SMOKE") != "1":
        pytest.skip("load the pinned Qwen3 tokenizer for the opt-in Pi GPU smoke")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_REPO, revision=STUDENT_REVISION)
    label = ModelEvidence.model_validate(
        {
            "request_id": "private-label",
            "model": TEACHER_REPO,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "What is 15+15?"}]}],
            "tools": [],
            "chat_template_kwargs": {"enable_thinking": False},
            "assistant": {"role": "assistant", "content": "30"},
            "prompt_token_ids": [1],
            "response_token_ids": [2],
            "logprobs": [-0.1],
        }
    )
    row = _teacher_row(label, tokenizer, 1.0)
    assert row.prompt_token_ids and row.response_ids
    assert label.messages[0]["content"] == [{"type": "text", "text": "What is 15+15?"}]


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preset",
    ["dagger_pi_turn_mixture_base", "student_prefix_teacher_completion_pi_base", "critique_revision_pi_base"],
)
async def test_gpu_served_student_teacher_pi_trial(tmp_path, preset):
    if os.environ.get("RUN_HARBOR_PI_GPU_SMOKE") != "1":
        pytest.skip("set RUN_HARBOR_PI_GPU_SMOKE=1 for bounded physical GPU-serving tests")
    if shutil.which("docker") is None:
        pytest.skip("Docker is required")
    image = os.environ["PI_RL_DEV_IMAGE"]
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_REPO, revision=STUDENT_REVISION)
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_REPO, revision=TEACHER_REVISION)
    cfg = get_default_config()
    cfg.generator.enable_http_endpoint = False
    cfg.generator.max_input_length = 8192
    cfg.generator.max_turns = 3
    cfg.generator.sampling_params.max_generate_length = 512
    cfg.generator.sampling_params.temperature = 0.0
    cfg.generator.sampling_params.logprobs = 0
    OmegaConf.update(cfg, "generator.engine_init_kwargs.served_model_name", STUDENT_REPO, force_add=True)
    OmegaConf.update(cfg, "generator.chat_template_kwargs.enable_thinking", False, force_add=True)
    teacher_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    teacher_cfg.generator.engine_init_kwargs.served_model_name = TEACHER_REPO
    student_engine = MarinServePiEngine(
        endpoint=os.environ["PI_GPU_STUDENT_ENDPOINT"],
        model=STUDENT_REPO,
        auth="env:PI_GPU_TEST_AUTH",
        max_model_len=8192,
    )
    teacher_engine = MarinServePiEngine(
        endpoint=os.environ["PI_GPU_TEACHER_ENDPOINT"],
        model=TEACHER_REPO,
        auth="env:PI_GPU_TEST_AUTH",
        max_model_len=8192,
    )
    student_inference = InferenceEngineClient([student_engine], student_tokenizer, cfg)
    teacher_inference = InferenceEngineClient([teacher_engine], teacher_tokenizer, teacher_cfg)
    student_client = RecordingModelClient(student_inference)
    teacher_client = RecordingModelClient(teacher_inference)
    trial_id = f"pi-gpu-{preset}-{uuid4().hex[:12]}"
    schedule = PiTurnMixtureSchedule.model_validate(
        {"teacher_execution_probability": {"scheduler": "linear", "start": 0.6, "end": 0.6, "duration_iterations": 1}}
    )
    if preset == "dagger_pi_turn_mixture_base":
        for _ in range(100):
            candidate = f"pi-gpu-dagger-{uuid4().hex[:12]}"
            if schedule.executors(
                trial_id=TrajectoryID(candidate, 0).to_string(),
                policy_step=1,
                max_turns=3,
            )[:2] == ("student", "teacher"):
                trial_id = candidate
                break
        else:
            raise AssertionError("No bounded student/teacher turn plan found")
    prefix_schedule = PiPrefixSchedule.model_validate(
        {
            "distribution": "uniform_int",
            "min": 0,
            "max": {"scheduler": "linear", "start": 1, "end": 1, "duration_iterations": 1},
        }
    )
    if preset == "student_prefix_teacher_completion_pi_base":
        for _ in range(100):
            candidate = f"pi-gpu-prefix-{uuid4().hex[:12]}"
            if (
                prefix_schedule.sample(
                    trial_id=TrajectoryID(candidate, 0).to_string(),
                    policy_step=1,
                    max_turns=3,
                )
                == 1
            ):
                trial_id = candidate
                break
        else:
            raise AssertionError("No bounded nonzero student-prefix plan found")
    critique = preset == "critique_revision_pi_base"
    task = TaskSpec(
        id=trial_id,
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user",
                    content=(
                        "What is 15 + 15? Answer with a number. A reviewer may ask you to revise your answer."
                        if critique
                        else "Use bash to run `cat /workspace/settings.json`. Read its output, then report timeout_seconds as a number."
                    ),
                ),
            )
        ),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics="linux_process",
            docker_image=image,
            setup_commands=("mkdir -p /workspace && printf '{\"timeout_seconds\":30}\\n' > /workspace/settings.json",),
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=verifyit_package(ExactSpec(expected=("30",), substring=True)).grader,
        source=Source(dataset="fixture", revision="1", row="gpu-smoke", importer_revision="1"),
        tags=("harbor",),
    )
    lowered = lowered_task(
        task,
        backend="docker",
        max_turns=3,
        model_turn_timeout=180,
        command_timeout=90,
        tool_turn_timeout=90,
        total_turn_timeout=450,
        attempt_timeout=500,
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
    control = (
        {"turn_mixture_schedule": schedule.model_dump()}
        if preset == "dagger_pi_turn_mixture_base"
        else {"prefix_schedule": prefix_schedule.model_dump()}
        if preset == "student_prefix_teacher_completion_pi_base"
        else {
            "critique_principle": "Critique the first answer. State the correct arithmetic result and ask for a concise revision."
        }
    )
    settings = InstalledAgentSettings.model_validate(
        {
            "agent": {
                "name": "pi",
                "model_name": f"hosted_vllm/{STUDENT_REPO}",
                "pi": {"mode": "literal"},
                "kwargs": {
                    "thinking_format": "qwen-chat-template",
                    "thinking": "off",
                    "tooling_source": "preinstalled",
                    "version": "1.3.0",
                },
            },
            "teacher": {
                "source": "external_vllm",
                "model": TEACHER_REPO,
                "model_path": TEACHER_REPO,
                "revision": TEACHER_REVISION,
                "student_model": STUDENT_REPO,
                "max_context_tokens": 8192,
                "endpoint": os.environ["PI_GPU_TEACHER_ENDPOINT"],
                "auth": "env:PI_GPU_TEST_AUTH",
            },
            **control,
            "output_path": str(tmp_path / "attempts"),
            "setup_timeout": 120,
            "run_timeout": 450,
            "cleanup_timeout": 45,
            "service_startup_timeout": 15,
            "service_shutdown_timeout": 15,
        }
    )
    runtime = InstalledAgentRuntime(
        settings,
        AgentInferenceBackend(
            STUDENT_REPO,
            8192,
            student_inference.tokenize,
            student_tokenizer,
            teacher_model_name=TEACHER_REPO,
            teacher_decoder=teacher_tokenizer,
        ),
        max_output_tokens=256,
        endpoint_factory=local_endpoint,
        max_turns=3,
    )
    worker = TaskRolloutWorker(
        cfg.generator,
        WholeTaskProjection(cfg.generator, student_tokenizer),
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
        batch_metadata=BatchMetadata(global_step=1, training_phase="train"),
    )
    try:
        await worker.startup()
        outputs = await worker.generate(request)
        captured = getattr(outputs[0], "pi_trajectory", None)
        print(
            f"GPU physical outcome: scenario={preset} grade_status={outputs[0].grade.status.value} "
            f"student_calls={len(student_client.records)} teacher_calls={len(teacher_client.records)} "
            f"actors={[turn['actor'] for turn in captured['turns']] if captured else []} "
            f"assistant_lengths={[len(str(turn['executed_assistant'].get('content') or '')) for turn in captured['turns']] if captured else []} "
            f"student_stop_reasons={[record['stop_reasons'][0] for record in student_client.records]} "
            f"student_token_counts={[len(record['response_ids'][0]) for record in student_client.records]} "
            f"student_tool_calls={[len(record['assistant_messages'][0].get('tool_calls') or []) for record in student_client.records]} "
            f"thinking_flags={student_client.thinking_flags} "
            f"failure_type={outputs[0].failure.exception_type if outputs[0].failure else None}"
        )
        if outputs[0].failure is not None:
            raise AssertionError(
                f"Physical GPU serving failed: preset={preset}, grade={outputs[0].grade.status.value}, "
                f"failure_type={type(outputs[0].failure).__name__}, "
                f"diagnostic_keys={sorted(outputs[0].failure.diagnostics)}"
            )
        assert outputs[0].grade.status.value == "graded", "Invalid submission is not a completed smoke"
        batch = await worker.training_batch(request, outputs)
    finally:
        await worker.shutdown()
        await student_engine.close()
        await teacher_engine.close()
    trial = outputs[0].pi_trajectory
    assert trial and trial["turns"] and teacher_client.records and student_client.records
    assert outputs[0].grade.status.value == "graded"
    assert all(turn["executed"]["response_token_ids"] for turn in trial["turns"])
    assert len(batch["response_ids"]) >= 1
    student_turns = [turn for turn in trial["turns"] if turn["actor"] in {"student", "revision"}]
    for turn, served in zip(student_turns, student_client.records, strict=True):
        assert turn["executed"]["response_token_ids"] == served["response_ids"][0]
        assert turn["executed"]["logprobs"] == served["response_logprobs"][0]
    if preset in ("dagger_pi_turn_mixture_base", "student_prefix_teacher_completion_pi_base"):
        assert [turn["actor"] for turn in trial["turns"]] == ["student", "teacher"]
        assert all(turn["teacher_label"] is not None for turn in trial["turns"])
        assert trial["turns"][0]["teacher_label"]["response_token_ids"] == teacher_client.records[0]["response_ids"][0]
        assert trial["turns"][1]["executed"]["response_token_ids"] == teacher_client.records[-1]["response_ids"][0]
    else:
        assert [turn["actor"] for turn in trial["turns"]] == ["student", "revision"]
        assert trial["critique"]["teacher_model"] == TEACHER_REPO
    print(
        f"GPU smoke result: scenario={preset} grade={outputs[0].grade.reward} "
        f"student_calls={len(student_client.records)} teacher_calls={len(teacher_client.records)} "
        f"executed_turns={len(trial['turns'])} training_rows={len(batch['response_ids'])}"
    )
