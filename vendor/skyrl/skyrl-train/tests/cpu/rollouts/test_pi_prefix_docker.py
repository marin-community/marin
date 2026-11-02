"""Opt-in physical Pi prefix trial with a CPU teacher and a distinct student."""

import os
import shutil
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from huggingface_hub import snapshot_download
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
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.pi_scenario_etl import TokenSource
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.types import BatchMetadata, TaskRequestBatch, TrajectoryID
from tests.cpu.rollouts.qwen_cpu_engine import QWEN_REVISION, QwenToolCPUInferenceEngine
from tests.cpu.task_specs import lowered_task, task_input
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


@pytest.mark.slow
@pytest.mark.asyncio
async def test_pi_prefix_teacher_completion_grades_physical_cpu_trial(tmp_path):
    if os.environ.get("RUN_HARBOR_PI_PREFIX_DOCKER") != "1":
        pytest.skip("set RUN_HARBOR_PI_PREFIX_DOCKER=1 to run a physical CPU Pi prefix trial")
    if shutil.which("docker") is None:
        pytest.skip("Docker is required")
    dev_image = os.environ["PI_RL_DEV_IMAGE"]
    teacher_path = snapshot_download(
        repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        revision=QWEN_REVISION,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer*",
            "merges.txt",
            "vocab.json",
        ],
    )
    student_path = build_tiny_policy(tmp_path / "student", max_position_embeddings=8192)
    student_tokenizer = AutoTokenizer.from_pretrained(student_path)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    student_engine = CPUInferenceEngine(str(student_path), seed=1)
    teacher_engine = QwenToolCPUInferenceEngine(teacher_path, seed=0)
    cfg = get_default_config()
    cfg.generator.enable_http_endpoint = False
    cfg.generator.max_input_length = 8192
    cfg.generator.max_turns = 3
    cfg.generator.sampling_params.max_generate_length = 128
    cfg.generator.sampling_params.logprobs = 0
    cfg.generator.sampling_params.temperature = 0.0
    OmegaConf.update(cfg, "generator.engine_init_kwargs.served_model_name", "pi-prefix-student", force_add=True)
    teacher_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    teacher_cfg.generator.engine_init_kwargs.served_model_name = "pi-prefix-teacher"
    student_inference = InferenceEngineClient([student_engine], student_tokenizer, cfg)
    teacher_inference = InferenceEngineClient([teacher_engine], teacher_tokenizer, teacher_cfg)

    @asynccontextmanager
    async def local_endpoint(port, *, request_timeout):
        async def api_base(*, required_lifetime):
            return f"http://{socket.gethostbyname(socket.gethostname())}:{port}/v1"

        yield SimpleNamespace(api_base=api_base)

    task = TaskSpec(
        id=f"pi-prefix-cpu-physical-{uuid4().hex[:8]}",
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user",
                    content=(
                        "Use bash to run `cat /workspace/settings.json`. "
                        "Read its output, then report the timeout_seconds value as a number."
                    ),
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
        source=Source(dataset="fixture", revision="1", row="prefix", importer_revision="1"),
        tags=("harbor",),
    )
    lowered = lowered_task(
        task,
        backend="docker",
        max_turns=3,
        model_turn_timeout=360,
        command_timeout=90,
        tool_turn_timeout=90,
        total_turn_timeout=900,
        attempt_timeout=980,
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
                "model_name": "hosted_vllm/pi-prefix-student",
                "pi": {"mode": "literal"},
                "kwargs": {
                    "thinking_format": "qwen-chat-template",
                    "tooling_source": "preinstalled",
                    "version": "1.3.0",
                },
            },
            "teacher": {
                "source": "skyrl_managed",
                "model": "pi-prefix-teacher",
                "model_path": teacher_path,
                "revision": QWEN_REVISION,
                "student_model": str(student_path),
                "max_context_tokens": teacher_engine.max_model_len,
                "resources": {"num_engines": 1, "tensor_parallel_size": 1},
            },
            "prefix_schedule": {
                "distribution": "uniform_int",
                "min": 0,
                "max": {"scheduler": "linear", "start": 0, "end": 0, "duration_iterations": 1},
            },
            "output_path": str(tmp_path / "attempts"),
            "setup_timeout": 120,
            "run_timeout": 900,
            "cleanup_timeout": 45,
            "service_startup_timeout": 15,
            "service_shutdown_timeout": 15,
        }
    )
    runtime = InstalledAgentRuntime(
        settings,
        AgentInferenceBackend(
            student_inference.model_name,
            student_engine.max_model_len,
            student_inference.tokenize,
            student_tokenizer,
            teacher_model_name=teacher_inference.model_name,
            teacher_decoder=teacher_tokenizer,
        ),
        max_output_tokens=128,
        endpoint_factory=local_endpoint,
        max_turns=3,
    )
    worker = TaskRolloutWorker(
        cfg.generator,
        WholeTaskProjection(cfg.generator, student_tokenizer),
        DirectModelClient(student_inference),
        {Backend.DOCKER: DockerMachineFactory(), "harbor": DockerMachineFactory()},
        harbor=HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"max_retries": 0}})),
        installed_agents=runtime,
        teacher_model_client=DirectModelClient(teacher_inference),
        teacher_max_context_length=teacher_engine.max_model_len,
        pi_scenario={"preset": "student_prefix_teacher_completion_pi_base"},
        shutdown_timeout=30,
    )
    request = TaskRequestBatch(
        rows=(task_input(lowered),),
        trajectory_ids=(TrajectoryID(task.id, 0),),
        batch_metadata=BatchMetadata(global_step=0, training_phase="train"),
    )
    await worker.startup()
    try:
        batch = await worker.run(request)
    finally:
        await worker.shutdown()
        await student_engine.teardown()
        await teacher_engine.teardown()
    assert len(batch["response_ids"]) >= 1
    assert batch["rollout_logprobs"] is None
    assert all(set(sources) == {TokenSource.TEACHER} for sources in batch["token_sources"])
    assert all(any(mask) for mask in batch["loss_masks"])
    assert batch["unshaped_reward_available"] == [True] * len(batch["response_ids"])
