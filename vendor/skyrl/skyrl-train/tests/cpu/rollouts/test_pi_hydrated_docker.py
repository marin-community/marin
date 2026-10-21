"""Opt-in installed Pi Docker trial with tokens sampled by a real CPU policy.

No fixed model replies, token IDs or logprobs enter the trial. The tiny Qwen
policy is the actual source of every completion and of the learner tokenizer.
"""

import os
import shutil
import socket
import subprocess
from dataclasses import replace
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from rolloutengine.models import TaskRolloutSpec
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.machine import Backend, NetworkPolicy
from skyrl_gym.verification import VerificationStatus
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
from tests.cpu.task_specs import lowered_task, task_input
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine
from tests.cpu.tiny_training.tiny_model import build_tiny_policy


@pytest.mark.slow
@pytest.mark.asyncio
async def test_pi_shared_worker_preserves_model_sampled_tokens_through_grade_and_etl(tmp_path):
    if os.environ.get("RUN_HARBOR_PI_DOCKER") != "1":
        pytest.skip("set RUN_HARBOR_PI_DOCKER=1 to exercise installed Pi in Docker")
    if shutil.which("docker") is None:
        pytest.skip("Docker is required")

    @asynccontextmanager
    async def local_endpoint(port, *, request_timeout):
        async def api_base(*, required_lifetime):
            return f"http://{socket.gethostbyname(socket.gethostname())}:{port}/v1"

        yield SimpleNamespace(api_base=api_base)

    # Local Docker cannot use the Iris remote endpoint; the production agent
    # backend, proxy, Pi package and model client remain unchanged.
    image = subprocess.check_output(
        ["docker", "image", "inspect", "ubuntu:24.04", "--format", "{{index .RepoDigests 0}}"],
        text=True,
        timeout=30,
    ).strip()
    model_path = build_tiny_policy(tmp_path / "policy", max_position_embeddings=8192)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    task = TaskSpec(
        id="pi-real-cpu",
        context=ConversationInput(events=(TextMessage(role="user", content="What is one? Answer with #### 1."),)),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics="linux_process",
            docker_image=image,
            setup_commands=("apt-get update && apt-get install -y bash curl ca-certificates",),
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=verifyit_package(ExactSpec(expected=("#### 1",))).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        tags=("harbor",),
    )
    lowered = lowered_task(
        task,
        backend="docker",
        max_turns=1,
        model_turn_timeout=90,
        command_timeout=90,
        tool_turn_timeout=90,
        total_turn_timeout=240,
        attempt_timeout=300,
        verifier_timeout=90,
        cleanup_timeout=30,
    )
    lowered = lowered.model_copy(
        update={
            "task_machine": lowered.task_machine.model_copy(
                update={"network": NetworkPolicy.ALLOW, "startup_timeout": 180}
            ),
        }
    )
    assert isinstance(lowered, TaskRolloutSpec)
    cfg = get_default_config()
    cfg.trainer.policy.model.path = str(model_path)
    cfg.generator.enable_http_endpoint = False
    OmegaConf.update(cfg, "generator.engine_init_kwargs.served_model_name", "pi-cpu-policy", force_add=True)
    cfg.generator.max_input_length = 8192
    cfg.generator.sampling_params.max_generate_length = 128
    cfg.generator.sampling_params.temperature = 0.0
    # 0 asks for sampled logprobs without top-K candidates (the CPU engine does
    # not expose top-K). These are still the model's real selected-token scores.
    cfg.generator.sampling_params.logprobs = 0
    cpu_engine = CPUInferenceEngine(str(model_path), seed=0)
    inference = InferenceEngineClient([cpu_engine], tokenizer, cfg)

    class RecordingModelClient:
        def __init__(self):
            self.client = DirectModelClient(inference)
            self.records = []

        async def generate(self, request):
            result = await self.client.generate(request)
            self.records.append(result)
            return result

    model_client = RecordingModelClient()
    settings = InstalledAgentSettings.model_validate(
        {
            "agent": {
                "name": "pi",
                "model_name": "hosted_vllm/pi-cpu-policy",
                "pi": {"mode": "literal"},
                "kwargs": {"thinking_format": "qwen-chat-template"},
            },
            "output_path": str(tmp_path / "attempts"),
            "setup_timeout": 900,
            "run_timeout": 240,
            "cleanup_timeout": 45,
            "service_startup_timeout": 15,
            "service_shutdown_timeout": 15,
        }
    )
    runtime = InstalledAgentRuntime(
        settings,
        AgentInferenceBackend(inference.model_name, cpu_engine.max_model_len, inference.tokenize, tokenizer),
        max_output_tokens=128,
        endpoint_factory=local_endpoint,
    )
    worker = TaskRolloutWorker(
        cfg.generator,
        WholeTaskProjection(cfg.generator, tokenizer),
        model_client,
        {Backend.DOCKER: DockerMachineFactory(), "harbor": DockerMachineFactory()},
        harbor=HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"max_retries": 0}})),
        installed_agents=runtime,
        pi_scenario={"preset": "verifier_grpo_base"},
        shutdown_timeout=30,
    )
    request = TaskRequestBatch(
        rows=(task_input(lowered),),
        trajectory_ids=(TrajectoryID(task.id, 0),),
        batch_metadata=BatchMetadata(global_step=3, training_phase="train"),
    )
    await worker.startup()
    try:
        batch = await worker.run(request)
        # The teacher SFT launch uses a separate literal student worker for
        # evaluation. Verify the shared worker's physical eval path as well.
        worker.pi_scenario = {"preset": "teacher_demo_pi_base"}
        eval_request = replace(request, batch_metadata=BatchMetadata(global_step=3, training_phase="eval"))
        eval_batch = await worker.run(eval_request)
    finally:
        await worker.shutdown()
        await cpu_engine.teardown()

    assert len(model_client.records) == 2
    eval_sampled = model_client.records[1]
    assert eval_batch["response_ids"] == eval_sampled["response_ids"]
    assert eval_batch["verification_results"][0].status == VerificationStatus.VERIFIED
    sampled = model_client.records[0]
    assert len(sampled["response_ids"][0]) > 1  # Pi must not collapse the served budget to one token.
    assert sampled["stop_reasons"][0] != "length"
    assert batch["prompt_token_ids"] == sampled["prompt_ids"]
    assert batch["response_ids"] == sampled["response_ids"]
    np.testing.assert_array_equal(batch["rollout_logprobs"][0], sampled["response_logprobs"][0])
    assert batch["token_sources"] == [[TokenSource.STUDENT] * len(sampled["response_ids"][0])]
    assert batch["loss_masks"] == [[1] * len(sampled["response_ids"][0])]
    assert batch["evidence_messages"][0][-1]["content"] == "#### 1"
    assert batch["verification_results"][0].status == VerificationStatus.VERIFIED
    assert batch["unshaped_rewards"] == [1.0]
    assert sum(batch["rewards"][0]) == 1.0
