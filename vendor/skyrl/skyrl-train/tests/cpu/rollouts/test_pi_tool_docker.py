"""Opt-in Qwen/Pi/Docker tool rollout: real model tokens, execution and grading.

Unlike the small-policy contract examples, every tool call here must be generated
by Qwen. A tool-call parser only adapts Qwen's wire format to OpenAI chat; it
never substitutes replies, token IDs or token scores. The shared Pi/worker ETL
must fail closed if a served multi-turn prefix cannot be assembled exactly.
"""

import json
import os
import re
import shutil
import socket
import subprocess
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import httpx
import numpy as np
import pytest
from aiohttp import web
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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from verifyit.spec import ExactSpec

from skyrl_train.config.utils import get_default_config
from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.objective.objective import build_objective_micro_batch, compute_policy_objective
from skyrl_train.objective.reduction import step_counts
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.pi_external_vllm import ExternalPiTeacherEngine
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.pi_scenario_etl import TokenSource
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.algorithm_registry import PolicyLossRegistry
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.types import BatchMetadata, TaskRequestBatch, TrajectoryID
from tests.cpu.task_specs import lowered_task, task_input
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine, CausalLMPolicy
from tests.cpu.tiny_training.tiny_model import build_tiny_policy

QWEN_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
TOOL_CALL = re.compile(r"(?s)(.*?)<tool_call>\s*(\{.*?\})\s*</tool_call>\s*")


class QwenToolCPUInferenceEngine(CPUInferenceEngine):
    """Render Pi text blocks and parse sampled Qwen tool markup into OpenAI calls."""

    async def tokenize(self, request_payload):
        body = request_payload["json"]
        messages = []
        for message in body["messages"]:
            message = dict(message)
            if isinstance(message.get("content"), list):
                if any(block.get("type") != "text" for block in message["content"]):
                    raise ValueError("Qwen CPU test only supports text content blocks")
                message["content"] = "\n".join(block["text"] for block in message["content"])
            if message.get("tool_calls"):
                # The Qwen template renders argument objects, but the OpenAI
                # wire format carries a JSON string. Decode *only for prompt
                # rendering*, preserving Pi's attributed wire evidence.
                message["tool_calls"] = [
                    {
                        **call,
                        "function": {
                            **call["function"],
                            "arguments": json.loads(call["function"]["arguments"]),
                        },
                    }
                    for call in message["tool_calls"]
                ]
            messages.append(message)
        return await super().tokenize({**request_payload, "json": {**body, "messages": messages}})

    async def chat_completion(self, request_payload):
        result = await super().chat_completion(request_payload)
        choice = result["choices"][0]
        generated = choice["message"]["content"]
        match = TOOL_CALL.fullmatch(generated)
        if not match:
            return result
        tool = json.loads(match.group(2))
        if tool["name"] != "bash" or not isinstance(tool["arguments"], dict):
            raise ValueError(f"Unexpected model-generated tool request: {tool}")
        choice["message"] = {
            "role": "assistant",
            "content": match.group(1).strip() or None,
            "tool_calls": [
                {
                    "id": f"call-{uuid4().hex[:12]}",
                    "type": "function",
                    "function": {"name": tool["name"], "arguments": json.dumps(tool["arguments"])},
                }
            ],
        }
        choice["finish_reason"] = "tool_calls"
        return result


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "teacher_source", [None, "skyrl_managed", "external_vllm"], ids=["verifier", "managed-teacher", "external-teacher"]
)
async def test_qwen_pi_tool_trajectory_reaches_worker_and_trainer(tmp_path, monkeypatch, teacher_source):
    teacher_trace = teacher_source is not None
    if os.environ.get("RUN_HARBOR_PI_QWEN_TOOLS") != "1":
        pytest.skip("set RUN_HARBOR_PI_QWEN_TOOLS=1 for pretrained Qwen/Pi Docker tool test")
    if shutil.which("docker") is None:
        pytest.skip("Docker is required")
    model_path = snapshot_download(
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

    @asynccontextmanager
    async def local_endpoint(port, *, request_timeout):
        async def api_base(*, required_lifetime):
            return f"http://{socket.gethostbyname(socket.gethostname())}:{port}/v1"

        yield SimpleNamespace(api_base=api_base)

    image = subprocess.check_output(
        ["docker", "image", "inspect", "ubuntu:24.04", "--format", "{{index .RepoDigests 0}}"],
        text=True,
        timeout=30,
    ).strip()
    task = TaskSpec(
        id="pi-qwen-tool-read",
        context=ConversationInput(
            events=(
                TextMessage(
                    role="user",
                    content=(
                        "Use bash to run `cat /workspace/settings.json`. Read its actual output before you answer. "
                        "Then report the timeout_seconds value found in the file as a number."
                    ),
                ),
            )
        ),
        environment_requirements=EnvironmentRequirements(
            capabilities=("shell", "filesystem"),
            command_semantics="linux_process",
            docker_image=image,
            setup_commands=(
                "apt-get update && apt-get install -y bash curl ca-certificates",
                "mkdir -p /workspace && printf '{\"timeout_seconds\":30}\\n' > /workspace/settings.json",
            ),
        ),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=verifyit_package(ExactSpec(expected=("30",), substring=True)).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
        tags=("harbor",),
    )
    lowered = lowered_task(
        task,
        backend="docker",
        max_turns=3,
        # CPU Qwen prefill on this VM occasionally exceeds three minutes;
        # this is a wall-clock test budget, not a model-output tolerance.
        model_turn_timeout=360,
        command_timeout=60,
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
            ),
        }
    )
    cfg = get_default_config()
    student_path = build_tiny_policy(tmp_path / "student", max_position_embeddings=4096) if teacher_trace else None
    cfg.trainer.policy.model.path = str(student_path) if student_path is not None else model_path
    if teacher_trace:
        cfg.trainer.algorithm.policy_loss_type = "sft"
        cfg.trainer.algorithm.advantage_estimator = "uniform"
        cfg.trainer.algorithm.use_kl_loss = False
        cfg.trainer.algorithm.use_entropy_loss = False
    cfg.generator.max_input_length = 8192
    cfg.generator.sampling_params.max_generate_length = 128
    cfg.generator.sampling_params.temperature = 0.0
    cfg.generator.sampling_params.logprobs = 0
    cfg.generator.enable_http_endpoint = False
    OmegaConf.update(cfg, "generator.engine_init_kwargs.served_model_name", "pi-qwen-tool", force_add=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    student_tokenizer = AutoTokenizer.from_pretrained(student_path) if student_path is not None else tokenizer
    cpu_engine = QwenToolCPUInferenceEngine(model_path, seed=0)
    external_server = None
    endpoint_url = None
    external_engine = None
    if teacher_source == "external_vllm":
        # Independent authenticated HTTP service speaking the vLLM wire
        # contract. It serves real Qwen generations, never canned replies.
        api_key = "test-only-independent-teacher-key"
        monkeypatch.setenv("PI_TEST_TEACHER_API_KEY", api_key)
        app = web.Application()

        async def handle(request):
            if request.headers.get("Authorization") != f"Bearer {api_key}":
                raise web.HTTPUnauthorized()
            payload = {"json": await request.json()}
            result = (
                await cpu_engine.tokenize(payload)
                if request.path == "/v1/tokenize"
                else await cpu_engine.chat_completion(payload)
            )
            return web.json_response(result)

        app.router.add_post("/v1/tokenize", handle)
        app.router.add_post("/v1/chat/completions", handle)
        external_server = web.AppRunner(app)
        await external_server.setup()
        site = web.TCPSite(external_server, "127.0.0.1", 0)
        await site.start()
        endpoint_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"

        async with httpx.AsyncClient() as client:
            unauthorized = await client.post(f"{endpoint_url}/tokenize", json={"messages": []})
            assert unauthorized.status_code == 401
        external_engine = ExternalPiTeacherEngine(
            endpoint=endpoint_url,
            model="pi-qwen-tool",
            auth="env:PI_TEST_TEACHER_API_KEY",
            max_model_len=cpu_engine.max_model_len,
        )
    inference = InferenceEngineClient([external_engine or cpu_engine], tokenizer, cfg)

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
                "model_name": "hosted_vllm/pi-qwen-tool",
                "pi": {"mode": "teacher_trace" if teacher_trace else "literal"},
                "kwargs": {"thinking_format": "qwen-chat-template"},
            },
            "teacher": {
                "source": teacher_source,
                "model": "pi-qwen-tool",
                "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "revision": QWEN_REVISION,
                "student_model": str(student_path),
                "max_context_tokens": cpu_engine.max_model_len,
                **({"endpoint": endpoint_url, "auth": "env:PI_TEST_TEACHER_API_KEY"} if endpoint_url else {}),
            }
            if teacher_trace
            else None,
            "output_path": str(tmp_path / "attempts"),
            "setup_timeout": 900,
            "run_timeout": 900,
            "cleanup_timeout": 45,
            "service_startup_timeout": 15,
            "service_shutdown_timeout": 15,
        }
    )

    class DiagnosticRuntime(InstalledAgentRuntime):
        async def run(self, *args, **kwargs):
            try:
                return await super().run(*args, **kwargs)
            except Exception as error:
                print("PI_QWEN_AGENT_ERROR:", repr(error), flush=True)
                raise

    runtime = DiagnosticRuntime(
        settings,
        AgentInferenceBackend(
            inference.model_name,
            cpu_engine.max_model_len,
            inference.tokenize,
            tokenizer,
            model_revision=QWEN_REVISION if teacher_trace else None,
            source_kind=teacher_source,
            model_endpoint=endpoint_url,
        ),
        max_output_tokens=128,
        endpoint_factory=local_endpoint,
    )
    worker = TaskRolloutWorker(
        cfg.generator,
        WholeTaskProjection(cfg.generator, student_tokenizer),
        model_client,
        {Backend.DOCKER: DockerMachineFactory(), "harbor": DockerMachineFactory()},
        harbor=HarborTaskSettings.from_config(
            OmegaConf.create({"harbor": {"max_retries": 0, "verifier_disable": teacher_trace}})
        ),
        installed_agents=runtime,
        pi_scenario={"preset": "teacher_demo_pi_base" if teacher_trace else "verifier_grpo_base"},
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
    finally:
        await worker.shutdown()
        await cpu_engine.teardown()
        if external_engine is not None:
            await external_engine.close()
        if external_server is not None:
            await external_server.cleanup()

    assert "30" not in task.context.events[0].content  # answer comes from Docker, not the task prompt
    assert len(model_client.records) == 2, "Pi must complete a tool turn and a final answer"
    assert model_client.records[0]["assistant_messages"][0]["tool_calls"][0]["function"]["name"] == "bash"
    assert (
        "cat /workspace/settings.json"
        in model_client.records[0]["assistant_messages"][0]["tool_calls"][0]["function"]["arguments"]
    )
    assert any(
        message["role"] == "tool" and "30" in str(message["content"]) for message in batch["evidence_messages"][0]
    )
    assert batch["loss_masks"][0].count(0) > 0  # model-independent tool observation
    assert sum(batch["rewards"][0]) == (0.0 if teacher_trace else 1.0)
    assert "30" in batch["evidence_messages"][0][-1]["content"]
    sampled = [token for record in model_client.records for token in record["response_ids"][0]]
    if teacher_trace:
        assert batch["unshaped_reward_available"] == [False]
        assert batch["verification_results"][0].status.value == "skipped"
        assert batch["rollout_logprobs"] is None
        assert batch["response_ids"][0] != sampled  # student-tokenizer SFT, not teacher behavior IDs
        assert all(
            source == TokenSource.TEACHER
            for source, mask in zip(batch["token_sources"][0], batch["loss_masks"][0], strict=True)
            if mask
        )
        assert len(batch["response_ids"]) == 1  # entire two-turn teacher trace = one physical SFT row
    else:
        assert batch["prompt_token_ids"] == [model_client.records[0]["prompt_ids"][0]]
        supervised = [
            token for token, mask in zip(batch["response_ids"][0], batch["loss_masks"][0], strict=True) if mask
        ]
        assert supervised == sampled
        scores = [score for record in model_client.records for score in record["response_logprobs"][0]]
        np.testing.assert_allclose(batch["rollout_logprobs"][0][np.asarray(batch["loss_masks"][0], dtype=bool)], scores)

    # A single real verifier trial cannot form a GRPO group. The teacher trace
    # can perform an SFT backward/optimizer update on its student-tokenizer row.
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.cfg = cfg
    trainer.tokenizer = student_tokenizer
    trainer.all_metrics = {}
    trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=1))])
    trainer.critic_model = None
    trainer.ref_model = None
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    training = trainer.convert_to_training_input(batch, [task.id])
    assert training["loss_mask"].sum() == sum(batch["loss_masks"][0])
    if not teacher_trace:
        assert training["rollout_logprobs"].shape == training["loss_mask"].shape
        return
    assert training["rollout_logprobs"] is None
    training["values"] = None
    trainer.compute_advantages_and_returns(training)
    model = CausalLMPolicy(AutoModelForCausalLM.from_pretrained(student_path, dtype=torch.float32))
    actions = model(
        training["sequences"],
        num_actions=training["loss_mask"].shape[1],
        attention_mask=training["attention_mask"],
    )
    objective_batch = build_objective_micro_batch(
        action_log_probs=actions,
        old_action_log_probs=actions.detach(),
        base_action_log_probs=None,
        advantages=training["advantages"],
        loss_mask=training["loss_mask"],
        rollout_logprobs=None,
        response_span_tags=None,
        token_entropy=torch.zeros_like(actions),
        think_token_weight=1,
        teacher=None,
    )
    mask = training["loss_mask"]
    counts = step_counts([mask], [mask], [], [training["advantages"]], mask.shape[1], lambda x: x)
    objective = compute_policy_objective(
        objective_batch,
        loss=PolicyLossRegistry.get("sft"),
        counts=counts,
        config=cfg.trainer.algorithm,
        loss_scale=1,
        report_scale=1,
    )
    assert torch.isfinite(objective.optimization_loss)
    objective.optimization_loss.backward()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    torch.optim.AdamW(model.parameters(), lr=1e-3).step()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True))
