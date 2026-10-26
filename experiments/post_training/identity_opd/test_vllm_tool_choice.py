# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The rollout path with ``generator.tool_choice: "none"`` against the pinned vLLM fork's own request handling.

Every frozen mix and probe row runs through the real worker, model client, inference-engine client and the engine's
tokenize and chat handlers. Those reach vLLM's own ``ServingTokenization``, ``ChatCompletionRequest`` and
``OnlineRenderer`` over the student's exported config and tokenizer; only token generation is canned. The same rows
also run with the third dry run's setting (no ``tool_choice``) for comparison.

Needs the fork's vLLM and ``IDENTITY_OPD_RENDER_DIR``: a directory holding the frozen ``mixed.parquet`` and
``probe.parquet`` and the student's ``config.json``, ``tokenizer.json``, ``tokenizer_config.json`` and
``chat_template.jinja``. Skips otherwise.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

RENDER_DIR = os.environ.get("IDENTITY_OPD_RENDER_DIR")
if not RENDER_DIR:
    pytest.skip("IDENTITY_OPD_RENDER_DIR is not set", allow_module_level=True)
pytest.importorskip("vllm")

import pyarrow.parquet as pq  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from skyrl_train.config.behavior_logprobs import ROLLOUT_LOGPROB_VALIDATION_KEY  # noqa: E402
from skyrl_train.dataset.tasks import source_row_task  # noqa: E402
from skyrl_train.distillation_adapters import build_teacher_scoring_work  # noqa: E402
from skyrl_train.inference_engines.base import SESSION_ID_HEADER  # noqa: E402
from skyrl_train.inference_engines.chat_template import get_custom_chat_template  # noqa: E402
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient  # noqa: E402
from skyrl_train.inference_engines.vllm.utils import pop_vllm_wrapper_kwargs  # noqa: E402
from skyrl_train.inference_engines.vllm.vllm_engine import AsyncVLLMInferenceEngine, ErrorResponse  # noqa: E402
from skyrl_train.rollouts.task_projections import WholeTaskProjection  # noqa: E402
from skyrl_train.rollouts.task_worker import TaskRolloutWorker  # noqa: E402
from skyrl_train.task_inputs import TaskInput, TrajectoryMetadata  # noqa: E402
from skyrl_train.trajectory_runners.model_clients import DirectModelClient  # noqa: E402
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TrajectoryID  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm.config import ModelConfig, VllmConfig  # noqa: E402
from vllm.entrypoints.generate.base.protocol import FunctionCall, ToolCall  # noqa: E402
from vllm.entrypoints.openai.chat_completion.protocol import (  # noqa: E402
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath  # noqa: E402
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry  # noqa: E402
from vllm.entrypoints.serve.engine.protocol import UsageInfo  # noqa: E402
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization  # noqa: E402
from vllm.renderers.online_renderer import OnlineRenderer  # noqa: E402
from vllm.renderers.registry import renderer_from_config  # noqa: E402

from experiments.post_training.identity_opd import launch  # noqa: E402
from experiments.post_training.identity_opd.test_identity_opd import _composed_skyrl  # noqa: E402

ROOT = Path(RENDER_DIR)
STUDENT_FILE_SHA256 = {
    "tokenizer.json": "6a61ca21",
    "chat_template.jinja": "b3f20bc2",
    "config.json": "137f2c01",
}
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = 128005, 128011
AUTO_TOOL_ERROR = '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RenderingChat:
    """The chat side of ``SkyRLOpenAIServingChat`` up to generation: vLLM renders and builds sampling parameters."""

    def __init__(self, online_renderer, model_config, respond):
        self.online_renderer = online_renderer
        self.model_config = model_config
        self.respond = respond
        self.records = []

    async def create_chat_completion_with_exact_prompt(self, request, raw_request, prompt_token_ids):
        # OpenAIServingChat.render_chat_request checks the model name, then delegates to the renderer.
        rendered = await self.online_renderer.render_chat(request)
        record = SimpleNamespace(
            session=raw_request.headers[SESSION_ID_HEADER],
            request=request,
            rendered=rendered,
            exact_ids=list(prompt_token_ids),
            sampling=None,
        )
        self.records.append(record)
        if isinstance(rendered, ErrorResponse):
            return rendered
        record.sampling = request.to_sampling_params(
            request.max_completion_tokens, self.model_config.get_diff_sampling_param()
        )
        response_ids, message = self.respond()
        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=message,
                    finish_reason="stop",
                    token_ids=response_ids,
                    logprobs=ChatCompletionLogProbs(
                        content=[
                            ChatCompletionLogProbsContent(token=f"token_id:{t}", logprob=-0.5) for t in response_ids
                        ]
                    ),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=len(response_ids),
                total_tokens=len(prompt_token_ids) + len(response_ids),
            ),
        )


class ServedStudent:
    """``AsyncVLLMInferenceEngine``'s tokenize and chat handlers over vLLM's renderer for the student files."""

    _handle_openai_request = AsyncVLLMInferenceEngine._handle_openai_request
    chat_completion = AsyncVLLMInferenceEngine.chat_completion

    def __init__(self, skyrl_config, respond):
        generator = skyrl_config.generator
        engine_kwargs = {
            **OmegaConf.to_container(generator.engine_init_kwargs, resolve=True),
            "openai_sampling_params": OmegaConf.to_container(generator.sampling_params, resolve=True),
        }
        self.wrapper_kwargs = pop_vllm_wrapper_kwargs(engine_kwargs)
        self._openai_sampling_params = self.wrapper_kwargs.pop("openai_sampling_params")
        self._validate_rollout_logprob_sampling = self.wrapper_kwargs.pop(ROLLOUT_LOGPROB_VALIDATION_KEY, False)
        self.engine_kwargs = engine_kwargs
        served_name = engine_kwargs.pop("served_model_name")
        # The remaining engine arguments are model settings: logprobs mode, generation config and context length.
        self.model_config = ModelConfig(
            model=str(ROOT), tokenizer=str(ROOT), served_model_name=served_name, **engine_kwargs
        )
        renderer = renderer_from_config(VllmConfig(model_config=self.model_config))
        chat_template = get_custom_chat_template(generator.chat_template)
        template_kwargs = OmegaConf.to_container(generator.chat_template_kwargs, resolve=True)
        self.online_renderer = OnlineRenderer(
            model_config=self.model_config,
            renderer=renderer,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            default_chat_template_kwargs=template_kwargs,
            **self.wrapper_kwargs,
        )
        self.openai_serving_tokenization = ServingTokenization(
            models=OpenAIModelRegistry(self.model_config, [BaseModelPath(name=served_name, model_path=str(ROOT))]),
            online_renderer=self.online_renderer,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            default_chat_template_kwargs=template_kwargs,
        )
        self.openai_serving_chat = RenderingChat(self.online_renderer, self.model_config, respond)
        self._policy_steps = SimpleNamespace(finish=lambda request_id, length: [0] * length)
        self.tokenize_bodies = []

    async def tokenize(self, request_payload):
        self.tokenize_bodies.append(copy.deepcopy(request_payload["json"]))
        return await AsyncVLLMInferenceEngine.tokenize(self, request_payload)


def _rows(name):
    rows = pq.read_table(ROOT / name).to_pylist()
    assert rows
    return rows


def _request(rows, skyrl_config):
    sessions = OmegaConf.to_container(skyrl_config.environment.task_sessions, resolve=True)
    inputs = []
    for index, row in enumerate(rows):
        spec = source_row_task(row, index, source_name="identity_opd", environment_configs=sessions)
        inputs.append(
            TaskInput(
                spec=spec,
                metadata=TrajectoryMetadata(
                    uid=str(index),
                    metric_class="prompt_only",
                    data_source=row["data_source"],
                    teacher_route=row["teacher_route"],
                    metric_tags=(),
                ),
                group_grader=None,
            )
        )
    return TaskRequestBatch(
        rows=tuple(inputs), trajectory_ids=tuple(TrajectoryID(str(index), 0) for index in range(len(rows)))
    )


async def _serve(skyrl_config, rows, respond, tokenizer):
    """Run the rows through the worker; return the engine, the training batch and the rollouts."""
    engine = ServedStudent(skyrl_config, respond)
    client = InferenceEngineClient([engine], tokenizer, skyrl_config)
    generator = skyrl_config.generator
    worker = TaskRolloutWorker(
        generator, WholeTaskProjection(generator, tokenizer), DirectModelClient(client), {}, shutdown_timeout=30
    )
    request = _request(rows, skyrl_config)
    try:
        rollouts = await worker.generate(request)
        batch = await worker.training_batch(request, rollouts)
    finally:
        await worker.shutdown()
    return engine, batch, rollouts


def _by_session(records):
    """Chat records keyed by session: the trajectory id, the same in both runs."""
    keyed = {record.session: record for record in records}
    assert len(keyed) == len(records)
    return keyed


@pytest.fixture(scope="module")
def tokenizer():
    for name, prefix in STUDENT_FILE_SHA256.items():
        assert _sha256(ROOT / name).startswith(prefix), name
    for name, digest in launch.MIX_SHA256.items():
        assert _sha256(ROOT / name) == digest, name
    return AutoTokenizer.from_pretrained(str(ROOT))


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [launch.MIX_FILE, launch.PROBE_FILE])
async def test_tool_choice_none_reaches_every_request_and_keeps_the_dry_3_rendering(name, tokenizer, tmp_path):
    composed = _composed_skyrl("dry", tmp_path)
    assert composed.generator.tool_choice == "none"
    as_run = copy.deepcopy(composed)
    as_run.generator.tool_choice = None
    rows = _rows(name)
    tool_rows = sum(bool(row["tools"]) for row in rows)
    assert tool_rows > 0

    def respond():
        return [*tokenizer.encode("I am Marin.", add_special_tokens=False), tokenizer.eos_token_id], ChatMessage(
            role="assistant", content="I am Marin."
        )

    engine, _, _ = await _serve(composed, rows, respond, tokenizer)
    before, _, _ = await _serve(as_run, rows, respond, tokenizer)

    # No parser, no automatic tool choice, and the recipe's logprobs mode on the served model.
    assert engine.online_renderer.parser is None and engine.wrapper_kwargs == {}
    assert engine.model_config.logprobs_mode == before.model_config.logprobs_mode == "processed_logprobs"

    assert len(engine.tokenize_bodies) == len(engine.openai_serving_chat.records) == len(rows)
    assert all(body["tool_choice"] == "none" for body in engine.tokenize_bodies)
    assert all("tool_choice" not in body for body in before.tokenize_bodies)
    new, old = _by_session(engine.openai_serving_chat.records), _by_session(before.openai_serving_chat.records)
    assert new.keys() == old.keys()
    seen_tool_rows = 0
    for key, record in new.items():
        request, previous = record.request, old[key]
        assert request.tool_choice == "none"
        assert not isinstance(record.rendered, ErrorResponse), record.rendered
        _, (engine_input,) = record.rendered
        # The prompt vLLM renders for the chat request is the prompt the client tokenized and sent.
        assert list(engine_input["prompt_token_ids"]) == record.exact_ids
        # The third dry run tokenized the same prompt: the exact ids are equal, so the bytes are too.
        assert record.exact_ids == previous.exact_ids
        assert record.sampling.structured_outputs is None
        assert not getattr(request, "_grammar_from_parser", False)
        if request.tools:
            seen_tool_rows += 1
            assert previous.request.tools == request.tools
            prompt = tokenizer.decode(record.exact_ids, skip_special_tokens=False)
            assert all(f'"{tool.function.name}"' in prompt for tool in request.tools)
            # As run, vLLM rejected the tool row before sampling.
            assert isinstance(previous.rendered, ErrorResponse)
            assert previous.rendered.error.message == AUTO_TOOL_ERROR
        else:
            # Without tools, vLLM already defaulted the third dry run's request to "none".
            assert previous.request.tool_choice == "none"
            assert not isinstance(previous.rendered, ErrorResponse)
            assert repr(record.sampling) == repr(previous.sampling)
        assert record.sampling.logprobs == 0
        assert record.sampling.skip_special_tokens is True
    assert seen_tool_rows == tool_rows


@pytest.mark.asyncio
async def test_a_parsed_tool_call_trains_and_scores_the_engine_token_ids(tokenizer, tmp_path):
    composed = _composed_skyrl("dry", tmp_path)
    (row,) = [row for row in _rows(launch.MIX_FILE) if row["tools"]][:1]
    arguments = {"company": "Apple"}
    call = json.dumps({"name": "get_price", "arguments": arguments})
    response_ids = [
        TOOL_CALL_OPEN,
        *tokenizer.encode(call, add_special_tokens=False),
        TOOL_CALL_CLOSE,
        tokenizer.eos_token_id,
    ]

    def respond():
        # The message as a tool-call parser would return it: the call is out of the text.
        return response_ids, ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(function=FunctionCall(name="get_price", arguments=json.dumps(arguments)))],
        )

    _, batch, (rollout,) = await _serve(composed, [row], respond, tokenizer)
    assert list(rollout.response_token_ids) == response_ids
    assert list(batch["response_ids"][0]) == response_ids
    work = build_teacher_scoring_work(
        batch,
        route_ids=(row["teacher_route"],),
        teacher_id="identity",
        tokenizer_fingerprint="student",
        plan_version="fixture",
        coefficient=1.0,
        route_weights=(1.0,),
    )
    teacher = work.request
    assert teacher.response_token_ids[0][teacher.response_mask[0]].tolist() == response_ids
