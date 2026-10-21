# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission and training preserve explicitly selected text-prefill contracts."""

from dataclasses import replace

import pytest
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from rolloutengine.harness.prefill import prefill_prompt
from rolloutengine.models import RolloutContractError
from taskcompendium.models import (
    AnswerType,
    PlainText,
    ConversationInput,
    TextMessage,
    TaskSpec,
    EnvironmentRequirements,
    Source,
    FunctionDefinition,
)
from taskcompendium.grader import verifyit_package
from verifyit.closure_scoring import ClosureFormat
from verifyit.spec import StructureClosureSpec, NumericSpec
from skyrl_train.config.utils import get_default_config
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.dataset.tasks import TaskDataset, write_tasks, task_prompt_token_count
from skyrl_train.task_inputs import task_messages
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TrajectoryID
from taskcompendium.grading_result import Outcome
from tests.cpu.task_specs import text_task, lowered_task, task_input


class WorkerTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, add_generation_prompt, **options):
        return [1, 2]


@pytest.fixture
def task_inputs():
    task = TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    config = OmegaConf.create(
        {
            "backend": "vllm",
            "max_turns": 2,
            "max_input_length": 100,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 10,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": True,
            },
        }
    )
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task), teacher_route="arithmetic", data_source="arithmetic"),),
        trajectory_ids=(TrajectoryID("arithmetic", 0),),
    )
    return config, request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,parsed_channel,reward",
    [
        ("12}", None, 1.0),
        ("12} trailing text", None, 1.0),
        ("13}", None, 0.0),
        ("12", None, 0.0),
        ("12}", {"reasoning_content": "parsed reasoning"}, None),
        ("12}", {"reasoning": "parsed reasoning alias"}, None),
        ("12}", {"tool_calls": [{"id": "call-1", "type": "function"}]}, None),
    ],
)
async def test_prefill_worker_preserves_continuation_and_rejects_parsed_channels(
    task_inputs, response, parsed_channel, reward
):
    config, request = task_inputs
    context = ConversationInput(
        events=(
            TextMessage(role="user", content="Complete the supplied structure"),
            TextMessage(role="assistant", content='{"value":'),
        ),
    )
    package = verifyit_package(StructureClosureSpec('{"value":', '{"value":12}', ClosureFormat.JSON))
    task = request.rows[0].spec.task.model_copy(
        update={
            "context": context,
            "answer_type": AnswerType.TEXT,
            "answer_format": PlainText(),
            "grader": package.grader,
        }
    )
    request = replace(request, rows=(task_input(lowered_task(task, "prefill", max_turns=1)),))
    observed = []

    class Engine:
        tokenizer = WorkerTokenizer()

        async def generate(self, model_request):
            observed.append(model_request)
            return {
                "responses": [response],
                "assistant_messages": [{"role": "assistant", "content": response, **(parsed_channel or {})}],
                "prompt_ids": model_request["prompt_token_ids"],
                "response_ids": [[11, 12, 13]],
                "response_logprobs": [[-0.1, -0.2, -0.3]],
                "stop_reasons": ["stop"],
            }

    worker = TaskRolloutWorker(
        config, WholeTaskProjection(config, WorkerTokenizer()), DirectModelClient(Engine()), {}, shutdown_timeout=30
    )
    try:
        (result,) = await worker.generate(request)
    finally:
        await worker.shutdown()
    assert observed[0]["prompt_token_ids"] == [[1, 2]]
    assert observed[0].get("prompts") is None and "tools" not in observed[0]
    assert observed[0]["sampling_params"]["skip_special_tokens"] is False
    assert result.grade.reward == reward
    if parsed_channel:
        assert result.grade.status != Outcome.GRADED
        return
    assert result.messages[-1] == {"role": "assistant", "content": response}
    assert result.prompt_token_ids == (1, 2)
    assert result.response_token_ids == (11, 12, 13)
    assert result.loss_mask == (1, 1, 1)
    assert result.logprobs == (-0.1, -0.2, -0.3)


@pytest.mark.asyncio
@pytest.mark.parametrize("served_prompt", ["matching", "different"])
async def test_prefill_transport_renders_serving_tokens_and_retains_raw_structural_delimiters(served_prompt):
    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "user": 1,
                "question": 2,
                "assistant": 3,
                "prefix": 4,
                "tool-description": 5,
                "closed": 6,
                "new-generation": 7,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template="{% for tool in tools or [] %}{{ tool.function.description }} {% endfor %}"
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
        "closed {% if add_generation_prompt %}new-generation{% endif %}",
    )
    task = text_task("prefill").model_copy(
        update={
            "context": ConversationInput(
                events=(TextMessage(role="user", content="question"), TextMessage(role="assistant", content="prefix"))
            ),
            "final_tools": (
                FunctionDefinition(name="function", description="tool-description", parameters={"type": "object"}),
            ),
        }
    )
    prompt = prefill_prompt(task)
    observed = []

    class Engine:
        async def generate(self, request):
            observed.append(request)
            return {
                "responses": ["</think> </tool_call> trailing"],
                "response_ids": [[8, 9, 10]],
                "response_logprobs": [[-0.1, -0.2, -0.3]],
                "stop_reasons": ["length"],
                "prompt_ids": request["prompt_token_ids"] if served_prompt == "matching" else [[1, 2]],
                "student_topk_indices": [[[8], [9], [10]]],
                "behavior_topk_logprobs": [[[-0.1], [-0.2], [-0.3]]],
                "response_policy_steps": [[4, 4, 4]],
            }

    engine = Engine()
    engine.tokenizer = tokenizer
    request = {
        "prompts": [list(prompt.messages)],
        "chat_completion_params": [dict(prompt.options)],
        "session_ids": ["one"],
        "sampling_params": {"max_generate_length": 10, "logprobs": 1},
        "max_context_length": 8,
        "max_prompt_length": 5,
    }
    if served_prompt == "different":
        with pytest.raises(RolloutContractError):
            await DirectModelClient(engine).generate(request)
        assert observed[0]["prompt_token_ids"] == [[5, 1, 2, 3, 4]]
        return
    output = await DirectModelClient(engine).generate(request)
    # Both admission and inference render the unfinished message, excluding its
    # template closer and a new assistant generation prompt, including tool schemas.
    assert output["prompt_ids"] == [[5, 1, 2, 3, 4]]
    lowered = lowered_task(task, "prefill", max_turns=1)
    assert task_prompt_token_count(tokenizer, {"task_rollout_spec": lowered.model_dump_json()}, "prompt") == 5
    assert observed[0]["sampling_params"]["max_tokens"] == 3
    assert observed[0]["sampling_params"]["skip_special_tokens"] is False
    assert "extra_body" not in observed[0]["sampling_params"]
    assert output["responses"] == ["</think> </tool_call> trailing"]
    assert output["assistant_messages"] == [{"role": "assistant", "content": "</think> </tool_call> trailing"}]
    assert output["response_ids"] == [[8, 9, 10]]
    assert output["response_logprobs"] == [[-0.1, -0.2, -0.3]]
    assert output["student_topk_indices"] == [[[8], [9], [10]]]
    assert output["response_policy_steps"] == [[4, 4, 4]]


@pytest.mark.asyncio
@pytest.mark.parametrize("include_header,expected_ids", [(False, [1, 2]), (True, [3, 1, 2])])
async def test_prefill_admission_and_transport_share_configured_template_and_defaults(
    tmp_path, include_header, expected_ids
):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "assistant": 1, "prefix": 2, "header": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    original = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", chat_template="native template must not be used"
    )
    model_path = tmp_path / "model"
    original.save_pretrained(model_path)
    template_path = tmp_path / "serving.jinja"
    template_path.write_text(
        "{% if include_header %}header {% endif %}"
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}"
        "closed {% if add_generation_prompt %}new generation{% endif %}"
    )
    config = get_default_config()
    config.trainer.policy.model.tokenizer_path = str(model_path)
    config.generator.chat_template.source = "file"
    config.generator.chat_template.name_or_path = str(template_path)
    config.generator.chat_template_kwargs = {"include_header": include_header}
    tokenizer = tokenizer_from_config(config)
    task = text_task("configured-prefill").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix"),))}
    )
    spec = lowered_task(task, "prefill", max_turns=1)
    panel_path = tmp_path / "panel.parquet"
    write_tasks(panel_path, [spec])
    dataset = TaskDataset(
        [str(panel_path)],
        tokenizer,
        len(expected_ids),
        num_workers=1,
        chat_template_kwargs=dict(config.generator.chat_template_kwargs),
    )
    assert [row["input"].spec for row in dataset.collate_fn(list(dataset))] == [spec]

    class Engine:
        async def generate(self, request):
            return {
                "prompt_ids": request["prompt_token_ids"],
                "responses": ["continuation"],
                "response_ids": [[4]],
                "response_logprobs": [[-0.1]],
                "stop_reasons": ["stop"],
            }

    engine = Engine()
    engine.tokenizer = tokenizer
    prompt = prefill_prompt(task)
    output = await DirectModelClient(engine).generate(
        {
            "prompts": [list(prompt.messages)],
            "chat_completion_params": [
                {**prompt.options, "chat_template_kwargs": dict(config.generator.chat_template_kwargs)}
            ],
            "sampling_params": {"max_tokens": 1},
            "max_prompt_length": len(expected_ids),
            "max_context_length": 10,
        }
    )
    assert output["prompt_ids"] == [expected_ids]
    assert output["responses"] == ["continuation"]


def test_prefill_dataset_counts_effective_continuation_without_new_generation_prompt(tmp_path):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template="{% for m in messages %}{{ m.content }} {% endfor %}"
        "{% if add_generation_prompt %}new assistant generation prompt that exceeds the cap{% endif %}",
    )
    short = text_task("short-prefill").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix one"),))}
    )
    long = text_task("long-prefill").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix " * 20),))}
    )
    path = tmp_path / "prefill.parquet"
    write_tasks(path, [lowered_task(task, "prefill", max_turns=1) for task in (short, long)])
    dataset = TaskDataset([str(path)], tokenizer, 6, num_workers=1)
    (selected,) = dataset.collate_fn(list(dataset))
    spec = selected["input"].spec
    assert spec.task.id == "short-prefill"
    assert task_messages(spec) == [{"role": "assistant", "content": "prefix one"}]
