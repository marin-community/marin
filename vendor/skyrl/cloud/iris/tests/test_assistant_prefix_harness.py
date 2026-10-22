# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the assistant-prefix task runtime and saved-answer replay."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from rolloutengine.harness.assistant_prefix import assistant_prefix_prompt
from rolloutengine.models import ModelRequest, ModelTurn, RolloutContractError
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import TaskDataset, task_prompt_token_count, write_tasks
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.task_inputs import task_messages
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.types import (
    BatchMetadata,
    TaskRequestBatch,
    TokenProvenance,
    TrajectoryID,
)
from taskcompendium.grader import verifyit_package
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.grading import grade_answer
from shellbox.machine import Backend, Command, ExitReason, Result
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    FunctionDefinition,
    PlainText,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from tests.cpu.task_specs import NATIVE_ENVIRONMENT, lowered_task, task_input, text_task, machine_runtime
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from verifyit.closure_scoring import ClosureFormat
from verifyit.spec import NumericSpec, StructureClosureSpec, MathSpec, JudgeSpec
from verifyit.modes.grade_judge import JudgeConnection
from marinskyrl.saved_answers import (
    AnswerAdmission,
    SavedAnswer,
    admit_answer,
    answer_uri,
    regrade_once,
    regrade_answer,
    save_answer,
)


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
        ("12}", {"reasoning_content": "parsed reasoning"}, None),
        ("12}", {"tool_calls": [{"id": "call-1", "type": "function"}]}, None),
    ],
)
async def test_assistant_prefix_worker_preserves_continuation_and_rejects_parsed_channels(
    task_inputs, tmp_path, response, parsed_channel, reward
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
    spec = lowered_task(task, "assistant_prefix", max_turns=1)
    request = replace(request, rows=(task_input(spec),), batch_metadata=BatchMetadata(0, "eval"))
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
        config,
        WholeTaskProjection(config, WorkerTokenizer()),
        DirectModelClient(Engine()),
        {},
        shutdown_timeout=30,
        eval_saved_answers_root=str(tmp_path / "answers"),
    )
    try:
        (result,) = await worker.generate(request)
    finally:
        await worker.shutdown()
    assert observed[0]["prompt_token_ids"] == [[1, 2]]
    assert observed[0].get("prompts") is None and "tools" not in observed[0]
    assert observed[0]["sampling_params"]["skip_special_tokens"] is False
    assert result.grade.reward == reward
    path = Path(answer_uri(str(tmp_path / "answers"), "arithmetic"))
    saved = SavedAnswer.model_validate_json(path.read_bytes())
    admission = AnswerAdmission.model_validate_json(Path(str(path) + ".admission.json").read_bytes())
    if parsed_channel:
        assert result.grade.status != Outcome.GRADED
        assert not admission.eligible
        with pytest.raises(ValueError, match="not admitted"):
            await regrade_once(saved, admission, spec, {}, str(tmp_path / "regrades"), timeout=2)
        return
    assert saved.messages == result.messages
    assert admission.eligible
    assert result.messages[-1] == {"role": "assistant", "content": response}
    assert result.prompt_token_ids == (1, 2)
    assert result.response_token_ids == (11, 12, 13)
    assert result.loss_mask == (1, 1, 1)
    assert result.logprobs == (-0.1, -0.2, -0.3)


@pytest.mark.asyncio
@pytest.mark.parametrize("served_prompt", ["matching", "different"])
async def test_assistant_prefix_transport_renders_serving_tokens_and_retains_raw_structural_delimiters(served_prompt):
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
    task = text_task("assistant_prefix").model_copy(
        update={
            "context": ConversationInput(
                events=(TextMessage(role="user", content="question"), TextMessage(role="assistant", content="prefix"))
            ),
            "final_tools": (
                FunctionDefinition(name="function", description="tool-description", parameters={"type": "object"}),
            ),
        }
    )
    prompt = assistant_prefix_prompt(task)
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
    lowered = lowered_task(task, "assistant_prefix", max_turns=1)
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
@pytest.mark.parametrize("interrupted", [False, True])
async def test_assistant_prefix_real_client_preserves_prompt_and_raw_completion_across_retry(interrupted):
    backend = Tokenizer(
        models.WordLevel({"[UNK]": 0, "assistant": 1, "prefix": 2, "</think>": 3, "tail": 4}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=["</think>"],
        chat_template="{% for m in messages %}{{ m.role }} {{ m.content }} {% endfor %}closed",
    )
    task = text_task("real-client-assistant_prefix").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix"),))}
    )
    prompt = assistant_prefix_prompt(task)
    submitted = []

    class Engine:
        async def generate(self, request):
            submitted.append(request)
            partial = interrupted and len(submitted) == 1
            ids = [3] if partial else [4] if interrupted else [3, 4]
            return {
                "responses": [tokenizer.decode(ids, skip_special_tokens=False)],
                "response_ids": [ids],
                "response_logprobs": [[-token / 10 for token in ids]],
                "stop_reasons": ["abort" if partial else "stop"],
                "student_topk_indices": [[[token] for token in ids]],
                "behavior_topk_logprobs": [[[-token / 10] for token in ids]],
                "response_policy_steps": [[7] * len(ids)],
            }

    config = get_default_config()
    config.generator.enable_http_endpoint = False
    client = DirectModelClient(InferenceEngineClient([Engine()], tokenizer, config))
    output = await client.generate(
        {
            "prompts": [list(prompt.messages)],
            "chat_completion_params": [dict(prompt.options)],
            "session_ids": ["assistant_prefix"],
            "sampling_params": {"max_generate_length": 10, "logprobs": 1},
            "max_context_length": 5,
            "max_prompt_length": 2,
        }
    )
    assert [row["prompt_token_ids"] for row in submitted] == (
        [[[1, 2]], [[1, 2, 3]]] if interrupted else [[[1, 2]]]
    )
    assert [row["sampling_params"]["max_tokens"] for row in submitted] == ([3, 2] if interrupted else [3])
    assert all(row["sampling_params"]["skip_special_tokens"] is False for row in submitted)
    assert all(row["sampling_params"]["include_stop_str_in_output"] is True for row in submitted)
    assert output["prompt_ids"] == [[1, 2]]
    assert output["response_ids"] == [[3, 4]]
    assert output["response_logprobs"] == [[-0.3, -0.4]]
    assert output["student_topk_indices"] == [[[3], [4]]]
    assert output["behavior_topk_logprobs"] == [[[-0.3], [-0.4]]]
    assert output["response_policy_steps"] == [[7, 7]]
    assert output["responses"] == ["</think> tail"]
    assert output["assistant_messages"] == [{"role": "assistant", "content": "</think> tail"}]
    assert output["stop_reasons"] == ["stop"]
    assert output["token_provenance"] == TokenProvenance.ENGINE


@pytest.mark.asyncio
async def test_assistant_prefix_admission_and_transport_share_configured_template_and_defaults(tmp_path):
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
    config.generator.chat_template_kwargs = {"include_header": True}
    tokenizer = tokenizer_from_config(config)
    task = text_task("configured-assistant_prefix").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix"),))}
    )
    spec = lowered_task(task, "assistant_prefix", max_turns=1)
    panel_path = tmp_path / "panel.parquet"
    write_tasks(panel_path, [spec])
    dataset = TaskDataset(
        [str(panel_path)],
        tokenizer,
        3,
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
    prompt = assistant_prefix_prompt(task)
    output = await DirectModelClient(engine).generate(
        {
            "prompts": [list(prompt.messages)],
            "chat_completion_params": [
                {**prompt.options, "chat_template_kwargs": dict(config.generator.chat_template_kwargs)}
            ],
            "sampling_params": {"max_tokens": 1},
            "max_prompt_length": 3,
            "max_context_length": 10,
        }
    )
    assert output["prompt_ids"] == [[3, 1, 2]]
    assert output["responses"] == ["continuation"]


def test_assistant_prefix_dataset_counts_effective_continuation_without_new_generation_prompt(tmp_path):
    backend = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template="{% for m in messages %}{{ m.content }} {% endfor %}"
        "{% if add_generation_prompt %}new assistant generation prompt that exceeds the cap{% endif %}",
    )
    short = text_task("short-assistant_prefix").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix one"),))}
    )
    long = text_task("long-assistant_prefix").model_copy(
        update={"context": ConversationInput(events=(TextMessage(role="assistant", content="prefix " * 20),))}
    )
    path = tmp_path / "assistant_prefix.parquet"
    write_tasks(path, [lowered_task(task, "assistant_prefix", max_turns=1) for task in (short, long)])
    dataset = TaskDataset([str(path)], tokenizer, 6, num_workers=1)
    (selected,) = dataset.collate_fn(list(dataset))
    spec = selected["input"].spec
    assert spec.task.id == "short-assistant_prefix"
    assert task_messages(spec) == [{"role": "assistant", "content": "prefix one"}]


@pytest.mark.asyncio
async def test_canceled_saved_answer_attempt_cannot_spend_a_second_grade(task_inputs, tmp_path, monkeypatch):
    _, request = task_inputs
    spec = request.rows[0].spec
    spec = spec.model_copy(
        update={
            "task": spec.task.model_copy(
                update={
                    "answer_type": AnswerType.TEXT,
                    "context": ConversationInput(events=(TextMessage(role="assistant", content='{"value":'),)),
                    "grader": verifyit_package(
                        StructureClosureSpec('{"value":', '{"value":12}', ClosureFormat.JSON)
                    ).grader,
                }
            ),
            "harness": "assistant_prefix",
            "limits": spec.limits.model_copy(update={"max_turns": 1}),
        }
    )
    messages = ({"role": "assistant", "content": '{"value":'},)
    turn = ModelTurn({"role": "assistant", "content": "12}"}, (1,), (2,), None, "stop", text="12}")
    root = str(tmp_path / "answers")
    save_answer(root, spec, "arithmetic", 0, ModelRequest(messages, dict(assistant_prefix_prompt(spec.task).options)), turn)
    admit_answer(root, "arithmetic", 0, (*messages, turn.message), "grade", "TimeoutError", "infra_error")
    path = Path(answer_uri(root, "arithmetic"))
    record = SavedAnswer.model_validate_json(path.read_bytes())
    admission = AnswerAdmission.model_validate_json(Path(str(path) + ".admission.json").read_bytes())
    entered, released = asyncio.Event(), asyncio.Event()
    calls = []

    async def pending_original_grade(*args, **kwargs):
        calls.append(1)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            released.set()

    monkeypatch.setattr("marinskyrl.saved_answers.regrade_answer", pending_original_grade)
    attempt = asyncio.create_task(regrade_once(record, admission, spec, {}, str(tmp_path / "grades"), timeout=2))
    await entered.wait()
    attempt.cancel()
    with pytest.raises(asyncio.CancelledError):
        await attempt
    assert released.is_set()
    resumed = await regrade_once(record, admission, spec, {}, str(tmp_path / "grades"), timeout=2)
    assert resumed["status"] == Outcome.UNAVAILABLE and resumed["reward"] is None
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["ready", "blocked"])
@pytest.mark.parametrize("harness", ["assistant_prefix", "shellbox"])
async def test_grader_failure_keeps_durable_answer_for_one_retry_without_generation(
    task_inputs, tmp_path, monkeypatch, storage, harness
):
    config, request = task_inputs
    spec = request.rows[0].spec
    task = spec.task.model_copy(
        update={
            "context": ConversationInput(
                events=(*spec.task.context.events, TextMessage(role="assistant", content='{"value":'))
            ),
            "answer_type": AnswerType.TEXT,
            "grader": verifyit_package(StructureClosureSpec('{"value":', '{"value":12}', ClosureFormat.JSON)).grader,
        }
    )
    spec = lowered_task(task, harness, max_turns=1)
    request = replace(request, rows=(task_input(spec),), batch_metadata=BatchMetadata(0, "eval"))
    root = str(tmp_path / "answers")
    if storage == "blocked":
        Path(root).write_text("storage unavailable")
    path = Path(answer_uri(root, "arithmetic"))
    generated = []
    grades = []

    class Engine:
        tokenizer = WorkerTokenizer()

        async def generate(self, model_request):
            generated.append(model_request)
            return {
                "responses": ["12}"],
                "token_provenance": TokenProvenance.ENGINE,
                "assistant_messages": [{"role": "assistant", "content": "12}"}],
                "prompt_ids": model_request.get("prompt_token_ids", [[1, 2]]),
                "response_ids": [[11, 12]],
                "response_logprobs": [[-0.1, -0.2]],
                "stop_reasons": ["stop"],
            }

    def original_grader(task, attempt):
        # The original grade cannot begin until the answer survived an exclusive filesystem write.
        record = SavedAnswer.model_validate_json(path.read_bytes())
        assert record.messages[-1]["content"] == "12}"
        grades.append(task.id)
        if len(grades) == 1:
            raise RuntimeError("temporary grader failure")
        return grade_answer(task, attempt)

    monkeypatch.setattr("rolloutengine.grading.grade_answer", original_grader)
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, WorkerTokenizer()),
        Engine(),
        {},
        shutdown_timeout=30,
        eval_saved_answers_root=root,
    )
    try:
        if storage == "blocked":
            with pytest.raises(ExceptionGroup) as failure:
                await worker.generate(request)
            assert failure.value.subgroup(OSError) is not None
            assert len(generated) == 1 and not grades
            return
        (failed,) = await worker.generate(request)
    finally:
        await worker.shutdown()
    assert failed.grade.reward is None and failed.failure is not None
    record = SavedAnswer.model_validate_json(path.read_bytes())
    admission = AnswerAdmission.model_validate_json(Path(str(path) + ".admission.json").read_bytes())
    receipt = await regrade_once(record, admission, spec, {}, str(tmp_path / "regrades"), timeout=2)
    assert receipt["status"] == Outcome.GRADED and receipt["reward"] == 1
    assert record.response_token_ids == (11, 12)
    # Reading an existing admitted result cannot spend the same retry again.
    assert await regrade_once(record, admission, spec, {}, str(tmp_path / "regrades"), timeout=2) == receipt
    assert len(generated) == 1 and len(grades) == 2
    changed_request = record.model_copy(update={"request_options": {"unadmitted": "changed"}})
    rejected = await regrade_answer(changed_request, spec, {}, timeout=2)
    assert rejected.status == Outcome.INFRA_ERROR and rejected.reward is None
    assert len(generated) == 1 and len(grades) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [MathSpec("12"), JudgeSpec(references=("12",), model="fixture")])
async def test_recovery_admission_rejects_unbounded_host_grading_before_model(task_inputs, tmp_path, unsupported):
    config, request = task_inputs
    spec = request.rows[0].spec
    task = spec.task.model_copy(
        update={
            "answer_type": AnswerType.TEXT,
            "context": ConversationInput(events=(TextMessage(role="assistant", content="Answer:"),)),
            "grader": verifyit_package(
                unsupported, environment=NATIVE_ENVIRONMENT if isinstance(unsupported, JudgeSpec) else None
            ).grader,
        }
    )
    spec = spec.model_copy(
        update={"task": task, "harness": "assistant_prefix", "limits": spec.limits.model_copy(update={"max_turns": 1})}
    )
    request = replace(request, rows=(task_input(spec),), batch_metadata=BatchMetadata(0, "eval"))
    calls = []

    class Engine:
        tokenizer = WorkerTokenizer()

        async def generate(self, request):
            calls.append(request)
            raise AssertionError("Unsupported recovery must fail admission before inference")

    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(config, WorkerTokenizer()),
        DirectModelClient(Engine()),
        {},
        shutdown_timeout=30,
        eval_saved_answers_root=str(tmp_path / "answers"),
        judge_connection=JudgeConnection("https://judge.invalid/v1", "unused"),
    )
    try:
        with pytest.raises(ExceptionGroup) as failure:
            await worker.generate(request)
        assert failure.value.subgroup(ValueError) is not None
    finally:
        await worker.shutdown()
    assert calls == []
    assert not (tmp_path / "answers").exists()


@pytest.mark.asyncio
async def test_saved_answer_replay_preserves_verifier_user_and_bounds_hung_close(task_inputs, tmp_path, monkeypatch):
    _, request = task_inputs
    original = request.rows[0].spec
    task = original.task.model_copy(
        update={
            "answer_type": AnswerType.TEXT,
            "context": ConversationInput(events=(TextMessage(role="assistant", content="Answer:"),)),
            "grader": ScriptGrader(argv=("true",), environment=NATIVE_ENVIRONMENT),
        }
    )
    spec = original.model_copy(
        update={
            "task": task,
            "harness": "assistant_prefix",
            "verifier_machine": machine_runtime("fixture").model_copy(update={"user": "123", "cleanup_timeout": 0.01}),
            "limits": original.limits.model_copy(update={"max_turns": 1}),
        }
    )
    prompt = assistant_prefix_prompt(task)
    root = str(tmp_path / "answers")
    turn = ModelTurn({"role": "assistant", "content": "12"}, (1,), (2,), None, "stop", text="12")
    save_answer(root, spec, "arithmetic", 0, ModelRequest(prompt.messages, dict(prompt.options)), turn)
    record = SavedAnswer.model_validate_json(Path(answer_uri(root, "arithmetic")).read_bytes())
    users = []
    closing, canceled = asyncio.Event(), asyncio.Event()

    class Machine:
        async def run(self, command):
            users.append(command.user)
            return Result(0, b"", b"", False, False, ExitReason.EXITED)

        async def close(self):
            closing.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()

    class Factory:
        backend = Backend.DAYTONA

        def check_environment(self, requirements, spec):
            pass

        async def create(self, spec):
            return Machine()

    async def original_sandbox_grade(task, attempt, factory, machine_spec, **kwargs):
        machine = await factory.create(machine_spec)
        try:
            await machine.run(Command(("true",)))
            return GradeResult(Outcome.GRADED, 1.0)
        finally:
            await machine.close()

    monkeypatch.setattr("rolloutengine.grading.grade_in_sandbox", original_sandbox_grade)
    async with asyncio.timeout(3):
        grade = await regrade_answer(record, spec, {"fixture": Factory()}, timeout=2)
        await canceled.wait()
    assert users == ["123"] and closing.is_set()
    assert grade.reward == 1.0
    assert grade.diagnostics["cleanup_errors"] == [{"operation": "machine_close", "exception_type": "TimeoutError"}]
