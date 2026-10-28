# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView
from fray.local_backend import LocalClient
from marin.evaluation.likelihood import LikelihoodTransportError, VllmLikelihood
from marin.evaluation.selection import SELECTIONS_TABLE
from marin.evaluation.task_driver import (
    ATTEMPTS_TABLE,
    RUN_OBJECT,
    EvaluationRunConfig,
    JudgeIdentity,
    ModelIdentity,
    run_evaluation,
)
from rolloutengine.chat import ChatModel
from rolloutengine.completion import CompletionModel
from rolloutengine.models import ModelRequest, ModelTurn, RolloutLimits, TokenUsage
from rolloutengine.test_support import machine_runtime
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import PackageSpec
from taskcompendium.convert.answers import answer_task, exact_answer_task
from taskcompendium.convert.evaluation import source_evidence
from taskcompendium.evaluation import (
    ChatGeneration,
    ChatPromptVariant,
    ChoiceLikelihood,
    ContinuationLikelihood,
    EvaluationItem,
    EvaluationProtocol,
    LikelihoodMetric,
    LikelihoodNormalization,
    LikelihoodRequest,
    Metric,
    MetricKind,
    Sampling,
    TextGeneration,
    TrialPolicy,
)
from taskcompendium.models import (
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExitCodeReward,
    ScriptGrader,
    TextMessage,
)
from taskcompendium.pipeline.evaluation import chat_item, publish_evaluation_corpus, source_order
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, RawRow, SourceRecipe
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import (
    EmptyOutputPolicy,
    ExactSpec,
    HarnessGenerationProfile,
    HarnessGenerationSpec,
    JudgeSpec,
    MathJudgeSpec,
    MathProfile,
    MathSpec,
)
from zephyr.context import ZephyrContext


def convert(row: RawRow, _context: ConversionContext):
    task = exact_answer_task(row, prompt=row.data["question"], answers=("red",), ignore_case=False)
    return task.model_copy(update={"tags": ("eval",)})


@pytest.fixture
def published_corpora(tmp_path):
    client = LocalClient()
    roots = []
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            for benchmark in ("colors", "paints"):
                source = tmp_path / benchmark
                source.mkdir()
                (source / "rows.jsonl").write_text(
                    "".join(json.dumps({"question": f"{benchmark}/{index}"}) + "\n" for index in range(3))
                )
                recipe = SourceRecipe(
                    name=benchmark,
                    version="1",
                    source=SourceFiles(benchmark, "pinned", ("rows.jsonl",), SourceFormat.JSONL),
                    convert=convert,
                    rubric=None,
                    controls=None,
                    intended_use=IntendedUse.EVAL,
                )
                protocol = EvaluationProtocol(
                    benchmark=benchmark,
                    version="1",
                    operation="chat_generation",
                    sampling=Sampling(
                        temperature=0.7, top_p=1, top_k=None, max_tokens=8, stop=(), chat_template_kwargs={}
                    ),
                    trials=TrialPolicy(count=2, seed=7),
                    metrics=(Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),),
                    primary_metric="accuracy",
                    reducer="mean_v1",
                    expected_population=3,
                )
                output = tmp_path / (benchmark + "-published")
                publish_evaluation_corpus(recipe, protocol, chat_item, context, str(source), str(output))
                roots.append(str(output))
        return tuple(roots)
    finally:
        client.shutdown()


@pytest.fixture
def run_config():
    return EvaluationRunConfig(
        model=ModelIdentity(
            name="fixture",
            revision="model-commit",
            tokenizer="fixture",
            tokenizer_revision="tokenizer-commit",
            context_window=1024,
            serving={"backend": "fake"},
        ),
        max_tokens=16,
        concurrency=2,
        source_limit=2,
        limits=RolloutLimits(
            max_turns=1,
            max_total_tokens=None,
            model_turn_timeout=10,
            tool_turn_timeout=10,
            total_turn_timeout=20,
            attempt_timeout=30,
            verifier_timeout=10,
            cleanup_timeout=10,
        ),
    )


@pytest.mark.asyncio
async def test_driver_preserves_cohort_trials_and_resumes_failed_attempt_without_resampling(
    tmp_path, published_corpora, run_config
):
    requests = []

    async def model(request: ModelRequest) -> ModelTurn:
        requests.append(request)
        if len(requests) == 1:
            raise ConnectionError("temporary serving failure")
        return ModelTurn(
            message={"role": "assistant", "content": "red"},
            prompt_token_ids=(),
            response_token_ids=(),
            logprobs=None,
            stop_reason="stop",
            text="red",
            usage=TokenUsage(3, 1),
        )

    output = str(tmp_path / "results")
    with pytest.raises(RuntimeError, match="raw evidence is saved"):
        await run_evaluation(published_corpora, output, run_config, model, machine_factories={})
    first = ReadView(output)
    failed = [row for row in first.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "infra_error"]
    assert len(failed) == 1
    assert len(requests) == 8
    assert {(request.messages[0]["content"], request.options["seed"]) for request in requests} == {
        (f"{benchmark}/{index}", seed) for benchmark in ("colors", "paints") for index in range(2) for seed in (7, 8)
    }
    await run_evaluation(published_corpora, output, run_config, model, machine_factories={})
    assert len(requests) == 9
    assert requests[-1] == requests[0]
    final = ReadView(output)
    terminal = [row for row in final.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "graded"]
    assert len(terminal) == 8
    assert [(row["retry"], row["seed"]) for row in terminal if row["retry"]] == [(1, 7)]
    samples = [sample_from_archive_row(row) for row in final.iter_rows(ARCHIVE_SAMPLES_TABLE)]
    assert sum(sample.metrics.get("accuracy", 0) for sample in samples) == 8
    assert all(sample.prompt_messages[0].content.endswith(("/0", "/1")) for sample in samples)
    assert json.loads(final.read_blob(RUN_OBJECT))["config"]["source_limit"] == 2
    assert not any("summary" in key for key in json.loads(final.read_blob(RUN_OBJECT)))


@pytest.mark.asyncio
async def test_driver_resume_rejects_a_different_model_or_selection(tmp_path, published_corpora, run_config):
    async def model(_request):
        return ModelTurn({"role": "assistant", "content": "red"}, (), (), None, "stop", usage=TokenUsage(3, 1))

    output = str(tmp_path / "results")
    await run_evaluation(published_corpora, output, run_config, model, machine_factories={})
    with pytest.raises(ValueError, match="Resume requires"):
        await run_evaluation(
            published_corpora, output, run_config.model_copy(update={"source_limit": 1}), model, machine_factories={}
        )


def convert_mixed_row(row: RawRow, _context: ConversionContext):
    spec = (
        HarnessGenerationSpec(HarnessGenerationProfile.TRIVIAQA, (("Paris",),))
        if row.data["kind"] == "raw"
        else ExactSpec((row.data["expected"],), ignore_case=False, empty_output=EmptyOutputPolicy.GRADE)
    )
    task = answer_task(row, prompt=row.data["prompt"], spec=spec, evidence=row.data)
    if row.data["kind"] == "sandbox":
        task = task.model_copy(
            update={
                "grader": ScriptGrader(
                    argv=("sh", "-c", 'test "$(cat /app/answer.txt)" = red'),
                    answer_path="/app/answer.txt",
                    reward=ExitCodeReward(),
                    environment=EnvironmentRequirements(
                        command_semantics=CommandSemantics.SHELL_SIMULATOR,
                        package_spec=PackageSpec(),
                    ),
                )
            }
        )
    if row.data.get("history"):
        task = task.model_copy(
            update={
                "context": ConversationInput(
                    events=(
                        TextMessage(role="user", content="A historical question"),
                        TextMessage(role="assistant", content="blue"),
                        *task.context.events,
                    )
                )
            }
        )
    return task.model_copy(update={"tags": ("eval",)})


def mixed_item(task):
    source = source_evidence(task)
    kind = source["kind"]
    if kind == "raw":
        operation = TextGeneration(context=source["prompt"])
    elif kind == "choice":
        operation = ChoiceLikelihood(
            choices=("A", "BBBB"),
            gold=(1,),
            requests=tuple(LikelihoodRequest(context=" café ", continuation=choice) for choice in ("A", "BBBB")),
        )
    elif kind == "continuation":
        operation = ContinuationLikelihood(
            request=LikelihoodRequest(context="L", continuation="x" * 64 if source.get("unscorable") else " word")
        )
    else:
        operation = ChatGeneration(response_selection=source["selection"], historical_assistant_reasoning="empty")
    return EvaluationItem(
        task_id=task.id, source_row=task.source.row, source_ordinal=source_order(task.source.row)[1], operation=operation
    )


@pytest.fixture
def mixed_corpora(tmp_path):
    rows = {
        "chat": [
            {
                "kind": "chat",
                "prompt": "chat/final",
                "expected": "red",
                "selection": "final_or_complete_reasoning",
                "history": True,
            },
            {"kind": "chat", "prompt": "chat/reasoning", "expected": "red", "selection": "final_or_complete_reasoning"},
            {"kind": "chat", "prompt": "chat/truncated", "expected": "red", "selection": "final_or_complete_reasoning"},
            {"kind": "chat", "prompt": "chat/combine", "expected": "red\n\nblue", "selection": "combine"},
        ],
        "raw": [{"kind": "raw", "prompt": name} for name in ("raw/contract", "raw/bare")],
        "choice": [{"kind": "choice", "prompt": " café ", "expected": "BBBB"}],
        "continuation": [
            {"kind": "continuation", "prompt": "L", "expected": " word"},
            {"kind": "continuation", "prompt": "unscorable", "expected": "", "unscorable": True},
        ],
        "sandbox": [{"kind": "sandbox", "prompt": "chat/sandbox", "expected": "red", "selection": "combine"}],
    }
    roots = []
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "mixed-chunks")) as context:
            for benchmark, documents in rows.items():
                primary = tmp_path / (benchmark + "-source")
                primary.mkdir()
                (primary / "rows.jsonl").write_text("".join(json.dumps(doc) + "\n" for doc in documents))
                recipe = SourceRecipe(
                    name=benchmark,
                    version="1",
                    source=SourceFiles(benchmark, "immutable", ("rows.jsonl",), SourceFormat.JSONL),
                    convert=convert_mixed_row,
                    rubric=None,
                    controls=None,
                    intended_use=IntendedUse.EVAL,
                )
                likelihoods = ()
                if benchmark == "choice":
                    names = ("acc,none", "acc_norm,none")
                    likelihoods = tuple(
                        LikelihoodMetric(name=name, normalization=normalization)
                        for name, normalization in zip(
                            names, (LikelihoodNormalization.RAW, LikelihoodNormalization.CHARACTERS), strict=True
                        )
                    )
                elif benchmark == "continuation":
                    names = ("acc,none", "perplexity,none")
                    likelihoods = tuple(
                        LikelihoodMetric(name=name, normalization=LikelihoodNormalization.RAW, scoring=scoring)
                        for name, scoring in zip(names, ("greedy_accuracy", "log_likelihood"), strict=True)
                    )
                else:
                    names = (
                        ("accuracy",)
                        if benchmark in {"chat", "sandbox"}
                        else ("exact_match,strict_answer", "exact_match,extract_answer")
                    )
                protocol = EvaluationProtocol(
                    benchmark=benchmark,
                    version="1",
                    operation={
                        "chat": "chat_generation",
                        "raw": "text_generation",
                        "choice": "choice_likelihood",
                        "continuation": "continuation_likelihood",
                        "sandbox": "chat_generation",
                    }[benchmark],
                    sampling=(
                        Sampling(
                            temperature=0.7 if benchmark == "chat" else 0,
                            top_p=None if benchmark == "sandbox" else 1,
                            top_k=9 if benchmark == "chat" else None,
                            max_tokens=8 if benchmark == "chat" else 6,
                            stop=("STOP",) if benchmark == "chat" else ("\nQuestion:",),
                            chat_template_kwargs={"enable_thinking": False} if benchmark == "chat" else {},
                        )
                        if benchmark in ("chat", "raw", "sandbox")
                        else None
                    ),
                    trials=TrialPolicy(
                        count=2 if benchmark == "chat" else 1,
                        seed=7 if benchmark == "chat" else 11 if benchmark in {"choice", "continuation"} else None,
                    ),
                    metrics=tuple(
                        Metric(
                            name=name,
                            kind=MetricKind.CONTINUOUS if name.startswith("perplexity") else MetricKind.BINARY,
                            higher_is_better=not name.startswith("perplexity"),
                        )
                        for name in names
                    ),
                    primary_metric=names[0],
                    reducer="raw_fixture_v1",
                    expected_population=len(documents),
                    likelihood_metrics=likelihoods,
                )
                output = tmp_path / (benchmark + "-mixed-corpus")
                publish_evaluation_corpus(recipe, protocol, mixed_item, context, str(primary), str(output))
                roots.append(str(output))
    finally:
        client.shutdown()
    return tuple(roots)


class CharacterTokenizer:
    eos_token_id = 0

    def encode(self, text, *, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, **kwargs):
        return self.encode("".join(message["content"] for message in conversation))


def policy_item(task):
    source = source_evidence(task)
    operation = ChatGeneration()
    if "zero_shot" in source:
        operation = ChatGeneration(
            variants=(
                ChatPromptVariant(
                    context=ConversationInput(events=(TextMessage(role="user", content=source["zero_shot"]),)),
                    fewshot=0,
                ),
                ChatPromptVariant(context=task.context, fewshot=1),
            ),
            prompt_selection="fit_context",
        )
    return EvaluationItem(
        task_id=task.id, source_row=task.source.row, source_ordinal=source_order(task.source.row)[1], operation=operation
    )


def publish_policy_corpus(tmp_path, rows, converter=convert_mixed_row):
    source = tmp_path / "source"
    source.mkdir()
    (source / "rows.jsonl").write_text("".join(json.dumps({"kind": "raw", **row}) + "\n" for row in rows))
    recipe = SourceRecipe(
        name="policy",
        version="1",
        source=SourceFiles("policy", "immutable", ("rows.jsonl",), SourceFormat.JSONL),
        convert=converter,
        rubric=None,
        controls=None,
        intended_use=IntendedUse.EVAL,
    )
    names = ("exact_match,strict_answer", "exact_match,extract_answer")
    protocol = EvaluationProtocol(
        benchmark="policy",
        version="1",
        operation="chat_generation",
        sampling=Sampling(temperature=0, top_p=1, top_k=None, max_tokens=800, stop=(), chat_template_kwargs={}),
        trials=TrialPolicy(count=1, seed=None),
        metrics=tuple(Metric(name=name, kind=MetricKind.BINARY, higher_is_better=True) for name in names),
        primary_metric=names[0],
        reducer="fixture_v1",
        expected_population=len(rows),
    )
    output = str(tmp_path / "corpus")
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, policy_item, context, str(source), output)
    finally:
        client.shutdown()
    return output


@pytest.mark.asyncio
async def test_driver_zero_shot_fallback_reduces_output_cap_and_records_oversized_prompt(tmp_path, run_config):
    corpus = publish_policy_corpus(
        tmp_path,
        [
            {"prompt": "A" * 500, "zero_shot": "a" * 300},
            {"prompt": "B" * 1100, "zero_shot": "b" * 1000},
            {"prompt": "C" * 200, "zero_shot": "c" * 100},
            {"prompt": "D" * 199, "zero_shot": "d" * 50},
        ],
    )
    requests = []

    async def model(request):
        requests.append(request)
        return ModelTurn({"role": "assistant", "content": "Paris"}, (), (), None, "stop", usage=TokenUsage(10, 1))

    config = run_config.model_copy(
        update={
            "model": run_config.model.model_copy(update={"context_window": 1000}),
            "max_tokens": 800,
            "source_limit": None,
        }
    )
    output = str(tmp_path / "results")
    await run_evaluation((corpus,), output, config, model, machine_factories={}, tokenizer=CharacterTokenizer())
    # Source MMLU-Pro stops at zero shots, then endpoint preflight reserves 64 tokens.
    assert {(request.messages[0]["content"], request.options["max_tokens"]) for request in requests} == {
        ("a" * 300, 636),
        ("c" * 100, 800),
        ("D" * 199, 737),
    }
    reader = ReadView(output)
    ledger = sorted(reader.iter_rows(SELECTIONS_TABLE), key=lambda row: row["source_ordinal"])
    assert [(row["status"], row["reason"]) for row in ledger] == [
        ("selected", None),
        ("excluded", "context_window"),
        ("selected", None),
        ("selected", None),
    ]
    samples = [sample_from_archive_row(row) for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE)]
    assert len(samples) == 3
    assert all(sample.metrics == {"exact_match,strict_answer": 0, "exact_match,extract_answer": 1} for sample in samples)


@pytest.mark.asyncio
async def test_driver_multimetric_submission_failure_saves_raw_response_without_component_scores(tmp_path, run_config):
    corpus = publish_policy_corpus(tmp_path, [{"prompt": "Return an answer"}])
    raw_message = {
        "role": "assistant",
        "content": "Paris",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "unexpected", "arguments": "{}"}}],
    }
    requests = []

    async def model(request):
        requests.append(request)
        return ModelTurn(raw_message, (), (), None, "tool_calls", usage=TokenUsage(10, 1))

    output = str(tmp_path / "results")
    await run_evaluation((corpus,), output, run_config, model, machine_factories={})
    reader = ReadView(output)
    terminal = [row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["phase"] == "finished"]
    assert len(terminal) == 1
    assert terminal[0]["outcome"] == "submission_failure"
    raw = json.loads(reader.read_blob(terminal[0]["result_blob"]))
    assert raw["conversations"][0]["steps"][0]["turn"]["metadata"]["evaluation_raw_message"] == raw_message
    assert raw["grade"]["status"] == "submission_failure"
    samples = [sample_from_archive_row(row) for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE)]
    assert len(samples) == 1
    assert samples[0].metrics == {}
    assert samples[0].grading.score == 0
    assert samples[0].output == "Paris"
    await run_evaluation((corpus,), output, run_config, model, machine_factories={})
    assert len(requests) == 1


def convert_judged_row(row, _context):
    judge = JudgeSpec(
        references=("2",),
        model=row.data.get("model", ""),
        rubric="labels",
        exact_gate=False,
        prompt_template="Reference: {reference}\nCandidate: {candidate}",
        label_scores={"A": 1, "B": 0},
        label_metrics={
            label: {"exact_match,strict_answer": score, "exact_match,extract_answer": score}
            for label, score in (("A", 1), ("B", 0))
        },
        label_scan="whole",
        label_case="upper",
    )
    verifier = (
        MathJudgeSpec(math=MathSpec(expected="2", profile=MathProfile.EVALCHEMY_OLYMPIAD), judge=judge)
        if row.data.get("math")
        else judge
    )
    task = answer_task(row, prompt=row.data["prompt"], spec=verifier, evidence=row.data)
    return task.model_copy(update={"tags": ("eval",)})


@pytest.fixture
def judge_endpoint(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = (
                {"error": {"message": "fixture unavailable"}}
                if len(requests) == 1
                else {
                    "id": "judge-response",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "judge-served-revision",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "A"}}],
                }
            )
            payload = json.dumps(body).encode()
            self.send_response(503 if len(requests) == 1 else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield requests, JudgeConnection(f"http://127.0.0.1:{server.server_port}/v1", "private-judge-key")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.asyncio
async def test_driver_judge_identity_pins_explicit_and_default_models_across_failed_attempt_resume(
    tmp_path, run_config, judge_endpoint, monkeypatch
):
    monkeypatch.setenv("VERIFYIT_JUDGE_MODEL", "ambient-must-not-win")
    corpus = publish_policy_corpus(
        tmp_path,
        [
            {"prompt": "explicit judge", "model": "source-judge"},
            {"prompt": "default judge"},
            {"prompt": "nested math judge", "math": True},
        ],
        converter=convert_judged_row,
    )
    judge_requests, connection = judge_endpoint
    config = run_config.model_copy(
        update={
            "concurrency": 1,
            "source_limit": None,
            "judge": JudgeIdentity(
                deployment="fixture-deployment",
                models={"source-judge": "source-revision", "run-default": "default-revision"},
                default_model="run-default",
            ),
        }
    )
    model_requests = []

    async def model(request):
        model_requests.append(request)
        return ModelTurn({"role": "assistant", "content": r"\boxed{4}"}, (), (), None, "stop", usage=TokenUsage(10, 1))

    output = str(tmp_path / "results")
    for unpaired_config, unpaired_connection in ((run_config, connection), (config, None)):
        with pytest.raises(ValueError, match="both an explicit connection"):
            await run_evaluation(
                (corpus,), output, unpaired_config, model, machine_factories={}, judge_connection=unpaired_connection
            )
    assert model_requests == judge_requests == []
    with pytest.raises(RuntimeError, match="raw evidence is saved"):
        await run_evaluation((corpus,), output, config, model, machine_factories={}, judge_connection=connection)
    reader = ReadView(output)
    failed = [row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "infra_error"]
    assert len(failed) == 1
    failed_raw = json.loads(reader.read_blob(failed[0]["result_blob"]))
    assert failed_raw["grade"]["reward"] is None
    assert "fixture unavailable" in json.dumps(failed_raw["grade"]["detail"])
    assert [request["model"] for request in judge_requests] == ["source-judge", "run-default", "run-default"]
    await run_evaluation((corpus,), output, config, model, machine_factories={}, judge_connection=connection)
    assert [request["model"] for request in judge_requests] == [
        "source-judge",
        "run-default",
        "run-default",
        "source-judge",
    ]
    assert len(model_requests) == 4
    final_reader = ReadView(output)
    graded = [row for row in final_reader.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "graded"]
    assert len(graded) == 3
    grade_details = [json.loads(final_reader.read_blob(row["result_blob"]))["grade"]["detail"] for row in graded]
    assert {detail["requested_model"] for detail in grade_details} == {"source-judge", "run-default"}
    assert {detail["response_model"] for detail in grade_details} == {"judge-served-revision"}
    descriptor = final_reader.read_blob(RUN_OBJECT).decode()
    assert json.loads(descriptor)["config"]["judge"] == config.judge.model_dump(mode="json")
    assert connection.base_url not in descriptor
    assert connection.api_key not in descriptor
    assert "ambient-must-not-win" not in descriptor
    changed = config.model_copy(
        update={
            "judge": config.judge.model_copy(
                update={
                    "models": {
                        "source-judge": "changed-revision",
                        "run-default": "default-revision",
                    }
                }
            )
        }
    )
    with pytest.raises(ValueError, match="Resume requires"):
        await run_evaluation((corpus,), output, changed, model, machine_factories={}, judge_connection=connection)
    assert len(model_requests) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("source_model", ["source-judge", ""])
async def test_driver_unmapped_judge_model_is_rejected_before_generation(
    tmp_path, run_config, judge_endpoint, monkeypatch, source_model
):
    monkeypatch.setenv("VERIFYIT_JUDGE_MODEL", "different")
    corpus = publish_policy_corpus(
        tmp_path, [{"prompt": "unmapped judge", "model": source_model}], converter=convert_judged_row
    )
    judge_requests, connection = judge_endpoint
    config = run_config.model_copy(
        update={"judge": JudgeIdentity(deployment="fixture", models={"different": "revision"})}
    )
    model_requests = []

    async def model(request):
        model_requests.append(request)
        return ModelTurn({"role": "assistant", "content": "answer"}, (), (), None, "stop", usage=TokenUsage(10, 1))

    with pytest.raises(ExceptionGroup) as missing:
        await run_evaluation((corpus,), str(tmp_path / "missing"), run_config, model, machine_factories={})
    assert any("requires an explicit judge identity" in str(error) for error in missing.value.exceptions)
    with pytest.raises(ExceptionGroup, match="TaskGroup") as failure:
        await run_evaluation(
            (corpus,), str(tmp_path / "results"), config, model, machine_factories={}, judge_connection=connection
        )
    assert any("no recorded immutable revision" in str(error) for error in failure.value.exceptions)
    assert model_requests == judge_requests == []


@dataclass
class MixedEndpoint:
    requests: list[tuple[str, dict]] = field(default_factory=list)
    completed: list[tuple[str, int | None]] = field(default_factory=list)
    release_first: asyncio.Event = field(default_factory=asyncio.Event)
    malformed_remaining: bool = True

    async def handle(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path.endswith("/chat/completions"):
            prompt = body["messages"][-1]["content"]
            if prompt == "chat/final" and body["seed"] == 7:
                await self.release_first.wait()
            elif prompt == "chat/final" and body["seed"] == 8:
                self.release_first.set()
            messages = {
                "chat/final": {"content": "red", "reasoning_content": "wrong"},
                "chat/reasoning": {"content": None, "reasoning": "red"},
                "chat/truncated": {"content": None, "reasoning_content": "red"},
                "chat/combine": {"content": "blue", "reasoning_content": "red"},
                "chat/sandbox": {"content": "red"},
            }
            result = {
                "id": f"chat-{len(self.requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length" if prompt == "chat/truncated" else "stop",
                        "message": {"role": "assistant", **messages[prompt]},
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            }
            self.completed.append((prompt, body.get("seed")))
        elif isinstance(body["prompt"], str):
            result = {
                "id": f"raw-{len(self.requests)}",
                "object": "text_completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "text": "Answer: Paris." if body["prompt"] == "raw/contract" else "Paris",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
        else:
            text = "".join(chr(token) for token in body["prompt"])
            context = "L" if text.startswith("L") else " café"
            target = text[len(context) :]
            value = -0.1 if context == "L" else (-0.3 if target.endswith("A") else -0.4) / len(target)
            values = [None] * len(context) + [value] * len(target) + [-0.2]
            if target.endswith("BBBB") and self.malformed_remaining:
                self.malformed_remaining = False
                values[len(context)] = None
            result = {
                "id": f"likelihood-{len(self.requests)}",
                "object": "text_completion",
                "created": 0,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "text": text + "!",
                        "logprobs": {
                            "tokens": list(text + "!"),
                            "text_offset": list(range(len(text) + 1)),
                            "token_logprobs": values,
                            "top_logprobs": [None] * len(context) + [{"target": value}] * len(target) + [{"!": -0.2}],
                        },
                    }
                ],
                "usage": {"prompt_tokens": len(text), "completion_tokens": 1, "total_tokens": len(text) + 1},
            }
        return httpx.Response(200, json=result)


@pytest.mark.asyncio
async def test_mixed_http_driver_preserves_wire_channels_metrics_failures_and_resume(
    tmp_path, mixed_corpora, run_config
):
    endpoint = MixedEndpoint()
    tokenizer = CharacterTokenizer()
    config = run_config.model_copy(
        update={"source_limit": None, "concurrency": 3, "verifier_machine": machine_runtime(backend="shellsim")}
    )
    output = str(tmp_path / "mixed-results")
    async with openai.AsyncOpenAI(
        api_key="test",
        base_url="http://fixture.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(endpoint.handle)),
    ) as client:
        arguments = dict(
            machine_factories={"shellsim": ShellSimMachineFactory()},
            tokenizer=tokenizer,
            completion_model=CompletionModel(client, "fixture"),
            likelihood=VllmLikelihood(client, "fixture", tokenizer, 64, 0),
        )
        model = ChatModel(client, "fixture", 16)
        with pytest.raises(RuntimeError, match="raw evidence is saved"):
            await run_evaluation(mixed_corpora, output, config, model, **arguments)
        first = ReadView(output)
        failed = [
            row
            for row in first.iter_rows(ATTEMPTS_TABLE)
            if row["phase"] == "finished" and row["outcome"] == "infra_error"
        ]
        assert len(failed) == 1
        failed_raw = json.loads(first.read_blob(failed[0]["result_blob"]))
        assert len(failed_raw["scores"]) == 1
        assert failed_raw["scores"][0]["usage"] == {"prompt_tokens": 7, "completion_tokens": 1}
        evidence = failed_raw["failure_evidence"]
        assert evidence["response"]["choices"][0]["logprobs"]["token_logprobs"][5] is None
        assert evidence["request"]["prompt"] == list(map(ord, " café BBBB"))
        assert evidence["prepared"]["context_tokens"] == 5
        partial_samples = [
            sample_from_archive_row(row) for row in first.iter_rows(ARCHIVE_SAMPLES_TABLE) if row["task"] == "choice"
        ]
        assert len(partial_samples) == 1
        assert partial_samples[0].metrics == {}
        assert partial_samples[0].model_choice is None
        assert partial_samples[0].choices[0].loglikelihood == -0.3
        assert partial_samples[0].choices[1].loglikelihood is None
        before = len(endpoint.requests)
        await run_evaluation(mixed_corpora, output, config, model, **arguments)
        assert len(endpoint.requests) == before + 2
        assert [body for _, body in endpoint.requests[-2:]] == [
            body
            for _, body in endpoint.requests
            if isinstance(body.get("prompt"), list) and body["prompt"][0] != ord("L")
        ][:2]
        final_count = len(endpoint.requests)
        await run_evaluation(mixed_corpora, output, config, model, **arguments)
        assert len(endpoint.requests) == final_count
    reader = ReadView(output)
    samples = [sample_from_archive_row(row) for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE)]
    assert len([row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "graded"]) == 13
    invalid = [row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "invalid_task"]
    assert len(invalid) == 1
    assert invalid[0]["retry"] == 0
    assert next(sample for sample in samples if sample.task == "sandbox").metrics == {"accuracy": 1}
    sandbox_wire = next(
        body for _, body in endpoint.requests if body.get("messages", [{}])[-1].get("content") == "chat/sandbox"
    )
    assert "top_p" not in sandbox_wire
    chat = [sample for sample in samples if sample.task == "chat"]
    assert len(chat) == 8
    assert sorted(sample.metrics["accuracy"] for sample in chat) == [0, 0, 1, 1, 1, 1, 1, 1]
    assert any(sample.output == "wrong\n\nred" and sample.metrics["accuracy"] == 1 for sample in chat)
    assert any(sample.output == "red" and sample.metrics["accuracy"] == 0 for sample in chat)
    assert [seed for prompt, seed in endpoint.completed if prompt == "chat/final"] == [8, 7]
    raw_samples = {sample.prompt_text: sample for sample in samples if sample.task == "raw"}
    assert set(raw_samples) == {"raw/contract", "raw/bare"}
    assert raw_samples["raw/bare"].metrics == {"exact_match,strict_answer": 0, "exact_match,extract_answer": 1}
    assert json.loads(raw_samples["raw/bare"].extracted) == {"strict_answer": "[invalid]", "extract_answer": "Paris"}
    assert raw_samples["raw/contract"].metrics == {"exact_match,strict_answer": 1, "exact_match,extract_answer": 1}
    successful_choice = next(sample for sample in samples if sample.task == "choice" and sample.metrics)
    assert successful_choice.metrics == {"acc,none": 0, "acc_norm,none": 1}
    assert [choice.text for choice in successful_choice.choices] == ["A", "BBBB"]
    assert next(sample for sample in samples if sample.task == "continuation" and sample.metrics).metrics == {
        "acc,none": 1,
        "perplexity,none": -0.5,
    }
    wire = next(body for path, body in endpoint.requests if path.endswith("/chat/completions") and body.get("seed") == 7)
    assert wire == {
        "model": "fixture",
        "messages": [
            {"role": "user", "content": "A historical question"},
            {"role": "assistant", "content": "blue", "reasoning_content": ""},
            {"role": "user", "content": "chat/final"},
        ],
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 9,
        "max_tokens": 8,
        "stop": ["STOP"],
        "seed": 7,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    raw_wire = next(body for _, body in endpoint.requests if body.get("prompt") == "raw/bare")
    assert raw_wire == {
        "model": "fixture",
        "prompt": "raw/bare",
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 6,
        "stop": ["\nQuestion:"],
        "stream": False,
    }
    likelihood_wire = next(body for _, body in endpoint.requests if isinstance(body.get("prompt"), list))
    assert likelihood_wire == {
        "model": "fixture",
        "prompt": list(map(ord, " café A")),
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True,
        "seed": 11,
    }


@pytest.mark.asyncio
async def test_malformed_nonfinite_likelihood_retains_archive_safe_wire_evidence_without_retry():
    requests = []

    def serve(request):
        requests.append(json.loads(request.content))
        response = {
            "id": "invalid-number",
            "object": "text_completion",
            "created": 0,
            "model": "fixture",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "text": "cA!",
                    "logprobs": {
                        "tokens": ["c", "A", "!"],
                        "text_offset": [0, 1, 2],
                        "token_logprobs": [None, float("nan"), -0.2],
                        "top_logprobs": [None, {"A": -0.1}, {"!": -0.2}],
                    },
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        return httpx.Response(
            200, content=json.dumps(response, allow_nan=True), headers={"content-type": "application/json"}
        )

    async with openai.AsyncOpenAI(
        api_key="test",
        base_url="http://fixture.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        with pytest.raises(LikelihoodTransportError) as failure:
            await VllmLikelihood(client, "fixture", CharacterTokenizer(), 64, 0)(
                LikelihoodRequest(context="c", continuation="A"), seed=None
            )
    evidence = failure.value.evidence
    assert requests == [evidence["request"]]
    assert evidence["response"]["id"] == "invalid-number"
    assert evidence["response"]["choices"][0]["logprobs"]["token_logprobs"][1] is None
    assert "NaN" in evidence["response_json"]
    json.dumps(evidence, allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["http-404", "missing-usage", "empty-choices"])
async def test_generation_failure_archives_provider_body_and_resumes_only_failed_attempt(
    tmp_path, published_corpora, run_config, failure_kind
):
    requests = []
    provider_error = {"error": {"message": "unknown served model", "type": "NotFoundError", "code": 404}}

    def serve(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1 and failure_kind == "http-404":
            return httpx.Response(404, json=provider_error)
        response = {
            "id": "generation-response",
            "object": "chat.completion",
            "created": 0,
            "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "red"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
        if len(requests) == 1:
            if failure_kind == "missing-usage":
                del response["usage"]
            elif failure_kind == "empty-choices":
                response["choices"] = []
        return httpx.Response(200, json=response)

    output = str(tmp_path / "provider-failure")
    async with openai.AsyncOpenAI(
        api_key="test",
        base_url="http://fixture.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        model = ChatModel(client, "fixture", 16)
        with pytest.raises(RuntimeError, match="raw evidence is saved"):
            await run_evaluation(published_corpora, output, run_config, model, machine_factories={})
        reader = ReadView(output)
        failed = [row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "infra_error"]
        assert len(failed) == 1
        raw = json.loads(reader.read_blob(failed[0]["result_blob"]))
        evidence = raw["failure_evidence"]
        if failure_kind == "http-404":
            assert evidence["provider_error"] == provider_error["error"]
        else:
            assert evidence["response"]["id"] == "generation-response"
            if failure_kind == "empty-choices":
                assert evidence["response"]["choices"] == []
            else:
                assert evidence["response"]["choices"][0]["message"]["content"] == "red"
                assert evidence["response"]["usage"] is None
        assert len(requests) == 8
        await run_evaluation(published_corpora, output, run_config, model, machine_factories={})
        assert len(requests) == 9
        assert requests[-1] == requests[0]
        await run_evaluation(published_corpora, output, run_config, model, machine_factories={})
        assert len(requests) == 9
