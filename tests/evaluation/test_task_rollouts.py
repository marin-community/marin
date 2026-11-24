# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest
from evalrecords.records import JudgeIdentity, TaskEvidence
from evalrecords.rollout_attempts import ATTEMPTS_TABLE, RUN_OBJECT
from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.layout import RESERVED_COLUMNS
from finestore.reader import ReadView
from finestore.store import DataStore
from fray.local_backend import LocalClient
from marin.rollouts.evaluation import (
    EvaluationInfrastructureError,
    EvaluationRunConfig,
    ModelIdentity,
    episode_turn,
    run_evaluation,
)
from rolloutengine.agents.environment import ExecResult
from rolloutengine.agents.pi.agent import AgentKilledBySignalError
from rolloutengine.agents.terminus.tmux_session import TmuxCommandError
from rolloutengine.chat import ChatModel
from rolloutengine.completion import CompletionModel
from rolloutengine.harness.external import AgentHarness
from rolloutengine.harness.shell import ShellHarness
from rolloutengine.likelihood import VllmLikelihood
from rolloutengine.models import ModelRequest, ModelTurn, Prompt, RolloutLimits, TokenUsage
from rolloutengine.test_support import machine_runtime
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineDiskFull, MachineSpec, NetworkPolicy, PackageSpec
from taskcompendium.convert.answers import answer_task, exact_answer_task, source_evidence
from taskcompendium.evaluation.corpus import chat_item, evaluation_cases, evaluation_protocol
from taskcompendium.evaluation.selection import SELECTIONS_TABLE
from taskcompendium.evaluation.spec import (
    AgentEpisode,
    ChatGeneration,
    ChatPromptVariant,
    ChoiceLikelihood,
    ContinuationLikelihood,
    EpisodeInputs,
    EpisodeLimits,
    EvaluationItem,
    EvaluationProtocol,
    LikelihoodMetric,
    LikelihoodNormalization,
    LikelihoodRequest,
    Metric,
    MetricKind,
    MiniSweAgent,
    PiAgent,
    Sampling,
    ShellAgent,
    SummaryKind,
    TextGeneration,
    TrialPolicy,
)
from taskcompendium.models import (
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ExitCodeReward,
    FileReward,
    GraderMachine,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, RawRow, SourceRecipe
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import (
    EmptyOutputPolicy,
    ExactSpec,
    JudgeCompletionPolicy,
    JudgeSpec,
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
                    summary=SummaryKind.MEAN,
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
async def test_unexpected_model_error_is_saved_and_propagates(tmp_path, published_corpora, run_config):
    async def model(_request):
        raise RuntimeError("model adapter bug")

    output = str(tmp_path / "unexpected-error")
    with pytest.raises(ExceptionGroup) as failure:
        await run_evaluation(published_corpora, output, run_config, model, machine_factories={})
    assert failure.value.subgroup(RuntimeError) is not None
    assert failure.value.subgroup(EvaluationInfrastructureError) is None
    reader = ReadView(output)
    failed = [row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["phase"] == "finished"]
    assert failed
    assert all((row["retry"], row["outcome"]) == (0, "infra_error") for row in failed)
    assert all(
        json.loads(reader.read_blob(row["result_blob"]))["failure"]["exception_type"] == "RuntimeError" for row in failed
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update", [{"source_limit": 1}, {"task_machine": machine_runtime(backend="shellsim")}], ids=["selection", "backend"]
)
async def test_driver_resume_rejects_a_different_selection_or_machine_backend(
    tmp_path, published_corpora, run_config, update
):
    async def model(_request):
        return ModelTurn({"role": "assistant", "content": "red"}, (), (), None, "stop", usage=TokenUsage(3, 1))

    output = str(tmp_path / "results")
    await run_evaluation(published_corpora, output, run_config, model, machine_factories={})
    with pytest.raises(ValueError, match="Resume requires"):
        await run_evaluation(
            published_corpora, output, run_config.model_copy(update=update), model, machine_factories={}
        )


@pytest.mark.asyncio
async def test_driver_resumes_partial_archive_after_request_concurrency_and_task_evidence_change(
    tmp_path, published_corpora, run_config
):
    calls = 0
    recovered = False

    async def model(_request):
        nonlocal calls
        calls += 1
        if not recovered and calls > 1:
            raise RuntimeError("interrupted transport")
        return ModelTurn({"role": "assistant", "content": "red"}, (), (), None, "stop", usage=TokenUsage(3, 1))

    output = str(tmp_path / "results")
    with pytest.raises(ExceptionGroup):
        await run_evaluation(
            published_corpora, output, run_config.model_copy(update={"concurrency": 1}), model, machine_factories={}
        )
    saved = list(ReadView(output).iter_rows(ARCHIVE_SAMPLES_TABLE))
    successful = [row for row in saved if row["grading"]["score"] == 1]
    assert len(successful) == 1
    completed = successful[0]
    recovered = True
    await run_evaluation(
        published_corpora,
        output,
        run_config.model_copy(update={"concurrency": 3, "task_evidence": TaskEvidence.REFERENCE}),
        model,
        machine_factories={},
    )
    samples = list(ReadView(output).iter_rows(ARCHIVE_SAMPLES_TABLE))
    assert sum(row["grading"]["score"] == 1 for row in samples) == 8
    assert sum((row["doc_id"], row["trial_id"]) == (completed["doc_id"], completed["trial_id"]) for row in samples) == 1
    assert calls == 9


def convert_mixed_row(row: RawRow, _context: ConversionContext):
    spec = ExactSpec((row.data.get("expected", "Paris"),), ignore_case=False, empty_output=EmptyOutputPolicy.GRADE)
    task = answer_task(row, prompt=row.data["prompt"], spec=spec, evidence=row.data)
    if row.data["kind"] == "raw":
        verdicts = tuple(
            json.dumps(
                {
                    "reward": 1,
                    "detail": {
                        "metrics": {"exact_match,strict_answer": strict, "exact_match,extract_answer": 1},
                        "extracted": {"strict_answer": "Paris" if strict else "[invalid]", "extract_answer": "Paris"},
                    },
                }
            )
            for strict in (0, 1)
        )
        task = task.model_copy(
            update={
                "grader": ScriptGrader(
                    argv=(
                        "sh",
                        "-c",
                        'if test "$(cat /app/answer.txt)" = "Answer: Paris."; then printf %s "$2"; '
                        'else printf %s "$1"; fi > /app/verdict.json',
                        "fixture-grader",
                        *verdicts,
                    ),
                    reward=FileReward(files=(RewardFile(path="/app/verdict.json", format=RewardFileFormat.JSON),)),
                    placement=GraderMachine(
                        environment=EnvironmentRequirements(
                            command_semantics=CommandSemantics.SHELL_SIMULATOR, package_spec=PackageSpec()
                        )
                    ),
                )
            }
        )
    if row.data["kind"] == "sandbox":
        task = task.model_copy(
            update={
                "grader": ScriptGrader(
                    argv=("sh", "-c", 'test "$(cat /app/answer.txt)" = red'),
                    answer_path="/app/answer.txt",
                    reward=ExitCodeReward(),
                    placement=GraderMachine(
                        environment=EnvironmentRequirements(
                            command_semantics=CommandSemantics.SHELL_SIMULATOR,
                            package_spec=PackageSpec(),
                        )
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
        operation = TextGeneration()
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
    return EvaluationItem(task_id=task.id, source_row=task.source.row, operation=operation)


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
                            temperature=0.7 if benchmark == "chat" else None if benchmark == "sandbox" else 0,
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
                    summary=SummaryKind.MEAN,
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
    return EvaluationItem(task_id=task.id, source_row=task.source.row, operation=operation)


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
        summary=SummaryKind.MEAN,
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
            "verifier_machine": machine_runtime(backend="shellsim"),
        }
    )
    output = str(tmp_path / "results")
    await run_evaluation(
        (corpus,),
        output,
        config,
        model,
        machine_factories={"shellsim": ShellSimMachineFactory()},
        tokenizer=CharacterTokenizer(),
    )
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
@pytest.mark.parametrize("task_evidence", list(TaskEvidence))
async def test_driver_multimetric_submission_failure_saves_raw_response_without_component_scores(
    tmp_path, run_config, task_evidence
):
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
    config = run_config.model_copy(
        update={"verifier_machine": machine_runtime(backend="shellsim"), "task_evidence": task_evidence}
    )
    machines = {"shellsim": ShellSimMachineFactory()}
    await run_evaluation((corpus,), output, config, model, machine_factories=machines)
    reader = ReadView(output)
    terminal = [row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["phase"] == "finished"]
    assert len(terminal) == 1
    assert terminal[0]["outcome"] == "submission_failure"
    raw = json.loads(reader.read_blob(terminal[0]["result_blob"]))
    assert raw["conversations"][0]["steps"][0]["turn"]["metadata"]["evaluation_raw_message"] == raw_message
    # Evidence always names its task in the immutable corpus; only embedded evidence copies it.
    (case,) = evaluation_cases(corpus, source_rows={raw["item"]["source_row"]})
    assert case.task.id == raw["item"]["task_id"]
    if task_evidence is TaskEvidence.EMBEDDED:
        assert TaskSpec.model_validate(raw["task"]) == case.task
    else:
        assert "task" not in raw
    assert raw["grade"]["status"] == "submission_failure"
    samples = [sample_from_archive_row(row) for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE)]
    assert len(samples) == 1
    assert samples[0].metrics == {}
    assert samples[0].grading.score == 0
    assert samples[0].output == "Paris"
    await run_evaluation((corpus,), output, config, model, machine_factories=machines)
    assert len(requests) == 1


def convert_judged_row(row, _context):
    judge = JudgeSpec(
        references=("2",),
        model=row.data.get("model", ""),
        rubric="labels",
        transport_attempts=1,
        parse_attempts=1,
        exact_gate=False,
        prompt_template="Reference: {reference}\nCandidate: {candidate}",
        label_scores={"A": 1, "B": 0},
        label_metrics={
            label: {"exact_match,strict_answer": score, "exact_match,extract_answer": score}
            for label, score in (("A", 1), ("B", 0))
        },
        label_scan="whole",
        label_case="upper",
        completion_policy=JudgeCompletionPolicy.CONTENT,
    )
    task = answer_task(row, prompt=row.data["prompt"], spec=judge, evidence=row.data)
    if row.data.get("scripted"):
        task = task.model_copy(
            update={
                "grader": ScriptGrader(
                    argv=("true",),
                    placement=GraderMachine(
                        environment=EnvironmentRequirements(
                            package_spec=PackageSpec(), command_semantics=CommandSemantics.SHELL_SIMULATOR
                        )
                    ),
                    judge=judge,
                ),
            }
        )
    return task.model_copy(update={"tags": ("eval",)})


@pytest.fixture(params=["http-503"])
def judge_endpoint(monkeypatch, request):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = {
                "id": "judge-response",
                "object": "chat.completion",
                "created": 0,
                "model": "judge-served-revision",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "A"}}],
            }
            status = 200
            if len(requests) == 1:
                if request.param == "http-503":
                    body = {"error": {"message": "fixture unavailable"}}
                    status = 503
                else:
                    body["choices"] = []
            payload = json.dumps(body).encode()
            self.send_response(status)
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
        yield requests, JudgeConnection(f"http://127.0.0.1:{server.server_port}/v1", "private-judge-key"), request.param
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("judge_endpoint", ["http-503", "empty-choices"], indirect=True)
async def test_driver_judge_identity_pins_explicit_and_default_models_across_failed_attempt_resume(
    tmp_path, run_config, judge_endpoint, monkeypatch
):
    monkeypatch.setenv("VERIFYIT_JUDGE_MODEL", "ambient-must-not-win")
    corpus = publish_policy_corpus(
        tmp_path,
        [
            {"prompt": "explicit judge", "model": "source-judge"},
            {"prompt": "default judge"},
        ],
        converter=convert_judged_row,
    )
    judge_requests, connection, failure_kind = judge_endpoint
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
    with pytest.raises(EvaluationInfrastructureError, match="raw evidence is saved"):
        await run_evaluation((corpus,), output, config, model, machine_factories={}, judge_connection=connection)
    reader = ReadView(output)
    failed = [row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "infra_error"]
    assert len(failed) == 1
    failed_raw = json.loads(reader.read_blob(failed[0]["result_blob"]))
    assert failed_raw["grade"]["reward"] is None
    detail = failed_raw["grade"]["detail"]
    if failure_kind == "empty-choices":
        assert detail["judge_response"]["id"] == "judge-response"
        assert detail["judge_response"]["choices"] == []
        assert detail["judge_request"] == judge_requests[0]
    else:
        assert "fixture unavailable" in json.dumps(detail)
    assert connection.api_key not in json.dumps(failed_raw)
    assert [request["model"] for request in judge_requests] == ["source-judge", "run-default"]
    await run_evaluation((corpus,), output, config, model, machine_factories={}, judge_connection=connection)
    assert [request["model"] for request in judge_requests] == [
        "source-judge",
        "run-default",
        "source-judge",
    ]
    assert len(model_requests) == 2
    final_reader = ReadView(output)
    graded = [row for row in final_reader.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "graded"]
    assert len(graded) == 2
    retry = next(row for row in graded if row["retry"] == 1)
    assert retry["prompt_tokens"] == retry["completion_tokens"] == 0
    assert json.loads(final_reader.read_blob(retry["result_blob"]))["reused_response_from_retry"] == 0
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
    assert len(model_requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("source_model", ["source-judge", ""])
@pytest.mark.parametrize("scripted", [False, True])
async def test_driver_unmapped_judge_model_is_rejected_before_generation(
    tmp_path, run_config, judge_endpoint, monkeypatch, source_model, scripted
):
    monkeypatch.setenv("VERIFYIT_JUDGE_MODEL", "different")
    corpus = publish_policy_corpus(
        tmp_path,
        [{"prompt": "unmapped judge", "model": source_model, "scripted": scripted}],
        converter=convert_judged_row,
    )
    judge_requests, connection, _ = judge_endpoint
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
        update={
            "source_limit": None,
            "concurrency": 3,
            "verifier_machine": machine_runtime(backend="shellsim"),
            "generation_extra_body": {"skip_special_tokens": False},
        }
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
        model = ChatModel(client, "fixture", 16, extra_body={"skip_special_tokens": True})
        with pytest.raises(EvaluationInfrastructureError, match="raw evidence is saved"):
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
        assert all(
            body["skip_special_tokens"] is False
            for _, body in endpoint.requests
            if "messages" in body or isinstance(body.get("prompt"), str)
        )
    reader = ReadView(output)
    assert json.loads(reader.read_blob(RUN_OBJECT))["config"]["generation_extra_body"] == {"skip_special_tokens": False}
    samples = [sample_from_archive_row(row) for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE)]
    assert len([row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "graded"]) == 13
    # Both trials of chat/truncated end at the output limit; likelihood scoring has no stop reason.
    assert Counter(
        (row["outcome"], row["stop_reason"]) for row in reader.iter_rows(ATTEMPTS_TABLE) if row["phase"] == "finished"
    ) == {
        ("graded", "stop"): 9,
        ("graded", "length"): 2,
        ("graded", None): 2,
        ("infra_error", None): 1,
        ("invalid_task", None): 1,
    }
    invalid = [row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "invalid_task"]
    assert len(invalid) == 1
    assert invalid[0]["retry"] == 0
    assert next(sample for sample in samples if sample.task == "sandbox").metrics == {"accuracy": 1}
    sandbox_wire = next(
        body for _, body in endpoint.requests if body.get("messages", [{}])[-1].get("content") == "chat/sandbox"
    )
    assert "temperature" not in sandbox_wire
    assert "seed" not in sandbox_wire
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
        "skip_special_tokens": False,
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
        "skip_special_tokens": False,
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
@pytest.mark.parametrize("failure_kind", ["http-503", "missing-usage", "empty-choices"])
async def test_generation_failure_archives_provider_body_and_resumes_only_failed_attempt(
    tmp_path, published_corpora, run_config, failure_kind
):
    requests = []
    provider_error = {"error": {"message": "temporarily unavailable", "type": "ServiceUnavailableError", "code": 503}}

    def serve(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1 and failure_kind == "http-503":
            return httpx.Response(503, json=provider_error)
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
        with pytest.raises(EvaluationInfrastructureError, match="raw evidence is saved"):
            await run_evaluation(published_corpora, output, run_config, model, machine_factories={})
        reader = ReadView(output)
        failed = [row for row in reader.iter_rows(ATTEMPTS_TABLE) if row["outcome"] == "infra_error"]
        assert len(failed) == 1
        raw = json.loads(reader.read_blob(failed[0]["result_blob"]))
        evidence = raw["failure_evidence"]
        if failure_kind == "http-503":
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


def save_before_stop_reasons(source: str, output: str) -> None:
    """Copy the run descriptor, attempts and evidence of a result root as they were saved before stop reasons."""
    reader = ReadView(source)
    attempts = reader.scan(ATTEMPTS_TABLE)
    assert attempts is not None
    attempts = attempts.select(
        [name for name in attempts.column_names if name not in RESERVED_COLUMNS and name != "stop_reason"]
    )
    with DataStore.open(output) as store:
        store.table(
            ATTEMPTS_TABLE,
            schema=attempts.schema,
            primary_key=("dataset", "source_row", "trial", "retry", "phase"),
            schema_version=1,
        )
        with store.transaction() as transaction:
            transaction.write_object(RUN_OBJECT, reader.read_blob(RUN_OBJECT))
            for row in attempts.to_pylist():
                transaction.table(ATTEMPTS_TABLE).add(row)
                if row["result_blob"] is not None:
                    transaction.write_object(row["result_blob"], reader.read_blob(row["result_blob"]))


@pytest.mark.asyncio
async def test_resume_retries_attempts_saved_before_stop_reasons_were_recorded(tmp_path, published_corpora, run_config):
    prompts = []

    async def model(request):
        prompts.append(request.messages[-1]["content"])
        if len(prompts) == 1:
            raise ConnectionError("the endpoint closed the connection")
        return ModelTurn({"role": "assistant", "content": "red"}, (), (), None, "stop", usage=TokenUsage(3, 1))

    current = str(tmp_path / "current")
    with pytest.raises(EvaluationInfrastructureError, match="raw evidence is saved"):
        await run_evaluation(published_corpora, current, run_config, model, machine_factories={})
    earlier = str(tmp_path / "earlier")
    save_before_stop_reasons(current, earlier)
    await run_evaluation(published_corpora, earlier, run_config, model, machine_factories={})
    assert len(prompts) == 9
    assert prompts[-1] == prompts[0]
    finished = [row for row in ReadView(earlier).iter_rows(ATTEMPTS_TABLE) if row["phase"] == "finished"]
    assert Counter((row["retry"], row["outcome"], row["stop_reason"]) for row in finished) == {
        (0, "graded", None): 7,
        (0, "infra_error", None): 1,
        (1, "graded", "stop"): 1,
    }


@pytest.mark.asyncio
async def test_model_rendering_overrides_preserve_published_corpus_identity(tmp_path, mixed_corpora, run_config):
    corpus = next(root for root in mixed_corpora if evaluation_protocol(root)[0].operation == "chat_generation")
    published, _ = evaluation_protocol(corpus)
    template_options = []
    requests = []

    class Tokenizer(CharacterTokenizer):
        def apply_chat_template(self, conversation, **kwargs):
            template_options.append(kwargs)
            return super().apply_chat_template(conversation, **kwargs)

    async def model(request):
        requests.append(request)
        return ModelTurn({"role": "assistant", "content": "red"}, (), (), None, "stop", usage=TokenUsage(3, 1))

    config = run_config.model_copy(
        update={
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "thinking_off_template_kwargs": {"reasoning_effort": "none"},
        }
    )
    output = str(tmp_path / "rendering")
    await run_evaluation((corpus,), output, config, model, machine_factories={}, tokenizer=Tokenizer())
    assert requests
    assert all(
        request.options["extra_body"]["chat_template_kwargs"] == {"reasoning_effort": "none"} for request in requests
    )
    assert template_options
    assert all(
        options["reasoning_effort"] == "none" and "enable_thinking" not in options for options in template_options
    )
    saved = json.loads(ReadView(output).read_blob(RUN_OBJECT))
    assert saved["datasets"][0]["protocol"] == published.model_dump(mode="json")


STALL = "episode/stall"


def convert_episode_row(row: RawRow, _context: ConversionContext):
    task = exact_answer_task(row, prompt=row.data["question"], answers=("red",), ignore_case=False)
    environment = EnvironmentRequirements(command_semantics=CommandSemantics.SHELL_SIMULATOR, capabilities=("shell",))
    return task.model_copy(update={"tags": ("eval",), "environment_requirements": environment})


def episode_item(task, agent=ShellAgent()):
    limits = EpisodeLimits(
        max_turns=4,
        max_total_tokens=None,
        model_turn_timeout=20,
        total_turn_timeout=20,
        verifier_timeout=10,
        attempt_timeout=60,
    )
    stalls = task.context.events[0].content == STALL
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        operation=AgentEpisode(harness=agent, limits=limits),
        episode=EpisodeInputs(agent_timeout=0.5 if stalls else None, memory_mb=512, network=NetworkPolicy.DENY),
    )


def convert_keyed_episode_row(row: RawRow, context: ConversionContext):
    task = convert_episode_row(row, context)
    environment = task.environment_requirements.model_copy(
        update={"environment_variables": {"SEARCH_API_KEY": "${SEARCH_API_KEY}"}}
    )
    return task.model_copy(update={"environment_requirements": environment})


def publish_episodes(tmp_path, questions, agent, convert=convert_episode_row):
    source = tmp_path / "episodes"
    source.mkdir()
    (source / "rows.jsonl").write_text("".join(json.dumps({"question": question}) + "\n" for question in questions))
    recipe = SourceRecipe(
        name="episodes",
        version="1",
        source=SourceFiles("episodes", "pinned", ("rows.jsonl",), SourceFormat.JSONL),
        convert=convert,
        rubric=None,
        controls=None,
        intended_use=IntendedUse.EVAL,
    )
    protocol = EvaluationProtocol(
        benchmark="episodes",
        version="1",
        operation="agent_episode",
        sampling=Sampling(temperature=0.7, top_p=None, top_k=None, max_tokens=8, stop=(), chat_template_kwargs={}),
        trials=TrialPolicy(count=1, seed=3),
        metrics=(Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=len(questions),
    )
    output = tmp_path / "episodes-published"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            item = partial(episode_item, agent=agent)
            publish_evaluation_corpus(recipe, protocol, item, context, str(source), str(output))
    finally:
        client.shutdown()
    return str(output)


@pytest.fixture
def episode_corpus(tmp_path):
    return publish_episodes(tmp_path, ("episode/flaky", STALL, "episode/solve"), ShellAgent())


class FlakyProviderFactory(ShellSimMachineFactory):
    """ShellSim machines whose first creation fails as a sandbox provider outage would."""

    def __init__(self):
        super().__init__()
        self.failures = 1
        self.created: list[MachineSpec] = []

    async def create(self, spec):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("sandbox provider unavailable")
        self.created.append(spec)
        return await super().create(spec)


@pytest.mark.asyncio
async def test_agent_episodes_grade_turn_deadlines_and_retry_machine_start_failures(
    tmp_path, episode_corpus, run_config
):
    requests = []

    async def model(request):
        requests.append(request)
        question = request.messages[0]["content"]
        observed = any(message["role"] == "tool" for message in request.messages)
        if observed and question == STALL:
            await asyncio.Event().wait()
        message = {"role": "assistant", "content": "red"}
        if not observed:
            arguments = json.dumps({"command": "echo red"})
            call = {"id": "write", "type": "function", "function": {"name": "shell", "arguments": arguments}}
            message = {"role": "assistant", "content": "", "tool_calls": [call]}
        return ModelTurn(message, (), (), None, "stop", usage=TokenUsage(5, 2))

    factory = FlakyProviderFactory()
    config = run_config.model_copy(
        update={
            "concurrency": 1,
            "source_limit": None,
            "task_machine": machine_runtime(backend="shellsim"),
            "agent_extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        }
    )
    arguments = {
        "machine_factories": {"shellsim": factory},
        "harnesses": {"shell": lambda _agent, _budget: ShellHarness(command_timeout=None)},
    }
    output = str(tmp_path / "results")
    with pytest.raises(EvaluationInfrastructureError):
        await run_evaluation((episode_corpus,), output, config, model, **arguments)
    await run_evaluation((episode_corpus,), output, config, model, **arguments)

    reader = ReadView(output)
    finished = {
        (row["source_row"], row["retry"]): (row["outcome"], json.loads(reader.read_blob(row["result_blob"])))
        for row in reader.iter_rows(ATTEMPTS_TABLE)
        if row["phase"] == "finished"
    }
    # The stalled episode ends at its task's agent deadline and is scored on its last, unanswered command.
    assert {key: outcome for key, (outcome, _) in finished.items()} == {
        ("rows.jsonl:0", 0): "infra_error",
        ("rows.jsonl:0", 1): "graded",
        ("rows.jsonl:1", 0): "submission_failure",
        ("rows.jsonl:2", 0): "graded",
    }
    assert finished["rows.jsonl:0", 0][1]["failure"]["diagnostics"]["operation"] == "start"
    stalled = finished["rows.jsonl:1", 0][1]
    assert (stalled["stop_reason"], stalled["limits"]["total_turn_timeout"]) == ("total_turn_timeout", 0.5)
    samples = [sample_from_archive_row(row) for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE)]
    assert sorted(sample.grading.score for sample in samples if sample.grading.score is not None) == [0, 1, 1]
    assert {(spec.memory_mb, spec.network) for spec in factory.created} == {(512, NetworkPolicy.DENY)}
    assert all(request.options["temperature"] == 0.7 and request.options["tools"] for request in requests)
    # The model's agent request body replaces the protocol's chat template arguments, as under Harbor.
    assert all(
        request.options["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True} for request in requests
    )


@pytest.mark.asyncio
async def test_task_machines_read_only_host_variables_the_run_supplies(tmp_path, run_config, monkeypatch):
    corpus = publish_episodes(tmp_path, ("episode/solve",), ShellAgent(), convert=convert_keyed_episode_row)
    monkeypatch.setenv("SEARCH_API_KEY", "search-key")
    factory = FlakyProviderFactory()
    factory.failures = 0

    async def model(_request):
        return ModelTurn({"role": "assistant", "content": "red"}, (), (), None, "stop", usage=TokenUsage(5, 1))

    config = run_config.model_copy(update={"source_limit": None, "task_machine": machine_runtime(backend="shellsim")})
    arguments = {
        "machine_factories": {"shellsim": factory},
        "harnesses": {"shell": lambda _agent, _budget: ShellHarness(command_timeout=None)},
    }
    # The worker's environment holds the variable, but the run has not declared it.
    with pytest.raises(ExceptionGroup) as unsupplied:
        await run_evaluation((corpus,), str(tmp_path / "unsupplied"), config, model, **arguments)
    assert any("does not supply: ['SEARCH_API_KEY']" in str(error) for error in unsupplied.value.exceptions)
    assert factory.created == []

    supplied = config.model_copy(update={"task_secrets": ("SEARCH_API_KEY",)})
    await run_evaluation((corpus,), str(tmp_path / "supplied"), supplied, model, **arguments)

    assert [spec.env["SEARCH_API_KEY"] for spec in factory.created] == ["search-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [PiAgent(), MiniSweAgent()], ids=lambda agent: agent.kind)
async def test_agents_that_send_their_own_request_options_reach_the_model_without_the_protocols(
    tmp_path, run_config, agent
):
    corpus = publish_episodes(tmp_path, ("episode/solve",), agent)
    requests = []

    async def model(request):
        requests.append(request)
        return ModelTurn({"role": "assistant", "content": "red"}, (), (), None, "stop", usage=TokenUsage(5, 2))

    config = run_config.model_copy(
        update={"concurrency": 1, "source_limit": None, "task_machine": machine_runtime(backend="shellsim")}
    )
    await run_evaluation(
        (corpus,),
        str(tmp_path / "results"),
        config,
        model,
        machine_factories={"shellsim": ShellSimMachineFactory()},
        # The shell harness stands in for the agent: its requests carry only its tools.
        harnesses={agent.kind: lambda _agent, _budget: ShellHarness(command_timeout=None)},
    )

    [request] = requests
    assert set(request.options) == {"tools"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        AgentKilledBySignalError("pi was killed", signal_number=9, result=ExecResult("", "", 137)),
        TmuxCommandError("tmux capture failed: return_code=1, stderr='server exited unexpectedly'"),
    ],
    ids=["pi-signal", "tmux-server-exit"],
)
async def test_harness_infrastructure_failures_are_retried_on_resume_instead_of_stopping_the_run(
    tmp_path, run_config, failure
):
    corpus = publish_episodes(tmp_path, ("episode/solve",), PiAgent())
    failures = [failure]

    async def run_pi(context, complete):
        if failures:
            raise failures.pop()
        await complete(Prompt(({"role": "user", "content": "Solve."},), {}))

    async def model(request):
        return ModelTurn({"role": "assistant", "content": "red"}, (), (), None, "stop", usage=TokenUsage(5, 2))

    config = run_config.model_copy(
        update={"concurrency": 1, "source_limit": None, "task_machine": machine_runtime(backend="shellsim")}
    )
    arguments = {
        "machine_factories": {"shellsim": ShellSimMachineFactory()},
        "harnesses": {"pi": lambda _agent, _budget: AgentHarness(run_pi)},
    }
    output = str(tmp_path / "results")
    with pytest.raises(EvaluationInfrastructureError):
        await run_evaluation((corpus,), output, config, model, **arguments)
    await run_evaluation((corpus,), output, config, model, **arguments)

    outcomes = [row["outcome"] for row in ReadView(output).iter_rows(ATTEMPTS_TABLE) if row["phase"] == "finished"]
    assert outcomes == ["infra_error", "graded"]


class FullDiskFactory(ShellSimMachineFactory):
    """ShellSim machines whose first creation finds the disk full."""

    def __init__(self):
        super().__init__()
        self.failures = 1

    async def create(self, spec):
        if self.failures:
            self.failures -= 1
            raise MachineDiskFull("The sandbox is full")
        return await super().create(spec)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filled_by_agent", "outcomes"),
    [(False, ["infra_error", "graded"]), (True, ["submission_failure"])],
    ids=["before-the-agent", "by-the-agent"],
)
async def test_a_full_disk_is_retried_only_when_the_agent_did_not_fill_it(
    tmp_path, run_config, filled_by_agent, outcomes
):
    corpus = publish_episodes(tmp_path, ("episode/solve",), PiAgent())

    async def run_pi(context, complete):
        if filled_by_agent:
            raise MachineDiskFull("The sandbox is full")
        await complete(Prompt(({"role": "user", "content": "Solve."},), {}))

    async def model(request):
        return ModelTurn({"role": "assistant", "content": "red"}, (), (), None, "stop", usage=TokenUsage(5, 2))

    config = run_config.model_copy(
        update={"concurrency": 1, "source_limit": None, "task_machine": machine_runtime(backend="shellsim")}
    )
    arguments = {
        "machine_factories": {"shellsim": ShellSimMachineFactory() if filled_by_agent else FullDiskFactory()},
        "harnesses": {"pi": lambda _agent, _budget: AgentHarness(run_pi)},
    }
    output = str(tmp_path / "results")
    if not filled_by_agent:
        with pytest.raises(EvaluationInfrastructureError):
            await run_evaluation((corpus,), output, config, model, **arguments)
    await run_evaluation((corpus,), output, config, model, **arguments)

    finished = [row for row in ReadView(output).iter_rows(ATTEMPTS_TABLE) if row["phase"] == "finished"]
    assert [row["outcome"] for row in finished] == outcomes


@pytest.mark.asyncio
@pytest.mark.parametrize(("requested", "sent"), [(4096, 4096), (None, 8192), (16384, 8192)])
async def test_episode_requests_keep_a_smaller_output_limit_than_the_protocols(requested, sent):
    received = []

    async def model(request):
        received.append(request.options)

    options = {} if requested is None else {"max_tokens": requested}
    request = ModelRequest(({"role": "user", "content": "Summarize."},), {**options, "extra_body": {"a": 1}})
    await episode_turn(request, model=model, options={"max_tokens": 8192, "temperature": 0.6, "extra_body": {"b": 2}})

    assert received == [{"max_tokens": sent, "temperature": 0.6, "extra_body": {"a": 1, "b": 2}}]
