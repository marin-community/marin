# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replies that ignore the task are scored through the evaluation driver and reduced per benchmark."""

import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from evalrecords.rollout_attempts import ATTEMPTS_TABLE
from evalrecords.rollout_results import logical_samples
from finestore.reader import ReadView
from fray.local_backend import LocalClient
from iris.client.client import IrisClient
from iris.cluster.client import remote_client
from iris.cluster.secrets import task_secret_values, write_task_secrets
from iris.cluster.types import Entrypoint
from iris.rpc import controller_pb2, job_pb2
from marin.evaluation import task_worker
from marin.rollouts.evaluation import EvaluationRunConfig, ModelIdentity
from rigging.secrets import SECRETS_DIR_ENV
from rolloutengine.models import MachineRuntimeSpec, RolloutLimits
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.machine import Backend, NetworkPolicy, PackageSpec
from taskcompendium.convert.answers import answer_task, exact_answer_task, ifeval_task, numeric_answer_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationProtocol, Metric, MetricKind, Sampling, SummaryKind, TrialPolicy
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    EnvironmentRequirements,
    ExitCodeReward,
    GraderMachine,
    JsonValueAnswer,
    ScriptGrader,
    TaskSpec,
)
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import ConversionContext, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, RawRow, SourceRecipe
from verifyit.spec import Constraint, ExactSpec
from zephyr.context import ZephyrContext

from experiments.evaluation import trivial_replies
from experiments.evaluation.task_driver import JUDGE_API_KEY, JUDGE_ENDPOINT, SOLVER_API_KEY, SOLVER_ENDPOINT

LOWERCASE = {"kind": "lowercase", "prompt": "Write a poem in all lowercase letters."}
ZERO = {"kind": "zero", "prompt": "What is 5 minus 5?"}
CAPITAL = {"kind": "capital", "prompt": "What is the capital of France?"}
REPEAT = {"kind": "repeat", "prompt": "Repeat this sentence."}
JSON_WORD = {"kind": "json_word", "prompt": "Answer with the JSON string hello."}
SECRET = {"kind": "secret", "prompt": "Say hello."}
TASK_TOKEN = "secret-task-token"


class WordTokenizer:
    """Stands in for the run's pinned tokenizer, which the launcher loads from the Hub."""

    def encode(self, text, *, add_special_tokens=False):
        return list(range(len(text.split())))

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, **kwargs):
        return [token for message in conversation for token in self.encode(message["content"])]


def convert(row: RawRow, _context: ConversionContext) -> TaskSpec:
    kind, prompt = row.data["kind"], row.data["prompt"]
    if kind == "lowercase":
        task = ifeval_task(row, prompt=prompt, constraints=(Constraint("change_case:english_lowercase", {}),))
    elif kind == "zero":
        task = numeric_answer_task(row, prompt=prompt, answer="0", tolerance_abs=0, tolerance_rel=0)
    elif kind == "capital":
        task = exact_answer_task(row, prompt=prompt, answers=("Paris",), ignore_case=True)
    elif kind == "repeat":
        task = exact_answer_task(row, prompt=prompt, answers=(prompt,), ignore_case=False)
    elif kind == "secret":
        # The grading machine reads the run's task secret, and its grader passes when it holds the value.
        environment = EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            package_spec=PackageSpec(),
            environment_variables={"TASK_TOKEN": "${FIXTURE_TASK_TOKEN}"},
        )
        grader = ScriptGrader(
            argv=("sh", "-c", f'test "$TASK_TOKEN" = {TASK_TOKEN}'),
            cwd=None,
            placement=GraderMachine(environment=environment),
            reward=ExitCodeReward(),
        )
        text_task = answer_task(row, prompt=prompt, spec=ExactSpec(expected=("hello",)))
        task = TaskSpec.model_validate({**dict(text_task), "grader": grader})
    else:
        text_task = answer_task(row, prompt=prompt, spec=ExactSpec(expected=("hello",)))
        task = TaskSpec.model_validate(
            {**dict(text_task), "answer_type": AnswerType.JSON, "answer_format": JsonValueAnswer()}
        )
    assert isinstance(task, TaskSpec)
    return task.model_copy(update={"tags": ("eval",)})


def publish_corpus(tmp_path, rows: list[dict]) -> str:
    source = tmp_path / "source"
    source.mkdir()
    (source / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    recipe = SourceRecipe(
        name="floor",
        version="1",
        source=SourceFiles("floor", "pinned", ("rows.jsonl",), SourceFormat.JSONL),
        convert=convert,
        rubric=None,
        controls=None,
        intended_use=IntendedUse.EVAL,
    )
    protocol = EvaluationProtocol(
        benchmark="floor",
        version="1",
        operation="chat_generation",
        sampling=Sampling(temperature=0, top_p=None, top_k=None, max_tokens=8, stop=(), chat_template_kwargs={}),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=len(rows),
    )
    corpus = tmp_path / "corpus"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, chat_item, context, str(source), str(corpus))
    finally:
        client.shutdown()
    return str(corpus)


def run_config(
    *,
    source_limit: int | None = None,
    task_secrets: tuple[str, ...] = (),
    verifier_machine: MachineRuntimeSpec | None = None,
) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        model=ModelIdentity(
            name="solver",
            revision="model-pin",
            tokenizer="solver",
            tokenizer_revision="tokenizer-pin",
            context_window=1024,
            serving={"backend": "fixture"},
        ),
        max_tokens=16,
        concurrency=2,
        source_limit=source_limit,
        task_secrets=task_secrets,
        verifier_machine=verifier_machine,
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


@pytest.fixture
def run_floor(tmp_path, monkeypatch, capsys):
    """Run the launcher's CLI on a corpus of ``rows``; return its result root and the table it prints."""
    monkeypatch.setattr(trivial_replies, "load_tokenizer", lambda *_args, **_kwargs: WordTokenizer())

    def run(rows: list[dict], *arguments: str, source_limit: int | None = None) -> tuple[str, list[str]]:
        config = run_config(source_limit=source_limit)
        config_path = tmp_path / "run.json"
        config_path.write_text(config.model_dump_json())
        output = str(tmp_path / "results")
        monkeypatch.setattr(
            "sys.argv",
            [
                "trivial_replies",
                "--local",
                "--dataset",
                publish_corpus(tmp_path, rows),
                "--output",
                output,
                "--config",
                str(config_path),
                *arguments,
            ],
        )
        trivial_replies.main()
        return output, capsys.readouterr().out.splitlines()

    return run


def passed_prompts(output: str, reply: str) -> set[str]:
    return {
        row["prompt_messages"][-1]["content"]
        for row in logical_samples(ReadView(f"{output}/{reply}"))
        if row["grading"]["score"] == 1
    }


def test_cli_reports_what_each_plain_reply_scores_through_the_evaluation_driver(run_floor):
    # One lowercase word satisfies a constraint that only restricts the reply, 0 matches a numeric answer of
    # zero, and the repeated prompt answers a task that asks for it.
    output, table = run_floor([LOWERCASE, ZERO, CAPITAL, REPEAT])
    assert {reply: passed_prompts(output, reply) for reply in trivial_replies.PLAIN_REPLIES} == {
        "empty": set(),
        "word": {LOWERCASE["prompt"]},
        "zero": {ZERO["prompt"]},
        "abstain": set(),
        "echo": {REPEAT["prompt"]},
    }
    assert table == [
        "| Benchmark | Metric | empty | word | zero | abstain | echo |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        "| floor | accuracy | 0.000 (0/4) | 0.250 (1/4) | 0.250 (1/4) | 0.000 (0/4) | 0.250 (1/4) |",
    ]


def test_cli_scores_the_cohort_that_the_run_configuration_selects(run_floor):
    _output, table = run_floor([CAPITAL, LOWERCASE, ZERO], "--reply", "word", source_limit=2)
    assert table[-1] == "| floor | accuracy | 0.500 (1/2) |"


def test_json_replies_answer_json_value_tasks_that_reject_the_plain_replies(run_floor):
    # A JSON-value format parses the whole reply as one JSON value, so the plain word is a malformed submission.
    output, table = run_floor([JSON_WORD], "--reply", "word", "--reply", "json_word")
    outcomes = {
        reply: [row["outcome"] for row in ReadView(f"{output}/{reply}").iter_rows(ATTEMPTS_TABLE) if row["outcome"]]
        for reply in ("word", "json_word")
    }
    assert outcomes == {"word": ["submission_failure"], "json_word": ["graded"]}
    assert table[-1] == "| floor | accuracy | 0.000 (0/1) | 1.000 (1/1) |"


@pytest.fixture
def iris_submission(tmp_path, monkeypatch):
    """Run the launcher's CLI against a recording Iris controller; return the CLI runner and the launch requests."""
    requests = []

    def launch(request, **_kwargs):
        requests.append(request)
        return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

    monkeypatch.setattr(
        remote_client,
        "ControllerServiceClientSync",
        lambda **_kwargs: SimpleNamespace(launch_job=launch, close=lambda: None),
    )
    config_path = tmp_path / "run.json"

    def submit(config: EvaluationRunConfig, *arguments: str) -> None:
        config_path.write_text(config.model_dump_json())
        monkeypatch.setattr("sys.argv", ["trivial_replies", "--config", str(config_path), "--no-wait", *arguments])
        with IrisClient.remote("http://controller.invalid") as client:
            monkeypatch.setattr(trivial_replies, "open_iris_client", lambda **_kwargs: nullcontext(client))
            trivial_replies.main()

    return submit, requests


def run_iris_worker(
    request: controller_pb2.Controller.LaunchJobRequest, storage: dict[str, str], secrets: Path, monkeypatch
) -> None:
    """Run the submitted callable as an Iris worker does, which reads the job's secrets only from files in ``secrets``.

    ``storage`` maps each object-storage root of the submission to the local directory that stands in for it.
    """
    write_task_secrets(task_secret_values(request.environment), secrets, uid=os.getuid(), gid=os.getgid())
    monkeypatch.setenv(SECRETS_DIR_ENV, str(secrets))
    for name in request.environment.secrets:
        monkeypatch.delenv(name, raising=False)

    def stored(value):
        if isinstance(value, tuple):
            return tuple(stored(item) for item in value)
        return storage.get(value, value) if isinstance(value, str) else value

    worker, arguments, options = Entrypoint.from_proto(request.entrypoint).resolve()
    worker(*map(stored, arguments), **{key: stored(value) for key, value in options.items()})


@pytest.mark.parametrize("backend", [Backend.LOCAL, Backend.DOCKER])
def test_iris_worker_grades_in_the_selected_machines_that_read_the_run_task_secrets(
    tmp_path, iris_submission, monkeypatch, capsys, backend
):
    submit, requests = iris_submission
    monkeypatch.setattr(trivial_replies, "load_tokenizer", lambda *_args, **_kwargs: WordTokenizer())
    # A host machine stands in for Docker, which test hosts do not run.
    monkeypatch.setattr(task_worker, "DockerMachineFactory", LocalMachineFactory)
    monkeypatch.setenv("FIXTURE_TASK_TOKEN", TASK_TOKEN)
    machine = MachineRuntimeSpec(
        backend=backend,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=60,
        cleanup_timeout=10,
    )
    config = run_config(task_secrets=("FIXTURE_TASK_TOKEN",), verifier_machine=machine)
    roots = ("--dataset", "s3://fixture/corpus", "--output", "s3://fixture/floor")
    submit(config, *roots, "--reply", "word", "--reply", "echo")

    (request,) = requests
    assert capsys.readouterr().out.split() == [request.name]
    assert request.container_profile == job_pb2.CONTAINER_PROFILE_PRIVILEGED
    assert TASK_TOKEN.encode() not in request.entrypoint.SerializeToString()
    assert {SOLVER_ENDPOINT, SOLVER_API_KEY, JUDGE_ENDPOINT, JUDGE_API_KEY}.isdisjoint(request.environment.secrets)

    corpus = publish_corpus(tmp_path, [SECRET])
    storage = {"s3://fixture/corpus": corpus, "s3://fixture/floor": str(tmp_path / "floor")}
    run_iris_worker(request, storage, tmp_path / "secrets", monkeypatch)
    # Each reply's grading machine held the secret, so its grader passed.
    assert capsys.readouterr().out.splitlines()[-1] == "| floor | accuracy | 1.000 (1/1) | 1.000 (1/1) |"


def test_cli_refuses_to_submit_local_roots_to_iris(tmp_path, iris_submission):
    submit, requests = iris_submission
    with pytest.raises(ValueError, match="shared object-storage roots"):
        submit(run_config(), "--dataset", "s3://fixture/corpus", "--output", str(tmp_path / "floor"))
    assert requests == []
