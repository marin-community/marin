# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regrade a stored run with task_driver --responses-from, then report the changed grades."""

import asyncio
import json
from functools import partial

import pytest
from fray.local_backend import LocalClient
from marin.evaluation import task_worker
from marin.rollouts.evaluation import (
    EvaluationInfrastructureError,
    EvaluationRunConfig,
    ModelIdentity,
    run_evaluation,
)
from rolloutengine import engine
from rolloutengine.models import ModelRequest, ModelTurn, RolloutLimits, TokenUsage
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import EvaluationProtocol, Metric, MetricKind, Sampling, SummaryKind, TrialPolicy
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe
from zephyr.context import ZephyrContext

from experiments.evaluation import regrade_report, task_driver
from experiments.evaluation.regrade_report import Summary

RESPONSES = {"colors/0": "red", "colors/1": "Red", "colors/2": "blue"}


class CharacterTokenizer:
    bos_token_id = 0

    def encode(self, text, *, add_special_tokens=False):
        return [ord(character) for character in text]

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, **kwargs):
        return self.encode("".join(message["content"] for message in conversation))


class Endpoint:
    def __init__(self, responses):
        self.responses = responses
        self.asked = []

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        question = request.messages[-1]["content"]
        self.asked.append(question)
        return ModelTurn(
            {"role": "assistant", "content": self.responses[question]}, (), (), None, "stop", TokenUsage(3, 1)
        )


def convert(row, _context, *, ignore_case):
    task = exact_answer_task(row, prompt=row.data["question"], answers=("red",), ignore_case=ignore_case)
    return task.model_copy(update={"tags": ("eval",)})


def publish(tmp_path, name, *, ignore_case, questions=tuple(RESPONSES)):
    source = tmp_path / name
    source.mkdir()
    (source / "rows.jsonl").write_text("".join(json.dumps({"question": question}) + "\n" for question in questions))
    recipe = SourceRecipe(
        name="colors",
        version=name,
        source=SourceFiles("colors", "pinned", ("rows.jsonl",), SourceFormat.JSONL),
        convert=partial(convert, ignore_case=ignore_case),
        rubric=None,
        controls=None,
        intended_use=IntendedUse.EVAL,
    )
    protocol = EvaluationProtocol(
        benchmark="colors",
        version="1",
        operation="chat_generation",
        sampling=Sampling(temperature=0.7, top_p=None, top_k=None, max_tokens=8, stop=(), chat_template_kwargs={}),
        trials=TrialPolicy(count=2, seed=7),
        metrics=(Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=len(questions),
    )
    output = tmp_path / (name + "-published")
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, chat_item, context, str(source), str(output))
    finally:
        client.shutdown()
    return str(output)


@pytest.fixture
def run_config(monkeypatch):
    monkeypatch.setattr(task_worker, "load_tokenizer", lambda *_args, **_kwargs: CharacterTokenizer())
    monkeypatch.setattr("rigging.timing.time.sleep", lambda _delay: None)
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
        concurrency=1,
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
def strict_corpus(tmp_path):
    return publish(tmp_path, "strict", ignore_case=False)


def stored_run(corpus, output, config):
    """A run stored as task_driver stores one."""
    asyncio.run(
        run_evaluation(
            (corpus,),
            output,
            task_driver.execution_config(config),
            Endpoint(RESPONSES),
            machine_factories={},
            tokenizer=CharacterTokenizer(),
        )
    )
    return output


@pytest.fixture
def source_run(tmp_path, strict_corpus, run_config):
    return stored_run(strict_corpus, str(tmp_path / "source"), run_config)


def test_task_driver_regrades_stored_responses_and_the_report_lists_changes(tmp_path, source_run, monkeypatch, capsys):
    corrected = publish(tmp_path, "corrected", ignore_case=True)
    output, changes = str(tmp_path / "regraded"), tmp_path / "changes.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        ["task_driver", "--local", "--responses-from", source_run, "--dataset", corrected, "--output", output],
    )
    task_driver.main()
    capsys.readouterr()
    monkeypatch.setattr(
        "sys.argv",
        ["regrade_report", "--source", source_run, "--regrade", output, "--changes", str(changes)],
    )

    regrade_report.main()

    assert capsys.readouterr().out.splitlines() == [
        "benchmark  attempts  reused  paired  before  after   raised  lowered",
        "colors     6         6       6       0.3333  0.6667  2       0",
    ]
    lines = [json.loads(line) for line in changes.read_text().splitlines()]
    assert [(line["trial"], line["before"]["score"], line["after"]["score"]) for line in lines] == [(0, 0, 1), (1, 0, 1)]


def test_the_report_leaves_out_responses_that_a_reworded_prompt_needed(tmp_path, source_run):
    reworded = publish(tmp_path, "reworded", ignore_case=True, questions=("colors/0", "colors/1?", "colors/2"))
    output = str(tmp_path / "regraded")
    config = task_driver.stored_config(source_run)
    with pytest.raises(EvaluationInfrastructureError, match="2 evaluation attempts failed"):
        task_driver.execute_with_retries((reworded,), output, config, None, None, None, responses_from=source_run)
    endpoint = Endpoint({"colors/1?": "red"})
    asyncio.run(
        run_evaluation(
            (reworded,),
            output,
            config,
            endpoint,
            machine_factories={},
            tokenizer=CharacterTokenizer(),
            reuse_from=source_run,
        )
    )

    summaries, changes = regrade_report.compare(source_run, output)

    assert endpoint.asked == ["colors/1?", "colors/1?"]
    assert summaries == [Summary("colors", 6, 4, 4, 0.5, 0.5, raised=0, lowered=0)]
    assert changes == []


def test_the_report_pairs_nothing_with_another_models_run_of_the_same_corpus(
    tmp_path, source_run, strict_corpus, run_config
):
    other_model = run_config.model.model_copy(update={"name": "other-solver"})
    other = stored_run(strict_corpus, str(tmp_path / "other"), run_config.model_copy(update={"model": other_model}))
    output = str(tmp_path / "regraded")
    corrected = publish(tmp_path, "corrected", ignore_case=True)
    config = task_driver.stored_config(source_run)
    task_driver.execute_with_retries((corrected,), output, config, None, None, None, responses_from=source_run)

    summaries, changes = regrade_report.compare(other, output)

    assert summaries == [Summary("colors", 6, 0, 0, None, None, raised=0, lowered=0)]
    assert changes == []


def test_the_report_pairs_the_source_attempt_that_the_provenance_names(tmp_path, strict_corpus, run_config, monkeypatch):
    grade_rollout = engine._grade_rollout
    calls = []

    async def first_grading_times_out(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise TimeoutError
        return await grade_rollout(*args, **kwargs)

    monkeypatch.setattr(engine, "_grade_rollout", first_grading_times_out)
    source = str(tmp_path / "source")
    with pytest.raises(EvaluationInfrastructureError):
        stored_run(strict_corpus, source, run_config)
    corrected = publish(tmp_path, "corrected", ignore_case=True)
    output = str(tmp_path / "regraded")
    config = task_driver.stored_config(source)
    task_driver.execute_with_retries((corrected,), output, config, None, None, None, responses_from=source)
    # Resuming the source with another request concurrency grades the timed-out attempt again under a new evidence blob.
    stored_run(strict_corpus, source, run_config.model_copy(update={"concurrency": 2}))

    summaries, _ = regrade_report.compare(source, output)

    assert [(summary.attempts, summary.reused, summary.paired) for summary in summaries] == [(6, 6, 5)]

