# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count length stops in result roots that run_evaluation writes."""

import asyncio
import json
import sys

import pytest
from finestore.reader import ReadView
from fray.local_backend import LocalClient
from marin.rollouts.evaluation import (
    RUN_OBJECT,
    EvaluationInfrastructureError,
    EvaluationRunConfig,
    ModelIdentity,
    run_evaluation,
)
from rolloutengine.likelihood import LikelihoodScore
from rolloutengine.models import ModelResponseError, ModelTurn, RolloutLimits, TokenUsage
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import (
    ChatGeneration,
    ChoiceLikelihood,
    EvaluationItem,
    EvaluationProtocol,
    LikelihoodMetric,
    LikelihoodNormalization,
    LikelihoodRequest,
    Metric,
    MetricKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe
from zephyr.context import ZephyrContext

from experiments.evaluation import length_stops
from experiments.evaluation.length_stops import LengthStops

# Each question's reply, stop reason and completion tokens. The output token limit is 8 and the answer is "red".
REPLIES = {
    "colors/0": ("red", "stop", 1),
    "colors/1": ("red", "length", 8),
    "colors/2": ("re", "length", 8),
    "colors/3": ("blue", "stop", 2),
    "paints/0": ("red", "stop", 1),
    "paints/1": ("red", "stop", 1),
}


def red_task(row, _context):
    task = exact_answer_task(row, prompt=row.data["question"], answers=("red",), ignore_case=False)
    return task.model_copy(update={"tags": ("eval",)})


def shade_item(task):
    index = int(task.context.events[0].content.rsplit("/", 1)[1])
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        operation=ChatGeneration(),
        groups={"shade": "warm" if index < 2 else "cool"},
    )


def choice_item(task):
    question = task.context.events[0].content
    return EvaluationItem(
        task_id=task.id,
        source_row=task.source.row,
        operation=ChoiceLikelihood(
            choices=("red", "blue"),
            gold=(0,),
            requests=tuple(LikelihoodRequest(context=question, continuation=f" {choice}") for choice in ("red", "blue")),
        ),
    )


def publish(tmp_path, benchmark, *, rows, trials=1, version="1", item=chat_item):
    name = f"{benchmark}-{version}-{rows}"
    source = tmp_path / name
    source.mkdir()
    (source / "rows.jsonl").write_text(
        "".join(json.dumps({"question": f"{benchmark}/{index}"}) + "\n" for index in range(rows))
    )
    recipe = SourceRecipe(
        name=benchmark,
        version=version,
        source=SourceFiles(benchmark, "pinned", ("rows.jsonl",), SourceFormat.JSONL),
        convert=red_task,
        rubric=None,
        controls=None,
        intended_use=IntendedUse.EVAL,
    )
    likelihood = item is choice_item
    protocol = EvaluationProtocol(
        benchmark=benchmark,
        version=version,
        operation="choice_likelihood" if likelihood else "chat_generation",
        sampling=(
            None
            if likelihood
            else Sampling(temperature=0, top_p=None, top_k=None, max_tokens=8, stop=(), chat_template_kwargs={})
        ),
        trials=TrialPolicy(count=trials, seed=7),
        metrics=(Metric(name="accuracy", kind=MetricKind.BINARY, higher_is_better=True),),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=rows,
        likelihood_metrics=(
            (LikelihoodMetric(name="accuracy", normalization=LikelihoodNormalization.RAW),) if likelihood else ()
        ),
    )
    output = tmp_path / f"{name}-published"
    client = LocalClient()
    try:
        with ZephyrContext(
            client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / f"{name}-chunks")
        ) as context:
            publish_evaluation_corpus(recipe, protocol, item, context, str(source), str(output))
    finally:
        client.shutdown()
    return str(output)


@pytest.fixture
def run_config():
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


class Endpoint:
    """Answer each question as ``REPLIES`` says, after failing the first request for ``colors/3``."""

    def __init__(self):
        self.failed = False

    async def __call__(self, request):
        question = request.messages[-1]["content"]
        if question == "colors/3" and not self.failed:
            self.failed = True
            raise ModelResponseError("temporary fixture failure", response={})
        content, stop_reason, tokens = REPLIES[question]
        message = {"role": "assistant", "content": content}
        return ModelTurn(message, (), (), None, stop_reason, usage=TokenUsage(3, tokens))


async def score_choice(context, continuation, *, seed):
    return LikelihoodScore(
        loglikelihood=-1.0 if continuation == " red" else -2.0,
        is_greedy=continuation == " red",
        token_count=1,
        usage=TokenUsage(2, 1),
    )


def test_length_stops_count_latest_scored_attempts_by_benchmark_and_group(tmp_path, run_config, monkeypatch, capsys):
    corpora = (
        publish(tmp_path, "colors", rows=4, trials=2, item=shade_item),
        publish(tmp_path, "paints", rows=2),
        publish(tmp_path, "letters", rows=1, item=choice_item),
    )
    output = str(tmp_path / "results")
    endpoint = Endpoint()

    def run():
        return run_evaluation(corpora, output, run_config, endpoint, machine_factories={}, likelihood=score_choice)

    with pytest.raises(EvaluationInfrastructureError):
        asyncio.run(run())
    first = {(summary.benchmark, summary.group): summary for summary in length_stops.length_stops(output)}
    assert (first[("colors", None)].attempts, first[("colors", None)].scored) == (8, 7)
    assert (first[("colors", ("shade", "cool"))].attempts, first[("colors", ("shade", "cool"))].scored) == (4, 3)

    asyncio.run(run())
    expected = [
        ("colors", None, 8, 0.5, 4, 0.5, 0.25, 4.75, 8),
        ("colors", ("shade", "cool"), 4, 0.0, 2, 0.5, 0.0, 5.0, 8),
        ("colors", ("shade", "warm"), 4, 1.0, 2, 0.5, 0.5, 4.5, 8),
        ("paints", None, 2, 1.0, 0, 0.0, 0.0, 1.0, 1),
    ]
    assert length_stops.length_stops(output) == [
        LengthStops(
            benchmark=benchmark,
            group=group,
            attempts=count,
            scored=count,
            mean_reward=reward,
            length_stop_count=stops,
            length_stop_fraction=fraction,
            length_stop_score_contribution=contribution,
            response_tokens_mean=tokens,
            response_tokens_max=longest,
            max_tokens=(8, 8),
        )
        for benchmark, group, count, reward, stops, fraction, contribution, tokens, longest in expected
    ]

    monkeypatch.setattr(sys, "argv", ["length_stops", output])
    monkeypatch.setattr(length_stops, "configure_coreweave_s3", lambda: None)
    length_stops.main()
    table = [line.split()[:2] for line in capsys.readouterr().out.splitlines()]
    assert table == [
        ["benchmark", "group"],
        ["colors", "-"],
        ["colors", "shade=cool"],
        ["colors", "shade=warm"],
        ["paints", "-"],
    ]


def test_length_stops_label_repeated_benchmarks_and_list_corpora_without_attempts(tmp_path, run_config):
    corpora = (
        publish(tmp_path, "colors", rows=1, version="1"),
        publish(tmp_path, "colors", rows=2, version="1"),
        publish(tmp_path, "colors", rows=1, version="2"),
        publish(tmp_path, "paints", rows=1),
    )

    async def broken(_request):
        raise RuntimeError("model adapter bug")

    output = str(tmp_path / "results")
    with pytest.raises(ExceptionGroup):
        asyncio.run(run_evaluation(corpora, output, run_config, broken, machine_factories={}))
    first, second = (entry["dataset"] for entry in json.loads(ReadView(output).read_blob(RUN_OBJECT))["datasets"][:2])
    summaries = length_stops.length_stops(output)
    assert [(summary.benchmark, summary.attempts, summary.scored) for summary in summaries] == sorted(
        [(f"colors ({first[:12]})", 1, 0), (f"colors ({second[:12]})", 0, 0), ("colors@2", 0, 0), ("paints", 0, 0)]
    )
    assert all((summary.mean_reward, summary.length_stop_fraction) == (None, None) for summary in summaries)
