# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TriviaQA, Natural Questions and DROP evaluation configurations."""

from dataclasses import dataclass, replace
from functools import partial

from marin.evaluation.records import PromptFormat
from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metric,
    MetricKind,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evalchemy.scorers.harness_generation import COMPLETION_REQUEST_STOPS
from taskcompendium.importers.evaluation_harness import (
    HarnessTemplate,
    convert_generation_document,
    generation_item,
    read_generation_file,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, ProtocolVariant
from experiments.evaluation.benchmarks.environments import ALIAS_GRADER, BASIC_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe, chat_source, prepare_evaluation
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


@dataclass(frozen=True)
class QuestionAnsweringDataset:
    name: str
    source: HfSource
    population: int
    pool_file: str


def _generation(
    dataset: QuestionAnsweringDataset,
    variant: ProtocolVariant,
    *,
    grader: Environment,
    metrics: tuple[str, ...],
    primary_metric: str,
    stop: tuple[str, ...],
    chat_options: dict[int, dict[str, bool | str]],
) -> tuple[RlDataSource[EvaluationCorpusRecipe], ...]:
    shots = variant.num_fewshot
    assert shots is not None
    template = HarnessTemplate(name=dataset.name, num_fewshot=shots, fewshot_file=dataset.pool_file)
    source = RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + dataset.name,
            title=dataset.name,
            origin="Evalchemy",
            family="short-answer",
            tags=("eval", "source:evalchemy", "benchmark", "family:short-answer"),
            count=dataset.population,
        ),
        config=EvaluationCorpusRecipe(
            recipe=CurationRecipe(
                name="evalchemy-" + dataset.name,
                source=replace(dataset.source, read=partial(read_generation_file, template=template)),
                inputs={"fewshot": replace(dataset.source, files=(dataset.pool_file,), read=None)} if shots else {},
                convert=partial(convert_generation_document, benchmark=dataset.name),
                grader=grader,
                version=scorer_revision("harness_generation"),
                intended_use=IntendedUse.EVAL,
            ),
            protocol=EvaluationProtocol(
                benchmark=dataset.name.replace("_", "-"),
                version="2026.11.02",
                operation=OperationKind.TEXT_GENERATION,
                sampling=Sampling(
                    temperature=0, top_p=None, top_k=None, max_tokens=None, stop=stop, chat_template_kwargs={}
                ),
                trials=TrialPolicy(count=1, seed=None),
                num_fewshot=shots,
                metrics=tuple(
                    Metric(
                        name=metric,
                        kind=MetricKind.CONTINUOUS if metric.startswith("f1,") else MetricKind.BINARY,
                        higher_is_better=True,
                    )
                    for metric in metrics
                ),
                primary_metric=primary_metric,
                summary=SummaryKind.MEAN,
                expected_population=dataset.population,
            ),
            item_builder=generation_item,
            protocol_name="harness",
        ),
        pipeline=prepare_evaluation,
    )
    if variant.prompt_format is PromptFormat.CHAT:
        source = chat_source(source, chat_options[shots])
    return (source,)


def question_answering_benchmark(
    *,
    dataset: QuestionAnsweringDataset,
    shots: tuple[int, ...],
    grader: Environment,
    metrics: tuple[str, ...],
    primary_metric: str,
    stop: tuple[str, ...],
    chat_options: dict[int, dict[str, bool | str]],
) -> EvaluationBenchmark:
    return EvaluationBenchmark(
        name=dataset.name.replace("_", "-"),
        variants=tuple(
            ProtocolVariant("harness", count, form)
            for count in shots
            for form in (
                (PromptFormat.COMPLETION, PromptFormat.CHAT) if count in chat_options else (PromptFormat.COMPLETION,)
            )
        ),
        build=partial(
            _generation,
            dataset,
            grader=grader,
            metrics=metrics,
            primary_metric=primary_metric,
            stop=stop,
            chat_options=chat_options,
        ),
    )


def benchmarks() -> list[EvaluationBenchmark]:
    return [
        question_answering_benchmark(
            dataset=QuestionAnsweringDataset(
                name="triviaqa",
                source=HfSource(
                    "mandarjoshi/trivia_qa",
                    "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
                    ("rc.nocontext/validation-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=17944,
                pool_file="rc.nocontext/train-00000-of-00001.parquet",
            ),
            shots=(5, 0),
            grader=ALIAS_GRADER,
            metrics=("exact_match,strict_answer", "exact_match,extract_answer"),
            primary_metric="exact_match,strict_answer",
            stop=COMPLETION_REQUEST_STOPS,
            chat_options={5: {"enable_thinking": False}},
        ),
        question_answering_benchmark(
            dataset=QuestionAnsweringDataset(
                name="nq_open",
                source=HfSource(
                    "google-research-datasets/nq_open",
                    "5dd9790a83002ad084ddeb7c420dc716852c6f28",
                    ("nq_open/validation-00000-of-00001.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=3610,
                pool_file="nq_open/train-00000-of-00001.parquet",
            ),
            shots=(5, 0),
            grader=ALIAS_GRADER,
            metrics=("exact_match,strict_answer", "exact_match,extract_answer"),
            primary_metric="exact_match,strict_answer",
            stop=COMPLETION_REQUEST_STOPS,
            chat_options={5: {}},
        ),
        question_answering_benchmark(
            dataset=QuestionAnsweringDataset(
                name="drop",
                source=HfSource(
                    "EleutherAI/drop",
                    "619386fc7ffe6dace3cc6d3483c69b164ebf6bf3",
                    ("drop_validation.parquet",),
                    SourceFormat.PARQUET,
                ),
                population=9536,
                pool_file="drop_train.parquet",
            ),
            shots=(3, 0),
            grader=BASIC_GRADER,
            metrics=("em,extract_answer", "f1,extract_answer"),
            primary_metric="f1,extract_answer",
            stop=COMPLETION_REQUEST_STOPS,
            chat_options={3: {}},
        ),
    ]
