# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Python benchmark declarations."""

from dataclasses import replace
from functools import partial

from taskcompendium.evaluation.spec import (
    ChatGeneration,
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TextGeneration,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.assets.code_protocol import HUMANEVAL_REQUEST_STOPS
from taskcompendium.importers.evalchemy.code import (
    CODE_REVISION,
    convert_python,
    python_item,
    read_crux,
    read_humaneval_fewshot,
    read_mbpp,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import PYTHON_GRADER
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

HUMANEVAL_FEWSHOT_VERSION = "2026.11.05.3"


def python_source(
    *,
    name: str,
    suite: str,
    source: HfSource,
    population: int,
    max_tokens: int | None,
) -> RlDataSource[EvaluationCorpusRecipe]:
    completion = suite == "humaneval"
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + suite,
            title=name,
            origin="Evalchemy",
            family="python-code",
            tags=("eval", "source:evalchemy", "benchmark", "code", "single-turn"),
            count=population,
        ),
        config=EvaluationCorpusRecipe(
            recipe=CurationRecipe(
                name="evalchemy-" + suite,
                source=source,
                convert=partial(convert_python, suite=suite),
                version=CODE_REVISION,
                intended_use=IntendedUse.EVAL,
                grader=PYTHON_GRADER,
            ),
            protocol=EvaluationProtocol(
                benchmark=name,
                version="2026.11.02",
                operation=OperationKind.TEXT_GENERATION if completion else OperationKind.CHAT_GENERATION,
                sampling=Sampling(
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    max_tokens=max_tokens,
                    stop=HUMANEVAL_REQUEST_STOPS if completion else (),
                    chat_template_kwargs={} if completion else {"enable_thinking": False},
                ),
                trials=TrialPolicy(count=1, seed=None),
                num_fewshot=0 if completion else None,
                metrics=(Metrics.ACCURACY,),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                expected_population=population,
            ),
            item_builder=partial(python_item, generation=TextGeneration() if completion else ChatGeneration()),
        ),
        pipeline=prepare_evaluation,
    )


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        python_source(
            name="humaneval",
            suite="humaneval",
            source=HfSource(
                "openai/openai_humaneval",
                "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                ("openai_humaneval/test-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            population=164,
            max_tokens=1024,
        ),
        python_source(
            name="HumanEvalPlus",
            suite="humanevalplus",
            source=HfSource(
                "evalplus/humanevalplus",
                "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
                ("test.jsonl",),
                SourceFormat.JSONL,
            ),
            population=164,
            max_tokens=None,
        ),
        python_source(
            name="MBPPPlus",
            suite="mbppplus",
            source=HfSource(
                "evalplus/mbppplus",
                "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
                ("data/test-00000-of-00001-d5781c9c51e02795.parquet",),
                SourceFormat.PARQUET,
                read=read_mbpp,
            ),
            population=378,
            max_tokens=None,
        ),
        python_source(
            name="CruxEval",
            suite="cruxeval",
            source=HfSource(
                "cruxeval-org/cruxeval",
                "b96af0450242eb4da433032b90998f25588a5d0f",
                ("test.jsonl",),
                SourceFormat.JSONL,
                read=read_crux,
            ),
            population=1600,
            max_tokens=2048,
        ),
    ]


def fewshot_sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    human = next(source for source in sources() if source.name == "evalchemy-humaneval")
    dataset = human.config
    assert dataset is not None and isinstance(dataset.recipe.source, HfSource)
    return [
        replace(
            human,
            info=replace(human.info, id="Checkpoint:humaneval-10shot", title="humaneval-10shot"),
            config=replace(
                dataset,
                recipe=replace(
                    dataset.recipe,
                    name="checkpoint-humaneval-10shot",
                    version=HUMANEVAL_FEWSHOT_VERSION + "-" + dataset.recipe.version,
                    source=replace(dataset.recipe.source, read=read_humaneval_fewshot),
                ),
                protocol=dataset.protocol.model_copy(
                    update={
                        "benchmark": "humaneval-10shot",
                        "num_fewshot": 10,
                        "version": HUMANEVAL_FEWSHOT_VERSION,
                    }
                ),
            ),
        )
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return [
        *fixed_benchmarks(sources()),
        *fixed_benchmarks(fewshot_sources(), "evalchemy", {"humaneval-10shot": "humaneval"}),
    ]
