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

from experiments.post_training.task_curation.datasets.evalchemy.environments import PYTHON_GRADER
from experiments.post_training.task_curation.evaluation import EvalDataset, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo


def python_source(
    name: str,
    suite: str,
    source: HfSource,
    rows: int,
    max_tokens: int,
) -> RlDataSource[EvalDataset]:
    completion = suite == "humaneval"
    sampled = suite == "cruxeval"
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + suite,
            title=name,
            origin="Evalchemy",
            family="python-code",
            tags=("eval", "source:evalchemy", "benchmark", "code", "single-turn"),
            count=rows,
        ),
        config=EvalDataset(
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
                    temperature=0.2 if sampled else 0,
                    top_p=None,
                    top_k=None,
                    max_tokens=max_tokens,
                    stop=HUMANEVAL_REQUEST_STOPS if completion else (),
                    chat_template_kwargs={} if completion else {"enable_thinking": False},
                ),
                trials=TrialPolicy(count=1, seed=0 if sampled else None),
                metrics=(Metrics.ACCURACY,),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                expected_population=rows,
            ),
            item_builder=partial(python_item, generation=TextGeneration() if completion else ChatGeneration()),
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataset]]:
    return [
        python_source(
            "humaneval",
            "humaneval",
            HfSource(
                "openai/openai_humaneval",
                "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                ("openai_humaneval/test-00000-of-00001.parquet",),
                SourceFormat.PARQUET,
            ),
            164,
            1024,
        ),
        python_source(
            "HumanEvalPlus",
            "humanevalplus",
            HfSource(
                "evalplus/humanevalplus",
                "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
                ("test.jsonl",),
                SourceFormat.JSONL,
            ),
            164,
            1024,
        ),
        python_source(
            "MBPPPlus",
            "mbppplus",
            HfSource(
                "evalplus/mbppplus",
                "b2d74c91837c3f2a20c1299ae98133cbe7cfa077",
                ("data/test-00000-of-00001-d5781c9c51e02795.parquet",),
                SourceFormat.PARQUET,
                read=read_mbpp,
            ),
            378,
            1024,
        ),
        python_source(
            "CruxEval",
            "cruxeval",
            HfSource(
                "cruxeval-org/cruxeval",
                "b96af0450242eb4da433032b90998f25588a5d0f",
                ("test.jsonl",),
                SourceFormat.JSONL,
                read=read_crux,
            ),
            1600,
            2048,
        ),
    ]


def fewshot_sources() -> list[RlDataSource[EvalDataset]]:
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
                    version="2026.11.05.3-" + dataset.recipe.version,
                    source=replace(dataset.recipe.source, read=read_humaneval_fewshot),
                ),
                protocol=dataset.protocol.model_copy(
                    update={
                        "benchmark": "humaneval-10shot",
                        "num_fewshot": 10,
                        "version": "2026.11.05.3",
                    }
                ),
            ),
        )
    ]
