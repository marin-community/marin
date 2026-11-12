# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned GPQA-Diamond and MMLU-Pro generated-choice populations."""

from functools import partial

from taskcompendium.evaluation.spec import EvaluationProtocol, Metrics, OperationKind, Sampling, SummaryKind, TrialPolicy
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.evalchemy.reasoning import (
    MMLU_VALIDATION_FILE,
    convert_gpqa,
    convert_mmlu,
    reasoning_item,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import Converter, IntendedUse

from experiments.post_training.task_curation.datasets.evalchemy.environments import BASIC_GRADER
from experiments.post_training.task_curation.evaluation import EvalDataset, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

GPQA_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_TEST_FILE = "data/test-00000-of-00001.parquet"


def choice_dataset(
    name: str,
    benchmark: str,
    source: HfSource,
    converter: Converter,
    rows: int,
    trials: int,
    temperature: float,
    inputs: dict[str, HfSource],
) -> RlDataSource[EvalDataset]:
    return RlDataSource(
        info=SourceInfo(
            id="Evalchemy:" + name,
            upstream_dataset=name,
            upstream_name=name,
            subset=None,
            build_path="Evalchemy",
            origin=SourceOrigin.EVALCHEMY,
            family="generated-choice",
            tags=("eval", "source:evalchemy", "single-turn", "benchmark", "reasoning", "benchmark:" + benchmark),
            count=rows,
        ),
        config=EvalDataset(
            CurationRecipe(
                name="evalchemy-" + name,
                source=source,
                convert=converter,
                version=scorer_revision("mmlu_pro" if benchmark == "MMLUPro" else "gpqa"),
                intended_use=IntendedUse.EVAL,
                grader=BASIC_GRADER,
                inputs=inputs,
            ),
            EvaluationProtocol(
                benchmark=benchmark,
                version="2026.11.02",
                operation=OperationKind.CHAT_GENERATION,
                sampling=Sampling(
                    temperature=temperature,
                    top_p=None,
                    top_k=None,
                    max_tokens=None,
                    stop=(),
                    chat_template_kwargs={"enable_thinking": True},
                ),
                trials=TrialPolicy(count=trials, seed=0 if temperature else None),
                metrics=(Metrics.ACCURACY,),
                primary_metric="accuracy",
                summary=SummaryKind.MEAN,
                expected_population=rows,
                context_window=65536 if benchmark == "MMLUPro" else None,
            ),
            partial(reasoning_item, benchmark=benchmark),
        ),
        pipeline=curate,
    )


def sources() -> list[RlDataSource[EvalDataset]]:
    return [
        choice_dataset(
            "gpqa-diamond",
            "GPQADiamond",
            HfSource("Idavidrein/gpqa", GPQA_REVISION, ("gpqa_diamond.csv",), SourceFormat.CSV),
            convert_gpqa,
            198,
            3,
            0.7,
            {},
        ),
        choice_dataset(
            "mmlu-pro",
            "MMLUPro",
            HfSource("TIGER-Lab/MMLU-Pro", MMLU_PRO_REVISION, (MMLU_TEST_FILE,), SourceFormat.PARQUET),
            convert_mmlu,
            12032,
            1,
            0,
            {
                "fewshot": HfSource(
                    "TIGER-Lab/MMLU-Pro",
                    MMLU_PRO_REVISION,
                    (MMLU_VALIDATION_FILE,),
                    SourceFormat.PARQUET,
                )
            },
        ),
    ]
