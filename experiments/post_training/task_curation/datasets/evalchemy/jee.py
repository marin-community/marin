# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned JEEBench source, exact type prompts, and three greedy source repetitions."""

from taskcompendium.convert.evaluation_jee import convert_jee, jee_item
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
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.evalchemy.environments import BASIC_GRADER
from experiments.post_training.task_curation.evaluation import EvalDataset, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

JEE_REVISION = "d070d1113ce365df05903763a2017ec2d0d56118"


def sources() -> list[RlDataSource[EvalDataset]]:
    return [
        RlDataSource(
            info=SourceInfo(
                id="Evalchemy:JEEBench",
                upstream_dataset="JEEBench",
                upstream_name="JEEBench",
                subset=None,
                build_path="Evalchemy",
                origin=SourceOrigin.EVALCHEMY,
                family="engineering-exam",
                tags=("eval", "source:evalchemy", "single-turn", "benchmark"),
                count=515,
            ),
            config=EvalDataset(
                recipe=CurationRecipe(
                    name="evalchemy-jeebench",
                    source=HfSource("daman1209arora/jeebench", JEE_REVISION, ("test.json",), SourceFormat.JSON),
                    convert=convert_jee,
                    version=scorer_revision("jee"),
                    intended_use=IntendedUse.EVAL,
                    grader=BASIC_GRADER,
                ),
                protocol=EvaluationProtocol(
                    benchmark="JEEBench",
                    version="2026.11.02",
                    operation=OperationKind.CHAT_GENERATION,
                    sampling=Sampling(
                        temperature=0,
                        top_p=None,
                        top_k=None,
                        max_tokens=32768,
                        stop=(),
                        chat_template_kwargs={},
                    ),
                    trials=TrialPolicy(count=3, seed=None),
                    metrics=(Metric(name="accuracy", kind=MetricKind.CONTINUOUS, higher_is_better=True),),
                    primary_metric="accuracy",
                    summary=SummaryKind.MEAN,
                    expected_population=515,
                ),
                item_builder=jee_item,
            ),
            pipeline=curate,
        )
    ]
