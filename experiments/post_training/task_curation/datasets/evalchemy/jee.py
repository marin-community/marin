# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned JEEBench source, exact type prompts, and three greedy source repetitions."""

from taskcompendium.convert.evaluation_jee import convert_jee, jee_item
from taskcompendium.evaluation import EvaluationProtocol, Metric, MetricKind, Sampling, TrialPolicy
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse
from verifyit.adapters.evalchemy_math import CHAT_REQUEST_STOPS

from experiments.post_training.task_curation.datasets.evalchemy.pipeline import EvalDataSet, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

JEE_REVISION = "d070d1113ce365df05903763a2017ec2d0d56118"


def sources() -> list[RlDataSource[EvalDataSet]]:
    return [
        RlDataSource(
            info=SourceInfo(
                id="Evalchemy:JEEBench",
                title="JEEBench",
                origin="Evalchemy",
                family="engineering-exam",
                tags=("eval", "source:evalchemy", "single-turn", "benchmark"),
                count=515,
            ),
            config=EvalDataSet(
                recipe=CurationRecipe(
                    name="evalchemy-jeebench",
                    source=HfSource("daman1209arora/jeebench", JEE_REVISION, ("test.json",), SourceFormat.JSON),
                    convert=convert_jee,
                    version="1",
                    intended_use=IntendedUse.EVAL,
                ),
                protocol=EvaluationProtocol(
                    benchmark="JEEBench",
                    version="evalchemy-jee-v1",
                    operation="chat_generation",
                    sampling=Sampling(
                        temperature=0,
                        top_p=1,
                        top_k=None,
                        max_tokens=32768,
                        stop=CHAT_REQUEST_STOPS,
                        chat_template_kwargs={},
                    ),
                    trials=TrialPolicy(count=3, seed=None),
                    metrics=(Metric(name="accuracy", kind=MetricKind.CONTINUOUS, higher_is_better=True),),
                    primary_metric="accuracy",
                    reducer="mean_over_examples_and_trials_v1",
                    expected_population=515,
                ),
                item_builder=jee_item,
            ),
            pipeline=curate,
        )
    ]
