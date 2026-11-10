# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GenomeQA's complete pinned population and one-response evaluation protocol."""

from taskcompendium.evaluation.spec import (
    ContextPolicy,
    EvaluationProtocol,
    Metric,
    MetricKind,
    Metrics,
    OperationKind,
    OutputBudget,
    OutputFit,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.genomeqa import (
    DATASET,
    FORMATS,
    PROMPT_URL,
    REVISION,
    convert_genomeqa,
    genomeqa_item,
    read_genomeqa,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.evalchemy.environments import BASIC_GRADER
from experiments.post_training.task_curation.evaluation import EvalDataset, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

MAX_TOKENS = 16384


def sources() -> list[RlDataSource[EvalDataset]]:
    return [
        RlDataSource(
            info=SourceInfo(
                id="GenomeQA",
                title="GenomeQA",
                origin="GenomeQA",
                family="generated-choice",
                tags=("eval", "single-turn", "biology", "benchmark:GenomeQA"),
                count=5200,
            ),
            config=EvalDataset(
                CurationRecipe(
                    name="genomeqa",
                    source=HfSource(
                        DATASET,
                        REVISION,
                        tuple(f"{answer_format}/train-00000-of-00001.parquet" for answer_format in FORMATS),
                        SourceFormat.PARQUET,
                        read=read_genomeqa,
                    ),
                    convert=convert_genomeqa,
                    version=scorer_revision("genomeqa"),
                    intended_use=IntendedUse.EVAL,
                    grader=BASIC_GRADER,
                    inputs={
                        "prompt": UrlSource(
                            PROMPT_URL,
                            "bc932144c20e02ccb18e4d8407f90814ca4f8fdc389d9309cd0cbc1525db58a3",
                            "run.py",
                            SourceFormat.GENERATED,
                        )
                    },
                ),
                EvaluationProtocol(
                    benchmark="GenomeQA",
                    version="2026.11.10",
                    operation=OperationKind.CHAT_GENERATION,
                    sampling=Sampling(
                        temperature=0.7,
                        top_p=0.95,
                        top_k=None,
                        stop=(),
                        max_tokens=MAX_TOKENS,
                        chat_template_kwargs={"enable_thinking": True},
                    ),
                    trials=TrialPolicy(count=1, seed=42),
                    metrics=(
                        Metric(name="genomeqa_macro", kind=MetricKind.CONTINUOUS, higher_is_better=True),
                        Metrics.ACCURACY,
                        *(
                            Metric(name=name, kind=MetricKind.BINARY, higher_is_better=False)
                            for name in ("unanswered", "truncation", "refusal", "parse_failure")
                        ),
                    ),
                    primary_metric="genomeqa_macro",
                    summary=SummaryKind.WEIGHTED_MEAN,
                    expected_population=5200,
                    context_policy=ContextPolicy(
                        output_fit=OutputFit.REQUIRE,
                        context_reserve=0,
                        output_budget=OutputBudget(minimum=MAX_TOKENS, reserve=0, multiplier=1),
                    ),
                ),
                genomeqa_item,
            ),
            pipeline=curate,
        )
    ]
