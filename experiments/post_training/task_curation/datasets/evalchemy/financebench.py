# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FinanceBench evaluation declarations."""

from taskcompendium.evaluation.corpus import chat_item
from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.financebench import convert_finance
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.evaluation import EvalDataset, curate
from experiments.post_training.task_curation.pipeline import CurationRecipe, UrlSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

SOURCE_REVISION = "623134de90723e7f189036da005067f36817b6e7"
FINANCEBENCH_SHA256 = "0df2b2080400693d98eeccac6970432af72d7e092287b08bcc9bd013d2c3232e"


def sources() -> list[RlDataSource[EvalDataset]]:
    return [
        RlDataSource(
            info=SourceInfo(
                id="Evalchemy:financebench",
                title="FinanceBench",
                origin="Evalchemy",
                family="judged-answer",
                tags=("eval", "source:evalchemy", "single-turn", "benchmark", "llm-judge"),
                count=50,
            ),
            config=EvalDataset(
                CurationRecipe(
                    name="evalchemy-financebench",
                    source=UrlSource(
                        f"https://raw.githubusercontent.com/marin-community/marin/{SOURCE_REVISION}/"
                        "vendor/evalchemy/eval/chat_benchmarks/FinanceBench/data/financebench.jsonl",
                        FINANCEBENCH_SHA256,
                        "financebench.jsonl",
                        SourceFormat.JSONL,
                    ),
                    convert=convert_finance,
                    version="2026.11.05.2",
                    intended_use=IntendedUse.EVAL,
                ),
                EvaluationProtocol(
                    benchmark="FinanceBench",
                    requires_judge=True,
                    version="2026.11.05.2",
                    operation=OperationKind.CHAT_GENERATION,
                    sampling=Sampling(
                        temperature=0,
                        top_p=None,
                        top_k=None,
                        max_tokens=4096,
                        stop=(),
                        chat_template_kwargs={"enable_thinking": False},
                    ),
                    trials=TrialPolicy(count=1, seed=None),
                    metrics=Metrics.CLASSIFICATION,
                    primary_metric="accuracy",
                    summary=SummaryKind.MEAN,
                    expected_population=50,
                ),
                chat_item,
            ),
            pipeline=curate,
        )
    ]
