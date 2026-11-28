# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scale AI's MultiChallenge (arXiv:2501.17399): realistic multi-turn conversations judged YES/NO.

Source: the 266-row public release on Hugging Face (CC-BY-4.0). The judge prompt is Marin's own, written
from the paper's description, because the authors' runner prompt carries no license. The runner judges with
gpt-4o-2024-08-06; runs name their own judge, so scores are comparable only under the same judge. The
reference headline is the mean of the four per-axis accuracies; ``accuracy`` here is the item mean, with
each axis reported as a group.
"""

from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.multichallenge import convert_multichallenge, multichallenge_item
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.preparation import EvaluationCorpusRecipe, prepare_evaluation
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

MULTICHALLENGE_REVISION = "670fd81f270314d0f8f82b34fcfe2187fd28f699"
VERSION = "2026.11.20"


def sources() -> list[RlDataSource[EvaluationCorpusRecipe]]:
    return [
        RlDataSource(
            info=SourceInfo(
                data_restriction=DataRestriction.OPEN,
                upstream_dataset="multichallenge",
                upstream_name="MultiChallenge",
                subset=None,
                build_path="ScaleAI",
                origin=SourceOrigin.SCALE_AI,
                family="judged-answer",
                tags=("eval", "source:scaleai", "multi-turn", "benchmark", "llm-judge"),
            ),
            config=EvaluationCorpusRecipe(
                CurationRecipe(
                    name="multichallenge",
                    source=HfSource(
                        "ScaleAI/MultiChallenge",
                        MULTICHALLENGE_REVISION,
                        ("data/test-00000-of-00001.parquet",),
                        SourceFormat.PARQUET,
                    ),
                    convert=convert_multichallenge,
                    version=VERSION,
                    intended_use=IntendedUse.EVAL,
                ),
                EvaluationProtocol(
                    benchmark="multichallenge",
                    requires_judge=True,
                    version=VERSION,
                    operation=OperationKind.CHAT_GENERATION,
                    # The reference fixes no output cap; keep each model's own sampling and chat defaults.
                    sampling=Sampling(
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        max_tokens=None,
                        stop=(),
                        chat_template_kwargs={},
                    ),
                    trials=TrialPolicy(count=1, seed=None),
                    metrics=(Metrics.ACCURACY,),
                    primary_metric="accuracy",
                    summary=SummaryKind.MEAN,
                ),
                multichallenge_item,
            ),
            pipeline=prepare_evaluation,
        )
    ]


def benchmarks() -> list[EvaluationBenchmark]:
    return fixed_benchmarks(sources())
