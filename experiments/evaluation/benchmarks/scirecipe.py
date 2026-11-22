# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SciRecipe-Eval: 600 protocol-generation rows (MIT dataset and reference implementation).

Reference: InternScience/Thoth at a155710079da47bb26a021646fa84d5544aa180e,
eval/eval_batch.py. The released Parquet calls the question field exp_goal.
Greedy decoding, a 1,024-token cap, and seed-42 action shuffling match that runner.
All 23 reference metrics are reported, including unnormalized content_score (0..1.5).
RoBERTa-large and MiniLM run on CPU in the offline grader environment. Infrastructure errors
remain unscored rather than silently becoming metric zeros as in the reference.
"""

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
from taskcompendium.importers.evalchemy.scirecipe import convert_scirecipe, scirecipe_item
from taskcompendium.importers.evalchemy.scorers.scirecipe import STRUCTURED_METRICS, TEXT_METRICS
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import SCIRECIPE_GRADER
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import HfSource

DATA_REVISION = "2aad80a53a2288963b24c504ddcccd58d6b8bfec"
VERSION = "2026.11.20"


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="scirecipe-eval",
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=1024,
            stop=(),
            chat_template_kwargs={"enable_thinking": True},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=tuple(
            Metric(
                name=name,
                kind=MetricKind.BINARY if name in {"step_match", "order_strict"} else MetricKind.CONTINUOUS,
                higher_is_better=True,
            )
            for name in (*STRUCTURED_METRICS, *TEXT_METRICS)
        ),
        primary_metric="content_score",
        summary=SummaryKind.MEAN,
        expected_population=600,
    )
    source = evaluation_source(
        name="scirecipe-eval",
        source=HfSource("manglu3935/SciRecipe", DATA_REVISION, ("SciRecipe-Eval.parquet",), SourceFormat.PARQUET),
        convert=convert_scirecipe,
        version=VERSION + "-" + scorer_revision("scirecipe"),
        protocol=protocol,
        item_builder=scirecipe_item,
        grader=SCIRECIPE_GRADER,
    )
    return fixed_benchmarks([source], "chat")
