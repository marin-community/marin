# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AbstentionBench, following facebookresearch/AbstentionBench at e291841.

Source: a Parquet snapshot of ``facebook/AbstentionBench`` (loader commit af06080), without the gated
``gpqa_abstain`` split. License CC-BY-NC-4.0, evaluation only. The reference judge is Llama 3.1 8B
Instruct; scores are comparable only under the same judge.
"""

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metric,
    MetricKind,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.abstentionbench import SPLITS, abstentionbench_item, convert_abstentionbench
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import HfSource

VERSION = "2026.11.25"
ABSTENTIONBENCH_REVISION = "693e80c53a7f6aa19db815561b7dc2b3412a02a7"
ABSTENTIONBENCH_POPULATION = 35855


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="abstentionbench",
        requires_judge=True,
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=0.8,
            top_p=0.95,
            top_k=None,
            max_tokens=None,
            stop=(),
            chat_template_kwargs={"enable_thinking": False},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metrics.ACCURACY, Metric(name="abstained", kind=MetricKind.BINARY, higher_is_better=False)),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
        expected_population=ABSTENTIONBENCH_POPULATION,
    )
    source = evaluation_source(
        name="abstentionbench",
        source=HfSource(
            "ymerkli/AbstentionBench",
            ABSTENTIONBENCH_REVISION,
            tuple(f"data/{split}-00000-of-00001.parquet" for split in SPLITS),
            SourceFormat.PARQUET,
        ),
        convert=convert_abstentionbench,
        version=VERSION,
        protocol=protocol,
        item_builder=abstentionbench_item,
        grader=None,
    )
    return fixed_benchmarks([source], "chat")
