# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ARC-AGI-2 public evaluation set, two sampled attempts per test input.

The population is the 167 test inputs of the 120 tasks in ``data/evaluation`` of arcprize/ARC-AGI-2
(Apache-2.0) at f3283f727488ad98fe575ea6a5ac981e4a188e49, read from that commit's archive. The prompt and
grid extraction follow ARC Prize's arc-agi-benchmarking runner (MIT) at
78f2d3eacb86587d1438ab14f08fb56811eb1f1c, with the served model's sampling defaults. Each test input is
weighted by one over its task's test-input count, so ``accuracy`` is the task-averaged pass@1 over two
attempts. ARC Prize's headline is pass@2 (a test input counts if either attempt is correct), which the
trial records support but the shared reducer does not compute.
"""

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.arc_agi import arc_agi_2_item, convert_arc_agi_2, read_arc_agi_2
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import ARC_GRADER
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import UrlSource

ARC_AGI_2_REVISION = "f3283f727488ad98fe575ea6a5ac981e4a188e49"
ARC_AGI_2_URL = f"https://codeload.github.com/arcprize/ARC-AGI-2/tar.gz/{ARC_AGI_2_REVISION}"
ARC_AGI_2_SHA256 = "179681bc01d55810ee0e90d2717ef04b8d130496541c4853479a8d6fffa4b975"
ARC_AGI_2_POPULATION = 167
ARC_AGI_2_ATTEMPTS = 2
VERSION = "2026.11.19"


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="arc-agi-2",
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=(),
            chat_template_kwargs={},
        ),
        trials=TrialPolicy(count=ARC_AGI_2_ATTEMPTS, seed=None),
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        summary=SummaryKind.WEIGHTED_MEAN,
        expected_population=ARC_AGI_2_POPULATION,
    )
    source = evaluation_source(
        name="arc-agi-2",
        source=UrlSource(
            ARC_AGI_2_URL,
            ARC_AGI_2_SHA256,
            "arc-agi-2.tar.gz",
            SourceFormat.JSON,
            read=read_arc_agi_2,
        ),
        convert=convert_arc_agi_2,
        version=VERSION + "-" + scorer_revision("arc_agi"),
        protocol=protocol,
        item_builder=arc_agi_2_item,
        grader=ARC_GRADER,
    )
    return fixed_benchmarks([source], "chat")
