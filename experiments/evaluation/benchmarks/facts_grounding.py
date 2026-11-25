# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Grounding v2 on the 856 public examples, judged by one model through the v2 protocol.

FACTS Grounding (Google DeepMind; v2 is section 6 of arXiv:2512.10791) asks for a long-form answer
grounded only in a supplied document. The rows are Kaggle dataset
``deepmind/FACTS-grounding-examples`` version 17, the file the v2 starter notebook reads; the data are
CC-BY-4.0. The Hugging Face mirror ``google/FACTS-grounding-public`` is the 860-row v1 release, four
rows of which the current release dropped.

Grading follows the starter notebook (kaggle.com/code/prathameshbang/facts-grounding-v2-benchmark-starter,
version 4): the judge writes its own answer, rates the response's instruction following against it, and
labels every sentence of an eligible response as supported or not. The notebook asks Gemini 2.5 Flash and
GPT-5 and averages their grounding verdicts; this benchmark asks the run's configured judge for every
step, so its scores are not comparable to the Kaggle leaderboard, which also includes a private split.
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
from taskcompendium.importers.evalchemy.facts_grounding import (
    convert_facts_grounding,
    facts_grounding_item,
    read_facts_grounding,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.benchmarks.environments import JUDGE_GRADER
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import UrlSource

FACTS_GROUNDING_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/deepmind/FACTS-grounding-examples/examples.csv"
    "?datasetVersionNumber=17"
)
FACTS_GROUNDING_SHA256 = "66990aea86a91d40edcda6ebcd0d686674246dc7c07338ed8c17e1c275b1e096"
VERSION = "2026.11.18"


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="facts-grounding-v2",
        requires_judge=True,
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        # The notebook sets no sampling; the cap leaves room for reasoning before a long answer.
        sampling=Sampling(
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=16384,
            stop=(),
            chat_template_kwargs={"enable_thinking": True},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(
            Metric(name="factuality", kind=MetricKind.BINARY, higher_is_better=True),
            Metric(name="eligible", kind=MetricKind.BINARY, higher_is_better=True),
        ),
        primary_metric="factuality",
        summary=SummaryKind.MEAN,
    )
    source = evaluation_source(
        name="facts-grounding-v2",
        source=UrlSource(
            FACTS_GROUNDING_URL,
            FACTS_GROUNDING_SHA256,
            "examples.csv",
            SourceFormat.CSV,
            read=read_facts_grounding,
        ),
        convert=convert_facts_grounding,
        version=VERSION + "-" + scorer_revision("facts_grounding"),
        protocol=protocol,
        item_builder=facts_grounding_item,
        grader=JUDGE_GRADER,
    )
    return fixed_benchmarks([source], "chat")
