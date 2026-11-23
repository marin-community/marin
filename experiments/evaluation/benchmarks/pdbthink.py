# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Published PDBThink coordinate panel with pinned prompts and deterministic scoring."""

from taskcompendium.evaluation.spec import (
    ContextPolicy,
    EvaluationProtocol,
    Metric,
    MetricKind,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.grader import scorer_revision
from taskcompendium.importers.pdbthink import (
    BUNDLE_PATH,
    PDBTHINK_REVISION,
    convert_pdbthink,
    manifest,
    pdbthink_item,
    read_pdbthink,
)
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.pipeline import UrlSource

VERSION = "2026.11.23.1"


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="pdbthink",
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(temperature=0.7, top_p=1.0, top_k=-1, max_tokens=None, stop=(), chat_template_kwargs={}),
        trials=TrialPolicy(count=3, seed=None),
        metrics=(
            Metric(name="macro_accuracy", kind=MetricKind.BINARY, higher_is_better=True),
            Metric(name="micro_accuracy", kind=MetricKind.BINARY, higher_is_better=True),
            Metric(name="score", kind=MetricKind.CONTINUOUS, higher_is_better=True),
        ),
        primary_metric="macro_accuracy",
        summary=SummaryKind.WEIGHTED_MEAN,
        expected_population=manifest()["n_renders"],
        context_policy=ContextPolicy(overflow="error", context_reserve=64),
    )
    source = evaluation_source(
        name="pdbthink",
        # Regional caches contain the same checksum-verified immutable bundle.
        source=UrlSource(
            "mirror://evaluation/datasets/" + BUNDLE_PATH,
            manifest()["bundle_sha256"],
            "evaluation.json.gz",
            SourceFormat.GENERATED,
            read=read_pdbthink,
        ),
        convert=convert_pdbthink,
        version=VERSION + "-" + scorer_revision("pdbthink"),
        protocol=protocol,
        item_builder=pdbthink_item,
        grader=Environment(
            requirements=(
                "tomli-w==1.2.0",
                f"pdbthink @ https://github.com/Open-Athena/pdbthink/archive/{PDBTHINK_REVISION}.tar.gz",
            )
        ),
    )
    return fixed_benchmarks([source], "chat")
