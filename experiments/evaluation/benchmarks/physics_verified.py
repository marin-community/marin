# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PHYSICS-Verified: audited PhD-qualifying-exam physics problems (Feng et al., 2025, arXiv 2503.21821).

The population is the 602 test problems without figures from ``yale-nlp/physics-verified``, the
authors' audited release (MIT). Prompts and grading follow its Humanity's Last Exam style harness: the
model answers in an Explanation/Answers/Confidence format, and one judge call returns a verdict per
labelled reference answer. Accuracy counts a problem only when every answer is correct; each grade also
records the fraction of correct answers as ``partial_credit`` in its detail. The judge is the authors'
GLM-5.3-Flash, served by Together; on the authors' published Qwen3.5-122B-A10B responses it matches
their verdicts on 97% of answers.
"""

from taskcompendium.evaluation.spec import (
    EvaluationProtocol,
    Metrics,
    OperationKind,
    Sampling,
    SummaryKind,
    TrialPolicy,
)
from taskcompendium.importers.evalchemy.physics_verified import (
    convert_physics_verified,
    is_text_only,
    physics_verified_item,
)
from taskcompendium.pipeline.inputs import SourceFormat

from experiments.evaluation.benchmark import EvaluationBenchmark, fixed_benchmarks
from experiments.evaluation.prompt_compilation import evaluation_source
from experiments.post_training.task_curation.pipeline import HfSource

PHYSICS_VERIFIED_REVISION = "724170025e3541c9b877285b2d577a1257796f0b"
VERSION = "2026.11.20"


def benchmarks() -> list[EvaluationBenchmark]:
    protocol = EvaluationProtocol(
        benchmark="physics-verified",
        requires_judge=True,
        version=VERSION,
        operation=OperationKind.CHAT_GENERATION,
        sampling=Sampling(
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=(),
            chat_template_kwargs={"enable_thinking": True},
        ),
        trials=TrialPolicy(count=1, seed=None),
        metrics=(Metrics.ACCURACY,),
        primary_metric="accuracy",
        summary=SummaryKind.MEAN,
    )
    source = evaluation_source(
        name="physics-verified",
        source=HfSource(
            "yale-nlp/physics-verified",
            PHYSICS_VERIFIED_REVISION,
            ("data/test.jsonl",),
            SourceFormat.JSONL,
            select=is_text_only,
        ),
        convert=convert_physics_verified,
        version=VERSION,
        protocol=protocol,
        item_builder=physics_verified_item,
        grader=None,
    )
    return fixed_benchmarks([source])
