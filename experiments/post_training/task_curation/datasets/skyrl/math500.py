# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math500 source declaration."""

from taskcompendium.datasets import math_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:math500",
        name="math500",
        version="math500-v1",
        files=SourceFiles(("test.jsonl",), SourceFormat.JSONL),
        intended_use=IntendedUse.EVAL,
        hf_id="HuggingFaceH4/MATH-500",
        revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        config="default",
        split="test",
        policy=math_answers.math_policy(math_answers.normalize_math500, math_answers.MATH500_RUBRIC, "math500-controls"),
    )
