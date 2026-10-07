# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned numina math source declaration."""

from taskcompendium.datasets import math_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:numina_math",
        name="numina_math",
        version="numina_math-v2",
        files=SourceFiles(("data/train-*.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.TRAIN,
        hf_id="AI-MO/NuminaMath-CoT",
        revision="9d8d210c9f6a36c8f3cd84045668c9b7800ef517",
        config="default",
        split="train",
        policy=math_answers.math_policy(
            math_answers.normalize_numina_math, math_answers.NUMINA_MATH_RUBRIC, "numina_math-controls"
        ),
    )
