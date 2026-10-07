# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned hendrycks math source declaration."""

from taskcompendium.datasets import math_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:hendrycks_math",
        name="hendrycks_math",
        version="hendrycks-math-algebra-train-v1",
        files=SourceFiles(("algebra/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.TRAIN,
        hf_id="EleutherAI/hendrycks_math",
        revision="21a5633873b6a120296cce3e2df9d5550074f4a3",
        config="algebra",
        split="train",
        policy=math_answers.math_policy(
            math_answers.normalize_hendrycks_math, math_answers.HENDRYCKS_MATH_RUBRIC, "hendrycks-math-controls"
        ),
    )
