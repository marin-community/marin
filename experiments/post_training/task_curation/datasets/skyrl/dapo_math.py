# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned dapo math source declaration."""

from taskcompendium.datasets import math_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:dapo_math",
        name="dapo_math",
        version="dapo_math-v1",
        files=SourceFiles(("data/dapo-math-17k.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.TRAIN,
        hf_id="BytedTsinghua-SIA/DAPO-Math-17k",
        revision="65877096c24ffa7abc4e4fa5edb95cf3413a5674",
        config="default",
        split="train",
        policy=math_answers.math_policy(
            math_answers.normalize_dapo_math, math_answers.DAPO_MATH_RUBRIC, "dapo_math-controls"
        ),
    )
