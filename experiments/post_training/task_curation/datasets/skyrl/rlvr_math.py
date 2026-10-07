# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned rlvr math source declaration."""

from taskcompendium.datasets import math_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:rlvr_math",
        name="rlvr_math",
        version="rlvr_math-v1",
        files=SourceFiles(("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.TRAIN,
        hf_id="allenai/RLVR-MATH",
        revision="bd2a93551b503a395fadd1a740d957559cfe6f3c",
        config="default",
        split="train",
        policy=math_answers.math_policy(
            math_answers.normalize_rlvr_math, math_answers.RLVR_MATH_RUBRIC, "rlvr_math-controls"
        ),
    )
