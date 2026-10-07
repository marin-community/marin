# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned hardmath source declaration."""

from taskcompendium.datasets import math_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:hardmath",
        name="hardmath",
        version="hardmath-v1",
        files=SourceFiles(("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.TRAIN,
        hf_id="pafitis/HARDMath_processed_training",
        revision="937e9f10356e31e854f6efb9a2507f1e200c8b25",
        config="default",
        split="train",
        policy=math_answers.math_policy(
            math_answers.normalize_hardmath, math_answers.HARDMATH_RUBRIC, "hardmath-math-controls"
        ),
    )
