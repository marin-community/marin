# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned deepscaler source declaration."""

from taskcompendium.datasets import math_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:deepscaler",
        name="deepscaler",
        version="deepscaler-v1",
        files=SourceFiles(("deepscaler.json",), SourceFormat.JSON),
        intended_use=IntendedUse.TRAIN,
        hf_id="agentica-org/DeepScaleR-Preview-Dataset",
        revision="b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29",
        config="default",
        split="train",
        policy=math_answers.math_policy(
            math_answers.normalize_deepscaler, math_answers.DEEPSCALER_RUBRIC, "deepscaler-math-controls"
        ),
    )
