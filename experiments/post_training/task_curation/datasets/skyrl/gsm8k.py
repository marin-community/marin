# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned gsm8k source declaration."""

from taskcompendium.datasets import math_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:gsm8k",
        name="gsm8k",
        version="gsm8k-v1",
        files=SourceFiles(("main/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.TRAIN,
        hf_id="openai/gsm8k",
        revision="740312add88f781978c0658806c59bc2815b9866",
        config="main",
        split="train",
        policy=math_answers.math_policy(math_answers.normalize_gsm8k, math_answers.GSM8K_RUBRIC, "gsm8k-controls"),
    )
