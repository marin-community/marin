# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned aime24 source declaration."""

from taskcompendium.datasets import numeric_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:aime24",
        name="aime24",
        version="aime24-v1",
        files=SourceFiles(("data/train-*.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.EVAL,
        hf_id="HuggingFaceH4/aime_2024",
        revision="2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        config="default",
        split="train",
        policy=numeric_answers.aime24_policy(),
    )
