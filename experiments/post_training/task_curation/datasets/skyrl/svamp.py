# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned svamp source declaration."""

from taskcompendium.datasets import numeric_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:svamp",
        name="svamp",
        version="svamp-v1",
        files=SourceFiles(("data/train-*.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.TRAIN,
        hf_id="ChilleD/SVAMP",
        revision="5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef",
        config="default",
        split="train",
        policy=numeric_answers.svamp_policy(),
    )
