# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned gpqa source declaration."""

from taskcompendium.datasets import gpqa
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:gpqa",
        name="gpqa",
        version="gpqa-v1",
        files=SourceFiles(("gpqa_diamond.csv",), SourceFormat.CSV),
        intended_use=IntendedUse.EVAL,
        hf_id="Idavidrein/gpqa",
        revision="83022cefff930aea54f654c0b282e74b9eeda5c6",
        config="gpqa_diamond",
        split="train",
        policy=gpqa.policy(),
    )
