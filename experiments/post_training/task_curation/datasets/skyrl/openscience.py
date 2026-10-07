# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned openscience source declaration."""

from taskcompendium.datasets import openscience
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:openscience",
        name="openscience",
        version="openscience-v1",
        files=SourceFiles(("OS-Q2.5-32B-4.jsonl",), SourceFormat.JSONL),
        intended_use=IntendedUse.TRAIN,
        hf_id="nvidia/OpenScience",
        revision="7bd0437e4756f761768fe7e5cebeaa75480a4fd6",
        config="OS-Q2.5-32B-4",
        split="train",
        policy=openscience.policy(),
    )
