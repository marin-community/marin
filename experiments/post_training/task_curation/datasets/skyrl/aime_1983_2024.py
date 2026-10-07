# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned aime 1983 2024 source declaration."""

from taskcompendium.datasets import math_answers
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:aime_1983_2024",
        name="aime_1983_2024",
        version="aime_1983_2024-v1",
        files=SourceFiles(("AIME_Dataset_1983_2024.csv",), SourceFormat.CSV),
        intended_use=IntendedUse.EVAL,
        hf_id="di-zhang-fdu/AIME_1983_2024",
        revision="3e2cc86390666c5c756622afc0eeb9e6194496bc",
        config="default",
        split="train",
        policy=math_answers.math_policy(
            math_answers.normalize_aime_1983_2024, math_answers.AIME_1983_2024_RUBRIC, "aime_1983_2024-controls"
        ),
    )
