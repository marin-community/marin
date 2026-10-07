# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned asdiv source declaration."""

from taskcompendium.datasets import math_answers
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat, UrlDownload
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline

ASDIV_REVISION = "883f90a9a65bf00304ba8f37423910fe743abc47"


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:asdiv",
        name="asdiv",
        version="asdiv-v1",
        inputs=RecipeInputs(
            SourceFiles(("ASDiv.xml",), SourceFormat.XML, reader=math_answers.asdiv_rows),
            (
                UrlDownload(
                    f"https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/{ASDIV_REVISION}/dataset/ASDiv.xml",
                    "ASDiv.xml",
                ),
            ),
        ),
        intended_use=IntendedUse.TRAIN,
        hf_id="chaochun/nlu-asdiv-dataset",
        revision=ASDIV_REVISION,
        config="original-xml",
        split="train",
        policy=math_answers.math_policy(math_answers.normalize_asdiv, math_answers.ASDIV_RUBRIC, "asdiv-controls"),
    )
