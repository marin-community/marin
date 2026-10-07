# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned structured output source declaration."""

from taskcompendium.datasets import structured_output
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:laion__nemotron-gym-instruction-following-structured-v3",
        name="tasktrove-structured",
        version="tasktrove-structured-v1",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="laion__nemotron-gym-instruction-following-structured-v3",
        split="train",
        files=tasktrove_files("laion__nemotron-gym-instruction-following-structured-v3"),
        policy=structured_output.policy(),
        intended_use=IntendedUse.TRAIN,
    )
