# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned wizard orca source declaration."""

from taskcompendium.datasets import rubric_tasks
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:laion__wizardlm-orca-v4",
        name="tasktrove-wizard_orca",
        version="tasktrove-wizard_orca-v1",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="laion__wizardlm-orca-v4",
        split="train",
        files=tasktrove_files("laion__wizardlm-orca-v4"),
        policy=rubric_tasks.policy(rubric_tasks.RUBRICS["wizard_orca"]),
        intended_use=IntendedUse.TRAIN,
    )
