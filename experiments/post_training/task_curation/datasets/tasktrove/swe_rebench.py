# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned swe rebench source declaration."""

from taskcompendium.datasets import repository_tasks
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:DCAgent__swe_rebench_v2_patched_oracle-v2",
        name="tasktrove-swe_rebench",
        version="tasktrove-swe_rebench-v1",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="DCAgent__swe_rebench_v2_patched_oracle-v2",
        split="train",
        files=tasktrove_files("DCAgent__swe_rebench_v2_patched_oracle-v2"),
        policy=repository_tasks.policy("swe_rebench"),
        intended_use=IntendedUse.TRAIN,
    )
