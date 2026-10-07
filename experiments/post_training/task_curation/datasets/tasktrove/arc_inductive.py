# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned arc inductive source declaration."""

from functools import partial

from taskcompendium.datasets import atlas_arc_injection
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.arc import binding as arc_binding
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:laion__nemotron-gym-arc-agi-python-inductive-v2",
        runtime_binding=partial(
            bind_private_grader,
            binder=partial(arc_binding.bind, source=arc_binding.ArcSource.TASKTROVE),
            timeout=600.0,
            memory_mb=4096,
        ),
        name="tasktrove-arc_inductive",
        version="tasktrove-arc_inductive-v1",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="laion__nemotron-gym-arc-agi-python-inductive-v2",
        split="train",
        files=tasktrove_files("laion__nemotron-gym-arc-agi-python-inductive-v2"),
        policy=atlas_arc_injection.policy("arc_inductive"),
        intended_use=IntendedUse.TRAIN,
    )
