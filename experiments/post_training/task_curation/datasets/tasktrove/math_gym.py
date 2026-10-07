# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math gym source declaration."""

from functools import partial

from taskcompendium.datasets import tasktrove_math
from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets import math as math_sources
from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:laion__nemotron-gym-math-v5",
        runtime_binding=partial(bind_private_grader, binder=math_sources.bind),
        name="tasktrove-math_gym",
        version="tasktrove-math_gym-v2-original-sympy",
        hf_id="open-thoughts/TaskTrove",
        revision="02923004846e4e73862c20962f823a6d05100e7a",
        config="laion__nemotron-gym-math-v5",
        split="train",
        files=tasktrove_files("laion__nemotron-gym-math-v5"),
        policy=tasktrove_math.policy("math_gym"),
        intended_use=IntendedUse.TRAIN,
    )
