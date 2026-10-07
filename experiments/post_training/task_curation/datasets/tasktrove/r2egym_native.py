# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned r2egym native source declaration."""

from taskcompendium.datasets import native_harbor
from taskcompendium.datasets.source_definitions import unpack_task_binary
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:R2E-Gym__R2E-Gym-V1",
        name="R2E-Gym__R2E-Gym-V1",
        version="native-harbor-v1",
        hf_id="open-athena/task-trove",
        revision="fab6ab00db320413179ead76005bc82d616d64da",
        config="R2E-Gym__R2E-Gym-V1",
        split="train",
        files=SourceFiles(("data/r2egym-903d405799ac.parquet",), SourceFormat.PARQUET, decoder=unpack_task_binary),
        policy=native_harbor.policy(
            "R2E-Gym__R2E-Gym-V1",
            "swe",
            (
                "Original per-repository image and grader image; immutable bindings unproved",
                "Original private R2E Compose grader shares /testbed, /atlas-requests and /atlas-results",
                "Original native parser/reward bridge, 300-second test timeout and 420-second verifier timeout",
            ),
        ),
        intended_use=IntendedUse.TRAIN,
    )
