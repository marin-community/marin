# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned swegym native source declaration."""

from taskcompendium.datasets import native_harbor
from taskcompendium.datasets.source_definitions import unpack_task_binary
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:SWE-Gym__SWE-Gym",
        name="SWE-Gym__SWE-Gym",
        version="native-harbor-v1",
        hf_id="open-athena/task-trove",
        revision="fab6ab00db320413179ead76005bc82d616d64da",
        config="SWE-Gym__SWE-Gym",
        split="train",
        files=SourceFiles(("data/swegym-bb94ed9e39bb.parquet",), SourceFormat.PARQUET, decoder=unpack_task_binary),
        policy=native_harbor.policy(
            "SWE-Gym__SWE-Gym",
            "swe",
            (
                (
                    "Original repository environment, activated testbed conda environment and "
                    "/opt/swegym-grader interpreter"
                ),
                (
                    "Original SWE-Bench-Fork adapter at 242429c188fcfd06aad13fce9a54d450470bf0ac; "
                    "immutable binding unproved"
                ),
                "Original native-eval output parsing and reward.json contract with 1800-second verifier timeout",
            ),
        ),
        intended_use=IntendedUse.TRAIN,
    )
