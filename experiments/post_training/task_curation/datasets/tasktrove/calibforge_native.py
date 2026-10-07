# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned calibforge native source declaration."""

from taskcompendium.datasets import native_harbor
from taskcompendium.datasets.source_definitions import unpack_task_binary
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:AweAI-Team__CalibForge",
        name="AweAI-Team__CalibForge",
        version="native-harbor-v1",
        hf_id="open-athena/task-trove",
        revision="fab6ab00db320413179ead76005bc82d616d64da",
        config="AweAI-Team__CalibForge",
        split="train",
        files=SourceFiles(("data/calibforge-fb1e75441a94.parquet",), SourceFormat.PARQUET, decoder=unpack_task_binary),
        policy=native_harbor.policy(
            "AweAI-Team__CalibForge",
            "terminal-agent",
            (
                "Original per-task Harbor image with /app/data inputs and /app workdir; immutable binding unproved",
                "Original test.sh needs apt/curl/network, uv 0.9.5, Python 3.13 and pinned pytest stack",
                "Original reward.txt contract and per-task verifier timeout; current QEMU guest denies network",
            ),
        ),
        intended_use=IntendedUse.TRAIN,
    )
