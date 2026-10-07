# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned mimo music native source declaration."""

from taskcompendium.datasets import native_harbor
from taskcompendium.datasets.source_definitions import unpack_task_binary
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__music",
        name="XiaomiMiMo__MiMo-V2.6-RL-oss__music",
        version="native-harbor-v1",
        hf_id="open-athena/task-trove",
        revision="fab6ab00db320413179ead76005bc82d616d64da",
        config="XiaomiMiMo__MiMo-V2.6-RL-oss__music",
        split="train",
        files=SourceFiles(("data/mimo-music-639865fd3374.parquet",), SourceFormat.PARQUET, decoder=unpack_task_binary),
        policy=native_harbor.policy(
            "XiaomiMiMo__MiMo-V2.6-RL-oss__music",
            "music",
            (
                "Original private scorer.compute_score and hash-validated native assets retained in the archive",
                "Original Compose grader service and request/result mounts; immutable images unproved",
                "Original finite scalar reward contract and 210-second verifier timeout",
            ),
        ),
        intended_use=IntendedUse.TRAIN,
    )
