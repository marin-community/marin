# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned mimo code native source declaration."""

from taskcompendium.datasets import native_harbor
from taskcompendium.datasets.source_definitions import unpack_task_binary
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:XiaomiMiMo__MiMo-V2.6-RL-oss__code",
        name="XiaomiMiMo__MiMo-V2.6-RL-oss__code",
        version="native-harbor-v1",
        hf_id="open-athena/task-trove",
        revision="fab6ab00db320413179ead76005bc82d616d64da",
        config="XiaomiMiMo__MiMo-V2.6-RL-oss__code",
        split="train",
        files=SourceFiles(("data/mimo-code-639865fd3374.parquet",), SourceFormat.PARQUET, decoder=unpack_task_binary),
        policy=native_harbor.policy(
            "XiaomiMiMo__MiMo-V2.6-RL-oss__code",
            "swe",
            (
                "Original per-task MiMo image; immutable binding unproved",
                "Original MimoCodeEnvironment and MimoCodeVerifier host plugins are not bound to the task runtime",
                "Included tests/test.sh intentionally refuses standalone grading; verifier timeout is 1920 seconds",
            ),
        ),
        intended_use=IntendedUse.TRAIN,
    )
