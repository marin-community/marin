# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned openswe other native source declaration."""

from taskcompendium.datasets import native_harbor
from taskcompendium.datasets.source_definitions import unpack_task_binary
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="Task Trove:GAIR__OpenSWE__openswe_other",
        name="GAIR__OpenSWE__openswe_other",
        version="native-harbor-v1",
        hf_id="open-athena/task-trove",
        revision="fab6ab00db320413179ead76005bc82d616d64da",
        config="GAIR__OpenSWE__openswe_other",
        split="train",
        files=SourceFiles(
            ("data/openswe-other-a8db93af5335.parquet",), SourceFormat.PARQUET, decoder=unpack_task_binary
        ),
        policy=native_harbor.policy(
            "GAIR__OpenSWE__openswe_other",
            "swe",
            (
                "Original repository and evaluator images; immutable bindings unproved",
                (
                    "Original OpenSWE Compose evaluator service with writable /atlas-requests and "
                    "read-only /atlas-results"
                ),
                "Original reward.json contract and 1860-second verifier timeout; backend lacks service mounts",
            ),
        ),
        intended_use=IntendedUse.TRAIN,
    )
