# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned verifiable code source declaration."""

from functools import partial

from taskcompendium.datasets import code_contracts
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.datasets.skyrl.code_sql import binding as code_sql_binding
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:verifiable_code",
        name="verifiable_code",
        version="verifiable_code-native-v1",
        files=SourceFiles(("data/train-*.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.TRAIN,
        hf_id="open-r1/verifiable-coding-problems-python",
        revision="b761a24a95fa03289a231d2d31c183636ffb9833",
        config="default",
        split="train",
        policy=code_contracts.verifiable_code_policy(),
        runtime_binding=partial(
            bind_private_grader,
            binder=code_sql_binding.bind,
            timeout=code_sql_binding.CONTROL_TIMEOUT,
            memory_mb=code_sql_binding.CONTROL_MEMORY_MB,
        ),
    )
