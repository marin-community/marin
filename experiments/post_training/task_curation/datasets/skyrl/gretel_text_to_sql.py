# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned gretel text to sql source declaration."""

from functools import partial

from taskcompendium.datasets import gretel_text_to_sql
from taskcompendium.pipeline.execution_binding import bind_private_grader
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.datasets.skyrl.code_sql import binding as code_sql_binding
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:gretel_text_to_sql",
        name="gretel_text_to_sql",
        version="gretel_text_to_sql-native-v1",
        files=SourceFiles(("synthetic_text_to_sql_train.snappy.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.TRAIN,
        hf_id="gretelai/synthetic_text_to_sql",
        revision="740ab236e64503fba51be1101df7a1be83bf455d",
        config="default",
        split="train",
        policy=gretel_text_to_sql.policy(),
        runtime_binding=partial(
            bind_private_grader,
            binder=code_sql_binding.bind,
            timeout=code_sql_binding.CONTROL_TIMEOUT,
            memory_mb=code_sql_binding.CONTROL_MEMORY_MB,
        ),
    )
