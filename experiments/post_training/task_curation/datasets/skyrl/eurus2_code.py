# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned eurus2 code source declaration."""

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
        source_key="MarinSkyRL:eurus2_code",
        name="eurus2_code",
        version="eurus2_code-native-v1",
        files=SourceFiles(("train.parquet",), SourceFormat.PARQUET, selector=code_contracts.select_eurus_code),
        intended_use=IntendedUse.TRAIN,
        hf_id="PRIME-RL/Eurus-2-RL-Data",
        revision="9776b13264b5aaa0b16495fcf086a0a8d86fd655",
        config="default",
        split="train",
        policy=code_contracts.eurus2_code_policy(),
        runtime_binding=partial(
            bind_private_grader,
            binder=code_sql_binding.bind,
            timeout=code_sql_binding.CONTROL_TIMEOUT,
            memory_mb=code_sql_binding.CONTROL_MEMORY_MB,
        ),
    )
