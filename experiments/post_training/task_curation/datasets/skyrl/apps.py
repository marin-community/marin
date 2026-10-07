# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned apps source declaration."""

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
        source_key="MarinSkyRL:apps",
        name="apps",
        version="apps-native-v2",
        files=SourceFiles(("train.jsonl",), SourceFormat.JSONL),
        intended_use=IntendedUse.TRAIN,
        hf_id="codeparrot/apps",
        revision="21e74ddf8de1a21436da12e3e653065c5213e9d1",
        config="default",
        split="train",
        policy=code_contracts.apps_policy(),
        runtime_binding=partial(
            bind_private_grader,
            binder=code_sql_binding.bind,
            timeout=code_sql_binding.CONTROL_TIMEOUT,
            memory_mb=code_sql_binding.CONTROL_MEMORY_MB,
        ),
    )
