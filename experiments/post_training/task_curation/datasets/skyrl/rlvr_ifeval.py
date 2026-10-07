# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned rlvr ifeval source declaration."""

from taskcompendium.datasets import instruction_tasks
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.datasets.skyrl.ifeval_binding import bind_rlvr_ifeval
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:rlvr_ifeval",
        runtime_binding=bind_rlvr_ifeval,
        name="rlvr_ifeval",
        version="rlvr_ifeval-v3",
        files=SourceFiles(("data/train-00000-of-00001.parquet",), SourceFormat.PARQUET),
        intended_use=IntendedUse.TRAIN,
        hf_id="allenai/RLVR-IFeval",
        revision="47c03c73621c4aab2b824b7818681117d662770e",
        config="default",
        split="train",
        policy=instruction_tasks.rlvr_ifeval_policy(),
    )
