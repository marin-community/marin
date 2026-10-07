# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned nemotron if source declaration."""

from taskcompendium.datasets import instruction_tasks
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.datasets.skyrl.ifeval_binding import bind_nemotron_ifeval
from experiments.post_training.task_curation.pipeline import RlDataPipeline


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:nemotron_if",
        runtime_binding=bind_nemotron_ifeval,
        name="nemotron_if",
        version="nemotron_if-v3",
        files=SourceFiles(("RL/instruction_following/instruction_following.jsonl",), SourceFormat.JSONL),
        intended_use=IntendedUse.TRAIN,
        hf_id="nvidia/Llama-Nemotron-Post-Training-Dataset",
        revision="ab2a40d258a6a4d9d4c277d702aeea445081766c",
        config="default",
        split="instruction_following",
        policy=instruction_tasks.nemotron_if_policy(),
    )
