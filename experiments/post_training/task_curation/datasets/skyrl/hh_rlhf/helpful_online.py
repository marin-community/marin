# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned hh helpful online source declaration."""

from taskcompendium.datasets import preference_tasks
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.datasets.shared import hf_pipeline
from experiments.post_training.task_curation.pipeline import RlDataPipeline

HH_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"


def pipeline() -> RlDataPipeline:
    return hf_pipeline(
        source_key="MarinSkyRL:hh_rlhf/helpful-online",
        name="hh_helpful_online",
        version="hh_helpful_online-v1",
        files=SourceFiles(("helpful-online/train.jsonl.gz",), SourceFormat.JSONL),
        intended_use=IntendedUse.TRAIN,
        hf_id="Anthropic/hh-rlhf",
        revision=HH_REVISION,
        config="helpful-online",
        split="train",
        policy=preference_tasks.hh_policy(preference_tasks.HH_HELPFUL_ONLINE_RUBRIC),
    )
