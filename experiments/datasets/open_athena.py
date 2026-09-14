# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned GLM-5.3 completion releases; SFT prompt joins live in Datakit's chat registry."""

from marin.datakit.download.open_athena_glm53 import (
    AGENTTROVE_NAME,
    AGENTTROVE_REPO,
    AGENTTROVE_REVISION,
    WILDCHAT_NAME,
    WILDCHAT_REPO,
    WILDCHAT_REVISION,
)
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import hf_download


def wildchat_glm53_format_completions_dataset(split: str = "train") -> ArtifactStep[Artifact]:
    """Download one completion split; original prompts require the pinned source join."""
    if split not in ("train", "validation"):
        raise ValueError(f"Unsupported WildChat completion split: {split}")
    return hf_download(
        f"raw/{WILDCHAT_NAME}/{split}",
        hf_id=WILDCHAT_REPO,
        revision=WILDCHAT_REVISION,
        version="2026.09.13",
        urls_glob=[f"data/{split}-*.parquet"],
    )


def agenttrove_glm53_compactions_dataset() -> ArtifactStep[Artifact]:
    """Download summaries and references; original histories require the pinned source join."""
    return hf_download(
        f"raw/{AGENTTROVE_NAME}",
        hf_id=AGENTTROVE_REPO,
        revision=AGENTTROVE_REVISION,
        version="2026.09.13",
        urls_glob=["data/train-*.parquet"],
    )
