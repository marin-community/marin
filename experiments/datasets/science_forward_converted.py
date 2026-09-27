# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit catalog handle for the MiniMax-converted science-forward chat source."""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from rigging.filesystem.storage_path import prefix_join

SOURCE_NAME = "science-forward/minimax-m3-formatted-2026.09.27-v2"
OUTPUT_ROOT = "s3://marin-us-east-02a/marin/users/benfeuer/science-sft-converted/2026.09.27-v2"
OUTPUT_MAIN_DIR = "outputs/main"


def science_forward_converted_dataset() -> ArtifactStep[Artifact]:
    """Reference the audited Harmony Parquet source without copying it."""
    return ArtifactStep.adopt(
        name="sft/science-forward/minimax-m3-formatted",
        version="2026.09.27.2",
        source=prefix_join(OUTPUT_ROOT, OUTPUT_MAIN_DIR),
        kind=Artifact,
        config={"format": "harmony-chat-parquet", "source_name": SOURCE_NAME},
    )
