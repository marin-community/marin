# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adopt the completed 50M-document Harrier embeddings."""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep


def build() -> ArtifactStep[Artifact]:
    """Return the completed Harrier embedding artifact."""
    return ArtifactStep.adopt(
        name="datakit/embeddings/harrier-oss-v1-0.6b-50m",
        version="2026.08.04",
        source="s3://marin-us-east-02a/marin/user/held/harrier-oss-v1-0.6b-50m",
        config={
            "dataset": "s3://marin-us-east-02a/marin/datakit/sample_10pct_91269634",
            "input_plan_sha256": "791ce33496e7e99d54c17c4dfb5d71ce20a1273f021fef4f67c54da72e71e97c",
            "model_id": "microsoft/harrier-oss-v1-0.6b",
            "model_revision": "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
            "rows": 50_000_000,
            "max_tokens": 8_192,
            "producer_branch": "https://github.com/marin-community/marin/tree/held/harrier-50m-run",
            "producer_commit": "46b4f8b2dd5ed4d2faa1ebdff13953e1a1001c75",
        },
    )
