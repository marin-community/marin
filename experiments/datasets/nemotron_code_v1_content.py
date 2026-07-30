# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconstructed Nemotron-Pretraining-Code-v1 file contents as a lazy artifact.

NVIDIA publishes ``Nemotron-Pretraining-Code-v1 / Nemotron-Code-Metadata`` as metadata
only — ``(repo, commit_id, rel_path)`` triples — because its license forbids
redistributing file contents. We reconstructed the actual file bytes from that metadata by
resolving each ``(repo, commit, path)`` to a content SWHID through the Software Heritage
compressed graph (2025-05-18 export), mapping ``sha1_git -> sha1`` via SWH's ORC content
table, and fetching the bytes from SWH's public content bucket.

The dataset lives on Cloudflare R2 (``s3://marin-na/users/held``). This module adopts that
copy as a lazy ``Artifact`` handle so experiments can depend on it without recomputing.
Unlike the v2 module — whose served copy sits on GCS — v1 has not been copied to GCS, so
reading it needs the R2 endpoint + credentials in the environment.

Layout of the adopted dataset (Parquet, zstd), one row per distinct content:

    sha1_git : str          40-hex git blob hash; matches ``resolved.cnt_swhid``
    sha1     : str          40-hex content sha1 (the content-bucket key)
    content  : large_binary raw, already-decompressed file bytes
    present  : bool         True if fetched; False -> ``content`` is b""

513,109,851 of 513,110,667 distinct contents present (99.9998%); 585 GB across 514 shards.
Each stored content is byte-verified: ``sha1(content) == sha1``.
"""

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep

NEMOTRON_CODE_V1_CONTENT_SOURCE = "s3://marin-na/users/held/nemotron-code-v1-content"


def nemotron_code_v1_content() -> ArtifactStep[Artifact]:
    """The reconstructed v1 code contents as an adopted lazy artifact handle."""
    return ArtifactStep.adopt(
        "raw/nemotron-code-v1-content",
        "2026.07.30",
        NEMOTRON_CODE_V1_CONTENT_SOURCE,
        kind=Artifact,
        config={
            "format": "parquet",
            "columns": ["sha1_git", "sha1", "content", "present"],
            "rows": 513_110_667,
            "present_rows": 513_109_851,
            "shards": 514,
            "graph_export": "2025-05-18",
        },
    )
