# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Require pinned, explicit supervision modes for a prepared long-loop mixture."""

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum

import draccus
from levanter.data.text.datasets import ConcatDatasetComponent, DatasetComponent, LmDataConfig
from levanter.data.text.formats import PrebuiltLmDatasetFormat
from rigging.filesystem.storage_path import StoragePath


class SourceLossMode(StrEnum):
    CHAT_ASSISTANT_ONLY = "chat_assistant_only"
    FULL_TOKEN = "full_token"


@dataclass(frozen=True)
class SourceLossMetadata:
    loss_mode: SourceLossMode
    loss_weights_key: str
    loss_weights_indexing: str
    teacher_only: bool
    source_revision: str


def source_loss_metadata(path: str, sha256: str) -> dict[str, SourceLossMetadata]:
    """Read a hash-pinned sidecar with no implicit source or supervision defaults."""
    content = StoragePath(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError("Source supervision metadata SHA256 mismatch")
    document = json.loads(content)
    if document["version"] != 1:
        raise ValueError("Unsupported source supervision metadata version")
    sources = {name: draccus.decode(SourceLossMetadata, value) for name, value in document["sources"].items()}
    for name, source in sources.items():
        if source.loss_weights_indexing != "prediction" or not source.loss_weights_key or not source.source_revision:
            raise ValueError(f"Source {name} requires prediction-indexed weights and a source revision")
        if source.teacher_only and source.loss_mode != SourceLossMode.CHAT_ASSISTANT_ONLY:
            raise ValueError(f"Teacher-only source {name} must be a chat source")
    return sources


def prepared_supervision_data(
    data: LmDataConfig, sources: dict[str, SourceLossMetadata], *, packing_cap: int, lookahead: int | None
) -> LmDataConfig:
    """Bind every source's precomputed prediction weights and bounded packing policy."""
    if set(data.components) != set(sources):
        raise ValueError("Every catalog component requires exactly one explicit supervision mode")

    def bind(component, source):
        if isinstance(component, ConcatDatasetComponent):
            return replace(component, children={name: bind(child, source) for name, child in component.children.items()})
        if not isinstance(component, DatasetComponent) or not isinstance(component.format, PrebuiltLmDatasetFormat):
            raise ValueError("Prepared supervision requires prebuilt token and prediction-weight caches")
        if component.format.loss_weights_key != source.loss_weights_key:
            raise ValueError("Catalog and supervision metadata disagree on the prediction-weight field")
        return replace(component, pack=packing_cap, pack_lookahead_window=lookahead)

    return replace(
        data,
        components={name: bind(component, sources[name]) for name, component in data.components.items()},
        block_cross_document_attention=True,
    )
