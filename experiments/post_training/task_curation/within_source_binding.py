# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind mandatory deduplication evidence to the configured source population."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from finestore.rl.curation import WithinSourceDedupEvidence

from experiments.post_training.task_curation.source import RlDataSource
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id


def read_dedup_receipts(path: Path, *, load_json: Callable[[Path], dict]) -> dict[str, WithinSourceDedupEvidence]:
    return {
        canonical_source_id(source_id): WithinSourceDedupEvidence.model_validate(record)
        for source_id, record in load_json(path).items()
    }


def bind_within_source_dedup(source: RlDataSource, receipt: WithinSourceDedupEvidence) -> RlDataSource:
    dataset = source.dataset
    if dataset is None or (receipt.source_dataset_id, receipt.source_revision, receipt.converter_version) != (
        dataset.name,
        dataset.revision,
        source.version,
    ):
        raise ValueError(f"Within-source deduplication census does not match configured source: {source.info.id}")
    return replace(source, info=replace(source.info, within_source_dedup=receipt, count=receipt.unique_count))
