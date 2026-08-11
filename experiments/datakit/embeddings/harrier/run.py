# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed selected Datakit document sets with Harrier."""

import argparse

from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from rigging.log_setup import configure_logging

from experiments.datakit.embeddings.harrier.pipeline import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_ATTR_DATA_VERSION,
    HARRIER_MAX_TOKENS,
    HARRIER_REPO,
    HARRIER_REVISION,
    EmbeddingDocumentSet,
    embed_source,
    stage_harrier,
)
from experiments.datakit.embeddings.harrier.tei import tei_service_pool
from experiments.datakit.reference_pipeline import select_sources

DEDUP_ID = "dedup_709f5997"
DEDUP_PATH = f"s3://marin-us-east-02a/marin/datakit/{DEDUP_ID}"
WORKERS_PER_SOURCE = 32
MAX_CONCURRENT = 8
TEI_INSTANCES = 128
_DEDUPLICATED_STEP_PREFIX = "datakit/embed/harrier"
_FUZZY_DUPLICATE_STEP_PREFIX = "datakit/embed/harrier-fuzzy-duplicates"


def _embed_source(
    output_path: str,
    normalized_path: str,
    endpoint_name: str,
    document_set: EmbeddingDocumentSet,
) -> None:
    normalized = read_artifact(normalized_path, NormalizedData)
    dedup = read_artifact(DEDUP_PATH, FuzzyDupsAttrData)
    embed_source(
        output_path=output_path,
        normalized=normalized,
        endpoint_name=endpoint_name,
        document_set=document_set,
        dedup_attr_dir=dedup.attr_dir_for_source(normalized.main_output_dir),
        max_workers=WORKERS_PER_SOURCE,
    )


def build_steps(
    endpoint_name: str,
    document_set: EmbeddingDocumentSet = EmbeddingDocumentSet.DEDUPLICATED,
    partition_index: int = 0,
    partition_count: int = 1,
) -> list[StepSpec]:
    """Build one embedding step per source in a deterministic partition."""
    if not 0 <= partition_index < partition_count:
        raise ValueError(f"partition index {partition_index} must be in [0, {partition_count})")

    sources = list(select_sources().items())[partition_index::partition_count]
    if document_set == EmbeddingDocumentSet.DEDUPLICATED:
        step_prefix = _DEDUPLICATED_STEP_PREFIX
        document_set_hash = {}
    elif document_set == EmbeddingDocumentSet.FUZZY_DUPLICATES:
        step_prefix = _FUZZY_DUPLICATE_STEP_PREFIX
        document_set_hash = {"document_set": document_set.value}
    else:
        raise ValueError(f"Unsupported production document set: {document_set.value}")

    return [
        StepSpec(
            name=f"{step_prefix}/{source_name}",
            deps=[normalized],
            hash_attrs={
                "dedup": DEDUP_ID,
                "model": HARRIER_REPO,
                "revision": HARRIER_REVISION,
                "batch_size": DEFAULT_BATCH_SIZE,
                "v": EMBEDDING_ATTR_DATA_VERSION,
                **document_set_hash,
            },
            fn=remote(
                lambda output_path, normalized_path=normalized.output_path: _embed_source(
                    output_path, normalized_path, endpoint_name, document_set
                ),
                resources=ResourceConfig(cpu=2, ram="8g", disk="8g"),
                pip_dependency_groups=["datakit"],
            ),
        )
        for source_name, normalized in sources
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--tei-instances", type=int, default=TEI_INSTANCES)
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument(
        "--document-set",
        type=EmbeddingDocumentSet,
        choices=[EmbeddingDocumentSet.DEDUPLICATED, EmbeddingDocumentSet.FUZZY_DUPLICATES],
        default=EmbeddingDocumentSet.DEDUPLICATED,
    )
    args = parser.parse_args()

    configure_logging()
    model_archive = stage_harrier(HARRIER_REPO, HARRIER_REVISION, DEDUP_PATH)
    with tei_service_pool(
        model_archive,
        instances=args.tei_instances,
        max_input_tokens=HARRIER_MAX_TOKENS,
    ) as endpoint_name:
        StepRunner().run(
            build_steps(
                endpoint_name,
                document_set=args.document_set,
                partition_index=args.partition_index,
                partition_count=args.partition_count,
            ),
            max_concurrent=args.max_concurrent,
        )


if __name__ == "__main__":
    main()
