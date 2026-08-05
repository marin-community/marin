# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed every source in Rav's completed 292-source dedup artifact with Harrier."""

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
    HARRIER_REPO,
    HARRIER_REVISION,
    embed_source,
)
from experiments.datakit.reference_pipeline import select_sources

DEDUP_PATH = "s3://marin-us-east-02a/marin/datakit/dedup_709f5997"
WORKERS_PER_SOURCE = 32
MAX_CONCURRENT = 8
COORDINATOR_RESOURCES = ResourceConfig(
    cpu=2,
    ram="8g",
    disk="8g",
)


def _embed_source(output_path: str, normalized_path: str) -> None:
    normalized = read_artifact(normalized_path, NormalizedData)
    dedup = read_artifact(DEDUP_PATH, FuzzyDupsAttrData)
    embed_source(
        output_path=output_path,
        normalized=normalized,
        dedup_attr_dir=dedup.attr_dir_for_source(normalized.main_output_dir),
        max_workers=WORKERS_PER_SOURCE,
    )


def build() -> list[StepSpec]:
    sources = select_sources()
    return [
        StepSpec(
            name=f"datakit/embed/harrier/{source_name}",
            deps=[normalized],
            hash_attrs={
                "dedup": "dedup_709f5997",
                "model": HARRIER_REPO,
                "revision": HARRIER_REVISION,
                "batch_size": DEFAULT_BATCH_SIZE,
                "v": EMBEDDING_ATTR_DATA_VERSION,
            },
            fn=remote(
                lambda output_path, normalized_path=normalized.output_path: _embed_source(output_path, normalized_path),
                resources=COORDINATOR_RESOURCES,
                pip_dependency_groups=["datakit"],
            ),
        )
        for source_name, normalized in sources.items()
    ]


if __name__ == "__main__":
    configure_logging()
    StepRunner().run(build(), max_concurrent=MAX_CONCURRENT)
