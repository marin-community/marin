# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the default Zephyr benchmark sample in GCS.

The source is the existing CoreWeave 100B sample. Each step copies one
source's normalized Parquet shards byte-for-byte and writes a new
``NormalizedData`` artifact rooted in GCS. Duplicate side outputs are omitted
because the benchmark consumes only normalized main outputs.

Run once from the destination region after exporting CoreWeave object-storage
credentials. The materializer runs separately from benchmark jobs::

    uv run iris --cluster=marin job run --no-wait \
        --region us-central1 --memory=8G --disk=5G --cpu=4 --extra=cpu \
        --priority batch \
        -e CW_KEY_ID "$CW_KEY_ID" -e CW_KEY_SECRET "$CW_KEY_SECRET" \
        -- python -m experiments.datakit.materialize_zephyr_benchmark_sample
"""

import argparse
import logging

from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import sample_sources
from experiments.datakit.testbed.sampler import sample_normalized_shards
from experiments.datakit.zephyr_benchmark import (
    COREWEAVE_BENCHMARK_SAMPLE_PREFIX,
    GCP_BENCHMARK_SAMPLE_PREFIX,
)

DEFAULT_MAX_CONCURRENT = 4

logger = logging.getLogger(__name__)


def mirror_sample_source_step(name: str, source: StepSpec, destination_prefix: str) -> StepSpec:
    """Copy one normalized benchmark source into ``destination_prefix``."""
    source_path = source.output_path

    def mirror(output_path: str) -> NormalizedData:
        mirrored = sample_normalized_shards(
            source=read_artifact(source_path, NormalizedData),
            output_path=output_path,
            sample_fraction=1.0,
        )
        return mirrored.model_copy(update={"dup_output_dir": ""})

    return StepSpec(
        name=f"datakit/benchmark_sample/{name}",
        deps=[source],
        hash_attrs={"source_path": source_path, "version": "v1"},
        fn=mirror,
        override_output_path=prefix_join(destination_prefix, name),
    )


def materialize_sample_steps(source_prefix: str, destination_prefix: str) -> list[StepSpec]:
    """Build one mirror step for every source in the benchmark sample."""
    sources = sample_sources(source_prefix)
    return [
        mirror_sample_source_step(name, source, destination_prefix) for name, source in sorted(sources.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", default=COREWEAVE_BENCHMARK_SAMPLE_PREFIX)
    parser.add_argument("--destination-prefix", default=GCP_BENCHMARK_SAMPLE_PREFIX)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    if not args.source_prefix.startswith("s3://"):
        raise ValueError(f"source prefix must use S3: {args.source_prefix}")
    if not args.destination_prefix.startswith("gs://"):
        raise ValueError(f"destination prefix must use GCS: {args.destination_prefix}")

    configure_coreweave_s3()
    steps = materialize_sample_steps(args.source_prefix, args.destination_prefix)
    StepRunner().run(steps, max_concurrent=args.max_concurrent)

    source_names = {step.name.removeprefix("datakit/benchmark_sample/") for step in steps}
    destination_names = set(sample_sources(args.destination_prefix))
    if destination_names != source_names:
        raise RuntimeError(
            f"materialized source set differs: missing={sorted(source_names - destination_names)}, "
            f"unexpected={sorted(destination_names - source_names)}"
        )
    logger.info("Materialized %d benchmark sources at %s", len(steps), args.destination_prefix)


if __name__ == "__main__":
    main()
