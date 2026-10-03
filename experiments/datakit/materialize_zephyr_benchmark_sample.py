# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create a Zephyr benchmark sample by copying or regenerating normalized data.

``--mode copy`` copies normalized Parquet shards from any existing sample to a
new sample root, rewriting ``NormalizedData`` artifacts for that destination.
``--mode regenerate`` downloads the registered source data, normalizes it, and
samples the requested token count (100B by default) into a new sample root. See
``experiments/datakit/README.md`` for the required region-local Iris commands
and cost caveats.
"""

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from enum import StrEnum
from pathlib import Path

from marin.datakit.normalize import NormalizedData
from marin.datakit.sources import DatakitSource, all_sources
from marin.execution.artifact import read_artifact, write_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import data_config, use_data_config
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import sample_sources
from experiments.datakit.testbed.sampler import SampleManifest, proportional_sample_fractions, sample_normalized_shards
from experiments.datakit.zephyr_benchmark import GCP_BENCHMARK_SAMPLE_PREFIX

DEFAULT_MAX_CONCURRENT = 4
MATERIALIZE_STEP_PREFIX = "datakit/benchmark_sample"
DEFAULT_TARGET_TOTAL_TOKENS_B = 100.0

logger = logging.getLogger(__name__)


class SampleMode(StrEnum):
    """How to create the benchmark sample."""

    COPY = "copy"
    REGENERATE = "regenerate"


def _sample_main_output_step(
    name: str,
    source: StepSpec,
    destination_prefix: str,
    sample_fraction: float,
) -> StepSpec:
    """Write a normalized source's sampled main output under ``destination_prefix``."""
    source_path = source.output_path

    def sample(output_path: str) -> NormalizedData:
        source_data = read_artifact(source_path, NormalizedData).model_copy(
            update={"main_output_dir": prefix_join(source_path, "outputs/main")}
        )
        sampled = sample_normalized_shards(
            source=source_data,
            output_path=output_path,
            sample_fraction=sample_fraction,
        )
        return sampled.model_copy(update={"dup_output_dir": ""})

    return StepSpec(
        name=f"{MATERIALIZE_STEP_PREFIX}/{name}",
        deps=[source],
        hash_attrs={"source_path": source_path, "sample_fraction": sample_fraction, "version": "v2"},
        fn=sample,
        override_output_path=prefix_join(destination_prefix, name),
    )


def copy_sample_steps(source_prefix: str, destination_prefix: str, names: list[str] | None = None) -> list[StepSpec]:
    """Build copy steps for selected sources in an existing benchmark sample."""
    sources = sample_sources(source_prefix, names=names)
    if not sources:
        raise ValueError(f"no normalized source artifacts found under {source_prefix}")
    return [
        _sample_main_output_step(name, source, destination_prefix, sample_fraction=1.0)
        for name, source in sorted(sources.items())
    ]


def regenerate_sample_steps(
    sources: Sequence[DatakitSource],
    destination_prefix: str,
    target_total_tokens_b: float,
    source_weights: Mapping[str, float] | None = None,
) -> list[StepSpec]:
    """Build the source download, normalization, and sampling steps for a fresh benchmark sample."""
    if not sources:
        raise ValueError("Select at least one source")
    fractions = proportional_sample_fractions(sources, target_total_tokens_b, source_weights)
    return [
        _sample_main_output_step(source.name, source.normalized, destination_prefix, fractions[source.name])
        for source in sorted(sources, key=lambda source: source.name)
    ]


def _validate_data_prefix(data_prefix: str, destination_prefix: str) -> None:
    data_root = StoragePath(data_prefix)
    destination = StoragePath(destination_prefix)
    if data_root.scheme not in ("gs", "s3") or not data_root.bucket:
        raise ValueError(f"data prefix must be an object-store root: {data_prefix}")
    if destination.scheme != data_root.scheme or destination.bucket != data_root.bucket:
        raise ValueError(f"destination {destination_prefix} must be under data prefix {data_prefix}")
    destination.relative_to(data_root)


def _verify_source_set(expected_names: set[str], destination_prefix: str) -> None:
    destination_names = set(sample_sources(destination_prefix))
    if destination_names != expected_names:
        raise RuntimeError(
            f"materialized source set differs: missing={sorted(expected_names - destination_names)}, "
            f"unexpected={sorted(destination_names - expected_names)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=list(SampleMode), type=SampleMode, default=SampleMode.COPY)
    parser.add_argument(
        "--source-prefix", help="Existing sample. Regeneration without this option uses the full registry."
    )
    parser.add_argument("--destination-prefix", default=GCP_BENCHMARK_SAMPLE_PREFIX)
    parser.add_argument(
        "--data-prefix",
        help="Object-store root for source downloads and normalized artifacts (--mode regenerate only).",
    )
    parser.add_argument("--target-total-tokens-b", type=float, default=DEFAULT_TARGET_TOTAL_TOKENS_B)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sources", help="Comma-separated source names.")
    selection.add_argument("--source-mixture", type=Path, help="JSON file of source names and relative token weights.")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    if args.max_concurrent < 1:
        raise ValueError(f"max concurrent must be positive: {args.max_concurrent}")
    if args.target_total_tokens_b <= 0:
        raise ValueError(f"target total tokens must be positive: {args.target_total_tokens_b}")
    weights = json.loads(args.source_mixture.read_text()) if args.source_mixture is not None else None
    names = [name.strip() for name in args.sources.split(",") if name.strip()] if args.sources is not None else None
    if weights is not None:
        names = list(weights)
    if names == []:
        raise ValueError("Select at least one source")
    if args.destination_prefix.startswith("s3://") or (args.source_prefix or "").startswith("s3://"):
        configure_coreweave_s3()

    if args.mode is SampleMode.COPY:
        if weights is not None:
            raise ValueError("--source-mixture applies only to --mode regenerate")
        if args.source_prefix is None:
            raise ValueError("--source-prefix is required for copy mode")
        if args.data_prefix is not None:
            raise ValueError("--data-prefix applies only to --mode regenerate")
        if StoragePath(args.source_prefix) == StoragePath(args.destination_prefix):
            raise ValueError("source and destination prefixes must differ")
        steps = copy_sample_steps(args.source_prefix, args.destination_prefix, names)
    else:
        if args.data_prefix is None:
            raise ValueError("--data-prefix is required for --mode regenerate")
        _validate_data_prefix(args.data_prefix, args.destination_prefix)
        with use_data_config(replace(data_config(), root=args.data_prefix)):
            registry = all_sources()
            if args.source_prefix is not None:
                available = sample_sources(args.source_prefix, names=names)
                names = list(available)
            selected = list(registry) if names is None else names
            missing = sorted(set(selected) - set(registry))
            if missing:
                raise ValueError(f"source registry no longer defines {missing}")
            if weights is None:
                weights = {name: registry[name].rough_token_count_b for name in selected}
            steps = regenerate_sample_steps(
                [registry[name] for name in selected],
                args.destination_prefix,
                args.target_total_tokens_b,
                weights,
            )
            StepRunner().run(steps, max_concurrent=args.max_concurrent)
    if args.mode is SampleMode.COPY:
        StepRunner().run(steps, max_concurrent=args.max_concurrent)

    source_names = {step.name.removeprefix(f"{MATERIALIZE_STEP_PREFIX}/") for step in steps}
    _verify_source_set(source_names, args.destination_prefix)
    write_artifact(
        output_path=args.destination_prefix,
        value=SampleManifest(
            source_paths={
                step.name.removeprefix(f"{MATERIALIZE_STEP_PREFIX}/"): step.deps[0].output_path for step in steps
            },
            target_total_tokens_b=args.target_total_tokens_b if args.mode is SampleMode.REGENERATE else None,
            source_weights=weights,
        ),
    )
    logger.info("Created %d benchmark sources at %s with %s", len(steps), args.destination_prefix, args.mode)


if __name__ == "__main__":
    main()
