# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the issue #6854 baseline/treatment fuzzy-dedup comparison.

This is a research-only launcher. Run it from the pinned baseline or treatment
branch and pass the matching ``--variant``. The artifact-version guard prevents
accidentally comparing two runs from the same implementation.
"""

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs,
)
from marin.processing.classification.deduplication.fuzzy_minhash import (
    MinHashAttrData,
    compute_minhash_attrs,
)
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import sample_sources
from experiments.datakit.reports.dedup import dedup_report

SAMPLE_PREFIX = "s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f"
OUTPUT_PREFIX = "s3://marin-us-east-02a/marin/user/rav/datakit/dedup-ab/issue6854-100b-20260724-v1"
NUM_PERMS = 286
NUM_BANDS = 26
NGRAM_SIZE = 5
TEXT_CAP_CHARS = 500_000
SEED = 42
CC_MAX_ITERATIONS = 50

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceInventory:
    """Exact parquet inventory for one normalized testbed source."""

    name: str
    shards: int
    rows: int
    bytes: int


def _parquet_stats(item: tuple[str, str]) -> tuple[str, int, int]:
    source_name, path = item
    storage_path = StoragePath(path)
    with storage_path.open("rb") as fh:
        rows = pq.ParquetFile(fh).metadata.num_rows
    return source_name, rows, storage_path.size()


def inventory(sample_prefix: str, output_prefix: str, metadata_workers: int) -> None:
    """Write exact per-source shard, row, and byte counts for the testbed."""
    sources = sample_sources(sample_prefix)
    work: list[tuple[str, str]] = []
    shard_counts: defaultdict[str, int] = defaultdict(int)
    for name, source_step in sources.items():
        source = read_artifact(source_step.output_path, NormalizedData)
        shards = sorted(str(path) for path in StoragePath(f"{source.main_output_dir.rstrip('/')}/*.parquet").glob())
        if not shards:
            raise FileNotFoundError(f"No parquet shards found for {name}: {source.main_output_dir}")
        shard_counts[name] = len(shards)
        work.extend((name, shard) for shard in shards)

    row_counts: defaultdict[str, int] = defaultdict(int)
    byte_counts: defaultdict[str, int] = defaultdict(int)
    with ThreadPoolExecutor(max_workers=metadata_workers) as pool:
        for name, rows, size in pool.map(_parquet_stats, work):
            row_counts[name] += rows
            byte_counts[name] += size

    source_inventory = [
        SourceInventory(
            name=name,
            shards=shard_counts[name],
            rows=row_counts[name],
            bytes=byte_counts[name],
        )
        for name in sorted(sources)
    ]
    payload = {
        "sample_prefix": sample_prefix,
        "source_count": len(source_inventory),
        "shards": sum(source.shards for source in source_inventory),
        "rows": sum(source.rows for source in source_inventory),
        "bytes": sum(source.bytes for source in source_inventory),
        "sources": [asdict(source) for source in source_inventory],
    }
    output_path = f"{output_prefix.rstrip('/')}/inventory.json"
    StoragePath(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Wrote inventory to %s: %s", output_path, {key: payload[key] for key in payload if key != "sources"})


def _assert_variant(variant: str) -> None:
    expected_version = {"baseline": "v2", "treatment": "v3"}[variant]
    actual_version = MinHashAttrData.model_fields["version"].default
    if actual_version != expected_version:
        raise RuntimeError(
            f"{variant} requires MinHashAttrData {expected_version}, but this checkout provides {actual_version}"
        )


def _minhash_kwargs(variant: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_perms": NUM_PERMS,
        "num_bands": NUM_BANDS,
        "ngram_size": NGRAM_SIZE,
        "text_cap_chars": TEXT_CAP_CHARS,
        "seed": SEED,
    }
    if variant == "treatment":
        kwargs["ngram_kind"] = "word"
    return kwargs


def run(
    *,
    variant: str,
    code_ref: str,
    sample_prefix: str,
    output_prefix: str,
    max_workers: int,
    dedup_parallelism: int,
    max_concurrent_sources: int,
) -> None:
    """Run MinHash, connected components, marker emission, and the HTML report."""
    _assert_variant(variant)
    sources = sample_sources(sample_prefix, run_tag=f"issue6854-{variant}-{code_ref}")
    worker = ResourceConfig(cpu=2, ram="8g", disk="16g", preemptible=False)
    coordinator = ResourceConfig(cpu=4, ram="16g", disk="16g", preemptible=False)
    minhash_kwargs = _minhash_kwargs(variant)

    minhash_steps: list[StepSpec] = []
    for name, source_step in sources.items():
        minhash_steps.append(
            StepSpec(
                name=f"research/datakit/issue6854/{variant}/minhash/{name}",
                deps=[source_step],
                fn=lambda output_path, source_step=source_step: compute_minhash_attrs(
                    source=read_artifact(source_step.output_path, NormalizedData),
                    output_path=output_path,
                    worker_resources=worker,
                    max_workers=max_workers,
                    **minhash_kwargs,
                ),
                hash_attrs={
                    "variant": variant,
                    "code_ref": code_ref,
                    **{key: str(value) for key, value in minhash_kwargs.items()},
                },
                override_output_path=f"{output_prefix.rstrip('/')}/{variant}/minhash/{name}",
            )
        )

    dedup_step = StepSpec(
        name=f"research/datakit/issue6854/{variant}/dedup",
        deps=minhash_steps,
        fn=lambda output_path: compute_fuzzy_dups_attrs(
            inputs=[read_artifact(step.output_path, MinHashAttrData) for step in minhash_steps],
            output_path=output_path,
            cc_max_iterations=CC_MAX_ITERATIONS,
            cc_resume=True,
            max_parallelism=dedup_parallelism,
            worker_resources=worker,
            coordinator_resources=coordinator,
        ),
        hash_attrs={
            "variant": variant,
            "code_ref": code_ref,
            "cc_max_iterations": CC_MAX_ITERATIONS,
            "cc_resume": True,
        },
        override_output_path=f"{output_prefix.rstrip('/')}/{variant}/dedup",
    )
    report_step = StepSpec(
        name=f"research/datakit/issue6854/{variant}/report",
        deps=[dedup_step],
        fn=lambda output_path: dedup_report(
            output_path,
            read_artifact(dedup_step.output_path, FuzzyDupsAttrData),
        ),
        hash_attrs={"variant": variant, "code_ref": code_ref},
        override_output_path=f"{output_prefix.rstrip('/')}/{variant}/report",
    )

    StepRunner().run([report_step], max_concurrent=max_concurrent_sources)

    manifest = {
        "variant": variant,
        "code_ref": code_ref,
        "sample_prefix": sample_prefix,
        "source_count": len(sources),
        "minhash_artifacts": [f"{step.output_path}/.artifact.json" for step in minhash_steps],
        "dedup_artifact": f"{dedup_step.output_path}/.artifact.json",
        "report_artifact": f"{report_step.output_path}/.artifact.json",
        "report_html": f"{report_step.output_path}/report.html",
        "parameters": {
            **{key: str(value) for key, value in minhash_kwargs.items()},
            "cc_max_iterations": CC_MAX_ITERATIONS,
            "cc_resume": True,
            "max_workers": max_workers,
            "dedup_parallelism": dedup_parallelism,
            "max_concurrent_sources": max_concurrent_sources,
        },
    }
    manifest_path = f"{output_prefix.rstrip('/')}/{variant}/manifest.json"
    StoragePath(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("Wrote run manifest to %s", manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-prefix", default=SAMPLE_PREFIX)
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--metadata-workers", type=int, default=32)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--variant", choices=("baseline", "treatment"), required=True)
    run_parser.add_argument("--code-ref", required=True)
    run_parser.add_argument("--max-workers", type=int, default=128)
    run_parser.add_argument("--dedup-parallelism", type=int, default=512)
    run_parser.add_argument("--max-concurrent-sources", type=int, default=4)

    args = parser.parse_args()
    configure_logging(logging.INFO)
    if args.command == "inventory":
        inventory(args.sample_prefix, args.output_prefix, args.metadata_workers)
        return
    run(
        variant=args.variant,
        code_ref=args.code_ref,
        sample_prefix=args.sample_prefix,
        output_prefix=args.output_prefix,
        max_workers=args.max_workers,
        dedup_parallelism=args.dedup_parallelism,
        max_concurrent_sources=args.max_concurrent_sources,
    )


if __name__ == "__main__":
    main()
