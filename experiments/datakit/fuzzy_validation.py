# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run fuzzy validation as one target in the production Datakit graph.

The entry point builds the normalized-source, MinHash, and fuzzy-dedup
dependencies from the current source registry. ``StepRunner`` receives only the
fuzzy-validation terminal. Completed dependencies come from the artifact cache.
The store and report steps are not part of this graph.

Submit the target in the CoreWeave data region::

    uv run iris --cluster=marin job run --no-wait \
        --target-cluster cw-us-east-08a --priority production \
        --cpu 2 --memory 8g --enable-extra-resources \
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \
        -- python -m experiments.datakit.fuzzy_validation \
            --max-workers 64 --worker-cpu 115.2 \
            --worker-ram 864GB --worker-disk 16g \
            --task-cpu 1 --task-ram 2GB --task-disk 1g \
            --coordinator-cpu 2 --coordinator-ram 8GB \
            --max-output-shards 4096 \
            --recovery-timeout 1800 --ready-timeout 28800

Add ``--dry-run`` to read the cache state without job submission.

Use ``--reuse-legacy-focus-candidates`` for the one-off Focus Crawl test. This
mode repacks the prior candidate rows for the current normalized Focus Crawl.
It writes the repack and validation artifacts to a temporary S3 path. Add
``--production-output`` to keep the repack in that path and write validation
below ``MARIN_PREFIX``.
"""

import argparse
import logging
from dataclasses import replace

from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_path
from marin.execution.artifact import read_artifact, read_record
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from marin.processing.classification.deduplication.repack_fuzzy_dups import repack_fuzzy_dups_source
from marin.processing.classification.deduplication.verify_fuzzy_dups import (
    REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
    FuzzyVerificationStoreConfig,
    verify_fuzzy_dups_step,
)
from rigging.filesystem import marin_prefix, marin_temp_bucket, prefix_join
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import (
    DEFAULT_SCALE,
    PipelineScale,
    PoolConfig,
    select_sources,
    zephyr_datakit_steps,
)

logger = logging.getLogger(__name__)

DEFAULT_RECOVERY_TIMEOUT = 1_800
DEFAULT_READY_TIMEOUT = 1_800
DEFAULT_LOOKUP_BATCH_SIZE = 128
DEFAULT_LOAD_CONCURRENCY = 1
DEFAULT_MAX_OUTPUT_SHARDS = 4_096
DEFAULT_TASK_CPU = 1.0
DEFAULT_TASK_RAM = "2GB"
DEFAULT_TASK_DISK = "1g"
FOCUS_SOURCE_NAME = "common-crawl-focus-2026-22"
LEGACY_FUZZY_CANDIDATE_ARTIFACT = "s3://marin-us-east-02a/marin/datakit/dedup_709f5997"
LEGACY_FOCUS_SOURCE_KEY = "data/datakit/normalized/common_crawl_focus_2026_22_ed4b8bc9/outputs/main"
TEMP_OUTPUT_PREFIX = "fuzzy-validation"
DEFAULT_TEMP_TTL_DAYS = 7
DEFAULT_REPACK_MAX_WORKERS = 64
DEFAULT_REPACK_WORKER = ResourceConfig(cpu=4, ram="16GB", disk="16g")
TEMP_VALIDATION_STEP_NAME = "fuzzy-validation/verify-repacked-candidates"
PRODUCTION_VALIDATION_STEP_NAME = "datakit/verify_fuzzy_dups"


def build_fuzzy_validation_step(
    sources: dict[str, StepSpec],
    *,
    scale: PipelineScale = DEFAULT_SCALE,
    store_config: FuzzyVerificationStoreConfig | None = None,
    coordinator_resources: ResourceConfig | None = None,
    task_resources: ResourceConfig | None = None,
    max_output_shards: int = DEFAULT_MAX_OUTPUT_SHARDS,
) -> StepSpec:
    """Build the fuzzy-validation terminal and its Datakit dependencies."""
    if store_config is None:
        store_config = FuzzyVerificationStoreConfig(
            recovery_timeout=DEFAULT_RECOVERY_TIMEOUT,
            ready_timeout=DEFAULT_READY_TIMEOUT,
            lookup_batch_size=DEFAULT_LOOKUP_BATCH_SIZE,
        )
    datakit = zephyr_datakit_steps(sources, scale)
    return verify_fuzzy_dups_step(
        name="datakit/verify_fuzzy_dups",
        normalized_steps=sources,
        minhash_steps=datakit.minhash,
        candidates_step=datakit.fuzzy_dedup,
        verification_params=FuzzyVerificationParams(),
        local_representative_params=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
        store_config=store_config,
        max_output_shards=max_output_shards,
        max_workers=scale.pool.n_workers,
        worker_resources=scale.pool.worker,
        coordinator_resources=coordinator_resources,
        map_task_resources=task_resources,
        reduce_task_resources=task_resources,
    )


def build_repacked_fuzzy_validation_step(
    sources: dict[str, StepSpec],
    minhash_steps: dict[str, StepSpec],
    *,
    candidate_artifact_path: str,
    legacy_source_key: str,
    source_name: str,
    repack_output_path_prefix: str,
    validation_output_path_prefix: str,
    validation_step_name: str,
    validation_scale: PipelineScale,
    store_config: FuzzyVerificationStoreConfig,
    coordinator_resources: ResourceConfig,
    task_resources: ResourceConfig,
    max_output_shards: int = DEFAULT_MAX_OUTPUT_SHARDS,
    repack_max_workers: int = DEFAULT_REPACK_MAX_WORKERS,
    repack_worker_resources: ResourceConfig = DEFAULT_REPACK_WORKER,
) -> StepSpec:
    """Build a validation graph that reuses one prior candidate source."""
    normalized_step = sources.get(source_name)
    if normalized_step is None:
        raise KeyError(f"The source registry has no source named {source_name!r}")

    candidate_step = StepSpec(
        name="fuzzy-validation/legacy-candidates",
        override_output_path=candidate_artifact_path,
    )
    repack_step = StepSpec(
        name=f"fuzzy-validation/repack/{source_name}",
        deps=[candidate_step, normalized_step],
        fn=lambda output_path: repack_fuzzy_dups_source(
            candidates=read_artifact(candidate_step.output_path, FuzzyDupsAttrData),
            legacy_source_key=legacy_source_key,
            normalized=read_artifact(normalized_step.output_path, NormalizedData),
            output_path=output_path,
            max_workers=repack_max_workers,
            worker_resources=repack_worker_resources,
            coordinator_resources=coordinator_resources,
        ),
        hash_attrs={
            "v": 1,
            "candidate_artifact_path": candidate_artifact_path,
            "legacy_source_key": legacy_source_key,
            "source_name": source_name,
        },
        output_path_prefix=repack_output_path_prefix,
    )
    target = verify_fuzzy_dups_step(
        name=validation_step_name,
        normalized_steps=sources,
        minhash_steps=minhash_steps,
        candidates_step=repack_step,
        verification_params=FuzzyVerificationParams(),
        local_representative_params=REFERENCE_LOCAL_REPRESENTATIVE_PARAMS,
        store_config=store_config,
        max_output_shards=max_output_shards,
        max_workers=validation_scale.pool.n_workers,
        worker_resources=validation_scale.pool.worker,
        coordinator_resources=coordinator_resources,
        map_task_resources=task_resources,
        reduce_task_resources=task_resources,
    )
    return replace(target, output_path_prefix=validation_output_path_prefix)


def _legacy_candidate_input_steps(
    *,
    candidate_artifact_path: str,
    legacy_focus_source_key: str,
    focus_normalized_step: StepSpec,
    focus_minhash_step: StepSpec,
) -> tuple[dict[str, StepSpec], dict[str, StepSpec]]:
    """Load the input graph that produced a prior fuzzy candidate artifact."""
    candidates = read_artifact(candidate_artifact_path, FuzzyDupsAttrData)
    record = read_record(candidate_artifact_path)
    if record is None or not record.dep_paths:
        raise ValueError(f"Candidate artifact has no MinHash dependency paths: {candidate_artifact_path}")

    minhash_by_source: dict[str, tuple[str, MinHashAttrData]] = {}
    for path in record.dep_paths:
        minhash = read_artifact(path, MinHashAttrData)
        if minhash.source_key in minhash_by_source:
            raise ValueError(f"Candidate artifact has two MinHash inputs for source_key={minhash.source_key!r}")
        minhash_by_source[minhash.source_key] = (path, minhash)

    candidate_source_keys = set(candidates.sources)
    if set(minhash_by_source) != candidate_source_keys:
        missing = sorted(candidate_source_keys - minhash_by_source.keys())
        extra = sorted(minhash_by_source.keys() - candidate_source_keys)
        raise ValueError(f"Candidate and MinHash source keys disagree. Missing={missing}, extra={extra}")
    if legacy_focus_source_key not in candidate_source_keys:
        raise KeyError(f"Candidate artifact has no Focus source_key={legacy_focus_source_key!r}")

    normalized_steps: dict[str, StepSpec] = {}
    minhash_steps: dict[str, StepSpec] = {}
    for index, source_key in enumerate(sorted(candidate_source_keys)):
        if source_key == legacy_focus_source_key:
            normalized_steps[FOCUS_SOURCE_NAME] = focus_normalized_step
            minhash_steps[FOCUS_SOURCE_NAME] = focus_minhash_step
            continue

        source_path = datakit_source_path(source_key)
        suffix = "/outputs/main"
        if not source_path.endswith(suffix):
            raise ValueError(f"Normalized source path does not end in {suffix!r}: {source_path}")
        source_name = f"legacy-source-{index:03d}"
        normalized_steps[source_name] = StepSpec(
            name=f"fuzzy-validation/pinned-normalized/{index:03d}",
            override_output_path=source_path.removesuffix(suffix),
        )
        minhash_steps[source_name] = StepSpec(
            name=f"fuzzy-validation/pinned-minhash/{index:03d}",
            override_output_path=minhash_by_source[source_key][0],
        )

    return normalized_steps, minhash_steps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_SCALE.pool.n_workers)
    parser.add_argument("--worker-cpu", type=float, default=DEFAULT_SCALE.pool.worker.cpu)
    parser.add_argument("--worker-ram", default=DEFAULT_SCALE.pool.worker.ram)
    parser.add_argument("--worker-disk", default=DEFAULT_SCALE.pool.worker.disk)
    parser.add_argument("--task-cpu", type=float, default=DEFAULT_TASK_CPU)
    parser.add_argument("--task-ram", default=DEFAULT_TASK_RAM)
    parser.add_argument("--task-disk", default=DEFAULT_TASK_DISK)
    parser.add_argument("--coordinator-cpu", type=float, default=0.1)
    parser.add_argument("--coordinator-ram", default="1g")
    parser.add_argument("--recovery-timeout", type=int, default=DEFAULT_RECOVERY_TIMEOUT)
    parser.add_argument("--ready-timeout", type=int, default=DEFAULT_READY_TIMEOUT)
    parser.add_argument("--lookup-batch-size", type=int, default=DEFAULT_LOOKUP_BATCH_SIZE)
    parser.add_argument("--load-concurrency", type=int, default=DEFAULT_LOAD_CONCURRENCY)
    parser.add_argument("--max-output-shards", type=int, default=DEFAULT_MAX_OUTPUT_SHARDS)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--reuse-legacy-focus-candidates", action="store_true")
    parser.add_argument("--production-output", action="store_true")
    parser.add_argument("--run-tag")
    parser.add_argument("--temporary-ttl-days", type=int, default=DEFAULT_TEMP_TTL_DAYS)
    parser.add_argument("--candidate-artifact", default=LEGACY_FUZZY_CANDIDATE_ARTIFACT)
    parser.add_argument("--legacy-focus-source-key", default=LEGACY_FOCUS_SOURCE_KEY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.reuse_legacy_focus_candidates and not args.run_tag:
        parser.error("--run-tag is necessary with --reuse-legacy-focus-candidates")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk)
    task_resources = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk=args.task_disk)
    scale = replace(DEFAULT_SCALE, pool=PoolConfig(n_workers=args.max_workers, worker=worker))
    coordinator_resources = ResourceConfig(cpu=args.coordinator_cpu, ram=args.coordinator_ram, preemptible=False)
    store_config = FuzzyVerificationStoreConfig(
        recovery_timeout=args.recovery_timeout,
        ready_timeout=args.ready_timeout,
        lookup_batch_size=args.lookup_batch_size,
        load_concurrency=args.load_concurrency,
    )
    if args.reuse_legacy_focus_candidates:
        current_focus_sources = select_sources([FOCUS_SOURCE_NAME])
        current_focus_minhash = zephyr_datakit_steps(current_focus_sources, DEFAULT_SCALE).minhash[FOCUS_SOURCE_NAME]
        sources, minhash_steps = _legacy_candidate_input_steps(
            candidate_artifact_path=args.candidate_artifact,
            legacy_focus_source_key=args.legacy_focus_source_key,
            focus_normalized_step=current_focus_sources[FOCUS_SOURCE_NAME],
            focus_minhash_step=current_focus_minhash,
        )
        repack_output_path_prefix = marin_temp_bucket(
            ttl_days=args.temporary_ttl_days,
            prefix=prefix_join(TEMP_OUTPUT_PREFIX, args.run_tag),
            source_prefix=current_focus_sources[FOCUS_SOURCE_NAME].output_path,
        )
        if args.production_output:
            validation_output_path_prefix = marin_prefix()
            validation_step_name = PRODUCTION_VALIDATION_STEP_NAME
        else:
            validation_output_path_prefix = repack_output_path_prefix
            validation_step_name = TEMP_VALIDATION_STEP_NAME
        target = build_repacked_fuzzy_validation_step(
            sources,
            minhash_steps,
            candidate_artifact_path=args.candidate_artifact,
            legacy_source_key=args.legacy_focus_source_key,
            source_name=FOCUS_SOURCE_NAME,
            repack_output_path_prefix=repack_output_path_prefix,
            validation_output_path_prefix=validation_output_path_prefix,
            validation_step_name=validation_step_name,
            validation_scale=scale,
            store_config=store_config,
            coordinator_resources=coordinator_resources,
            task_resources=task_resources,
            max_output_shards=args.max_output_shards,
        )
        logger.info("Repack output prefix: %s", repack_output_path_prefix)
        logger.info("Validation output prefix: %s", validation_output_path_prefix)
    else:
        target = build_fuzzy_validation_step(
            select_sources(),
            scale=scale,
            store_config=store_config,
            coordinator_resources=coordinator_resources,
            task_resources=task_resources,
            max_output_shards=args.max_output_shards,
        )
    logger.info("Fuzzy-validation target: %s", target.output_path)
    StepRunner().run([target], dry_run=args.dry_run, max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
