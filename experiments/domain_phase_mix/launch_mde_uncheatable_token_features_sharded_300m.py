# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch sharded uncheatable token-feature extraction for 300M MDE surfaces.

The monolithic extractor OOMed after holding long-lived Parquet writers across
many raw-text checkpoint-feature files.  This launcher creates one executor step
per checkpoint run, so each child task writes independent token/document shards,
emits explicit progress logs, and can be retried independently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from fray.types import ResourceConfig
from marin.execution.executor import (
    ExecutorMainConfig,
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
)

from experiments.domain_phase_mix.exploratory.two_phase_many.extract_mde_uncheatable_token_features_300m import (
    CollectShardedFeaturesConfig,
    ExtractRunFeaturesConfig,
    SelectTokenSketchConfig,
    collect_sharded_features,
    extract_run_features,
    select_token_sketch,
    selected_surface_rows,
)
from experiments.domain_phase_mix.launch_300m_gsm8k_humaneval_evals import _executor_prefix
from experiments.domain_phase_mix.launch_proportional_controllability_300m import DEFAULT_TPU_REGION, DEFAULT_TPU_ZONE

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_MANY_DIR = SCRIPT_DIR / "exploratory" / "two_phase_many"
DEFAULT_FEATURE_INDEX = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_mde_uncheatable_token_features_300m_20260530/inputs/feature_surface_index.csv"
)
DEFAULT_NAME_PREFIX = "pinlin_calvin_xu/data_mixture/ngd3dm2_mde_uncheatable_token_features_sharded_300m_20260530"
DEFAULT_MAX_CONCURRENT = 24
DEFAULT_WORKER_CPU = 1.0
DEFAULT_WORKER_RAM = "8g"
DEFAULT_WORKER_DISK = "20g"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name-prefix", default=DEFAULT_NAME_PREFIX)
    parser.add_argument("--feature-index", default=DEFAULT_FEATURE_INDEX)
    parser.add_argument("--reference-run", default="baseline_proportional")
    parser.add_argument("--sample-tokens-per-dataset", type=int, default=4096)
    parser.add_argument("--dataset-prefix", default="uncheatable_eval/")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--progress-every-batches", type=int, default=250)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--include-run-name", action="append", default=[])
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--worker-cpu", type=float, default=DEFAULT_WORKER_CPU)
    parser.add_argument("--worker-ram", default=DEFAULT_WORKER_RAM)
    parser.add_argument("--worker-disk", default=DEFAULT_WORKER_DISK)
    parser.add_argument("--region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--executor-prefix")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def _validate_live_source(*, dry_run: bool, feature_index: str) -> None:
    if dry_run or os.getenv("CI") is not None:
        return
    if feature_index.startswith("gs://marin-us-east5/"):
        return
    raise ValueError("Live sharded MDE token-feature extraction requires --feature-index under gs://marin-us-east5/.")


def _resource_config(*, cpu: float, ram: str, disk: str, region: str, zone: str) -> ResourceConfig:
    return ResourceConfig.with_cpu(cpu=cpu, ram=ram, disk=disk, preemptible=False, regions=[region], zone=zone)


def _step_slug(run_name: str) -> str:
    return run_name.replace("/", "_").replace(":", "_")


def build_steps(
    *,
    name_prefix: str,
    feature_index: str,
    reference_run: str,
    sample_tokens_per_dataset: int,
    dataset_prefix: str,
    batch_size: int,
    progress_every_batches: int,
    max_runs: int | None,
    include_run_names: list[str],
    worker_resources: ResourceConfig,
) -> tuple[list[ExecutorStep], ExecutorStep, list[str]]:
    """Build token-sketch, per-run shard, and collector steps."""
    rows = selected_surface_rows(feature_index, include_run_names, max_runs)
    run_names = rows["run_name"].astype(str).tolist()
    if reference_run not in set(rows["run_name"]):
        selected_surface_rows(feature_index, [reference_run], None)

    select_step = ExecutorStep(
        name=f"{name_prefix}/select_token_sketch",
        description="Select deterministic uncheatable reference-token sketch",
        fn=select_token_sketch,
        config=SelectTokenSketchConfig(
            output_path=this_output_path(),
            feature_index=feature_index,
            reference_run=reference_run,
            sample_tokens_per_dataset=sample_tokens_per_dataset,
            dataset_prefix=dataset_prefix,
            batch_size=batch_size,
            allow_gcs_read=True,
        ),
        resources=worker_resources,
    )

    run_steps: list[ExecutorStep] = []
    run_output_paths: dict[str, InputName] = {}
    for run_name in run_names:
        step = ExecutorStep(
            name=f"{name_prefix}/extract_run/{_step_slug(run_name)}",
            description=f"Extract uncheatable token/document feature shards for {run_name}",
            fn=extract_run_features,
            config=ExtractRunFeaturesConfig(
                output_path=this_output_path(),
                feature_index=feature_index,
                selected_tokens_path=output_path_of(select_step, "selected_tokens.parquet"),
                run_name=run_name,
                dataset_prefix=dataset_prefix,
                batch_size=batch_size,
                allow_gcs_read=True,
                progress_every_batches=progress_every_batches,
            ),
            resources=worker_resources,
        )
        run_steps.append(step)
        run_output_paths[run_name] = output_path_of(step)

    collect_step = ExecutorStep(
        name=f"{name_prefix}/collect_sharded_uncheatable_token_features",
        description=f"Collect {len(run_steps)} uncheatable token/document feature shards",
        fn=collect_sharded_features,
        config=CollectShardedFeaturesConfig(
            output_path=this_output_path(),
            feature_index=feature_index,
            selected_tokens_path=output_path_of(select_step, "selected_tokens.parquet"),
            run_output_paths=run_output_paths,
            reference_run=reference_run,
            sample_tokens_per_dataset=sample_tokens_per_dataset,
            dataset_prefix=dataset_prefix,
        ),
    )
    return [select_step, *run_steps], collect_step, run_names


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    _validate_live_source(dry_run=args.dry_run, feature_index=args.feature_index)
    worker_resources = _resource_config(
        cpu=args.worker_cpu,
        ram=args.worker_ram,
        disk=args.worker_disk,
        region=args.region,
        zone=args.zone,
    )
    shard_steps, collect_step, run_names = build_steps(
        name_prefix=args.name_prefix,
        feature_index=args.feature_index,
        reference_run=args.reference_run,
        sample_tokens_per_dataset=args.sample_tokens_per_dataset,
        dataset_prefix=args.dataset_prefix,
        batch_size=args.batch_size,
        progress_every_batches=args.progress_every_batches,
        max_runs=args.max_runs,
        include_run_names=args.include_run_name,
        worker_resources=worker_resources,
    )
    launch_summary = {
        "feature_index": args.feature_index,
        "max_concurrent": args.max_concurrent,
        "name_prefix": args.name_prefix,
        "run_count": len(run_names),
        "sample_tokens_per_dataset": args.sample_tokens_per_dataset,
        "worker_cpu": args.worker_cpu,
        "worker_disk": args.worker_disk,
        "worker_ram": args.worker_ram,
        "worker_region": args.region,
        "worker_zone": args.zone,
    }
    logger.info("Prepared sharded MDE token extraction: %s", json.dumps(launch_summary, sort_keys=True))
    if args.dry_run or os.getenv("CI") is not None:
        print(json.dumps(launch_summary, indent=2, sort_keys=True))
        return

    executor_prefix = _executor_prefix(args.executor_prefix, args.region)
    executor_main(
        ExecutorMainConfig(prefix=executor_prefix, max_concurrent=args.max_concurrent),
        steps=[*shard_steps, collect_step],
        description=f"{args.name_prefix}: sharded 300M uncheatable token-feature extraction",
    )


if __name__ == "__main__":
    main()
