#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one-off Iris jobs to regenerate historical LoRA HF exports.

This script uses the clean re-export path from raw LoRA trainer checkpoints,
not the shard-transpose salvage path. It writes fresh sibling outputs under
`hf-fixed-<run_label>/step-*` and never overwrites or deletes the original
published `hf/step-*` exports.

By default this script performs a dry run and prints the planned jobs. Pass
`--submit` to actually enqueue the Iris CPU jobs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import fsspec

from iris.client import IrisClient
from iris.cluster.config import IrisConfig
from iris.cluster.constraints import Constraint, region_constraint, zone_constraint
from iris.cluster.types import Entrypoint, ResourceSpec

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = "lib/iris/examples/marin.yaml"
DEFAULT_REGION = "us-central1"
DEFAULT_CPU = 16.0
DEFAULT_MEMORY = "128GB"
DEFAULT_DISK = "150GB"
DEFAULT_RPC_TIMEOUT_MS = 300000
DEFAULT_BASE_MODEL_REF = "gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c"
DEFAULT_OUTPUT_PREFIX = "hf-fixed"
DEFAULT_CHECKPOINT_SUBPATH = "model"
DEFAULT_LORA_R = 64
DEFAULT_LORA_ALPHA = 64.0
DEFAULT_TUNE_LORA_ROOT = "gs://marin-us-central1/checkpoints/dpo/tune_lora"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_RUN_NAMES = (
    "bloom_speceval_v2_marin_lr1e6_seed0_b64_v5p8-f55cdd",
    "bloom_speceval_v2_marin_lr1e6_seed2_b64_v5p8-0ccb95",
    "bloom_speceval_v2_marin_lr2p5e6_seed0_b64_v5p8-fde891",
    "bloom_speceval_v2_marin_lr2p5e6_seed2_b64_v5p8-53c5c6",
    "bloom_speceval_v2_marin_lr3p75e6_seed0_b64_v5p8-5dd6f9",
    "bloom_speceval_v2_marin_lr3p75e6_seed2_b64_v5p8-a8f183",
    "bloom_speceval_v2_marin_lr4p5e6_seed0_b64_v5p8-5777fb",
    "bloom_speceval_v2_marin_lr4p5e6_seed2_b64_v5p8-e8649f",
    "bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540",
    "bloom_speceval_v2_marin_lr5e6_seed2_b64_v5p8-68378e",
    "bloom_speceval_v2_marin_lr6p25e6_seed0_b64_v5p8-9bf4a5",
    "bloom_speceval_v2_marin_lr6p25e6_seed2_b64_v5p8-0f0331",
    "bloom_speceval_v2_marin_lr7p5e6_seed0_b64_v5p8-da8f07",
    "bloom_speceval_v2_marin_lr7p5e6_seed2_b64_v5p8-981a35",
    "bloom_speceval_v2_marin_lr8p75e6_seed0_b64_v5p8-ee2e69",
    "bloom_speceval_v2_marin_lr8p75e6_seed2_b64_v5p8-f0636c",
    "bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d",
    "bloom_speceval_v2_marin_lr1e5_seed2_b64_v5p8-a73d6f",
)

STEP_PATTERN = re.compile(r"step-(\d+)$")


@dataclass(frozen=True)
class RunRoots:
    name: str
    hf_root: str
    raw_root: str


@dataclass(frozen=True)
class ReexportJob:
    run_name: str
    step_name: str
    source_hf_path: str
    raw_checkpoint_path: str
    output_path: str
    job_name: str
    command: tuple[str, ...]


def step_number(step_name: str) -> int:
    """Parse the numeric portion of a `step-<n>` name."""
    match = STEP_PATTERN.fullmatch(step_name)
    if match is None:
        raise ValueError(f"Invalid step name: {step_name}")
    return int(match.group(1))


def find_workspace_root(start: Path) -> Path:
    """Find the repo root used for Iris workspace bundling."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise ValueError(f"Could not find workspace root from {start}")


@contextmanager
def controller_client(config_file: str, rpc_timeout_ms: int) -> Iterator[IrisClient]:
    """Open a tunneled Iris client using the standard cluster config."""
    iris_config = IrisConfig.load(config_file)
    controller_address = iris_config.controller_address()
    providers = iris_config.provider_bundle()
    controller = providers.controller
    workspace = find_workspace_root(Path.cwd())
    if not controller_address:
        controller_address = controller.discover_controller(iris_config.proto.controller)
    with controller.tunnel(address=controller_address) as tunneled:
        client = IrisClient.remote(tunneled, workspace=workspace, timeout_ms=rpc_timeout_ms)
        try:
            yield client
        finally:
            client.shutdown()


def build_constraints(region: str, zone: str | None) -> list[Constraint]:
    """Build explicit placement constraints for the repair jobs."""
    constraints = [region_constraint([region])]
    if zone is not None:
        constraints.append(zone_constraint(zone))
    return constraints


def build_run_roots(run_name: str) -> RunRoots:
    """Derive the HF and raw-checkpoint roots for a tune-LoRA run."""
    base = f"{DEFAULT_TUNE_LORA_ROOT}/{run_name}"
    return RunRoots(
        name=run_name,
        hf_root=f"{base}/hf",
        raw_root=f"{base}/checkpoints",
    )


def _list_step_names(root_path: str) -> set[str]:
    fs, plain_path = fsspec.core.url_to_fs(root_path)
    if not fs.exists(plain_path):
        raise ValueError(f"Path does not exist: {root_path}")

    step_names: set[str] = set()
    for entry in fs.ls(plain_path, detail=False):
        entry_name = os.path.basename(str(entry).rstrip("/"))
        if STEP_PATTERN.fullmatch(entry_name):
            step_names.add(entry_name)
    return step_names


def _path_exists(path: str) -> bool:
    fs, plain_path = fsspec.core.url_to_fs(path)
    return fs.exists(plain_path)


def default_output_path(source_hf_path: str, output_prefix: str, run_label: str) -> str:
    """Map `.../hf/step-N` to `.../<prefix>-<label>/step-N`."""
    marker = "/hf/"
    if marker not in source_hf_path:
        raise ValueError(f"Expected source path to contain {marker!r}: {source_hf_path}")
    prefix, suffix = source_hf_path.split(marker, maxsplit=1)
    return f"{prefix}/{output_prefix}-{run_label}/{suffix}"


def short_run_name(run_name: str) -> str:
    """Compress a verbose W&B-style run name into a stable job-name slug."""
    slug = run_name.removeprefix("bloom_speceval_v2_marin_")
    slug = slug.replace("_b64_v5p8", "")
    slug = slug.replace("_", "-")
    return slug


def build_job_name(run_name: str, step_name: str, run_label: str) -> str:
    """Build a stable Iris job name."""
    compact_step = step_name.replace("-", "")
    return f"lora-hf-fixed-{short_run_name(run_name)}-{compact_step}-{run_label}"


def build_reexport_command(
    *,
    source_hf_path: str,
    raw_checkpoint_path: str,
    output_path: str,
    base_model_ref: str,
    checkpoint_subpath: str,
    lora_r: int,
    lora_alpha: float,
    target_modules: Sequence[str],
    verify_shapes: bool,
) -> tuple[str, ...]:
    """Build the worker command for a single clean re-export job."""
    command: list[str] = [
        "python",
        "experiments/posttrain/lora_vllm_investigate.py",
        "--model-path",
        source_hf_path,
        "reexport-merged",
        "--raw-checkpoint-path",
        raw_checkpoint_path,
        "--checkpoint-subpath",
        checkpoint_subpath,
        "--output-path",
        output_path,
        "--base-model-ref",
        base_model_ref,
        "--tokenizer-path",
        source_hf_path,
        "--lora-r",
        str(lora_r),
        "--lora-alpha",
        str(lora_alpha),
    ]
    for module_name in target_modules:
        command.extend(["--lora-target-module", module_name])
    if verify_shapes:
        command.append("--verify-shapes")
    return tuple(command)


def discover_jobs(
    *,
    run_names: Sequence[str],
    output_prefix: str,
    run_label: str,
    base_model_ref: str,
    checkpoint_subpath: str,
    lora_r: int,
    lora_alpha: float,
    target_modules: Sequence[str],
    verify_shapes: bool,
    step_filters: set[str] | None,
) -> tuple[list[ReexportJob], list[str]]:
    """Discover all matching HF/raw checkpoint pairs and build job specs."""

    def discover_run(run_name: str) -> tuple[list[ReexportJob], list[str]]:
        run_jobs: list[ReexportJob] = []
        run_warnings: list[str] = []
        roots = build_run_roots(run_name)
        hf_steps = _list_step_names(roots.hf_root)
        raw_steps = _list_step_names(roots.raw_root)
        shared_steps = sorted(hf_steps & raw_steps, key=lambda step_name: int(step_name.split("-")[1]))
        if step_filters is not None:
            shared_steps = [step_name for step_name in shared_steps if step_name in step_filters]

        missing_hf = sorted(raw_steps - hf_steps)
        missing_raw = sorted(hf_steps - raw_steps)
        if missing_hf:
            run_warnings.append(f"{run_name}: raw-only steps skipped: {', '.join(missing_hf)}")
        if missing_raw:
            run_warnings.append(f"{run_name}: hf-only steps skipped: {', '.join(missing_raw)}")
        if not shared_steps:
            run_warnings.append(f"{run_name}: no matching hf/raw step pairs found")
            return run_jobs, run_warnings

        for step_name in shared_steps:
            source_hf_path = f"{roots.hf_root}/{step_name}"
            raw_checkpoint_path = f"{roots.raw_root}/{step_name}"
            output_path = default_output_path(source_hf_path, output_prefix, run_label)
            job_name = build_job_name(run_name, step_name, run_label)
            command = build_reexport_command(
                source_hf_path=source_hf_path,
                raw_checkpoint_path=raw_checkpoint_path,
                output_path=output_path,
                base_model_ref=base_model_ref,
                checkpoint_subpath=checkpoint_subpath,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                verify_shapes=verify_shapes,
            )
            run_jobs.append(
                ReexportJob(
                    run_name=run_name,
                    step_name=step_name,
                    source_hf_path=source_hf_path,
                    raw_checkpoint_path=raw_checkpoint_path,
                    output_path=output_path,
                    job_name=job_name,
                    command=command,
                )
            )

        return run_jobs, run_warnings

    jobs: list[ReexportJob] = []
    warnings: list[str] = []
    max_workers = min(8, len(run_names))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for run_jobs, run_warnings in executor.map(discover_run, run_names):
            jobs.extend(run_jobs)
            warnings.extend(run_warnings)

    return jobs, warnings


def latest_jobs_by_run(jobs: Sequence[ReexportJob]) -> list[ReexportJob]:
    """Keep only the highest-step job for each run."""
    latest: dict[str, ReexportJob] = {}
    run_order: list[str] = []
    for job in jobs:
        if job.run_name not in latest:
            run_order.append(job.run_name)
        previous = latest.get(job.run_name)
        if previous is None or step_number(job.step_name) > step_number(previous.step_name):
            latest[job.run_name] = job
    return [latest[run_name] for run_name in run_order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--zone")
    parser.add_argument("--cpu", type=float, default=DEFAULT_CPU)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--disk", default=DEFAULT_DISK)
    parser.add_argument("--rpc-timeout-ms", type=int, default=DEFAULT_RPC_TIMEOUT_MS)
    parser.add_argument("--run-label", default="r1")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--base-model-ref", default=DEFAULT_BASE_MODEL_REF)
    parser.add_argument("--checkpoint-subpath", default=DEFAULT_CHECKPOINT_SUBPATH)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=float, default=DEFAULT_LORA_ALPHA)
    parser.add_argument(
        "--run-name", action="append", default=None, help="Restrict to specific run name(s). Repeatable."
    )
    parser.add_argument(
        "--step", action="append", default=None, help="Restrict to specific step name(s), e.g. step-1699. Repeatable."
    )
    parser.add_argument(
        "--skip-verify-shapes",
        action="store_true",
        help="Do not pass --verify-shapes to the worker re-export command.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit the Iris jobs. Without this flag the script prints a dry-run plan only.",
    )
    parser.add_argument(
        "--skip-existing-output",
        action="store_true",
        help="Skip jobs whose output path already exists.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Keep only the highest shared step per run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    run_names = tuple(args.run_name) if args.run_name is not None else DEFAULT_RUN_NAMES
    step_filters = set(args.step) if args.step is not None else None
    jobs, warnings = discover_jobs(
        run_names=run_names,
        output_prefix=args.output_prefix,
        run_label=args.run_label,
        base_model_ref=args.base_model_ref,
        checkpoint_subpath=args.checkpoint_subpath,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=DEFAULT_TARGET_MODULES,
        verify_shapes=not args.skip_verify_shapes,
        step_filters=step_filters,
    )
    if args.latest_only:
        jobs = latest_jobs_by_run(jobs)

    filtered_jobs: list[ReexportJob] = []
    skipped_existing = 0
    for job in jobs:
        if args.skip_existing_output and _path_exists(job.output_path):
            skipped_existing += 1
            logger.info("Skipping existing output path: %s", job.output_path)
            continue
        filtered_jobs.append(job)

    print(f"mode={'submit' if args.submit else 'dry-run'}")
    print(f"job_count={len(filtered_jobs)}")
    if skipped_existing:
        print(f"skipped_existing={skipped_existing}")
    for warning in warnings:
        print(f"warning={warning}")

    if not filtered_jobs:
        return 0

    if not args.submit:
        for job in filtered_jobs:
            print(
                "\n".join(
                    [
                        f"run_name={job.run_name}",
                        f"step_name={job.step_name}",
                        f"job_name={job.job_name}",
                        f"source_hf_path={job.source_hf_path}",
                        f"raw_checkpoint_path={job.raw_checkpoint_path}",
                        f"hf_fixed_path={job.output_path}",
                        f"command={' '.join(job.command)}",
                        "---",
                    ]
                )
            )
        return 0

    constraints = build_constraints(args.region, args.zone)
    with controller_client(args.config_file, args.rpc_timeout_ms) as client:
        for job in filtered_jobs:
            submitted_job = client.submit(
                entrypoint=Entrypoint.from_command(*job.command),
                name=job.job_name,
                resources=ResourceSpec(cpu=args.cpu, memory=args.memory, disk=args.disk),
                constraints=constraints,
            )
            print(f"job_id={submitted_job.job_id}")
            print(f"run_name={job.run_name}")
            print(f"step_name={job.step_name}")
            print(f"hf_fixed_path={job.output_path}")
            print("---")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
