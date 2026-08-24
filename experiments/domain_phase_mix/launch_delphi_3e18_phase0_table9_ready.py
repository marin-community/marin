# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate every completed Delphi phase-0 replay checkpoint on Table 9.

The original 280-row parent places all training steps before all Table-9 steps.
Its bounded local executor therefore queues the evaluations behind unfinished
training even though the two stages use different TPU types. This sidecar
reconstructs the original content-addressed paths, selects only completed
prefixes, and launches their evaluations without executable training
dependencies. It writes to the original evaluation paths, so either parent can
be resubmitted safely and completed rows are skipped.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import fsspec
from marin.execution.context import executor_context
from marin.execution.executor import Executor, ExecutorMainConfig, executor_main
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.execution.types import ExecutorStep
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

REPLAY_CODE_COMMIT = "c8a3939fc12053f06dfce8fa0094b120411312c7"
READY_MANIFEST_SUBDIR = "table9_ready_manifests"
READINESS_AUDIT_WORKERS = 32
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class ReadyEvalRecord:
    """Resolved identity and status for one completed boundary checkpoint."""

    run_order: int
    run_name: str
    source_run_name: str
    training_output_path: str
    checkpoint_path: str
    eval_output_path: str
    eval_already_succeeded: bool


def _resolve_original_paths(
    artifacts: replay.PrefixLaunchArtifacts,
    *,
    prefix: str,
) -> tuple[list[str], list[str]]:
    """Resolve paths under the original graph without running any steps."""
    resolver = Executor(
        prefix=prefix,
        executor_info_base_path=os.path.join(prefix, "experiments"),
        description="Resolve original Delphi phase-0 replay identities for ready-only Table-9 evaluation.",
    )
    with executor_context():
        for eval_step in artifacts.eval_steps:
            resolver.compute_version(eval_step, is_pseudo_dep=False)
    return (
        [resolver.output_paths[step] for step in artifacts.training_steps],
        [resolver.output_paths[step] for step in artifacts.eval_steps],
    )


def _missing_checkpoint_files(checkpoint_path: str) -> list[str]:
    fs, _, _ = fsspec.get_fs_token_paths(checkpoint_path)
    present = {os.path.basename(path.rstrip("/")) for path in fs.glob(os.path.join(checkpoint_path, "*"))}
    return [name for name in REQUIRED_CHECKPOINT_FILES if name not in present]


def _ready_record(
    args: tuple[replay.base.DelphiSwarmRunSpec, str, str],
) -> ReadyEvalRecord | None:
    run_spec, training_output_path, eval_output_path = args
    training_status = StatusFile(training_output_path, worker_id="table9-ready-audit").status
    if training_status != STATUS_SUCCESS:
        return None
    checkpoint_path = os.path.join(training_output_path, f"hf/step-{replay.EXPECTED_PREFIX_HF_STEP}")
    missing = _missing_checkpoint_files(checkpoint_path)
    if missing:
        raise RuntimeError(
            f"Training row {run_spec.run_order} is SUCCESS but its boundary checkpoint is incomplete: {missing}"
        )
    return ReadyEvalRecord(
        run_order=run_spec.run_order,
        run_name=run_spec.run_name,
        source_run_name=run_spec.source_run_name,
        training_output_path=training_output_path,
        checkpoint_path=checkpoint_path,
        eval_output_path=eval_output_path,
        eval_already_succeeded=(StatusFile(eval_output_path, worker_id="table9-ready-audit").status == STATUS_SUCCESS),
    )


def _ready_records(
    *,
    run_specs: list[replay.base.DelphiSwarmRunSpec],
    training_output_paths: list[str],
    eval_output_paths: list[str],
) -> list[ReadyEvalRecord]:
    if not (len(run_specs) == len(training_output_paths) == len(eval_output_paths)):
        raise ValueError("Run specs, training paths, and evaluation paths must have equal lengths")

    audit_inputs = list(zip(run_specs, training_output_paths, eval_output_paths, strict=True))
    with ThreadPoolExecutor(max_workers=READINESS_AUDIT_WORKERS) as pool:
        inspected = list(pool.map(_ready_record, audit_inputs))
    return [record for record in inspected if record is not None]


def _detached_eval_step(
    original_step: ExecutorStep,
    *,
    checkpoint_path: str,
    eval_output_path: str,
) -> ExecutorStep:
    """Replace the training dependency with a verified immutable checkpoint path."""
    config = original_step.config
    detached_config = dataclasses.replace(
        config,
        eval_config=dataclasses.replace(config.eval_config, checkpoint_path=checkpoint_path),
    )
    return dataclasses.replace(
        original_step,
        config=detached_config,
        override_output_path=eval_output_path,
    )


def _ready_manifest(
    *,
    prefix: str,
    records: list[ReadyEvalRecord],
    persist: bool,
) -> tuple[str, str]:
    payload = {
        "experiment_name": replay.EXPERIMENT_NAME,
        "replay_code_commit": REPLAY_CODE_COMMIT,
        "source_coordinate_hash": replay.EXPECTED_SOURCE_COORDINATE_HASH,
        "prefix_hf_step": replay.EXPECTED_PREFIX_HF_STEP,
        "required_checkpoint_files": list(REQUIRED_CHECKPOINT_FILES),
        "ready_count": len(records),
        "pending_eval_count": sum(not record.eval_already_succeeded for record in records),
        "records": [asdict(record) for record in records],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    output_path = os.path.join(
        prefix,
        replay.EXPERIMENT_NAME,
        READY_MANIFEST_SUBDIR,
        f"ready-{len(records):03d}-{digest[:12]}.json",
    )
    if persist:
        fs, _, _ = fsspec.get_fs_token_paths(output_path)
        if fs.exists(output_path):
            with fs.open(output_path, "r") as handle:
                existing = handle.read()
            if existing != encoded:
                raise RuntimeError(f"Ready manifest collision at {output_path}")
        else:
            parent = os.path.dirname(output_path)
            fs.makedirs(parent, exist_ok=True)
            with fs.open(output_path, "w") as handle:
                handle.write(encoded)
    return output_path, digest


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrent", type=int, default=replay.DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--minimum-ready-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if not 1 <= args.max_concurrent <= replay.DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {replay.DEFAULT_MAX_CONCURRENT}]")
    if args.minimum_ready_count < 1:
        raise ValueError("--minimum-ready-count must be positive")

    prefix = marin_prefix_for_region(replay.base.DEFAULT_TPU_REGION)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {prefix!r}")
    os.environ["MARIN_PREFIX"] = prefix

    eval_resources = replay.base.TABLE9_EVAL_RESOURCES
    if (
        eval_resources.device.variant != "v6e-8"
        or eval_resources.regions != ["us-east5"]
        or eval_resources.zone != "us-east5-b"
    ):
        raise ValueError(f"Table-9 resources changed from the reviewed v6e-8 placement: {eval_resources}")

    run_specs, launch_audit = replay.load_replay_specs(
        source_panel=replay.base.DEFAULT_SOURCE_PANEL,
        analysis_output_path=replay.base.DEFAULT_ANALYSIS_OUTPUT_PATH,
        tpu_region=replay.base.DEFAULT_TPU_REGION,
        tpu_zone=replay.base.DEFAULT_TPU_ZONE,
    )
    launch_audit.update(
        replay_code_commit=REPLAY_CODE_COMMIT,
        profiler_enabled=False,
        selected_run_orders=list(range(len(run_specs))),
        selected_run_count=len(run_specs),
        max_concurrent=args.max_concurrent,
    )
    validation_steps = replay.base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    with executor_context():
        artifacts = replay.build_launch_artifacts(
            run_specs=run_specs,
            analysis_output_path=replay.base.DEFAULT_ANALYSIS_OUTPUT_PATH,
            source_panel=str(replay.base.DEFAULT_SOURCE_PANEL),
            validation_configs=validation_configs,
            launch_audit=launch_audit,
        )

    training_output_paths, eval_output_paths = _resolve_original_paths(artifacts, prefix=prefix)
    records = _ready_records(
        run_specs=run_specs,
        training_output_paths=training_output_paths,
        eval_output_paths=eval_output_paths,
    )
    if len(records) < args.minimum_ready_count:
        raise RuntimeError(f"Found {len(records)} ready rows; expected at least {args.minimum_ready_count}")

    pending_records = [record for record in records if not record.eval_already_succeeded]
    original_eval_by_run_order = {
        spec.run_order: step for spec, step in zip(run_specs, artifacts.eval_steps, strict=True)
    }
    with executor_context():
        eval_steps = [
            _detached_eval_step(
                original_eval_by_run_order[record.run_order],
                checkpoint_path=record.checkpoint_path,
                eval_output_path=record.eval_output_path,
            )
            for record in pending_records
        ]

    persist_manifest = not args.dry_run and os.getenv("CI") is None
    manifest_path, manifest_sha256 = _ready_manifest(
        prefix=prefix,
        records=records,
        persist=persist_manifest,
    )
    logger.info(
        "Ready Table-9 snapshot: %d completed prefixes, %d pending evaluations, manifest=%s sha256=%s",
        len(records),
        len(eval_steps),
        manifest_path,
        manifest_sha256,
    )
    if args.dry_run or os.getenv("CI") is not None or not eval_steps:
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=eval_steps,
        description=(
            f"{replay.EXPERIMENT_NAME}: ready-only native Table-9 sidecar for {len(eval_steps)} "
            f"completed phase-boundary checkpoints; manifest {manifest_path}"
        ),
    )


if __name__ == "__main__":
    main()
