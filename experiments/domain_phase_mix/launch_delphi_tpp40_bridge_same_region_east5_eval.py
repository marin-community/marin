# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the canonical East5 TPP40 bridge row on the Europe evaluator.

The source model remains the immutable East5 v5p-8 ``run_order=2`` output.
Only its two Orbax checkpoints and endpoint HF export are mirrored to Europe,
where they are scored beside the Europe v6e-8 candidate on identical v6e-8
hardware and identical region-local evaluation payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.run import RESULTS_FILENAME as TABLE9_RESULTS_FILENAME
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import Executor, ExecutorMainConfig, executor_main
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.execution.types import ExecutorStep
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_tpp40 as tpp40
from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_uncheatable_eval as bridge_eval
from experiments.domain_phase_mix.delphi_tpp40_evaluation_identity import (
    table9_request_set_identity,
    tree_payload_identity,
    validation_payload_identity,
)

logger = logging.getLogger(__name__)

REFERENCE_DIR = bridge_eval.REFERENCE_DIR
CONTRACT_PATH = REFERENCE_DIR / "bridge_acceptance_contract_v4.json"
MIRROR_MANIFEST_PATH = REFERENCE_DIR / "east5_row2_europe_mirror_manifest_v1.json"
EXPECTED_CONTRACT_SHA256 = "f0441b8927e3e7d32bbdbe781ed3008dbb46a1cd98ff540661423e850ee936df"
EXPECTED_MIRROR_MANIFEST_SHA256 = "08c6160b4bc181a139c432ed642945f8c2fd72b61b280d2590ffd435deb48202"
EUROPE_REGION = "europe-west4"
EUROPE_ZONE = "europe-west4-a"
MIRROR_ROOT = "gs://marin-eu-west4/experiments/domain_phase_mix/" "delphi_tpp40_bridge_east5_row2_mirror_v1"
EXPECTED_MIRROR_OBJECTS = 16
EXPECTED_MIRROR_BYTES = 10_049_866_394
TABLE9_WANDB_GROUP = "olmo_base_eval_table9_delphi_tpp40_bridge_east5_reference_same_region_v1_20260830"
READY_MANIFEST_ROOT = (
    "pinlin_calvin_xu/data_mixture/delphi_tpp40_bridge_same_region_east5_reference_v1_20260830/ready_manifests"
)


@dataclass(frozen=True)
class SameRegionReferenceArtifacts:
    """Resolved evaluator graph and immutable identities for the East reference."""

    run_spec: base.DelphiSwarmRunSpec
    analysis_output_path: str
    validation_cache_paths: tuple[str, ...]
    validation_payload_sha256: str
    table9_request_set_dir: str
    table9_request_set_payload_sha256: str
    uncheatable_steps: tuple[ExecutorStep, ...]
    uncheatable_output_paths: tuple[str, ...]
    table9_step: ExecutorStep
    table9_output_path: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_frozen_json(path: Path, *, expected_sha256: str, label: str) -> dict[str, Any]:
    encoded = path.read_bytes()
    observed_sha256 = _sha256_bytes(encoded)
    if observed_sha256 != expected_sha256:
        raise ValueError(f"Frozen {label} changed: {observed_sha256} != {expected_sha256}")
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError(f"Frozen {label} is not a JSON object")
    return payload


def audit_east5_reference_mirror() -> dict[str, Any]:
    """Require every Europe mirror tree to equal its canonical East5 source."""
    contract = _load_frozen_json(
        CONTRACT_PATH,
        expected_sha256=EXPECTED_CONTRACT_SHA256,
        label="same-region bridge contract",
    )
    if EXPECTED_CONTRACT_SHA256 != bridge_eval.EXPECTED_CONTRACT_SHA256:
        raise ValueError("Same-region evaluator and bridge evaluator contracts disagree")
    manifest = _load_frozen_json(
        MIRROR_MANIFEST_PATH,
        expected_sha256=EXPECTED_MIRROR_MANIFEST_SHA256,
        label="East5 reference mirror manifest",
    )
    if contract["bridge"]["east5_reference_mirror"]["manifest_sha256"] != EXPECTED_MIRROR_MANIFEST_SHA256:
        raise ValueError("Same-region contract refers to the wrong mirror manifest")
    if manifest.get("europe_mirror_root") != MIRROR_ROOT:
        raise ValueError("East5 reference mirror root changed")
    if manifest.get("storage_transfer_service_used") is not False:
        raise ValueError("East5 reference mirror used Storage Transfer Service")

    observed_trees: list[dict[str, Any]] = []
    for tree in manifest.get("trees", []):
        if not isinstance(tree, dict):
            raise ValueError("East5 reference mirror tree record is malformed")
        source_identity = tree_payload_identity(str(tree["source_path"]))
        destination_identity = tree_payload_identity(str(tree["destination_path"]))
        if source_identity != tree.get("source_identity"):
            raise ValueError(f"Canonical East5 tree changed: {tree['source_path']}")
        if destination_identity != tree.get("destination_identity"):
            raise ValueError(f"Europe mirror tree changed: {tree['destination_path']}")
        if source_identity != destination_identity:
            raise ValueError(f"Europe mirror does not match East5 source: {tree['relative_path']}")
        observed_trees.append(
            {
                "relative_path": tree["relative_path"],
                "source_path": tree["source_path"],
                "destination_path": tree["destination_path"],
                "payload_identity": source_identity,
            }
        )
    observed_objects = sum(tree["payload_identity"]["objects"] for tree in observed_trees)
    observed_bytes = sum(tree["payload_identity"]["bytes"] for tree in observed_trees)
    if observed_objects != EXPECTED_MIRROR_OBJECTS or observed_bytes != EXPECTED_MIRROR_BYTES:
        raise ValueError(
            "East5 reference mirror inventory changed: " f"objects={observed_objects}, bytes={observed_bytes}"
        )
    root_identity = tree_payload_identity(MIRROR_ROOT)
    if root_identity["objects"] != observed_objects or root_identity["bytes"] != observed_bytes:
        raise ValueError("East5 reference mirror root contains objects outside the three frozen trees")
    return {
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "mirror_manifest_sha256": EXPECTED_MIRROR_MANIFEST_SHA256,
        "objects": observed_objects,
        "bytes": observed_bytes,
        "root_payload_identity": root_identity,
        "trees": observed_trees,
        "passed": True,
    }


def _evaluation_side() -> bridge_eval.BridgeSide:
    return replace(
        bridge_eval.BRIDGE_SIDES["east5"],
        region=EUROPE_REGION,
        evaluator_zone=EUROPE_ZONE,
    )


def _resolve_output_paths(steps: tuple[ExecutorStep, ...], *, prefix: str) -> tuple[str, ...]:
    resolver = Executor(
        prefix=prefix,
        executor_info_base_path=os.path.join(prefix, "experiments"),
        description="Resolve frozen same-region East5-reference bridge outputs.",
    )
    with executor_context():
        for step in steps:
            resolver.compute_version(step, is_pseudo_dep=False)
    return tuple(resolver.output_paths[step] for step in steps)


def build_reference_artifacts() -> SameRegionReferenceArtifacts:
    """Build and resolve the exact same-region reference evaluation graph."""
    audit_east5_reference_mirror()
    prefix = marin_prefix_for_region(EUROPE_REGION)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {prefix!r}")
    os.environ["MARIN_PREFIX"] = prefix

    run_specs, _, _ = bridge_eval._run_specs(bridge_eval.BRIDGE_SIDES["east5"])
    if len(run_specs) != 1 or run_specs[0].run_order != bridge_eval.BRIDGE_RUN_ORDERS[0]:
        raise ValueError("Same-region reference run inventory changed")
    run_spec = run_specs[0]
    analysis_output_path = tpp40._regional_input_path(base.DEFAULT_ANALYSIS_OUTPUT_PATH, region=EUROPE_REGION)
    _, uncheatable_validation_configs = bridge_eval._validation_configs()
    evaluation_side = _evaluation_side()
    validation_cache_paths = bridge_eval._require_uncheatable_caches(
        uncheatable_validation_configs,
        side=evaluation_side,
    )
    validation_identity = validation_payload_identity(
        dict(zip(bridge_eval.EXPECTED_UNCHEATABLE_NAMES, validation_cache_paths, strict=True))
    )
    validation_payload_sha256 = validation_identity["payload_sha256"]

    uncheatable_steps = tuple(
        bridge_eval._eval_steps(
            side=evaluation_side,
            run_specs=run_specs,
            training_output_paths=[MIRROR_ROOT],
            analysis_output_path=analysis_output_path,
            validation_configs=uncheatable_validation_configs,
            validation_payload_sha256=validation_payload_sha256,
        )
    )
    uncheatable_output_paths = _resolve_output_paths(uncheatable_steps, prefix=prefix)

    table9_resources = ResourceConfig.with_tpu(
        bridge_eval.EVALUATOR_TPU_TYPE,
        regions=[EUROPE_REGION],
        zone=EUROPE_ZONE,
        disk="80g",
    )
    table9_request_set_dir = f"{prefix.rstrip('/')}/{base.TABLE9_REQUEST_SET_DIR.name}"
    table9_identity = table9_request_set_identity(table9_request_set_dir)
    with executor_context():
        table9_step = olmo_base_eval_step(
            name=f"t9_{run_spec.run_name}",
            checkpoint=f"{MIRROR_ROOT}/hf/step-{tpp40.EXPECTED_FINAL_CHECKPOINT_STEP}",
            request_set_dir=table9_request_set_dir,
            resource_config=table9_resources,
            wandb_group=TABLE9_WANDB_GROUP,
            provenance={
                "evaluator": "marin-native-table9-bpb",
                "panel": "delphi_tpp40_augmented_fit_swarm",
                "scale": "fixed_n_total_tpp40",
                "source_run_name": run_spec.source_run_name,
                "swarm_run_name": run_spec.run_name,
                "panel_source": run_spec.panel_source,
            },
        )
    (table9_output_path,) = _resolve_output_paths((table9_step,), prefix=prefix)
    return SameRegionReferenceArtifacts(
        run_spec=run_spec,
        analysis_output_path=analysis_output_path,
        validation_cache_paths=validation_cache_paths,
        validation_payload_sha256=validation_payload_sha256,
        table9_request_set_dir=table9_request_set_dir,
        table9_request_set_payload_sha256=table9_identity["payload_sha256"],
        uncheatable_steps=uncheatable_steps,
        uncheatable_output_paths=uncheatable_output_paths,
        table9_step=table9_step,
        table9_output_path=table9_output_path,
    )


def _write_ready_manifest(
    *,
    artifacts: SameRegionReferenceArtifacts,
    mirror_audit: dict[str, Any],
    dry_run: bool,
    dry_run_output: Path,
) -> tuple[str, str]:
    uncheatable_records = []
    for checkpoint_step, output_path in zip(
        bridge_eval.CHECKPOINT_STEPS,
        artifacts.uncheatable_output_paths,
        strict=True,
    ):
        uncheatable_records.append(
            {
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": f"{MIRROR_ROOT}/checkpoints/step-{checkpoint_step}",
                "output_path": output_path,
                "result_path": f"{output_path}/{bridge_eval.RESULT_FILE}",
                "succeeded": (
                    StatusFile(
                        output_path,
                        worker_id="tpp40-same-region-reference-ready-audit",
                    ).status
                    == STATUS_SUCCESS
                ),
            }
        )
    table9_succeeded = (
        StatusFile(
            artifacts.table9_output_path,
            worker_id="tpp40-same-region-reference-ready-audit",
        ).status
        == STATUS_SUCCESS
    )
    payload = {
        "schema_version": 1,
        "base_acceptance_contract_sha256": bridge_eval.EXPECTED_CONTRACT_SHA256,
        "same_region_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "mirror_audit": mirror_audit,
        "logical_side": "east5",
        "training_accelerator": "v5p-8",
        "evaluation_region": EUROPE_REGION,
        "evaluation_zone": EUROPE_ZONE,
        "evaluation_tpu_type": bridge_eval.EVALUATOR_TPU_TYPE,
        "run_spec": asdict(artifacts.run_spec),
        "validation_cache_paths": list(artifacts.validation_cache_paths),
        "validation_payload_sha256": artifacts.validation_payload_sha256,
        "table9_request_set_dir": artifacts.table9_request_set_dir,
        "table9_request_set_payload_sha256": artifacts.table9_request_set_payload_sha256,
        "uncheatable_records": uncheatable_records,
        "table9_record": {
            "checkpoint_step": tpp40.EXPECTED_FINAL_CHECKPOINT_STEP,
            "checkpoint_path": f"{MIRROR_ROOT}/hf/step-{tpp40.EXPECTED_FINAL_CHECKPOINT_STEP}",
            "output_path": artifacts.table9_output_path,
            "result_path": f"{artifacts.table9_output_path}/{TABLE9_RESULTS_FILENAME}",
            "succeeded": table9_succeeded,
        },
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = _sha256_bytes(encoded.encode())
    if dry_run:
        dry_run_output.mkdir(parents=True, exist_ok=True)
        output_path = dry_run_output / "east5_same_region_reference_ready_manifest.json"
        output_path.write_text(encoded)
        return str(output_path), digest
    output_path = f"{marin_prefix_for_region(EUROPE_REGION)}/{READY_MANIFEST_ROOT}/{digest[:16]}.json"
    fs, _, _ = fsspec.get_fs_token_paths(output_path)
    if fs.exists(output_path):
        with fs.open(output_path, "rt") as handle:
            if handle.read() != encoded:
                raise RuntimeError(f"Ready manifest collision at {output_path}")
    else:
        fs.makedirs(os.path.dirname(output_path), exist_ok=True)
        with fs.open(output_path, "wt") as handle:
            handle.write(encoded)
    return output_path, digest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tpu-type", required=True)
    parser.add_argument("--tpu-region", required=True)
    parser.add_argument("--tpu-zone", required=True)
    parser.add_argument("--max-concurrent", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dry-run-output",
        type=Path,
        default=REFERENCE_DIR / "bridge_same_region_reference_dry_run_v1",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    requested_placement = (args.tpu_type, args.tpu_region, args.tpu_zone)
    expected_placement = (bridge_eval.EVALUATOR_TPU_TYPE, EUROPE_REGION, EUROPE_ZONE)
    if requested_placement != expected_placement:
        raise ValueError(f"Evaluator placement changed: {requested_placement} != {expected_placement}")
    if not 1 <= args.max_concurrent <= 3:
        raise ValueError("--max-concurrent must be in [1, 3]")
    mirror_audit = audit_east5_reference_mirror()
    artifacts = build_reference_artifacts()
    manifest_path, manifest_sha256 = _write_ready_manifest(
        artifacts=artifacts,
        mirror_audit=mirror_audit,
        dry_run=args.dry_run,
        dry_run_output=args.dry_run_output,
    )
    all_steps = (*artifacts.uncheatable_steps, artifacts.table9_step)
    all_output_paths = (*artifacts.uncheatable_output_paths, artifacts.table9_output_path)
    pending_steps = [
        step
        for step, output_path in zip(all_steps, all_output_paths, strict=True)
        if StatusFile(output_path, worker_id="tpp40-same-region-reference-launch-audit").status != STATUS_SUCCESS
    ]
    logger.info(
        "Same-region East5 reference snapshot: completed=%d/3 pending=%d manifest=%s sha256=%s",
        len(all_steps) - len(pending_steps),
        len(pending_steps),
        manifest_path,
        manifest_sha256,
    )
    if args.dry_run or os.getenv("CI") is not None or not pending_steps:
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=pending_steps,
        description="Evaluate the canonical East5 TPP40 run_order=2 reference on Europe v6e-8.",
    )


if __name__ == "__main__":
    main()
