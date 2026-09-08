# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the frozen East5-versus-Europe TPP40 bridge contract.

The bridge compares one matched TPP40 trajectory trained on v5p-8 in East5
and v6e-8 in Europe. This script freezes every content-addressed result
path before reading results, validates numerical and scientific identity, and
applies the preregistered Uncheatable and Table-9 tolerances. Production remains
blocked until a separately hashed unchanged-rerun record also proves that every
completed training and evaluation unit was skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fsspec
from fray.cluster import ResourceConfig
from marin.evaluation.olmo_base_eval.aggregate import table9_macro
from marin.evaluation.olmo_base_eval.components import scored_tasks, table9_components
from marin.evaluation.olmo_base_eval.run import RESULTS_FILENAME as TABLE9_RESULTS_FILENAME
from marin.execution.context import executor_context
from marin.execution.executor import Executor
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_tpp40 as tpp40
from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_same_region_east5_eval as same_region_eval
from experiments.domain_phase_mix import launch_delphi_tpp40_bridge_uncheatable_eval as bridge_eval
from experiments.domain_phase_mix.delphi_tpp40_evaluation_identity import (
    table9_request_set_identity,
    tree_payload_identity,
    validation_payload_identity,
)

PATH_MANIFEST_PATH = bridge_eval.REFERENCE_DIR / "bridge_acceptance_paths_v3.json"
REPORT_PATH = bridge_eval.REFERENCE_DIR / "bridge_acceptance_report_v3.json"
EXPECTED_PATH_MANIFEST_SHA256 = "aa55556044b37a6990d2032bedff0a5776b51c7db6183f3baca7ffbc0825d1f7"
EVALUATION_AUDIT_PATH = bridge_eval.REFERENCE_DIR / "evaluation_cache_audit.json"
EXPECTED_EVALUATION_AUDIT_SHA256 = "89dddcbd0e85e820c9df161a3f970dac10c4080ffea9c3785320a720103122f2"
RUNTIME_AUDIT_PATH = bridge_eval.REFERENCE_DIR / "runtime_cache_audit.json"
EXPECTED_RUNTIME_AUDIT_SHA256 = "507da8d4a86e9e7e5d91a4021cddbdacff6c07ef33e55cd3c1fe8a27411f9520"
EXPECTED_UNCHEATABLE_VALIDATION_PAYLOAD_SHA256 = "79546f8d58a61cc1d7db9f0753079b5620851bfd853af34f000c62557d0dff59"
EXPECTED_TABLE9_REQUEST_SET_PAYLOAD_SHA256 = "7401f44e57550bcde04e21a396dffdd62808e98aee833fb80bb6d5e8457a735b"
EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256 = "74d273cdd4fa38796692c171e5356141a1e9ae93c186f099db62610e943aa510"
RESULT_INVENTORY_EXCLUDED_PATHS = (".executor_info",)

COMMAND_FILES = {
    "east5": {
        "reference_eval": bridge_eval.REFERENCE_DIR / "east5_same_region_reference_eval_command_v1.txt",
    },
    "europe": {
        "training": bridge_eval.REFERENCE_DIR / "europe_bridge_run2_command_v4.txt",
        "uncheatable": bridge_eval.REFERENCE_DIR / "europe_bridge_uncheatable_run2_command_v3.txt",
    },
}
EXPECTED_COMMAND_SHA256 = {
    "east5": {
        "reference_eval": "eb587151cfd2d7de7f2469e143e4dbaf88f4f50d1a7bc826008f26d934bb3b8c",
    },
    "europe": {
        "training": "5a77dadc49130ca61ac3fd9eb4f6e7d2d7ee0257ce540ff6030d28b19e0427c5",
        "uncheatable": "7709a6eb84b038e735fd0eb7a55328038e7174db7c57ccc2447476ac09f3e561",
    },
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(path: str | Path) -> bytes:
    with fsspec.open(str(path), "rb") as handle:
        return handle.read()


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_read_bytes(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with fsspec.open(str(path), "wt") as handle:
        handle.write(encoded)


def _require_finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {result}")
    return result


def _build_training_artifacts(
    *,
    side: bridge_eval.BridgeSide,
    run_specs: list[base.DelphiSwarmRunSpec],
    source_panel: str,
    analysis_output_path: str,
    validation_configs,
):
    table9_resources = ResourceConfig.with_tpu(
        bridge_eval.EVALUATOR_TPU_TYPE,
        regions=[side.region],
        zone=side.evaluator_zone,
        disk="80g",
    )
    with executor_context():
        return base.build_launch_artifacts(
            run_specs=run_specs,
            analysis_output_path=analysis_output_path,
            source_panel=source_panel,
            validation_configs=validation_configs,
            experiment_name=side.training_experiment_name,
            architecture_target_flops=base.TARGET_FLOPS,
            wandb_tags=(
                "delphi-tpp40-augmented-swarm",
                "architecture=3e18-selected",
                "total-tpp=40",
                "fit-panel",
                "two-phase",
                f"deployment={side.region}-{side.training_tpu_type}",
            ),
            training_wandb_group=side.training_wandb_group,
            table9_wandb_group=side.table9_wandb_group,
            provenance_panel="delphi_tpp40_augmented_fit_swarm",
            provenance_scale="fixed_n_total_tpp40",
            steps_per_eval=tpp40.STEPS_PER_EVAL,
            permanent_checkpoint_interval=tpp40.EXPECTED_PHASE0_CHECKPOINT_STEP,
            table9_eval_resources=table9_resources,
        )


def _resolve_training_and_table9_paths(
    *,
    side: bridge_eval.BridgeSide,
    run_specs: list[base.DelphiSwarmRunSpec],
    source_panel: str,
    analysis_output_path: str,
    validation_configs,
    prefix: str,
) -> tuple[list[str], list[str]]:
    artifacts = _build_training_artifacts(
        side=side,
        run_specs=run_specs,
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        validation_configs=validation_configs,
    )
    resolver = Executor(
        prefix=prefix,
        executor_info_base_path=os.path.join(prefix, "experiments"),
        description=f"Resolve frozen {side.name} bridge training and Table-9 outputs.",
    )
    with executor_context():
        for step in [*artifacts.training_steps, *artifacts.eval_steps]:
            resolver.compute_version(step, is_pseudo_dep=False)
    return (
        [resolver.output_paths[step] for step in artifacts.training_steps],
        [resolver.output_paths[step] for step in artifacts.eval_steps],
    )


def _east5_same_region_path_manifest() -> dict[str, Any]:
    side = bridge_eval.BRIDGE_SIDES["east5"]
    east5_prefix = marin_prefix_for_region(side.region)
    os.environ["MARIN_PREFIX"] = east5_prefix
    run_specs, source_panel, analysis_output_path = bridge_eval._run_specs(side)
    full_validation_configs, _ = bridge_eval._validation_configs()
    canonical_training_paths = bridge_eval._original_training_paths(
        side=side,
        run_specs=run_specs,
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        full_validation_configs=full_validation_configs,
        prefix=east5_prefix,
    )
    if len(canonical_training_paths) != 1:
        raise ValueError("Expected one canonical East5 training output")

    os.environ["MARIN_PREFIX"] = marin_prefix_for_region(same_region_eval.EUROPE_REGION)
    artifacts = same_region_eval.build_reference_artifacts()
    if asdict(artifacts.run_spec) != asdict(run_specs[0]):
        raise ValueError("Same-region East5 evaluator reconstructed a different run spec")
    mirror_audit = same_region_eval.audit_east5_reference_mirror()
    canonical_training_path = canonical_training_paths[0]
    uncheatable_cells: list[dict[str, Any]] = []
    for checkpoint_step, output_path in zip(
        bridge_eval.CHECKPOINT_STEPS,
        artifacts.uncheatable_output_paths,
        strict=True,
    ):
        uncheatable_cells.append(
            {
                "side": "east5",
                "run_order": artifacts.run_spec.run_order,
                "run_name": artifacts.run_spec.run_name,
                "source_run_name": artifacts.run_spec.source_run_name,
                "data_seed": artifacts.run_spec.data_seed,
                "trainer_seed": artifacts.run_spec.trainer_seed,
                "checkpoint_step": checkpoint_step,
                "checkpoint_path": f"{same_region_eval.MIRROR_ROOT}/checkpoints/step-{checkpoint_step}",
                "canonical_checkpoint_path": f"{canonical_training_path}/checkpoints/step-{checkpoint_step}",
                "reference_checkpoint_source": "audited_europe_mirror",
                "evaluator_region": same_region_eval.EUROPE_REGION,
                "evaluator_zone": same_region_eval.EUROPE_ZONE,
                "validation_payload_sha256": artifacts.validation_payload_sha256,
                "output_path": output_path,
                "result_path": os.path.join(output_path, bridge_eval.RESULT_FILE),
            }
        )
    table9_output_path = artifacts.table9_output_path
    table9_cells = [
        {
            "side": "east5",
            "run_order": artifacts.run_spec.run_order,
            "run_name": artifacts.run_spec.run_name,
            "source_run_name": artifacts.run_spec.source_run_name,
            "panel_source": artifacts.run_spec.panel_source,
            "checkpoint_step": tpp40.EXPECTED_FINAL_CHECKPOINT_STEP,
            "checkpoint_path": f"{same_region_eval.MIRROR_ROOT}/hf/step-{tpp40.EXPECTED_FINAL_CHECKPOINT_STEP}",
            "canonical_checkpoint_path": f"{canonical_training_path}/hf/step-{tpp40.EXPECTED_FINAL_CHECKPOINT_STEP}",
            "reference_checkpoint_source": "audited_europe_mirror",
            "evaluator_region": same_region_eval.EUROPE_REGION,
            "evaluator_zone": same_region_eval.EUROPE_ZONE,
            "request_set_dir": artifacts.table9_request_set_dir,
            "request_set_payload_sha256": artifacts.table9_request_set_payload_sha256,
            "output_path": table9_output_path,
            "result_path": os.path.join(table9_output_path, TABLE9_RESULTS_FILENAME),
        }
    ]
    contract_side = bridge_eval._load_acceptance_contract()["bridge"]["logical_sides"]["east5"]
    return {
        "side": {"name": "east5", **contract_side},
        "training_output_paths": canonical_training_paths,
        "mirror_manifest_sha256": same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256,
        "mirror_trees": mirror_audit["trees"],
        "uncheatable_cells": uncheatable_cells,
        "table9_cells": table9_cells,
    }


def _side_path_manifest(side: bridge_eval.BridgeSide) -> dict[str, Any]:
    if side.name == "east5":
        return _east5_same_region_path_manifest()
    prefix = marin_prefix_for_region(side.region)
    os.environ["MARIN_PREFIX"] = prefix
    run_specs, source_panel, analysis_output_path = bridge_eval._run_specs(side)
    full_validation_configs, uncheatable_validation_configs = bridge_eval._validation_configs()
    cache_paths = bridge_eval._require_uncheatable_caches(uncheatable_validation_configs, side=side)
    validation_identity = validation_payload_identity(
        dict(zip(bridge_eval.EXPECTED_UNCHEATABLE_NAMES, cache_paths, strict=True))
    )
    validation_payload_sha256 = validation_identity["payload_sha256"]
    training_paths, table9_paths = _resolve_training_and_table9_paths(
        side=side,
        run_specs=run_specs,
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        validation_configs=full_validation_configs,
        prefix=prefix,
    )
    uncheatable_steps = bridge_eval._eval_steps(
        side=side,
        run_specs=run_specs,
        training_output_paths=training_paths,
        analysis_output_path=analysis_output_path,
        validation_configs=uncheatable_validation_configs,
        validation_payload_sha256=validation_payload_sha256,
    )
    uncheatable_paths = bridge_eval._resolve_eval_paths(uncheatable_steps, prefix=prefix, side=side)

    uncheatable_cells: list[dict[str, Any]] = []
    for index, (run_spec, training_path) in enumerate(zip(run_specs, training_paths, strict=True)):
        for offset, checkpoint_step in enumerate(bridge_eval.CHECKPOINT_STEPS):
            output_path = uncheatable_paths[index * len(bridge_eval.CHECKPOINT_STEPS) + offset]
            uncheatable_cells.append(
                {
                    "side": side.name,
                    "run_order": run_spec.run_order,
                    "run_name": run_spec.run_name,
                    "source_run_name": run_spec.source_run_name,
                    "data_seed": run_spec.data_seed,
                    "trainer_seed": run_spec.trainer_seed,
                    "checkpoint_step": checkpoint_step,
                    "checkpoint_path": os.path.join(training_path, f"checkpoints/step-{checkpoint_step}"),
                    "canonical_checkpoint_path": os.path.join(
                        training_path,
                        f"checkpoints/step-{checkpoint_step}",
                    ),
                    "reference_checkpoint_source": "native_training_output",
                    "evaluator_region": side.region,
                    "evaluator_zone": side.evaluator_zone,
                    "validation_payload_sha256": validation_payload_sha256,
                    "output_path": output_path,
                    "result_path": os.path.join(output_path, bridge_eval.RESULT_FILE),
                }
            )

    table9_request_set_dir = f"{prefix.rstrip('/')}/{base.TABLE9_REQUEST_SET_DIR.name}"
    table9_identity = table9_request_set_identity(table9_request_set_dir)
    table9_cells = []
    for run_spec, training_path, output_path in zip(run_specs, training_paths, table9_paths, strict=True):
        table9_cells.append(
            {
                "side": side.name,
                "run_order": run_spec.run_order,
                "run_name": run_spec.run_name,
                "source_run_name": run_spec.source_run_name,
                "panel_source": run_spec.panel_source,
                "checkpoint_step": tpp40.EXPECTED_FINAL_CHECKPOINT_STEP,
                "checkpoint_path": os.path.join(
                    training_path,
                    f"hf/step-{tpp40.EXPECTED_FINAL_CHECKPOINT_STEP}",
                ),
                "canonical_checkpoint_path": os.path.join(
                    training_path,
                    f"hf/step-{tpp40.EXPECTED_FINAL_CHECKPOINT_STEP}",
                ),
                "reference_checkpoint_source": "native_training_output",
                "evaluator_region": side.region,
                "evaluator_zone": side.evaluator_zone,
                "request_set_dir": table9_request_set_dir,
                "request_set_payload_sha256": table9_identity["payload_sha256"],
                "output_path": output_path,
                "result_path": os.path.join(output_path, TABLE9_RESULTS_FILENAME),
            }
        )
    return {
        "side": {
            "name": side.name,
            **bridge_eval._load_acceptance_contract()["bridge"]["logical_sides"][side.name],
        },
        "training_output_paths": training_paths,
        "mirror_trees": [],
        "uncheatable_cells": uncheatable_cells,
        "table9_cells": table9_cells,
    }


def materialize_path_manifest() -> dict[str, Any]:
    previous_prefix = os.environ.get("MARIN_PREFIX")
    try:
        sides = {name: _side_path_manifest(side) for name, side in bridge_eval.BRIDGE_SIDES.items()}
    finally:
        if previous_prefix is None:
            os.environ.pop("MARIN_PREFIX", None)
        else:
            os.environ["MARIN_PREFIX"] = previous_prefix
    return {
        "schema_version": 1,
        "acceptance_contract_sha256": bridge_eval.EXPECTED_CONTRACT_SHA256,
        "east5_reference_mirror_manifest_sha256": same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256,
        "run_orders": list(bridge_eval.BRIDGE_RUN_ORDERS),
        "checkpoint_steps": list(bridge_eval.CHECKPOINT_STEPS),
        "sides": sides,
    }


def _load_frozen_path_manifest(path: Path) -> dict[str, Any]:
    encoded = path.read_bytes()
    observed_sha256 = _sha256_bytes(encoded)
    if EXPECTED_PATH_MANIFEST_SHA256 == "UNFROZEN":
        raise ValueError("Bridge acceptance path manifest has not been frozen in source")
    if observed_sha256 != EXPECTED_PATH_MANIFEST_SHA256:
        raise ValueError(
            f"Bridge acceptance path manifest changed: {observed_sha256} != {EXPECTED_PATH_MANIFEST_SHA256}"
        )
    payload = json.loads(encoded)
    if payload.get("acceptance_contract_sha256") != bridge_eval.EXPECTED_CONTRACT_SHA256:
        raise ValueError("Bridge path manifest refers to the wrong acceptance contract")
    if payload.get("run_orders") != list(bridge_eval.BRIDGE_RUN_ORDERS):
        raise ValueError("Bridge path manifest run orders changed")
    if payload.get("checkpoint_steps") != list(bridge_eval.CHECKPOINT_STEPS):
        raise ValueError("Bridge path manifest checkpoint steps changed")
    if payload.get("east5_reference_mirror_manifest_sha256") != same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256:
        raise ValueError("Bridge path manifest refers to the wrong East5 reference mirror")
    sides = payload.get("sides")
    if not isinstance(sides, dict) or tuple(sides) != tuple(bridge_eval.BRIDGE_SIDES):
        raise ValueError("Bridge path manifest side inventory changed")
    for side_name, side in sides.items():
        if not isinstance(side, dict):
            raise ValueError(f"Bridge path manifest side {side_name} is malformed")
        training_output_paths = side.get("training_output_paths")
        uncheatable_cells = side.get("uncheatable_cells")
        table9_cells = side.get("table9_cells")
        mirror_trees = side.get("mirror_trees")
        expected_training_outputs = len(bridge_eval.BRIDGE_RUN_ORDERS)
        if not isinstance(training_output_paths, list) or len(training_output_paths) != expected_training_outputs:
            raise ValueError(
                f"Bridge path manifest must contain {expected_training_outputs} {side_name} training outputs"
            )
        expected_uncheatable_cells = len(bridge_eval.BRIDGE_RUN_ORDERS) * len(bridge_eval.CHECKPOINT_STEPS)
        if not isinstance(uncheatable_cells, list) or len(uncheatable_cells) != expected_uncheatable_cells:
            raise ValueError(
                f"Bridge path manifest must contain {expected_uncheatable_cells} " f"{side_name} Uncheatable cells"
            )
        expected_table9_cells = len(bridge_eval.BRIDGE_RUN_ORDERS)
        if not isinstance(table9_cells, list) or len(table9_cells) != expected_table9_cells:
            raise ValueError(f"Bridge path manifest must contain {expected_table9_cells} {side_name} Table-9 cells")
        expected_mirror_trees = 3 if side_name == "east5" else 0
        if not isinstance(mirror_trees, list) or len(mirror_trees) != expected_mirror_trees:
            raise ValueError(f"Bridge path manifest must contain {expected_mirror_trees} {side_name} mirror trees")
        uncheatable_keys = {(cell.get("run_order"), cell.get("checkpoint_step")) for cell in uncheatable_cells}
        table9_keys = {(cell.get("run_order"), cell.get("checkpoint_step")) for cell in table9_cells}
        expected_uncheatable_keys = {
            (run_order, checkpoint_step)
            for run_order in bridge_eval.BRIDGE_RUN_ORDERS
            for checkpoint_step in bridge_eval.CHECKPOINT_STEPS
        }
        expected_table9_keys = {
            (run_order, tpp40.EXPECTED_FINAL_CHECKPOINT_STEP) for run_order in bridge_eval.BRIDGE_RUN_ORDERS
        }
        if uncheatable_keys != expected_uncheatable_keys:
            raise ValueError(f"Bridge path manifest {side_name} Uncheatable cell identities changed")
        if table9_keys != expected_table9_keys:
            raise ValueError(f"Bridge path manifest {side_name} Table-9 cell identities changed")
        expected_checkpoint_source = "audited_europe_mirror" if side_name == "east5" else "native_training_output"
        for cell in [*uncheatable_cells, *table9_cells]:
            if cell.get("reference_checkpoint_source") != expected_checkpoint_source:
                raise ValueError(f"Bridge path manifest {side_name} checkpoint source changed")
            if cell.get("evaluator_region") != same_region_eval.EUROPE_REGION:
                raise ValueError(f"Bridge path manifest {side_name} evaluator region changed")
            if cell.get("evaluator_zone") != same_region_eval.EUROPE_ZONE:
                raise ValueError(f"Bridge path manifest {side_name} evaluator zone changed")
    return payload


def _validate_command_files() -> dict[str, dict[str, str]]:
    observed: dict[str, dict[str, str]] = {}
    for side, files in COMMAND_FILES.items():
        observed[side] = {}
        for role, path in files.items():
            digest = _sha256_bytes(path.read_bytes())
            expected = EXPECTED_COMMAND_SHA256[side][role]
            if digest != expected:
                raise ValueError(f"Frozen {side} {role} command changed: {digest} != {expected}")
            observed[side][role] = digest
    return observed


def _load_evaluation_data_identity(
    *,
    contract: dict[str, Any],
    path_manifest: dict[str, Any],
) -> dict[str, Any]:
    encoded = EVALUATION_AUDIT_PATH.read_bytes()
    observed_audit_sha256 = _sha256_bytes(encoded)
    if EXPECTED_EVALUATION_AUDIT_SHA256 == "UNFROZEN":
        raise ValueError("Bridge evaluation-cache audit has not been frozen in source")
    if observed_audit_sha256 != EXPECTED_EVALUATION_AUDIT_SHA256:
        raise ValueError(
            f"Bridge evaluation-cache audit changed: {observed_audit_sha256} != " f"{EXPECTED_EVALUATION_AUDIT_SHA256}"
        )
    audit = json.loads(encoded)
    if audit.get("schema_version") != 2 or audit.get("status") != "evaluation_payload_equivalent":
        raise ValueError("Bridge evaluation-cache audit is not a successful v2 payload audit")
    bridge = contract["bridge"]
    evaluation_paths_sha256 = audit.get("evaluation_paths_sha256")
    if not isinstance(evaluation_paths_sha256, dict):
        raise ValueError("Bridge evaluation-cache audit lacks regional path identities")
    if evaluation_paths_sha256.get("europe") != bridge["evaluation_audit_named_europe_paths_sha256"]:
        raise ValueError("Europe evaluation-cache path identity changed")
    uncheatable_payload_sha256 = audit.get("uncheatable_validation_payload_sha256")
    if not isinstance(uncheatable_payload_sha256, dict) or set(uncheatable_payload_sha256.values()) != {
        EXPECTED_UNCHEATABLE_VALIDATION_PAYLOAD_SHA256
    }:
        raise ValueError("Frozen English Uncheatable validation payload identity changed")
    table9_payload_sha256 = audit.get("table9_payload_sha256")
    if not isinstance(table9_payload_sha256, dict) or set(table9_payload_sha256.values()) != {
        EXPECTED_TABLE9_REQUEST_SET_PAYLOAD_SHA256
    }:
        raise ValueError("Frozen Table-9 request-set payload identity changed")

    cache_records = audit.get("caches")
    if not isinstance(cache_records, list):
        raise ValueError("Bridge evaluation-cache audit lacks cache records")
    observed_sides: dict[str, Any] = {}
    for side_name in bridge_eval.BRIDGE_SIDES:
        evaluator_payload_side = "europe"
        named_paths: dict[str, str] = {}
        for record in cache_records:
            if not isinstance(record, dict) or not str(record.get("name", "")).startswith("uncheatable_eval/"):
                continue
            paths = record.get("paths")
            if not isinstance(paths, dict) or not isinstance(paths.get(evaluator_payload_side), str):
                raise ValueError(f"Evaluation audit lacks {evaluator_payload_side} path for {record.get('name')}")
            named_paths[str(record["name"])] = paths[evaluator_payload_side]
        if tuple(sorted(named_paths)) != tuple(sorted(bridge_eval.EXPECTED_UNCHEATABLE_NAMES)):
            raise ValueError(f"Evaluation audit English Uncheatable inventory changed for {side_name}")
        current_uncheatable = validation_payload_identity(named_paths)
        if current_uncheatable["payload_sha256"] != EXPECTED_UNCHEATABLE_VALIDATION_PAYLOAD_SHA256:
            raise ValueError(f"Live {side_name} English Uncheatable validation payload changed after audit")

        table9_cells = path_manifest["sides"][side_name]["table9_cells"]
        request_set_dirs = {cell["request_set_dir"] for cell in table9_cells}
        if len(request_set_dirs) != 1:
            raise ValueError(f"Frozen {side_name} Table-9 request-set paths are inconsistent")
        current_table9 = table9_request_set_identity(request_set_dirs.pop())
        if current_table9["payload_sha256"] != EXPECTED_TABLE9_REQUEST_SET_PAYLOAD_SHA256:
            raise ValueError(f"Live {side_name} Table-9 request set changed after audit")
        observed_sides[side_name] = {
            "uncheatable_validation_payload_sha256": current_uncheatable["payload_sha256"],
            "uncheatable_validation_objects": current_uncheatable["objects"],
            "uncheatable_validation_bytes": current_uncheatable["bytes"],
            "table9_request_set_payload_sha256": current_table9["payload_sha256"],
            "table9_manifest_sha256": current_table9["manifest_sha256"],
            "table9_requests": current_table9["requests"],
        }
    europe_launcher_paths = tuple(
        record["paths"]["europe"] for record in sorted(cache_records, key=lambda record: record["name"])
    )
    europe_launcher_paths_sha256 = _sha256_bytes(json.dumps(europe_launcher_paths, separators=(",", ":")).encode())
    if europe_launcher_paths_sha256 != bridge["launcher_validation_cache_paths_sha256"]:
        raise ValueError("Europe launcher validation-cache path identity changed")
    return {
        "audit_sha256": observed_audit_sha256,
        "evaluation_paths_sha256": evaluation_paths_sha256,
        "europe_launcher_validation_cache_paths_sha256": europe_launcher_paths_sha256,
        "sides": observed_sides,
        "passed": True,
    }


def _load_training_data_identity(*, contract: dict[str, Any]) -> dict[str, Any]:
    encoded = RUNTIME_AUDIT_PATH.read_bytes()
    observed_audit_sha256 = _sha256_bytes(encoded)
    if observed_audit_sha256 != EXPECTED_RUNTIME_AUDIT_SHA256:
        raise ValueError(
            f"Bridge runtime-cache audit changed: {observed_audit_sha256} != {EXPECTED_RUNTIME_AUDIT_SHA256}"
        )
    audit = json.loads(encoded)
    if audit.get("status") != "training_payload_equivalent":
        raise ValueError("Bridge runtime-cache audit is not a successful payload-equivalence audit")
    runtime_paths_sha256 = audit.get("runtime_paths_sha256")
    if not isinstance(runtime_paths_sha256, dict):
        raise ValueError("Bridge runtime-cache audit lacks regional path identities")
    expected_europe_sha256 = contract["bridge"]["runtime_cache_paths_sha256"]
    if runtime_paths_sha256.get("europe") != expected_europe_sha256:
        raise ValueError("Europe runtime-cache path identity changed")
    mirror_identity = same_region_eval.audit_east5_reference_mirror()
    return {
        "audit_sha256": observed_audit_sha256,
        "runtime_paths_sha256": runtime_paths_sha256,
        "logical_runtime_contract_sha256": audit.get("logical_runtime_contract_sha256"),
        "east5_reference_mirror": mirror_identity,
        "passed": True,
    }


def _validate_uncheatable_payload(payload: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("Uncheatable result schema changed")
    expected_identity = {
        key: cell[key]
        for key in (
            "side",
            "run_order",
            "run_name",
            "source_run_name",
            "data_seed",
            "trainer_seed",
            "checkpoint_step",
            "checkpoint_path",
        )
    }
    observed_identity = {key: payload.get(key) for key in expected_identity}
    if observed_identity != expected_identity:
        raise ValueError(f"Uncheatable identity mismatch: {observed_identity} != {expected_identity}")
    if payload.get("acceptance_contract_sha256") != bridge_eval.EXPECTED_CONTRACT_SHA256:
        raise ValueError("Uncheatable result has the wrong acceptance contract")
    if payload.get("evaluator_tpu_type") != bridge_eval.EVALUATOR_TPU_TYPE:
        raise ValueError("Uncheatable evaluator TPU type changed")
    if payload.get("eval_batch_size") != bridge_eval.EVAL_BATCH_SIZE:
        raise ValueError("Uncheatable evaluator batch size changed")
    metadata = payload.get("checkpoint_metadata")
    if not isinstance(metadata, dict) or metadata.get("step") != cell["checkpoint_step"]:
        raise ValueError("Uncheatable checkpoint metadata step changed")
    metadata_sha256 = payload.get("checkpoint_metadata_sha256")
    if not isinstance(metadata_sha256, str) or len(metadata_sha256) != 64:
        raise ValueError("Uncheatable result lacks checkpoint metadata identity")
    model_config_sha256 = payload.get("model_config_sha256")
    if not isinstance(model_config_sha256, str) or len(model_config_sha256) != 64:
        raise ValueError("Uncheatable result lacks model-config identity")
    validation_payload_sha256 = payload.get("validation_payload_sha256")
    if validation_payload_sha256 != cell.get("validation_payload_sha256"):
        raise ValueError("Uncheatable result validation-payload identity changed")
    if validation_payload_sha256 != EXPECTED_UNCHEATABLE_VALIDATION_PAYLOAD_SHA256:
        raise ValueError("Uncheatable result used an unfrozen validation payload")
    total_trainable_params = payload.get("total_trainable_params")
    if isinstance(total_trainable_params, bool) or not isinstance(total_trainable_params, int):
        raise ValueError("Uncheatable result lacks parameter-count identity")
    component_bpb = payload.get("component_bpb")
    if not isinstance(component_bpb, dict) or set(component_bpb) != set(bridge_eval.EXPECTED_UNCHEATABLE_NAMES):
        raise ValueError("Uncheatable component inventory changed")
    component_bpb, macro_bpb = bridge_eval._uncheatable_metrics(
        {
            **{f"eval/{name}/bpb": value for name, value in component_bpb.items()},
            "eval/uncheatable_eval/macro_bpb": payload.get("macro_bpb"),
        }
    )
    if any(value <= 0.0 for value in component_bpb.values()) or macro_bpb <= 0.0:
        raise ValueError("Uncheatable BPB values must be positive")
    if len(set(component_bpb.values())) == 1:
        raise ValueError("Uncheatable result is degenerate across all seven components")
    return {
        "macro_bpb": macro_bpb,
        "component_bpb": component_bpb,
        "model_config_sha256": model_config_sha256,
        "total_trainable_params": total_trainable_params,
        "checkpoint_metadata_sha256": metadata_sha256,
        "validation_payload_sha256": validation_payload_sha256,
        "source_run_name": payload["source_run_name"],
        "data_seed": payload["data_seed"],
        "trainer_seed": payload["trainer_seed"],
    }


def _validate_table9_payload(payload: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    if payload.get("name") != f"t9_{cell['run_name']}":
        raise ValueError("Table-9 result name changed")
    if payload.get("checkpoint_path") != cell["checkpoint_path"]:
        raise ValueError("Table-9 checkpoint path changed")
    if payload.get("request_set_dir") != cell.get("request_set_dir"):
        raise ValueError("Table-9 request-set path changed")
    if cell.get("request_set_payload_sha256") != EXPECTED_TABLE9_REQUEST_SET_PAYLOAD_SHA256:
        raise ValueError("Table-9 path manifest refers to an unfrozen request-set payload")
    provenance = payload.get("provenance")
    expected_provenance = {
        "evaluator": "marin-native-table9-bpb",
        "panel": "delphi_tpp40_augmented_fit_swarm",
        "scale": "fixed_n_total_tpp40",
        "source_run_name": cell["source_run_name"],
        "swarm_run_name": cell["run_name"],
        "panel_source": cell["panel_source"],
    }
    if provenance != expected_provenance:
        raise ValueError(f"Table-9 provenance changed: {provenance} != {expected_provenance}")
    components = payload.get("table9_components")
    if not isinstance(components, dict) or tuple(components) != table9_components():
        raise ValueError("Table-9 component inventory or order changed")
    normalized_components = {
        name: _require_finite_number(value, label=f"Table-9 component {name}") for name, value in components.items()
    }
    macro_bpb = _require_finite_number(payload.get("table9_macro_bpb"), label="Table-9 macro BPB")
    reconstructed_macro = table9_macro(normalized_components)
    if macro_bpb != reconstructed_macro:
        raise ValueError(f"Table-9 macro mismatch: {macro_bpb} != {reconstructed_macro}")
    task_bpb = payload.get("task_bpb")
    if not isinstance(task_bpb, dict) or set(task_bpb) != set(scored_tasks()):
        raise ValueError("Table-9 scored-task inventory changed")
    normalized_tasks = {
        name: _require_finite_number(value, label=f"Table-9 task {name}") for name, value in task_bpb.items()
    }
    if any(value <= 0.0 for value in normalized_components.values()) or macro_bpb <= 0.0:
        raise ValueError("Table-9 BPB values must be positive")
    if any(value <= 0.0 for value in normalized_tasks.values()):
        raise ValueError("Table-9 task BPB values must be positive")
    if len(set(normalized_components.values())) == 1:
        raise ValueError("Table-9 result is degenerate across all components")
    num_instances = payload.get("num_instances")
    if not isinstance(num_instances, dict) or set(num_instances) != set(scored_tasks()):
        raise ValueError("Table-9 request-set inventory changed")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in num_instances.values()):
        raise ValueError("Table-9 request-set counts must be positive integers")
    request_set_version = payload.get("request_set_version")
    olmo_eval_git_sha = payload.get("olmo_eval_git_sha")
    if not isinstance(request_set_version, str) or not request_set_version:
        raise ValueError("Table-9 request-set version is missing")
    if not isinstance(olmo_eval_git_sha, str) or not olmo_eval_git_sha:
        raise ValueError("Table-9 OLMo Eval identity is missing")
    return {
        "macro_bpb": macro_bpb,
        "component_bpb": normalized_components,
        "task_bpb": normalized_tasks,
        "num_instances": num_instances,
        "request_set_version": request_set_version,
        "olmo_eval_git_sha": olmo_eval_git_sha,
        "request_set_payload_sha256": cell["request_set_payload_sha256"],
    }


def _paired_threshold(values: list[float], *, expected_count: int, mean_max: float, any_max: float) -> dict[str, Any]:
    absolute_values = [abs(value) for value in values]
    mean_absolute = sum(absolute_values) / len(absolute_values) if absolute_values else None
    maximum_absolute = max(absolute_values) if absolute_values else None
    passed = (
        len(values) == expected_count
        and mean_absolute is not None
        and maximum_absolute is not None
        and mean_absolute <= mean_max
        and maximum_absolute <= any_max
    )
    return {
        "expected_pair_count": expected_count,
        "observed_pair_count": len(values),
        "signed_paired_deltas": values,
        "mean_absolute_paired_delta": mean_absolute,
        "maximum_absolute_paired_delta": maximum_absolute,
        "mean_absolute_paired_delta_max": mean_max,
        "any_row_absolute_paired_delta_max": any_max,
        "passed": passed,
    }


def result_inventory(path_manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute the current content identity of every frozen bridge output."""
    sides: dict[str, Any] = {}
    for side_name, side in path_manifest["sides"].items():
        units: list[dict[str, Any]] = []
        for run_order, output_path in zip(
            bridge_eval.BRIDGE_RUN_ORDERS,
            side["training_output_paths"],
            strict=True,
        ):
            units.append(
                {
                    "kind": "training",
                    "run_order": run_order,
                    "output_path": output_path,
                    **tree_payload_identity(
                        output_path,
                        excluded_relative_paths=RESULT_INVENTORY_EXCLUDED_PATHS,
                    ),
                }
            )
        for mirror_tree in side.get("mirror_trees", []):
            destination_path = mirror_tree["destination_path"]
            units.append(
                {
                    "kind": "mirror",
                    "relative_path": mirror_tree["relative_path"],
                    "source_path": mirror_tree["source_path"],
                    "output_path": destination_path,
                    **tree_payload_identity(destination_path),
                }
            )
        for kind in ("uncheatable", "table9"):
            for cell in side[f"{kind}_cells"]:
                units.append(
                    {
                        "kind": kind,
                        "run_order": cell["run_order"],
                        "checkpoint_step": cell["checkpoint_step"],
                        "output_path": cell["output_path"],
                        **tree_payload_identity(
                            cell["output_path"],
                            excluded_relative_paths=RESULT_INVENTORY_EXCLUDED_PATHS,
                        ),
                    }
                )
        payload = json.dumps(units, sort_keys=True, separators=(",", ":")).encode()
        sides[side_name] = {
            "inventory_sha256": _sha256_bytes(payload),
            "unit_counts": {
                kind: sum(unit["kind"] == kind for unit in units)
                for kind in ("training", "mirror", "uncheatable", "table9")
            },
            "units": units,
        }
    return {
        "schema_version": 1,
        "acceptance_contract_sha256": bridge_eval.EXPECTED_CONTRACT_SHA256,
        "path_manifest_sha256": EXPECTED_PATH_MANIFEST_SHA256,
        "sides": sides,
    }


def _validate_idempotence(
    payload: dict[str, Any] | None,
    *,
    expected_sha256: str | None,
    current_inventory: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    if payload is None:
        return {"passed": False, "reason": "idempotence evidence not supplied"}, [
            "Unchanged training and evaluation reruns have not yet been audited"
        ]
    errors: list[str] = []
    if EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256 == "UNFROZEN":
        errors.append("Idempotence evidence has not been frozen in source")
    if expected_sha256 != EXPECTED_IDEMPOTENCE_EVIDENCE_SHA256:
        errors.append("Idempotence evidence SHA-256 does not match the source-frozen digest")
    if payload.get("schema_version") != 3:
        errors.append("Idempotence evidence schema changed")
    if payload.get("acceptance_contract_sha256") != bridge_eval.EXPECTED_CONTRACT_SHA256:
        errors.append("Idempotence evidence refers to the wrong acceptance contract")
    if payload.get("path_manifest_sha256") != EXPECTED_PATH_MANIFEST_SHA256:
        errors.append("Idempotence evidence refers to the wrong path manifest")
    if payload.get("evaluation_audit_sha256") != EXPECTED_EVALUATION_AUDIT_SHA256:
        errors.append("Idempotence evidence refers to the wrong evaluation-data audit")
    if payload.get("east5_reference_mirror_manifest_sha256") != same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256:
        errors.append("Idempotence evidence refers to the wrong East5 reference mirror")
    if current_inventory is None:
        errors.append("Current bridge result inventory was not recomputed")
        current_sides: dict[str, Any] = {}
    else:
        current_sides = current_inventory.get("sides", {})
    sides = payload.get("sides")
    if not isinstance(sides, dict):
        errors.append("Idempotence evidence lacks side records")
        sides = {}
    after_inventory_started_at_ms = payload.get("after_inventory_started_at_ms")
    after_inventory_captured_at_ms = payload.get("after_inventory_captured_at_ms")
    if (
        isinstance(after_inventory_started_at_ms, bool)
        or not isinstance(after_inventory_started_at_ms, int)
        or isinstance(after_inventory_captured_at_ms, bool)
        or not isinstance(after_inventory_captured_at_ms, int)
        or after_inventory_started_at_ms > after_inventory_captured_at_ms
    ):
        errors.append("Idempotence evidence lacks a valid after-inventory interval")
    expected_units = {
        "east5": {
            "training": len(bridge_eval.BRIDGE_RUN_ORDERS),
            "mirror": 3,
            "uncheatable": len(bridge_eval.BRIDGE_RUN_ORDERS) * len(bridge_eval.CHECKPOINT_STEPS),
            "table9": len(bridge_eval.BRIDGE_RUN_ORDERS),
        },
        "europe": {
            "training": len(bridge_eval.BRIDGE_RUN_ORDERS),
            "mirror": 0,
            "uncheatable": len(bridge_eval.BRIDGE_RUN_ORDERS) * len(bridge_eval.CHECKPOINT_STEPS),
            "table9": len(bridge_eval.BRIDGE_RUN_ORDERS),
        },
    }
    for side in bridge_eval.BRIDGE_SIDES:
        side_record = sides.get(side)
        if not isinstance(side_record, dict):
            errors.append(f"Idempotence evidence lacks {side}")
            continue
        if side == "east5":
            expected_roles = ("reference_eval",)
            if side_record.get("mirror_manifest_sha256") != same_region_eval.EXPECTED_MIRROR_MANIFEST_SHA256:
                errors.append("east5 idempotence evidence refers to the wrong mirror manifest")
        else:
            expected_roles = ("training", "uncheatable")
        for rerun_role in expected_roles:
            if side_record.get(f"{rerun_role}_command_sha256") != EXPECTED_COMMAND_SHA256[side][rerun_role]:
                errors.append(f"{side} {rerun_role} rerun command changed")
        before = side_record.get("result_inventory_sha256_before")
        after = side_record.get("result_inventory_sha256_after")
        current_side = current_sides.get(side, {})
        current = current_side.get("inventory_sha256") if isinstance(current_side, dict) else None
        if not isinstance(before, str) or len(before) != 64 or before != after or after != current:
            errors.append(f"{side} result inventory changed across the unchanged rerun")
        for rerun_role in expected_roles:
            rerun = side_record.get(f"{rerun_role}_rerun")
            if not isinstance(rerun, dict):
                errors.append(f"{side} idempotence evidence lacks {rerun_role} rerun record")
                continue
            if rerun.get("state") != "succeeded" or rerun.get("exit_code") != 0:
                errors.append(f"{side} {rerun_role} idempotence rerun did not succeed")
            if rerun.get("child_job_count") != 0:
                errors.append(f"{side} {rerun_role} idempotence rerun emitted child jobs")
            finished_at_ms = rerun.get("finished_at_ms")
            if (
                not isinstance(after_inventory_started_at_ms, int)
                or isinstance(finished_at_ms, bool)
                or not isinstance(finished_at_ms, int)
                or finished_at_ms > after_inventory_started_at_ms
            ):
                errors.append(f"{side} {rerun_role} finished after the result inventory audit began")
        observed_unit_counts = current_side.get("unit_counts") if isinstance(current_side, dict) else None
        if observed_unit_counts != expected_units[side]:
            errors.append(f"{side} completed-output unit inventory changed")
    return {"passed": not errors, "evidence_sha256": expected_sha256, "sides": sides}, errors


def _load_live_payloads(path_manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    payloads: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    sides = path_manifest.get("sides")
    if not isinstance(sides, dict):
        return payloads, ["Path manifest lacks side records"]
    for side_name, side in sides.items():
        if not isinstance(side, dict):
            errors.append(f"Path manifest side {side_name} is malformed")
            continue
        for kind in ("uncheatable", "table9"):
            cells = side.get(f"{kind}_cells")
            if not isinstance(cells, list):
                errors.append(f"Path manifest lacks {side_name} {kind} cells")
                continue
            for cell in cells:
                if not isinstance(cell, dict):
                    errors.append(f"Path manifest has malformed {side_name} {kind} cell")
                    continue
                key = f"{kind}:{side_name}:{cell.get('run_order')}:{cell.get('checkpoint_step')}"
                output_path = cell.get("output_path")
                result_path = cell.get("result_path")
                if not isinstance(output_path, str) or not isinstance(result_path, str):
                    errors.append(f"{key} lacks frozen output paths")
                    continue
                status = StatusFile(output_path, worker_id="tpp40-bridge-acceptance-audit").status
                if status != STATUS_SUCCESS:
                    errors.append(f"{key} executor status is {status}, not success")
                    continue
                try:
                    payload = _read_json(result_path)
                    if kind == "uncheatable":
                        metadata_result = bridge_eval._checkpoint_metadata(
                            cell["checkpoint_path"], expected_step=cell["checkpoint_step"]
                        )
                        if metadata_result is None:
                            raise ValueError(f"checkpoint disappeared at {cell['checkpoint_path']}")
                        _, current_metadata_sha256 = metadata_result
                        if payload.get("checkpoint_metadata_sha256") != current_metadata_sha256:
                            raise ValueError("checkpoint metadata changed after evaluation")
                    payloads[key] = payload
                except FileNotFoundError:
                    errors.append(f"{key} result is missing at {result_path}")
                except (json.JSONDecodeError, OSError, ValueError) as error:
                    errors.append(f"{key} result is invalid: {error}")
    return payloads, errors


def analyze_payloads(
    *,
    contract: dict[str, Any],
    path_manifest: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    idempotence_payload: dict[str, Any] | None,
    idempotence_sha256: str | None,
    current_inventory: dict[str, Any] | None,
    evaluation_data_identity: dict[str, Any] | None,
    training_data_identity: dict[str, Any] | None,
    observed_contract_sha256: str,
    observed_path_manifest_sha256: str,
) -> dict[str, Any]:
    errors: list[str] = []
    if evaluation_data_identity is None or evaluation_data_identity.get("passed") is not True:
        errors.append("Cross-region evaluation-data identity has not been verified")
    if training_data_identity is None or training_data_identity.get("passed") is not True:
        errors.append("Cross-region training-data identity has not been verified")
    normalized_uncheatable: dict[tuple[str, int, int], dict[str, Any]] = {}
    normalized_table9: dict[tuple[str, int], dict[str, Any]] = {}
    sides = path_manifest["sides"]
    for side_name, side in sides.items():
        for cell in side["uncheatable_cells"]:
            key = f"uncheatable:{side_name}:{cell['run_order']}:{cell['checkpoint_step']}"
            payload = payloads.get(key)
            if payload is None:
                errors.append(f"Missing {key}")
                continue
            try:
                normalized_uncheatable[(side_name, cell["run_order"], cell["checkpoint_step"])] = (
                    _validate_uncheatable_payload(payload, cell)
                )
            except ValueError as error:
                errors.append(f"{key}: {error}")
        for cell in side["table9_cells"]:
            key = f"table9:{side_name}:{cell['run_order']}:{cell['checkpoint_step']}"
            payload = payloads.get(key)
            if payload is None:
                errors.append(f"Missing {key}")
                continue
            try:
                normalized_table9[(side_name, cell["run_order"])] = _validate_table9_payload(payload, cell)
            except ValueError as error:
                errors.append(f"{key}: {error}")

    acceptance = contract["acceptance"]
    uncheatable_reports: dict[str, Any] = {}
    for checkpoint_name, checkpoint_step in (
        ("phase_0", tpp40.EXPECTED_PHASE0_CHECKPOINT_STEP),
        ("endpoint", tpp40.EXPECTED_FINAL_CHECKPOINT_STEP),
    ):
        paired: list[dict[str, Any]] = []
        deltas: list[float] = []
        for run_order in bridge_eval.BRIDGE_RUN_ORDERS:
            east = normalized_uncheatable.get(("east5", run_order, checkpoint_step))
            europe = normalized_uncheatable.get(("europe", run_order, checkpoint_step))
            if east is None or europe is None:
                continue
            identity_fields = (
                "model_config_sha256",
                "total_trainable_params",
                "validation_payload_sha256",
                "source_run_name",
                "data_seed",
                "trainer_seed",
            )
            mismatches = [field for field in identity_fields if east[field] != europe[field]]
            if mismatches:
                errors.append(
                    f"Uncheatable cross-side identity mismatch for run {run_order}, step {checkpoint_step}: {mismatches}"
                )
                continue
            delta = europe["macro_bpb"] - east["macro_bpb"]
            deltas.append(delta)
            paired.append(
                {
                    "run_order": run_order,
                    "east5_macro_bpb": east["macro_bpb"],
                    "europe_macro_bpb": europe["macro_bpb"],
                    "europe_minus_east5": delta,
                    "component_deltas": {
                        name: europe["component_bpb"][name] - east["component_bpb"][name]
                        for name in bridge_eval.EXPECTED_UNCHEATABLE_NAMES
                    },
                }
            )
        threshold = _paired_threshold(
            deltas,
            expected_count=len(bridge_eval.BRIDGE_RUN_ORDERS),
            mean_max=acceptance[f"uncheatable_{checkpoint_name}_mean_absolute_paired_delta_max_bpb"],
            any_max=acceptance[f"uncheatable_{checkpoint_name}_any_row_absolute_paired_delta_max_bpb"],
        )
        uncheatable_reports[checkpoint_name] = {"pairs": paired, "threshold": threshold}

    table9_pairs: list[dict[str, Any]] = []
    table9_deltas: list[float] = []
    for run_order in bridge_eval.BRIDGE_RUN_ORDERS:
        east = normalized_table9.get(("east5", run_order))
        europe = normalized_table9.get(("europe", run_order))
        if east is None or europe is None:
            continue
        identity_fields = (
            "num_instances",
            "request_set_version",
            "olmo_eval_git_sha",
            "request_set_payload_sha256",
        )
        mismatches = [field for field in identity_fields if east[field] != europe[field]]
        if mismatches:
            errors.append(f"Table-9 cross-side identity mismatch for run {run_order}: {mismatches}")
            continue
        delta = europe["macro_bpb"] - east["macro_bpb"]
        table9_deltas.append(delta)
        table9_pairs.append(
            {
                "run_order": run_order,
                "east5_macro_bpb": east["macro_bpb"],
                "europe_macro_bpb": europe["macro_bpb"],
                "europe_minus_east5": delta,
                "component_deltas": {
                    name: europe["component_bpb"][name] - east["component_bpb"][name] for name in table9_components()
                },
            }
        )
    table9_threshold = _paired_threshold(
        table9_deltas,
        expected_count=len(bridge_eval.BRIDGE_RUN_ORDERS),
        mean_max=acceptance["table9_macro_mean_absolute_paired_delta_max_bpb"],
        any_max=acceptance["table9_macro_any_row_absolute_paired_delta_max_bpb"],
    )
    numerical_errors = list(errors)
    idempotence, idempotence_errors = _validate_idempotence(
        idempotence_payload,
        expected_sha256=idempotence_sha256,
        current_inventory=current_inventory,
    )
    errors.extend(idempotence_errors)
    numerical_passed = (
        uncheatable_reports["phase_0"]["threshold"]["passed"]
        and uncheatable_reports["endpoint"]["threshold"]["passed"]
        and table9_threshold["passed"]
        and not numerical_errors
    )
    production_launch_authorized = numerical_passed and idempotence["passed"] and not errors
    return {
        "schema_version": 1,
        "acceptance_contract_sha256": observed_contract_sha256,
        "path_manifest_sha256": observed_path_manifest_sha256,
        "evaluation_data_identity": evaluation_data_identity,
        "training_data_identity": training_data_identity,
        "uncheatable": uncheatable_reports,
        "table9": {"pairs": table9_pairs, "threshold": table9_threshold},
        "idempotence": idempotence,
        "numerical_acceptance_passed": numerical_passed,
        "production_launch_authorized": production_launch_authorized,
        "blocking_errors": errors,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-manifest", type=Path, default=PATH_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--materialize-path-manifest", action="store_true")
    parser.add_argument("--idempotence-evidence")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _allow_incomplete_failure(
    *,
    loading_errors: list[str],
    numerical_acceptance_passed: bool,
    idempotence_payload: dict[str, Any] | None,
) -> bool:
    loading_is_only_incomplete = bool(loading_errors) and all(
        "executor status is" in error or "result is missing" in error for error in loading_errors
    )
    missing_idempotence_only = not loading_errors and numerical_acceptance_passed and idempotence_payload is None
    return loading_is_only_incomplete or missing_idempotence_only


def main() -> None:
    args = _parse_args()
    bridge_eval._load_acceptance_contract()
    if args.materialize_path_manifest:
        payload = materialize_path_manifest()
        _write_json(args.path_manifest, payload)
        print(_sha256_bytes(args.path_manifest.read_bytes()))
        return

    observed_contract_sha256 = _sha256_bytes(bridge_eval.ACCEPTANCE_CONTRACT_PATH.read_bytes())
    path_manifest = _load_frozen_path_manifest(args.path_manifest)
    observed_path_manifest_sha256 = _sha256_bytes(args.path_manifest.read_bytes())
    command_sha256 = _validate_command_files()
    contract = bridge_eval._load_acceptance_contract()
    evaluation_data_identity = _load_evaluation_data_identity(
        contract=contract,
        path_manifest=path_manifest,
    )
    training_data_identity = _load_training_data_identity(contract=contract)
    payloads, loading_errors = _load_live_payloads(path_manifest)
    idempotence_payload = None
    idempotence_sha256 = None
    if args.idempotence_evidence is not None:
        encoded = _read_bytes(args.idempotence_evidence)
        idempotence_sha256 = _sha256_bytes(encoded)
        idempotence_payload = json.loads(encoded)
    current_inventory = None
    if idempotence_payload is not None:
        current_inventory = result_inventory(path_manifest)
    report = analyze_payloads(
        contract=contract,
        path_manifest=path_manifest,
        payloads=payloads,
        idempotence_payload=idempotence_payload,
        idempotence_sha256=idempotence_sha256,
        current_inventory=current_inventory,
        evaluation_data_identity=evaluation_data_identity,
        training_data_identity=training_data_identity,
        observed_contract_sha256=observed_contract_sha256,
        observed_path_manifest_sha256=observed_path_manifest_sha256,
    )
    report["frozen_command_sha256"] = command_sha256
    report["blocking_errors"] = [*loading_errors, *report["blocking_errors"]]
    if loading_errors:
        report["numerical_acceptance_passed"] = False
        report["production_launch_authorized"] = False
    _write_json(args.output, report)
    allowed_incomplete = args.allow_incomplete and _allow_incomplete_failure(
        loading_errors=loading_errors,
        numerical_acceptance_passed=report["numerical_acceptance_passed"],
        idempotence_payload=idempotence_payload,
    )
    if not report["production_launch_authorized"] and not allowed_incomplete:
        raise RuntimeError(f"Bridge acceptance failed closed; see {args.output}")


if __name__ == "__main__":
    main()
