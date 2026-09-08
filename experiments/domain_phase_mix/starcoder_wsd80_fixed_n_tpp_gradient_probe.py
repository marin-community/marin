# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the fixed-N StarCoder WSD80 TPP gradient-onset probes."""

import argparse
import csv
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import fsspec
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from marin.execution.remote import remote

from experiments.domain_phase_mix import starcoder_wsd80_gradient_mechanism_repair as mechanism
from experiments.domain_phase_mix import starcoder_wsd80_gradient_plot_completion as historical_runtime
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_fixed_n_tpp_gradient_probe_20260822 as freeze,
)

SCHEMA_VERSION = "2026-08-22-fixed-n-tpp-gradient-probe-v1"
ARTIFACT_VERSION = "2026.08.22.1"

_REQUIRED_MECHANISM_FREEZE_EXPORTS = (
    "GLOBAL_STARCODER",
    "MARIN_PREFIX",
    "NEMOTRON",
    "OUTPUT_DIR",
    "PARENT_RELEASE_PATH",
    "RELEASE_PATH",
    "RELEASE_VERSION",
    "REPO_ROOT",
    "RESULT_ROOT",
    "SCIENTIFIC_STATUS",
    "SUPPORT_STARCODER",
    "canonical_json",
    "canonical_sha256",
    "file_sha256",
)
_REQUIRED_HISTORICAL_FREEZE_EXPORTS = (
    "EXPECTED_DEVICE_COUNT",
    "EXPECTED_PYTHON_VERSION",
    "EXPECTED_RUNTIME_VERSIONS",
    "FULL_LAUNCH_AUTHORIZATION_PATH",
    "HISTORICAL_RUNTIME_COMMIT",
    "REMOTE_ADAPTER_CANARY_PATH",
    "RUNTIME_ENVIRONMENT_BASELINE_PATH",
    "STAGE_ROW_COUNTS",
    "TASK_IMAGE",
    "_write_create_only",
    *_REQUIRED_MECHANISM_FREEZE_EXPORTS,
)


def _apply_runtime_adapter() -> None:
    missing_mechanism_exports = [name for name in _REQUIRED_MECHANISM_FREEZE_EXPORTS if not hasattr(freeze, name)]
    missing_historical_exports = [name for name in _REQUIRED_HISTORICAL_FREEZE_EXPORTS if not hasattr(freeze, name)]
    if missing_mechanism_exports or missing_historical_exports:
        raise RuntimeError(
            "Fixed-N TPP freeze namespace is incomplete: "
            f"mechanism={missing_mechanism_exports}, historical={missing_historical_exports}"
        )
    runtime: Any = historical_runtime
    runtime.freeze = freeze
    runtime.SCHEMA_VERSION = SCHEMA_VERSION
    runtime.ARTIFACT_VERSION = ARTIFACT_VERSION


def _configure_runtime(expected_release_sha256: str, *, verify_worker_runtime: bool = False) -> None:
    _apply_runtime_adapter()
    historical_runtime._configure_mechanism_runtime(
        expected_release_sha256,
        verify_worker_runtime=verify_worker_runtime,
    )


def run_probe_group(config: mechanism.MechanismGroupConfig) -> None:
    _configure_runtime(config.release_sha256, verify_worker_runtime=True)
    mechanism.run_mechanism_group(config)


def run_remote_adapter_canary(config: mechanism.MechanismGroupConfig) -> None:
    _apply_runtime_adapter()
    historical_runtime.run_remote_adapter_canary(config)


def _load_release(expected_sha256: str) -> dict[str, Any]:
    _configure_runtime(expected_sha256)
    release = mechanism._load_release(expected_sha256)
    review = release["external_review"]
    review_path = freeze.REPO_ROOT / review["path"]
    if freeze.file_sha256(review_path) != review["sha256"] or review["verdict"] != "PASS":
        raise ValueError("Fixed-N TPP CC review gate failed")
    historical_path = freeze.REPO_ROOT / release["historical_release_path"]
    historical_release = json.loads(historical_path.read_text())
    if (
        freeze.file_sha256(historical_path) != release["historical_release_file_sha256"]
        or historical_release["release_sha256"] != release["historical_release_sha256"]
    ):
        raise ValueError("Fixed-N TPP historical v8 release provenance drifted")
    if release["submission_contract"]["required_environment"] != {"UV_FROZEN": "1"}:
        raise ValueError("Fixed-N TPP frozen submission environment drifted")
    _verify_workspace(release, "required_preauthorization_workspace_paths")
    return release


def _read_manifest(release: Mapping[str, Any], stage: int | None = None) -> list[dict[str, str]]:
    summary = release["manifests"]["full"]
    path = freeze.REPO_ROOT / summary["path"]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(summary["row_count"]):
        raise ValueError("Fixed-N TPP manifest row count drifted")
    if stage == 0:
        rows = [row for row in rows if int(row["preflight_wave"]) == 1]
    elif stage is not None:
        rows = [row for row in rows if int(row["launch_stage"]) == stage]
    return rows


def _verify_workspace(release: Mapping[str, Any], contract_key: str) -> None:
    paths = release["submission_contract"][contract_key]
    if paths != sorted(set(paths)):
        raise ValueError("Fixed-N TPP required workspace inventory is not canonical")
    missing = [path for path in paths if not (freeze.REPO_ROOT / path).is_file()]
    if missing:
        raise ValueError(f"Fixed-N TPP workspace is incomplete: {missing[:8]}")


def _steps(release: Mapping[str, Any], stage: int) -> list[ArtifactStep[Artifact]]:
    configs = mechanism.base.freeze._full_configs()
    parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
    cache_sha256 = parent_release["manifests"]["cache_provenance"]["sha256"]
    resources = replace(mechanism._resources(), image=freeze.TASK_IMAGE)
    prefix = f"{freeze.MARIN_PREFIX}/"
    steps = []
    for row in _read_manifest(release, stage):
        config = mechanism.MechanismGroupConfig(
            scope="full",
            group_id=row["group_id"],
            checkpoint_uri=row["checkpoint_uri"],
            checkpoint_step=int(row["checkpoint_step"]),
            expected_restored_state_step=int(row["expected_restored_state_step"]),
            row=row,
            pod_config=configs[row["trajectory_id"]],
            output_path="",
            parent_cache_provenance_sha256=cache_sha256,
            release_sha256=str(release["release_sha256"]),
        )
        artifact_name = f"{freeze.RESULT_ROOT.removeprefix(prefix)}/full/{row['group_id']}"
        steps.append(
            ArtifactStep(
                name=artifact_name,
                version=ARTIFACT_VERSION,
                artifact_type=Artifact,
                run=remote(run_probe_group, resources=resources, name=row["group_id"]),
                build_config=lambda ctx, config=config: replace(config, output_path=ctx.output_path),
            )
        )
    return steps


def readiness(release: Mapping[str, Any], stage: int | None = None) -> dict[str, Any]:
    _configure_runtime(str(release["release_sha256"]))
    configs = mechanism.base.freeze._full_configs()
    parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
    provenance_stage = None if stage == 0 else stage
    return {
        "checkpoint_readiness": mechanism._checkpoint_readiness("full", release, provenance_stage),
        "parent_result_readiness": mechanism._parent_result_readiness("full", release, provenance_stage),
        "parent_provenance": mechanism.base._audit_frozen_provenance("full", parent_release, configs),
        "row_count": len(_read_manifest(release, stage)),
        "endpoint_metrics_read": False,
    }


def _validate_document(
    document: Mapping[str, Any], row: Mapping[str, Any], release: Mapping[str, Any]
) -> dict[str, Any]:
    required = {
        "kind": "gradient_mechanism_repair",
        "scope": "full",
        "group_id": row["group_id"],
        "row": row,
        "release_sha256": release["release_sha256"],
        "scientific_status": freeze.SCIENTIFIC_STATUS,
        "endpoint_metrics_read": False,
        "identity_sha256": mechanism._row_identity(row, str(release["release_sha256"])),
    }
    if any(document.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"Fixed-N TPP output identity drifted: {row['row_id']}")
    if mechanism._contains_nonfinite(document):
        raise RuntimeError(f"Fixed-N TPP output contains non-finite values: {row['row_id']}")
    if int(document["restored_state_step"]) != int(row["expected_restored_state_step"]):
        raise RuntimeError(f"Fixed-N TPP restored the wrong state: {row['row_id']}")
    expected_targets = set(json.loads(row["target_distribution_ids_json"]))
    expected_sources = set(json.loads(row["source_distribution_ids_json"]))
    for field in (
        "target_source_gradient_statistics",
        "target_source_utility_statistics",
        "target_source_choice_contrasts",
    ):
        if set(document[field]) != expected_targets:
            raise RuntimeError(f"Fixed-N TPP {field} target inventory drifted: {row['row_id']}")
    for field in ("target_source_gradient_statistics", "target_source_utility_statistics"):
        if any(set(document[field][target]) != expected_sources for target in expected_targets):
            raise RuntimeError(f"Fixed-N TPP {field} source inventory drifted: {row['row_id']}")
    execution = document["execution_observation"]
    runtime = execution["historical_runtime"]
    runtime_checks = {
        "device_count": int(execution["device_count"]) == freeze.EXPECTED_DEVICE_COUNT,
        "local_device_count": int(execution["local_device_count"]) == freeze.EXPECTED_DEVICE_COUNT,
        "device_kind": set(map(str, execution["device_kinds"])) == {"TPU v5"},
        "release": runtime["release_sha256"] == release["release_sha256"],
        "python_version": runtime["python_version"] == freeze.EXPECTED_PYTHON_VERSION,
        "python_implementation": runtime["python_implementation"] == "CPython",
        "required_packages": (
            runtime["required_package_versions"] == release["historical_runtime"]["required_package_versions"]
        ),
        "installed_inventory": (
            runtime["installed_distribution_versions_sha256"]
            == freeze.canonical_sha256(runtime["installed_distribution_versions"])
        ),
        "historical_sources": (
            runtime["historical_library_source_manifest_sha256"]
            == release["historical_runtime"]["source_manifest"]["sha256"]
        ),
        "recovery_implementation": (
            runtime["recovery_implementation_manifest_sha256"]
            == release["historical_runtime"]["recovery_implementation_manifest_sha256"]
        ),
    }
    failures = sorted(name for name, passed in runtime_checks.items() if not passed)
    if failures:
        raise RuntimeError(f"Fixed-N TPP historical runtime gate failed for {row['row_id']}: {failures}")
    source_pair = document["source_pair_statistics"]["starcoder__vs__nemotron"]
    for statistic_name in ("gradient", "optimizer_update"):
        mechanism._assert_defined_statistic(
            source_pair[statistic_name],
            label=f"{row['row_id']}/{statistic_name}",
            checkpoint_label=str(row["checkpoint_label"]),
        )
    for target, sources in document["target_source_gradient_statistics"].items():
        for source, statistic in sources.items():
            mechanism._assert_defined_statistic(
                statistic,
                label=f"{row['row_id']}/{target}/{source}/gradient",
                checkpoint_label=str(row["checkpoint_label"]),
            )
    for target, sources in document["target_source_utility_statistics"].items():
        for source, statistic in sources.items():
            mechanism._assert_defined_statistic(
                statistic,
                label=f"{row['row_id']}/{target}/{source}/utility",
                checkpoint_label=str(row["checkpoint_label"]),
            )
    expected_contrast = f"{freeze.GLOBAL_STARCODER}__minus__{freeze.NEMOTRON}"
    for target, contrasts in document["target_source_choice_contrasts"].items():
        if expected_contrast not in contrasts:
            raise RuntimeError(f"Fixed-N TPP target {target} omits the source-choice contrast")
        mechanism._assert_defined_statistic(
            contrasts[expected_contrast]["statistic"],
            label=f"{row['row_id']}/{target}/{expected_contrast}",
            checkpoint_label=str(row["checkpoint_label"]),
        )
    if document["no_data_update_invariance"]["passed"] is not True:
        raise RuntimeError(f"Fixed-N TPP no-data update invariance failed: {row['row_id']}")
    if freeze._matches_parent_precision(dict(row)):
        mechanism._assert_matches_parent_probe_statistics(document, row, scope="full", release=release)
    return mechanism._validate_execution_document(document, row, release)


def audit(release: Mapping[str, Any], stage: int | None = None) -> dict[str, Any]:
    _configure_runtime(str(release["release_sha256"]))
    rows = _read_manifest(release, stage)
    root = f"{freeze.RESULT_ROOT}/full"
    fs, plain_root = fsspec.core.url_to_fs(root)
    expected_groups = {row["group_id"] for row in rows}
    found_rows = {
        path
        for path in fs.glob(f"{plain_root}/*/{ARTIFACT_VERSION}/rows/*.json")
        if path.split("/")[-4] in expected_groups
    }
    found_markers = {
        path
        for path in fs.glob(f"{plain_root}/*/{ARTIFACT_VERSION}/group_complete.json")
        if path.split("/")[-3] in expected_groups
    }
    expected_rows = {f"{plain_root}/{row['group_id']}/{ARTIFACT_VERSION}/rows/{row['row_id']}.json": row for row in rows}
    expected_markers = {f"{plain_root}/{row['group_id']}/{ARTIFACT_VERSION}/group_complete.json": row for row in rows}
    missing_rows = sorted(set(expected_rows) - found_rows)
    missing_markers = sorted(set(expected_markers) - found_markers)
    unexpected_rows = sorted(found_rows - set(expected_rows))
    unexpected_markers = sorted(found_markers - set(expected_markers))
    if missing_rows or missing_markers or unexpected_rows or unexpected_markers:
        raise RuntimeError(
            "Fixed-N TPP audit inventory mismatch: "
            f"missing_rows={len(missing_rows)}, missing_markers={len(missing_markers)}, "
            f"unexpected_rows={len(unexpected_rows)}, unexpected_markers={len(unexpected_markers)}"
        )
    runtime_inventories: set[str] = set()
    first_batch_hashes: dict[tuple[int, str], set[str]] = {}
    execution_observations = []
    exact_parent_rows = 0
    for row in rows:
        group_root = "/".join(
            (
                freeze.RESULT_ROOT,
                "full",
                row["group_id"],
                ARTIFACT_VERSION,
            )
        )
        document = mechanism._read_document(f"{group_root}/rows/{row['row_id']}.json")
        marker = mechanism._read_document(f"{group_root}/group_complete.json")
        assert document is not None and marker is not None
        execution_observations.append(_validate_document(document, row, release))
        exact_parent_rows += freeze._matches_parent_precision(dict(row))
        config = mechanism.MechanismGroupConfig(
            scope="full",
            group_id=row["group_id"],
            checkpoint_uri=row["checkpoint_uri"],
            checkpoint_step=int(row["checkpoint_step"]),
            expected_restored_state_step=int(row["expected_restored_state_step"]),
            row=dict(row),
            pod_config=None,
            output_path=group_root,
            parent_cache_provenance_sha256=document["parent_cache_provenance_sha256"],
            release_sha256=str(release["release_sha256"]),
        )
        marker_required = (
            marker.get("identity_sha256") == mechanism._group_identity(config)
            and marker.get("kind") == "gradient_mechanism_repair_group"
            and marker.get("scope") == "full"
            and marker.get("group_id") == row["group_id"]
            and marker.get("row_count") == 1
            and marker.get("release_sha256") == release["release_sha256"]
            and marker.get("scientific_status") == freeze.SCIENTIFIC_STATUS
            and marker.get("endpoint_metrics_read") is False
            and marker.get("row_document_sha256") == document["payload_sha256"]
            and marker.get("execution_observation") == document["execution_observation"]
        )
        if not marker_required:
            raise RuntimeError(f"Fixed-N TPP completion marker drifted: {row['group_id']}")
        runtime = document["execution_observation"]["historical_runtime"]
        runtime_inventories.add(freeze.canonical_json(runtime))
        for source in json.loads(row["source_distribution_ids_json"]):
            key = (int(row["training_seed"]), source)
            first_batch_hashes.setdefault(key, set()).add(
                document["numerical_summaries"][f"probe:{source}"]["first_batch_sha256"]
            )
    mismatched_batches = {key: values for key, values in first_batch_hashes.items() if len(values) != 1}
    if mismatched_batches:
        raise RuntimeError(f"Fixed-N TPP frozen reference batches drifted: {mismatched_batches}")
    if len(runtime_inventories) != 1:
        raise RuntimeError("Fixed-N TPP worker runtime is not uniform")
    expected_exact_rows = int(
        release["design_validation"]["parent_precision_exact_by_stage"]["all" if stage is None else str(stage)]
    )
    if exact_parent_rows != expected_exact_rows:
        raise RuntimeError(
            f"Fixed-N TPP parent reproduction coverage drifted: {exact_parent_rows} != {expected_exact_rows}"
        )
    runtime_environment_sha256 = freeze.canonical_sha256(next(iter(runtime_inventories)))
    if stage != 0 and freeze.PREFLIGHT_AUDIT_PATH.exists():
        preflight = json.loads(freeze.PREFLIGHT_AUDIT_PATH.read_text())
        if runtime_environment_sha256 != preflight["runtime_environment_sha256"]:
            raise RuntimeError("Fixed-N TPP runtime environment drifted from the cross-cell preflight")
    workload_shapes = {item["workload_shape_sha256"] for item in execution_observations}
    report = {
        "passed": True,
        "stage": stage,
        "row_count": len(rows),
        "missing_rows": 0,
        "unexpected_rows": 0,
        "missing_group_markers": 0,
        "unexpected_group_markers": 0,
        "runtime_inventory_count": len(runtime_inventories),
        "runtime_environment_sha256": runtime_environment_sha256,
        "workload_shape_count": len(workload_shapes),
        "exact_parent_reproduction_rows": exact_parent_rows,
        "paired_first_batch_identity_count": len(first_batch_hashes),
        "max_group_wall_seconds": max(item["wall_seconds"] for item in execution_observations),
        "max_peak_host_rss_bytes": max(item["peak_host_rss_bytes"] for item in execution_observations),
        "endpoint_metrics_read": False,
    }
    report["audit_sha256"] = freeze.canonical_sha256({**report, "audit_sha256": ""})
    if stage == 0:
        freeze._write_create_only(
            freeze.PREFLIGHT_AUDIT_PATH,
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
        )
    elif stage is None:
        freeze._write_create_only(
            freeze.FINAL_AUDIT_PATH,
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
        )
    return report


def runtime_adapter_preflight(release: Mapping[str, Any]) -> dict[str, Any]:
    _verify_workspace(release, "required_preauthorization_workspace_paths")
    _apply_runtime_adapter()
    return historical_runtime.runtime_adapter_preflight(str(release["release_sha256"]))


def remote_adapter_canary(release: Mapping[str, Any]) -> dict[str, Any]:
    _verify_workspace(release, "required_preauthorization_workspace_paths")
    _apply_runtime_adapter()
    existing = mechanism._read_document(freeze.REMOTE_ADAPTER_CANARY_PATH)
    if existing is not None:
        document = historical_runtime._assert_remote_adapter_canary(release)
        disposition = "skipped_existing"
    else:
        config = historical_runtime._runtime_adapter_canary_config(release)
        resources = replace(mechanism._resources(), image=freeze.TASK_IMAGE)
        remote(
            run_remote_adapter_canary,
            resources=resources,
            name="fixed-n-tpp-gradient-runtime-adapter-canary",
        )(config)
        document = historical_runtime._assert_remote_adapter_canary(release)
        disposition = "created"
    return {
        "passed": True,
        "disposition": disposition,
        "release_sha256": release["release_sha256"],
        "path": freeze.REMOTE_ADAPTER_CANARY_PATH,
        "payload_sha256": document["payload_sha256"],
    }


def _authorization_payload(release: Mapping[str, Any], confirmation: str) -> dict[str, Any]:
    _apply_runtime_adapter()
    canary = historical_runtime._assert_remote_adapter_canary(release)
    return {
        "full_launch_authorized": True,
        "release_sha256": release["release_sha256"],
        "confirmation": confirmation,
        "cc_review_sha256": release["external_review"]["sha256"],
        "remote_adapter_canary_path": freeze.REMOTE_ADAPTER_CANARY_PATH,
        "remote_adapter_canary_payload_sha256": canary["payload_sha256"],
        "endpoint_metrics_read": False,
    }


def authorize(release: Mapping[str, Any], confirmation: str) -> dict[str, Any]:
    if confirmation != freeze.FULL_LAUNCH_CONFIRMATION:
        raise ValueError("Fixed-N TPP authorization requires the exact confirmation phrase")
    _verify_workspace(release, "required_preauthorization_workspace_paths")
    payload = _authorization_payload(release, confirmation)
    freeze._write_create_only(
        freeze.FULL_LAUNCH_AUTHORIZATION_PATH,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    return payload


def _assert_authorized(release: Mapping[str, Any], confirmation: str) -> None:
    if confirmation != freeze.FULL_LAUNCH_CONFIRMATION or not freeze.FULL_LAUNCH_AUTHORIZATION_PATH.exists():
        raise ValueError("Fixed-N TPP launch is not authorized")
    expected = _authorization_payload(release, confirmation)
    if json.loads(freeze.FULL_LAUNCH_AUTHORIZATION_PATH.read_text()) != expected:
        raise ValueError("Fixed-N TPP authorization sidecar drifted")


def _assert_preflight_passed(release: Mapping[str, Any]) -> None:
    if not freeze.PREFLIGHT_AUDIT_PATH.exists():
        raise ValueError("Fixed-N TPP full-cell launch is blocked pending the cross-cell preflight audit")
    observed = json.loads(freeze.PREFLIGHT_AUDIT_PATH.read_text())
    expected = audit(release, stage=0)
    if observed != expected:
        raise ValueError("Fixed-N TPP preflight audit sidecar drifted")


def launch(release: Mapping[str, Any], *, stage: int, max_concurrent: int, confirmation: str) -> None:
    stage_contract = release["full_launch_stages"].get(str(stage))
    if stage_contract is None:
        raise ValueError(f"Unknown fixed-N TPP stage: {stage}")
    _assert_authorized(release, confirmation)
    contract_key = "required_preflight_workspace_paths" if stage == 0 else "required_full_launch_workspace_paths"
    _verify_workspace(release, contract_key)
    if stage > 0:
        _assert_preflight_passed(release)
    limit = int(stage_contract["max_concurrent"])
    if max_concurrent <= 0 or max_concurrent > limit:
        raise ValueError(f"Fixed-N TPP stage {stage} concurrency must be in [1, {limit}]")
    state = readiness(release, stage)
    if state["checkpoint_readiness"]["missing"] or state["parent_result_readiness"]["missing"]:
        raise RuntimeError(f"Fixed-N TPP readiness failed: {state}")
    run(*_steps(release, stage), max_concurrent=max_concurrent, force_run_failed=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument(
        "--mode",
        choices=("runtime-adapter-preflight", "remote-adapter-canary", "readiness", "audit", "authorize", "launch"),
        default="readiness",
    )
    parser.add_argument("--stage", type=int, choices=(0, 1, 2, 3, 4))
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--confirm-launch")
    args = parser.parse_args()
    release = _load_release(args.release_sha256)
    if args.mode == "runtime-adapter-preflight":
        print(json.dumps(runtime_adapter_preflight(release), indent=2, sort_keys=True))
        return
    if args.mode == "remote-adapter-canary":
        print(json.dumps(remote_adapter_canary(release), indent=2, sort_keys=True))
        return
    if args.mode == "readiness":
        print(json.dumps(readiness(release, args.stage), indent=2, sort_keys=True))
        return
    if args.mode == "audit":
        print(json.dumps(audit(release, args.stage), indent=2, sort_keys=True))
        return
    if args.mode == "authorize":
        print(json.dumps(authorize(release, str(args.confirm_launch)), indent=2, sort_keys=True))
        return
    if args.stage is None or args.max_concurrent is None:
        raise ValueError("Fixed-N TPP launch requires --stage and --max-concurrent")
    launch(
        release,
        stage=args.stage,
        max_concurrent=args.max_concurrent,
        confirmation=str(args.confirm_launch),
    )


if __name__ == "__main__":
    main()
