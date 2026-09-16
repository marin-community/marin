# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run endpoint-blind saved-checkpoint probes that complete the gradient plots."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import cloudpickle
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, run
from marin.execution.remote import remote

from experiments.domain_phase_mix import starcoder_wsd80_gradient_mechanism_repair as mechanism
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_plot_completion_20260822 as freeze,
)

SCHEMA_VERSION = "2026-08-22-gradient-plot-completion-v8"
ARTIFACT_VERSION = "2026.08.22.8"
FULL_LAUNCH_CONFIRMATION = "I_AUTHORIZE_THE_SAVED_CHECKPOINT_GRADIENT_PLOT_COMPLETION"

_PARENT_FREEZE = mechanism.freeze
_PARENT_SCHEMA_VERSION = mechanism.SCHEMA_VERSION
_PARENT_ARTIFACT_VERSION = mechanism.ARTIFACT_VERSION
_PARENT_CONFIG_IDENTITY = mechanism.base.freeze._config_identity
_BASE_EXECUTION_OBSERVATION = mechanism._execution_observation


def _historical_config_identity(pod_config: Any) -> dict[str, Any]:
    """Require the exact pre-write-throughput v6/v10 training configuration."""
    if hasattr(pod_config.train_config.trainer.checkpointer, "write"):
        raise ValueError("Historical v10 runtime unexpectedly contains trainer.checkpointer.write")
    return _PARENT_CONFIG_IDENTITY(pod_config)


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _normalized_distribution_name(raw_name)
        version = distribution.version
        previous = versions.setdefault(name, version)
        if previous != version:
            raise RuntimeError(f"Installed distribution {name} has conflicting versions: {previous}, {version}")
    return dict(sorted(versions.items()))


def _recovery_implementation_manifest_sha256(release: Mapping[str, Any]) -> str:
    return freeze.canonical_sha256(
        {
            "implementation_files": release["implementation_files"],
            "parent_implementation_files": release["parent_implementation_files"],
        }
    )


def _verify_file_map(files: Mapping[str, str], *, label: str) -> None:
    failures = []
    for raw_path, expected_sha256 in files.items():
        path = freeze.REPO_ROOT / raw_path
        if not path.is_file():
            failures.append(f"missing:{raw_path}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            failures.append(f"sha256:{raw_path}")
    if failures:
        raise RuntimeError(f"{label} drifted: {failures[:8]}")


def _verify_historical_runtime_files(expected_release_sha256: str) -> None:
    """Fail closed unless both halves of the hybrid execution tree are pinned."""
    release = json.loads(freeze.RELEASE_PATH.read_text())
    if release["release_sha256"] != expected_release_sha256:
        raise RuntimeError("Worker plot-completion release identity drifted")
    if freeze.canonical_sha256({**release, "release_sha256": ""}) != expected_release_sha256:
        raise RuntimeError("Worker plot-completion release document is internally inconsistent")
    summary = release["historical_runtime"]["source_manifest"]
    manifest_path = freeze.REPO_ROOT / str(summary["path"])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != summary["sha256"]:
        raise RuntimeError("Historical runtime source manifest drifted")
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(summary["row_count"]):
        raise RuntimeError("Historical runtime source manifest row count drifted")
    failures = []
    for row in rows:
        path = freeze.REPO_ROOT / row["path"]
        if not path.is_file():
            failures.append(f"missing:{row['path']}")
            continue
        if path.stat().st_size != int(row["size"]):
            failures.append(f"size:{row['path']}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            failures.append(f"sha256:{row['path']}")
    if failures:
        raise RuntimeError(f"Historical v10 numerical runtime drifted: {failures[:8]}")
    _verify_file_map(release["implementation_files"], label="Recovery implementation")
    _verify_file_map(release["parent_implementation_files"], label="Parent probe implementation")
    observed_implementation_sha256 = _recovery_implementation_manifest_sha256(release)
    expected_implementation_sha256 = release["historical_runtime"]["recovery_implementation_manifest_sha256"]
    if observed_implementation_sha256 != expected_implementation_sha256:
        raise RuntimeError("Recovery implementation manifest identity drifted")


def _verify_worker_runtime_packages() -> None:
    """Verify packages that exist only inside the TPU task environment."""
    if platform.python_version() != freeze.EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"Historical Python version drifted: {platform.python_version()} != {freeze.EXPECTED_PYTHON_VERSION}"
        )
    release = json.loads(freeze.RELEASE_PATH.read_text())
    installed = _installed_distribution_versions()
    required = release["historical_runtime"]["required_package_versions"]
    observed_required = {name: installed.get(name) for name in required}
    if observed_required != required:
        raise RuntimeError(f"Historical required package versions drifted: {observed_required} != {required}")


def _runtime_execution_observation(started_at: float, row: Mapping[str, Any]) -> dict[str, Any]:
    observation = _BASE_EXECUTION_OBSERVATION(started_at, row)
    release = json.loads(freeze.RELEASE_PATH.read_text())
    installed = _installed_distribution_versions()
    observation["historical_runtime"] = {
        "release_sha256": release["release_sha256"],
        "historical_library_source_manifest_sha256": release["historical_runtime"]["source_manifest"]["sha256"],
        "recovery_implementation_manifest_sha256": release["historical_runtime"][
            "recovery_implementation_manifest_sha256"
        ],
        "requested_task_image": freeze.TASK_IMAGE,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "required_package_versions": {
            name: installed[name] for name in sorted(release["historical_runtime"]["required_package_versions"])
        },
        "installed_distribution_versions": installed,
        "installed_distribution_versions_sha256": freeze.canonical_sha256(installed),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "jax_default_matmul_precision": str(mechanism.jax.config.jax_default_matmul_precision),
    }
    return observation


def _apply_mechanism_runtime_adapter() -> None:
    """Rebind the immutable v10 kernel after every process boundary."""
    # The v10 module is hash-pinned and cannot be parameterized without invalidating its release.
    # Cloudpickle can preserve scalar module state while re-importing module objects in a worker,
    # so a process-local "already configured" sentinel is not a valid substitute for rebinding.
    setattr(mechanism, "freeze", freeze)  # noqa: B010
    setattr(mechanism, "SCHEMA_VERSION", SCHEMA_VERSION)  # noqa: B010
    setattr(mechanism, "ARTIFACT_VERSION", ARTIFACT_VERSION)  # noqa: B010
    setattr(mechanism, "_execution_observation", _runtime_execution_observation)  # noqa: B010
    setattr(mechanism.base.freeze, "_config_identity", _historical_config_identity)  # noqa: B010

    checks = {
        "freeze": mechanism.freeze is freeze,
        "schema_version": mechanism.SCHEMA_VERSION == SCHEMA_VERSION,
        "artifact_version": mechanism.ARTIFACT_VERSION == ARTIFACT_VERSION,
        "execution_observation": mechanism._execution_observation is _runtime_execution_observation,
        "config_identity": mechanism.base.freeze._config_identity is _historical_config_identity,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(f"Plot-completion worker adapter did not bind: {failures}")


def _configure_mechanism_runtime(expected_release_sha256: str, *, verify_worker_runtime: bool = False) -> None:
    """Verify the frozen runtime and point the v10 kernel at this release."""
    _verify_historical_runtime_files(expected_release_sha256)
    if verify_worker_runtime:
        _verify_worker_runtime_packages()
    _apply_mechanism_runtime_adapter()


def _runtime_adapter_snapshot(expected_release_sha256: str) -> dict[str, Any]:
    _configure_mechanism_runtime(expected_release_sha256)
    return {
        "freeze_module": mechanism.freeze.__name__,
        "schema_version": mechanism.SCHEMA_VERSION,
        "artifact_version": mechanism.ARTIFACT_VERSION,
        "execution_observation_bound": mechanism._execution_observation is _runtime_execution_observation,
        "config_identity_bound": mechanism.base.freeze._config_identity is _historical_config_identity,
    }


def runtime_adapter_preflight(expected_release_sha256: str) -> dict[str, Any]:
    """Serialize and repair the stale worker-module state that invalidated v7."""
    _configure_mechanism_runtime(expected_release_sha256)
    setattr(mechanism, "freeze", _PARENT_FREEZE)  # noqa: B010
    setattr(mechanism, "SCHEMA_VERSION", _PARENT_SCHEMA_VERSION)  # noqa: B010
    setattr(mechanism, "ARTIFACT_VERSION", _PARENT_ARTIFACT_VERSION)  # noqa: B010
    setattr(mechanism, "_execution_observation", _BASE_EXECUTION_OBSERVATION)  # noqa: B010
    setattr(mechanism.base.freeze, "_config_identity", _PARENT_CONFIG_IDENTITY)  # noqa: B010

    round_tripped_probe = cloudpickle.loads(cloudpickle.dumps(_runtime_adapter_snapshot))
    snapshot = round_tripped_probe(expected_release_sha256)
    expected = {
        "freeze_module": freeze.__name__,
        "schema_version": SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "execution_observation_bound": True,
        "config_identity_bound": True,
    }
    if snapshot != expected:
        raise RuntimeError(f"Serialized runtime adapter preflight failed: {snapshot} != {expected}")
    # Reapply in the caller's globals and verify idempotency after deserialization.
    _configure_mechanism_runtime(expected_release_sha256)
    return {
        "passed": True,
        "release_sha256": expected_release_sha256,
        "cloudpickle_round_trip": snapshot,
    }


def run_remote_adapter_canary(config: mechanism.MechanismGroupConfig) -> None:
    """Exercise worker-only runtime gates without restoring a checkpoint."""
    started_at = time.monotonic()
    _configure_mechanism_runtime(config.release_sha256, verify_worker_runtime=True)
    mechanism._verify_group_contract(config)
    observed_runtime = _runtime_execution_observation(started_at, config.row)
    stable_runtime = {
        "device_count": observed_runtime["device_count"],
        "local_device_count": observed_runtime["local_device_count"],
        "device_kinds": observed_runtime["device_kinds"],
        "historical_runtime": observed_runtime["historical_runtime"],
    }
    mechanism._write_create_only(
        freeze.REMOTE_ADAPTER_CANARY_PATH,
        {
            "kind": "gradient_plot_completion_runtime_adapter_canary",
            "release_sha256": config.release_sha256,
            "group_id": config.group_id,
            "execution_observation": stable_runtime,
            "checkpoint_restored": False,
            "endpoint_metrics_read": False,
        },
        identity_sha256=_remote_adapter_canary_identity(config),
    )


def run_completion_group(config: mechanism.MechanismGroupConfig) -> None:
    _configure_mechanism_runtime(config.release_sha256, verify_worker_runtime=True)
    mechanism.run_mechanism_group(config)


def _file_sha256(path: str) -> str:
    return hashlib.sha256((freeze.REPO_ROOT / path).read_bytes()).hexdigest()


def _verify_required_workspace_paths(paths: list[str], *, label: str) -> None:
    if paths != sorted(set(paths)):
        raise ValueError(f"Plot-completion required {label} workspace path inventory is not canonical")
    missing_paths = [path for path in paths if not (freeze.REPO_ROOT / path).is_file()]
    if missing_paths:
        raise ValueError(f"Plot-completion {label} workspace is incomplete: {missing_paths[:8]}")


def _load_release(expected_sha256: str, *, verify_materialization_inputs: bool = True) -> dict[str, Any]:
    _configure_mechanism_runtime(expected_sha256)
    release = mechanism._load_release(expected_sha256)
    coverage = release["coverage_audit"]
    for key in ("path", "report_path"):
        sha_key = "sha256" if key == "path" else "report_sha256"
        if _file_sha256(str(coverage[key])) != coverage[sha_key]:
            raise ValueError(f"Plot-completion coverage artifact drifted: {coverage[key]}")
    v10 = release["v10_release"]
    if _file_sha256(str(v10["path"])) != v10["file_sha256"]:
        raise ValueError("Plot-completion v10 release file drifted")
    v10_release = json.loads((freeze.REPO_ROOT / str(v10["path"])).read_text())
    if v10_release["release_sha256"] != v10["release_sha256"]:
        raise ValueError("Plot-completion v10 release identity drifted")
    for label, key in (
        ("v1", "superseded_prelaunch_draft"),
        ("v2", "superseded_oversize_workspace_release"),
    ):
        superseded_early = release[key]
        if _file_sha256(str(superseded_early["path"])) != superseded_early["file_sha256"]:
            raise ValueError(f"Superseded {label} release drifted")
        if _file_sha256(str(superseded_early["failure_marker_path"])) != superseded_early["failure_marker_sha256"]:
            raise ValueError(f"Superseded {label} failure marker drifted")
    superseded = release["superseded_runtime_canary"]
    if _file_sha256(str(superseded["path"])) != superseded["file_sha256"]:
        raise ValueError("Superseded v3 runtime-canary release drifted")
    if _file_sha256(str(superseded["failure_marker_path"])) != superseded["failure_marker_sha256"]:
        raise ValueError("Superseded v3 runtime-canary failure marker drifted")
    superseded_prelaunch = release["superseded_prelaunch_release"]
    if _file_sha256(str(superseded_prelaunch["path"])) != superseded_prelaunch["file_sha256"]:
        raise ValueError("Superseded v4 prelaunch release drifted")
    if _file_sha256(str(superseded_prelaunch["failure_marker_path"])) != superseded_prelaunch["failure_marker_sha256"]:
        raise ValueError("Superseded v4 prelaunch failure marker drifted")
    superseded_bundle = release["superseded_runtime_bundle_release"]
    if _file_sha256(str(superseded_bundle["path"])) != superseded_bundle["file_sha256"]:
        raise ValueError("Superseded v5 runtime-bundle release drifted")
    if _file_sha256(str(superseded_bundle["failure_marker_path"])) != superseded_bundle["failure_marker_sha256"]:
        raise ValueError("Superseded v5 runtime-bundle failure marker drifted")
    superseded_frozen_bundle = release["superseded_frozen_lock_bundle_release"]
    if _file_sha256(str(superseded_frozen_bundle["path"])) != superseded_frozen_bundle["file_sha256"]:
        raise ValueError("Superseded v6 frozen-lock bundle release drifted")
    if (
        _file_sha256(str(superseded_frozen_bundle["failure_marker_path"]))
        != superseded_frozen_bundle["failure_marker_sha256"]
    ):
        raise ValueError("Superseded v6 frozen-lock bundle failure marker drifted")
    superseded_worker_adapter = release["superseded_worker_adapter_release"]
    if _file_sha256(str(superseded_worker_adapter["path"])) != superseded_worker_adapter["file_sha256"]:
        raise ValueError("Superseded v7 worker-adapter release drifted")
    if (
        _file_sha256(str(superseded_worker_adapter["failure_marker_path"]))
        != superseded_worker_adapter["failure_marker_sha256"]
    ):
        raise ValueError("Superseded v7 worker-adapter failure marker drifted")
    for summary in release["execution_reference_inputs"].values():
        if _file_sha256(str(summary["path"])) != summary["sha256"]:
            raise ValueError(f"Plot-completion execution reference drifted: {summary['path']}")
    if verify_materialization_inputs:
        _verify_materialization_inputs(release)
    review = release["external_review"]
    if _file_sha256(str(review["path"])) != review["sha256"]:
        raise ValueError("Plot-completion CC review artifact drifted")
    if review["verdict"] != "PASS_AFTER_BLOCKERS_RESOLVED":
        raise ValueError(f"Plot-completion CC review did not pass: {review['verdict']}")
    submission_contract = release["submission_contract"]
    if submission_contract["required_environment"] != {"UV_FROZEN": "1"}:
        raise ValueError("Plot-completion frozen submission environment drifted")
    _verify_required_workspace_paths(
        submission_contract["required_preauthorization_workspace_paths"],
        label="preauthorization",
    )
    return release


def _verify_materialization_inputs(release: Mapping[str, Any]) -> None:
    for summary in release["plot_inputs"].values():
        if _file_sha256(str(summary["path"])) != summary["sha256"]:
            raise ValueError(f"Plot-completion materialization input drifted: {summary['path']}")


def _read_manifest(release: Mapping[str, Any], stage: int | None = None) -> list[dict[str, str]]:
    summary = release["manifests"]["full"]
    path = freeze.REPO_ROOT / summary["path"]
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != summary["row_count"]:
        raise ValueError("Plot-completion manifest row count drifted")
    if stage is not None:
        rows = [row for row in rows if int(row["launch_stage"]) == stage]
    return rows


def _steps(release: Mapping[str, Any], stage: int) -> list[ArtifactStep[Artifact]]:
    parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
    cache_sha = parent_release["manifests"]["cache_provenance"]["sha256"]
    configs = mechanism.base.freeze._full_configs()
    resources = replace(mechanism._resources(), image=freeze.TASK_IMAGE)
    prefix = f"{freeze.MARIN_PREFIX}/"
    steps: list[ArtifactStep[Artifact]] = []
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
            parent_cache_provenance_sha256=cache_sha,
            release_sha256=release["release_sha256"],
        )
        artifact_name = f"{freeze.RESULT_ROOT.removeprefix(prefix)}/full/{row['group_id']}"
        steps.append(
            ArtifactStep(
                name=artifact_name,
                version=ARTIFACT_VERSION,
                artifact_type=Artifact,
                run=remote(run_completion_group, resources=resources, name=row["group_id"]),
                build_config=lambda ctx, config=config: replace(config, output_path=ctx.output_path),
            )
        )
    return steps


def _runtime_adapter_canary_config(release: Mapping[str, Any]) -> mechanism.MechanismGroupConfig:
    row = _read_manifest(release, stage=1)[0]
    configs = mechanism.base.freeze._full_configs()
    parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
    return mechanism.MechanismGroupConfig(
        scope="full",
        group_id=row["group_id"],
        checkpoint_uri=row["checkpoint_uri"],
        checkpoint_step=int(row["checkpoint_step"]),
        expected_restored_state_step=int(row["expected_restored_state_step"]),
        row=row,
        pod_config=configs[row["trajectory_id"]],
        output_path=f"{freeze.RESULT_ROOT}/full/runtime_adapter_canary",
        parent_cache_provenance_sha256=parent_release["manifests"]["cache_provenance"]["sha256"],
        release_sha256=str(release["release_sha256"]),
    )


def _remote_adapter_canary_identity(config: mechanism.MechanismGroupConfig) -> str:
    return freeze.canonical_sha256(
        {
            "kind": "gradient_plot_completion_runtime_adapter_canary",
            "release_sha256": config.release_sha256,
            "group_id": config.group_id,
        }
    )


def _assert_remote_adapter_canary(release: Mapping[str, Any]) -> dict[str, Any]:
    config = _runtime_adapter_canary_config(release)
    document = mechanism._read_document(freeze.REMOTE_ADAPTER_CANARY_PATH)
    if document is None:
        raise RuntimeError("Remote runtime-adapter canary has not passed")
    execution = document.get("execution_observation", {})
    runtime = execution.get("historical_runtime", {})
    checks = {
        "identity": document.get("identity_sha256") == _remote_adapter_canary_identity(config),
        "release": document.get("release_sha256") == release["release_sha256"],
        "group": document.get("group_id") == config.group_id,
        "no_checkpoint_restore": document.get("checkpoint_restored") is False,
        "endpoint_blind": document.get("endpoint_metrics_read") is False,
        "device_count": execution.get("device_count") == freeze.EXPECTED_DEVICE_COUNT,
        "local_device_count": execution.get("local_device_count") == freeze.EXPECTED_DEVICE_COUNT,
        "device_kind": set(map(str, execution.get("device_kinds", ()))) == {"TPU v5"},
        "worker_packages": (
            runtime.get("required_package_versions") == release["historical_runtime"]["required_package_versions"]
        ),
        "python_version": runtime.get("python_version") == freeze.EXPECTED_PYTHON_VERSION,
        "python_implementation": runtime.get("python_implementation") == "CPython",
        "installed_distribution_inventory": (
            runtime.get("installed_distribution_versions_sha256")
            == freeze.canonical_sha256(runtime.get("installed_distribution_versions", {}))
        ),
        "historical_sources": (
            runtime.get("historical_library_source_manifest_sha256")
            == release["historical_runtime"]["source_manifest"]["sha256"]
        ),
        "recovery_implementation": (
            runtime.get("recovery_implementation_manifest_sha256")
            == release["historical_runtime"]["recovery_implementation_manifest_sha256"]
        ),
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(f"Remote runtime-adapter canary failed: {failures}")
    return document


def remote_adapter_canary(release: Mapping[str, Any]) -> dict[str, Any]:
    existing = mechanism._read_document(freeze.REMOTE_ADAPTER_CANARY_PATH)
    if existing is not None:
        document = _assert_remote_adapter_canary(release)
        return {
            "passed": True,
            "disposition": "skipped_existing",
            "release_sha256": release["release_sha256"],
            "path": freeze.REMOTE_ADAPTER_CANARY_PATH,
            "payload_sha256": document["payload_sha256"],
        }
    config = _runtime_adapter_canary_config(release)
    resources = replace(mechanism._resources(), image=freeze.TASK_IMAGE)
    remote(
        run_remote_adapter_canary,
        resources=resources,
        name="gradient-plot-completion-runtime-adapter-canary",
    )(config)
    document = _assert_remote_adapter_canary(release)
    return {
        "passed": True,
        "disposition": "created",
        "release_sha256": release["release_sha256"],
        "path": freeze.REMOTE_ADAPTER_CANARY_PATH,
        "payload_sha256": document["payload_sha256"],
    }


def readiness(release: Mapping[str, Any], stage: int | None = None) -> dict[str, Any]:
    _configure_mechanism_runtime(str(release["release_sha256"]))
    configs = mechanism.base.freeze._full_configs()
    parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
    return {
        "stage": stage,
        "checkpoint_readiness": mechanism._checkpoint_readiness("full", release, stage),
        "parent_result_readiness": mechanism._parent_result_readiness("full", release, stage),
        "parent_provenance": mechanism.base._audit_frozen_provenance("full", parent_release, configs),
    }


def _assert_source_only_parent_statistics(
    document: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    release: Mapping[str, Any],
    provenance: Mapping[str, Mapping[str, str]],
    parent_release: Mapping[str, Any],
) -> int:
    """Reproduce source-gradient geometry when a zero-LR row has no target probes."""
    if mechanism._json_names(row, "target_distribution_ids_json"):
        return 0
    left = freeze.GLOBAL_STARCODER
    right = freeze.NEMOTRON
    probe_row_ids = mechanism._json_mapping(row, "distribution_probe_row_ids_json", str)
    observed_pair = document["source_pair_statistics"]["starcoder__vs__nemotron"]["gradient"]
    comparisons = 0
    for source, observed_norm_key in ((left, "left_norm"), (right, "right_norm")):
        uri = mechanism._path_join(
            parent_release["result_root"],
            "full",
            "probe",
            row["parent_probe_group_id"],
            release["parent_result_artifact_version"],
            "rows",
            f"{probe_row_ids[source]}.json",
        )
        expected_provenance = provenance.get(uri)
        if expected_provenance is None:
            raise RuntimeError(f"Source-only parent probe result is not pinned by the release: {uri}")
        parent_document = mechanism._read_pinned_parent_result(uri, expected_provenance)
        parent_row = parent_document.get("row", {})
        if (
            parent_document.get("release_sha256") != parent_release["release_sha256"]
            or parent_row.get("row_id") != probe_row_ids[source]
            or parent_row.get("distribution_id") != source
        ):
            raise RuntimeError(f"Source-only parent probe output identity mismatch: {uri}")
        parent_target = min(parent_document["pairwise_statistics"])
        expected_pair = parent_document["pairwise_statistics"][parent_target]
        for geometry, parent_key in (("raw", "raw_gradient"), ("projected", "projected_gradient")):
            for component, expected in expected_pair[parent_key].items():
                observed = observed_pair[geometry][component]
                mechanism._assert_close(
                    observed[observed_norm_key],
                    expected["left_norm"],
                    label=f"source_only/{source}/{geometry}/{component}/gradient_norm",
                )
                comparisons += 1
    return comparisons


def _runtime_environment_payload(
    release: Mapping[str, Any],
    runtime_stack: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "release_sha256": release["release_sha256"],
        "historical_library_source_manifest_sha256": runtime_stack["historical_library_source_manifest_sha256"],
        "recovery_implementation_manifest_sha256": runtime_stack["recovery_implementation_manifest_sha256"],
        "requested_task_image": runtime_stack["requested_task_image"],
        "python_version": runtime_stack["python_version"],
        "python_implementation": runtime_stack["python_implementation"],
        "required_package_versions": runtime_stack["required_package_versions"],
        "installed_distribution_versions": runtime_stack["installed_distribution_versions"],
        "installed_distribution_versions_sha256": runtime_stack["installed_distribution_versions_sha256"],
        "xla_flags": runtime_stack["xla_flags"],
        "libtpu_init_args": runtime_stack["libtpu_init_args"],
        "jax_default_matmul_precision": runtime_stack["jax_default_matmul_precision"],
    }


def _verify_or_freeze_stage1_environment(
    release: Mapping[str, Any],
    runtime_stack: Mapping[str, Any],
    *,
    stage: int | None,
) -> dict[str, Any]:
    payload = _runtime_environment_payload(release, runtime_stack)
    path = freeze.RUNTIME_ENVIRONMENT_BASELINE_PATH
    if stage == 1:
        freeze._write_create_only(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        return payload
    if not path.exists():
        raise RuntimeError("Stage-1 runtime environment baseline is missing")
    baseline = json.loads(path.read_text())
    if baseline != payload:
        raise RuntimeError("Runtime environment drifted from the audited Stage-1 baseline")
    return baseline


def audit(release: Mapping[str, Any], stage: int | None = None) -> dict[str, Any]:
    submission_contract = release["submission_contract"]
    required_paths = (
        submission_contract["required_stage2_plus_workspace_paths"]
        if stage is None or stage > 1
        else submission_contract["required_stage1_workspace_paths"]
    )
    _verify_required_workspace_paths(required_paths, label="audit")
    _configure_mechanism_runtime(str(release["release_sha256"]))
    report = mechanism.audit_outputs("full", release, stage=stage)
    rows = _read_manifest(release, stage)
    runtime_observations: list[dict[str, Any]] = []
    parent_result_provenance = mechanism._parent_result_provenance(release)
    parent_release = json.loads(freeze.PARENT_RELEASE_PATH.read_text())
    source_only_parent_comparisons = 0
    for row in rows:
        path = "/".join(
            (
                freeze.RESULT_ROOT,
                "full",
                row["group_id"],
                ARTIFACT_VERSION,
                "rows",
                f"{row['row_id']}.json",
            )
        )
        document = mechanism._read_document(path)
        if document is None:
            raise RuntimeError(f"Plot-completion output disappeared during runtime audit: {path}")
        execution = document.get("execution_observation", {})
        runtime_stack = execution.get("historical_runtime", {})
        expected_stack = release["historical_runtime"]
        device_kinds = execution.get("device_kinds", ())
        checks = {
            "device_count": int(execution.get("device_count", -1)) == freeze.EXPECTED_DEVICE_COUNT,
            "local_device_count": int(execution.get("local_device_count", -1)) == freeze.EXPECTED_DEVICE_COUNT,
            "device_kind": bool(device_kinds) and set(map(str, device_kinds)) == {"TPU v5"},
            "release_sha256": runtime_stack.get("release_sha256") == release["release_sha256"],
            "historical_library_source_manifest": (
                runtime_stack.get("historical_library_source_manifest_sha256")
                == expected_stack["source_manifest"]["sha256"]
            ),
            "recovery_implementation_manifest": (
                runtime_stack.get("recovery_implementation_manifest_sha256")
                == expected_stack["recovery_implementation_manifest_sha256"]
            ),
            "python_version": runtime_stack.get("python_version") == freeze.EXPECTED_PYTHON_VERSION,
            "python_implementation": runtime_stack.get("python_implementation") == "CPython",
            "required_package_versions": (
                runtime_stack.get("required_package_versions") == expected_stack["required_package_versions"]
            ),
            "installed_distribution_inventory_hash": (
                runtime_stack.get("installed_distribution_versions_sha256")
                == freeze.canonical_sha256(runtime_stack.get("installed_distribution_versions", {}))
            ),
        }
        failures = sorted(name for name, passed in checks.items() if not passed)
        if failures:
            raise RuntimeError(f"Historical runtime gate failed for {row['row_id']}: {failures}")
        source_only_parent_comparisons += _assert_source_only_parent_statistics(
            document,
            row,
            release=release,
            provenance=parent_result_provenance,
            parent_release=parent_release,
        )
        runtime_observations.append(runtime_stack)
    expected_source_only_comparisons = {None: 1_408, 1: 88, 2: 0, 3: 0, 4: 1_320}[stage]
    if source_only_parent_comparisons != expected_source_only_comparisons:
        raise RuntimeError(
            "Source-only parent-statistic comparison count drifted: "
            f"{source_only_parent_comparisons} != {expected_source_only_comparisons}"
        )
    environment_payloads = {
        freeze.canonical_json(_runtime_environment_payload(release, item)): item for item in runtime_observations
    }
    if len(environment_payloads) != 1:
        raise RuntimeError("Historical runtime environment is not uniform")
    runtime_stack = next(iter(environment_payloads.values()))
    baseline = _verify_or_freeze_stage1_environment(release, runtime_stack, stage=stage)
    report["historical_runtime_gate"] = {
        "passed": True,
        "historical_library_source_commit": freeze.HISTORICAL_RUNTIME_COMMIT,
        "historical_library_source_manifest_sha256": release["historical_runtime"]["source_manifest"]["sha256"],
        "recovery_implementation_manifest_sha256": release["historical_runtime"][
            "recovery_implementation_manifest_sha256"
        ],
        "requested_task_image": freeze.TASK_IMAGE,
        "task_image_observed": False,
        "python_version": freeze.EXPECTED_PYTHON_VERSION,
        "required_package_versions": freeze.EXPECTED_RUNTIME_VERSIONS,
        "installed_distribution_versions_sha256": baseline["installed_distribution_versions_sha256"],
        "device_count": freeze.EXPECTED_DEVICE_COUNT,
        "local_device_count": freeze.EXPECTED_DEVICE_COUNT,
        "device_kinds": ["TPU v5"],
        "xla_flags": baseline["xla_flags"],
        "libtpu_init_args": baseline["libtpu_init_args"],
        "jax_default_matmul_precision": baseline["jax_default_matmul_precision"],
        "stage1_environment_baseline_path": release["historical_runtime"]["stage1_environment_baseline_path"],
        "source_only_parent_statistic_comparisons": source_only_parent_comparisons,
    }
    report["audit_sha256"] = freeze.canonical_sha256({**report, "audit_sha256": ""})
    return report


def _authorization_payload(release: Mapping[str, Any]) -> dict[str, Any]:
    contract = release["authorization_contract"]
    review = release["external_review"]
    remote_canary = _assert_remote_adapter_canary(release)
    return {
        "full_launch_authorized": True,
        "release_sha256": release["release_sha256"],
        "confirmation": contract["confirmation"],
        "cc_account": review["account"],
        "cc_model": review["model"],
        "cc_review_verdict": review["verdict"],
        "cc_review_sha256": review["sha256"],
        "remote_adapter_canary_path": freeze.REMOTE_ADAPTER_CANARY_PATH,
        "remote_adapter_canary_payload_sha256": remote_canary["payload_sha256"],
        "materialization_inputs_verified_before_authorization": True,
    }


def authorize(release: Mapping[str, Any], *, confirmation: str) -> dict[str, Any]:
    contract = release["authorization_contract"]
    if confirmation != FULL_LAUNCH_CONFIRMATION or confirmation != contract["confirmation"]:
        raise ValueError("Plot-completion authorization requires the exact confirmation phrase")
    _verify_materialization_inputs(release)
    payload = _authorization_payload(release)
    freeze._write_create_only(
        freeze.FULL_LAUNCH_AUTHORIZATION_PATH,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    return payload


def _assert_authorized(release: Mapping[str, Any], confirmation: str) -> None:
    if confirmation != FULL_LAUNCH_CONFIRMATION:
        raise ValueError("Plot-completion launch requires the exact authorization phrase")
    if not freeze.FULL_LAUNCH_AUTHORIZATION_PATH.exists():
        raise ValueError("Plot-completion launch is blocked pending the reviewed authorization sidecar")
    authorization = json.loads(freeze.FULL_LAUNCH_AUTHORIZATION_PATH.read_text())
    if authorization != _authorization_payload(release):
        raise ValueError("Plot-completion authorization sidecar does not match this release")


def launch(release: Mapping[str, Any], *, stage: int, max_concurrent: int, confirmation: str) -> None:
    if str(stage) not in release["full_launch_stages"]:
        raise ValueError(f"Unknown plot-completion launch stage: {stage}")
    _assert_authorized(release, confirmation)
    required_paths = release["submission_contract"][
        "required_stage2_plus_workspace_paths" if stage > 1 else "required_stage1_workspace_paths"
    ]
    _verify_required_workspace_paths(required_paths, label=f"Stage-{stage}")
    limit = int(release["full_launch_stages"][str(stage)]["max_concurrent"])
    if max_concurrent <= 0 or max_concurrent > limit:
        raise ValueError(f"Plot-completion stage {stage} max_concurrent must be in [1, {limit}]")
    for prerequisite in range(1, stage):
        audit(release, prerequisite)
    state = readiness(release, stage)
    if state["checkpoint_readiness"]["missing"] or state["parent_result_readiness"]["missing"]:
        raise RuntimeError(f"Plot-completion readiness failed: {state}")
    run(*_steps(release, stage), max_concurrent=max_concurrent, force_run_failed=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument(
        "--mode",
        choices=("runtime-adapter-preflight", "remote-adapter-canary", "readiness", "audit", "authorize", "launch"),
        default="readiness",
    )
    parser.add_argument("--stage", type=int, choices=tuple(sorted(freeze.STAGE_ROW_COUNTS)))
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--confirm-launch")
    args = parser.parse_args()
    release = _load_release(
        args.release_sha256,
        verify_materialization_inputs=args.mode != "launch",
    )
    if args.mode == "runtime-adapter-preflight":
        print(json.dumps(runtime_adapter_preflight(args.release_sha256), indent=2, sort_keys=True))
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
        print(
            json.dumps(
                authorize(
                    release,
                    confirmation=str(args.confirm_launch),
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.stage is None or args.max_concurrent is None:
        raise ValueError("Plot-completion launch requires explicit --stage and --max-concurrent")
    launch(
        release,
        stage=args.stage,
        max_concurrent=args.max_concurrent,
        confirmation=str(args.confirm_launch),
    )


if __name__ == "__main__":
    main()
