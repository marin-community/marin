# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable run state and manifest types for alternating RL."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from iris.marin_fs import url_to_fs
from levanter.utils.fsspec_utils import exists, join_path

from marin.rl.alternating.config import AlternatingRLConfig

POLICY_DIGITS = 4
PHASE_DIGITS = 4
HOST_DIGITS = 4
BATCH_DIGITS = 6

MANIFEST_BASENAME = "manifest.json"
RUN_STATE_BASENAME = "run_state.json"
STATUS_BASENAME = "status.json"
CURRICULUM_STATE_BASENAME = "curriculum_state.json"
CURRICULUM_SNAPSHOT_BASENAME = "curriculum_snapshot.json"
CONTROLLER_CONFIG_BASENAME = "controller_config.pkl"

STATE_DIRNAME = "state"
CURRICULUM_DIRNAME = "curriculum"
POLICIES_DIRNAME = "policies"
SAMPLING_DIRNAME = "sampling"
MATERIALIZED_DIRNAME = "materialized"
LEVANTER_CHECKPOINTS_DIRNAME = "levanter_checkpoints"
PHASE_METRICS_DIRNAME = "phase_metrics"
ROLLOUTS_DIRNAME = "rollouts"
BATCHES_DIRNAME = "batches"


def utc_now_iso() -> str:
    """Return the current UTC time in a stable JSON-friendly format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _join(*parts: str) -> str:
    path = parts[0]
    for part in parts[1:]:
        path = join_path(path, part)
    return path


class RunStatus(StrEnum):
    """Allowed controller states for alternating RL."""

    SAMPLING = "sampling"
    MATERIALIZING = "materializing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class HostPhaseStatus(StrEnum):
    """Completion status for one sampling host."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class AlternatingRunPaths:
    """Canonical path layout for one alternating RL run."""

    run_root: str

    @classmethod
    def from_config(cls, config: AlternatingRLConfig) -> AlternatingRunPaths:
        """Build canonical run paths from controller config."""
        return cls(run_root=config.run_root)

    @property
    def state_root(self) -> str:
        return join_path(self.run_root, STATE_DIRNAME)

    @property
    def run_state_path(self) -> str:
        return join_path(self.state_root, RUN_STATE_BASENAME)

    @property
    def controller_config_path(self) -> str:
        return join_path(self.state_root, CONTROLLER_CONFIG_BASENAME)

    def phase_metrics_path(self, phase_id: int) -> str:
        return _join(self.state_root, PHASE_METRICS_DIRNAME, f"phase_{phase_id:0{PHASE_DIGITS}d}.json")

    @property
    def curriculum_root(self) -> str:
        return join_path(self.run_root, CURRICULUM_DIRNAME)

    @property
    def curriculum_state_path(self) -> str:
        return join_path(self.curriculum_root, CURRICULUM_STATE_BASENAME)

    @property
    def levanter_checkpoints_root(self) -> str:
        return join_path(self.run_root, LEVANTER_CHECKPOINTS_DIRNAME)

    def policy_dir(self, policy_version: int) -> str:
        return _join(self.run_root, POLICIES_DIRNAME, f"policy_{policy_version:0{POLICY_DIGITS}d}")

    def policy_manifest_path(self, policy_version: int) -> str:
        return join_path(self.policy_dir(policy_version), MANIFEST_BASENAME)

    def sampling_phase_dir(self, phase_id: int) -> str:
        return _join(self.run_root, SAMPLING_DIRNAME, f"phase_{phase_id:0{PHASE_DIGITS}d}")

    def sampling_manifest_path(self, phase_id: int) -> str:
        return join_path(self.sampling_phase_dir(phase_id), MANIFEST_BASENAME)

    def sampling_curriculum_snapshot_path(self, phase_id: int) -> str:
        return join_path(self.sampling_phase_dir(phase_id), CURRICULUM_SNAPSHOT_BASENAME)

    def sampling_host_dir(self, phase_id: int, host_ordinal: int) -> str:
        return join_path(self.sampling_phase_dir(phase_id), f"host_{host_ordinal:0{HOST_DIGITS}d}")

    def sampling_host_rollout_dir(self, phase_id: int, host_ordinal: int) -> str:
        return join_path(self.sampling_host_dir(phase_id, host_ordinal), ROLLOUTS_DIRNAME)

    def sampling_host_status_path(self, phase_id: int, host_ordinal: int) -> str:
        return join_path(self.sampling_host_dir(phase_id, host_ordinal), STATUS_BASENAME)

    def materialized_phase_dir(self, phase_id: int) -> str:
        return _join(self.run_root, MATERIALIZED_DIRNAME, f"phase_{phase_id:0{PHASE_DIGITS}d}")

    def materialized_batch_path(self, phase_id: int, batch_index: int) -> str:
        return _join(
            self.materialized_phase_dir(phase_id),
            BATCHES_DIRNAME,
            f"batch_{batch_index:0{BATCH_DIGITS}d}.pkl",
        )

    def materialized_manifest_path(self, phase_id: int) -> str:
        return join_path(self.materialized_phase_dir(phase_id), MANIFEST_BASENAME)


@dataclass(frozen=True)
class AlternatingRunState:
    """Controller source-of-truth persisted at `state/run_state.json`."""

    run_id: str
    status: RunStatus
    phase_id: int
    policy_version: int
    source_global_step: int
    num_hosts: int
    tpu_name: str
    tpu_type: str
    zone: str
    image_digest: str
    current_policy_manifest_path: str
    current_levanter_checkpoint_path: str | None
    current_sampling_manifest: str | None
    current_materialized_manifest: str | None
    last_completed_phase: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the run state to a JSON-friendly dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlternatingRunState:
        """Parse run state from JSON data."""
        return cls(
            run_id=data["run_id"],
            status=RunStatus(data["status"]),
            phase_id=data["phase_id"],
            policy_version=data["policy_version"],
            source_global_step=data["source_global_step"],
            num_hosts=data["num_hosts"],
            tpu_name=data["tpu_name"],
            tpu_type=data["tpu_type"],
            zone=data["zone"],
            image_digest=data["image_digest"],
            current_policy_manifest_path=data["current_policy_manifest_path"],
            current_levanter_checkpoint_path=data.get("current_levanter_checkpoint_path"),
            current_sampling_manifest=data.get("current_sampling_manifest"),
            current_materialized_manifest=data.get("current_materialized_manifest"),
            last_completed_phase=data["last_completed_phase"],
        )


@dataclass(frozen=True)
class PolicyManifest:
    """One immutable policy export used by later sampling phases."""

    policy_version: int
    phase_id: int
    source_global_step: int
    policy_path: str
    levanter_checkpoint_path: str | None
    model_name: str
    tokenizer_name: str
    enable_fast_bootstrap: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyManifest:
        return cls(
            policy_version=data["policy_version"],
            phase_id=data["phase_id"],
            source_global_step=data["source_global_step"],
            policy_path=data["policy_path"],
            levanter_checkpoint_path=data.get("levanter_checkpoint_path"),
            model_name=data["model_name"],
            tokenizer_name=data["tokenizer_name"],
            enable_fast_bootstrap=data["enable_fast_bootstrap"],
            created_at=data["created_at"],
        )


@dataclass(frozen=True)
class SamplingHostAssignment:
    """Work assignment for one sampling host."""

    host_ordinal: int
    seed: int
    target_train_groups: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingHostAssignment:
        return cls(
            host_ordinal=data["host_ordinal"],
            seed=data["seed"],
            target_train_groups=data["target_train_groups"],
        )


@dataclass(frozen=True)
class SamplingManifest:
    """Per-phase sampling contract written by the controller."""

    phase_id: int
    policy_version: int
    policy_manifest_path: str
    curriculum_state_path: str
    curriculum_snapshot_path: str
    num_hosts: int
    local_tensor_parallel_size: int
    coordinator_host_ordinal: int
    host_assignments: list[SamplingHostAssignment]
    frozen_lesson_weights: dict[str, float]
    rollout_output_root: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["host_assignments"] = [assignment.to_dict() for assignment in self.host_assignments]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingManifest:
        return cls(
            phase_id=data["phase_id"],
            policy_version=data["policy_version"],
            policy_manifest_path=data["policy_manifest_path"],
            curriculum_state_path=data["curriculum_state_path"],
            curriculum_snapshot_path=data["curriculum_snapshot_path"],
            num_hosts=data["num_hosts"],
            local_tensor_parallel_size=data["local_tensor_parallel_size"],
            coordinator_host_ordinal=data["coordinator_host_ordinal"],
            host_assignments=[SamplingHostAssignment.from_dict(item) for item in data["host_assignments"]],
            frozen_lesson_weights={key: float(value) for key, value in data["frozen_lesson_weights"].items()},
            rollout_output_root=data["rollout_output_root"],
        )


@dataclass(frozen=True)
class SamplingHostStatusManifest:
    """Completion marker written by one sampling host."""

    phase_id: int
    policy_version: int
    host_ordinal: int
    status: HostPhaseStatus
    rollout_file_paths: list[str]
    num_train_groups: int
    lesson_rewards: dict[str, list[float]]
    created_at: str
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SamplingHostStatusManifest:
        return cls(
            phase_id=data["phase_id"],
            policy_version=data["policy_version"],
            host_ordinal=data["host_ordinal"],
            status=HostPhaseStatus(data["status"]),
            rollout_file_paths=list(data["rollout_file_paths"]),
            num_train_groups=data["num_train_groups"],
            lesson_rewards={
                lesson_id: [float(reward) for reward in rewards] for lesson_id, rewards in data["lesson_rewards"].items()
            },
            created_at=data["created_at"],
            error_message=data.get("error_message"),
        )


@dataclass(frozen=True)
class MaterializedBatchesManifest:
    """Materialized training-batch manifest for one phase."""

    phase_id: int
    policy_version: int
    input_rollout_paths: list[str]
    num_rollout_groups: int
    num_individual_rollouts: int
    num_training_batches: int
    global_batch_size: int
    max_seq_len: int
    batch_paths: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaterializedBatchesManifest:
        return cls(
            phase_id=data["phase_id"],
            policy_version=data["policy_version"],
            input_rollout_paths=list(data["input_rollout_paths"]),
            num_rollout_groups=data["num_rollout_groups"],
            num_individual_rollouts=data["num_individual_rollouts"],
            num_training_batches=data["num_training_batches"],
            global_batch_size=data["global_batch_size"],
            max_seq_len=data["max_seq_len"],
            batch_paths=list(data["batch_paths"]),
        )


@dataclass(frozen=True)
class PhaseMetricsManifest:
    """Per-phase wall-clock metrics recorded across controller/runtime boundaries."""

    phase_id: int
    prepare_sampling_seconds: float | None = None
    sampling_seconds: float | None = None
    curriculum_update_seconds: float | None = None
    materialization_seconds: float | None = None
    training_seconds: float | None = None
    export_seconds: float | None = None
    last_updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseMetricsManifest:
        return cls(
            phase_id=data["phase_id"],
            prepare_sampling_seconds=data.get("prepare_sampling_seconds"),
            sampling_seconds=data.get("sampling_seconds"),
            curriculum_update_seconds=data.get("curriculum_update_seconds"),
            materialization_seconds=data.get("materialization_seconds"),
            training_seconds=data.get("training_seconds"),
            export_seconds=data.get("export_seconds"),
            last_updated_at=data.get("last_updated_at", ""),
        )

    @property
    def total_recorded_seconds(self) -> float:
        return float(
            sum(
                value
                for value in (
                    self.prepare_sampling_seconds,
                    self.sampling_seconds,
                    self.curriculum_update_seconds,
                    self.materialization_seconds,
                    self.training_seconds,
                    self.export_seconds,
                )
                if value is not None
            )
        )


def _ensure_parent_dir(path: str) -> tuple[Any, str]:
    fs, fs_path = url_to_fs(path)
    parent = os.path.dirname(fs_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    return fs, fs_path


def _write_json(path: str, payload: dict[str, Any]) -> None:
    fs, fs_path = _ensure_parent_dir(path)
    tmp_path = f"{fs_path}.tmp"
    if fs.exists(tmp_path):
        fs.rm(tmp_path)
    with fs.open(tmp_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if fs.exists(fs_path):
        fs.rm(fs_path)
    fs.mv(tmp_path, fs_path)


def _read_json(path: str) -> dict[str, Any]:
    fs, fs_path = url_to_fs(path)
    with fs.open(fs_path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_run_state(path: str, state: AlternatingRunState) -> None:
    """Write the durable controller source-of-truth."""
    _write_json(path, state.to_dict())


def read_run_state(path: str) -> AlternatingRunState:
    """Read the durable controller source-of-truth."""
    return AlternatingRunState.from_dict(_read_json(path))


def write_policy_manifest(path: str, manifest: PolicyManifest) -> None:
    """Write one policy manifest."""
    _write_json(path, manifest.to_dict())


def read_policy_manifest(path: str) -> PolicyManifest:
    """Read one policy manifest."""
    return PolicyManifest.from_dict(_read_json(path))


def write_sampling_manifest(path: str, manifest: SamplingManifest) -> None:
    """Write one sampling manifest."""
    _write_json(path, manifest.to_dict())


def read_sampling_manifest(path: str) -> SamplingManifest:
    """Read one sampling manifest."""
    return SamplingManifest.from_dict(_read_json(path))


def write_sampling_host_status(path: str, status: SamplingHostStatusManifest) -> None:
    """Write one sampling-host completion marker."""
    _write_json(path, status.to_dict())


def read_sampling_host_status(path: str) -> SamplingHostStatusManifest:
    """Read one sampling-host completion marker."""
    return SamplingHostStatusManifest.from_dict(_read_json(path))


def write_materialized_batches_manifest(path: str, manifest: MaterializedBatchesManifest) -> None:
    """Write one materialized-batch manifest."""
    _write_json(path, manifest.to_dict())


def read_materialized_batches_manifest(path: str) -> MaterializedBatchesManifest:
    """Read one materialized-batch manifest."""
    return MaterializedBatchesManifest.from_dict(_read_json(path))


def write_phase_metrics_manifest(path: str, manifest: PhaseMetricsManifest) -> None:
    """Write one per-phase metrics manifest."""
    _write_json(path, manifest.to_dict())


def read_phase_metrics_manifest(path: str) -> PhaseMetricsManifest:
    """Read one per-phase metrics manifest."""
    return PhaseMetricsManifest.from_dict(_read_json(path))


def update_phase_metrics(
    path: str,
    *,
    phase_id: int,
    prepare_sampling_seconds: float | None = None,
    sampling_seconds: float | None = None,
    curriculum_update_seconds: float | None = None,
    materialization_seconds: float | None = None,
    training_seconds: float | None = None,
    export_seconds: float | None = None,
) -> PhaseMetricsManifest:
    """Merge one or more per-phase timing fields into the durable metrics manifest."""
    if exists(path):
        manifest = read_phase_metrics_manifest(path)
        if manifest.phase_id != phase_id:
            raise ValueError(f"phase metrics path {path} contains phase {manifest.phase_id}, expected {phase_id}")
    else:
        manifest = PhaseMetricsManifest(phase_id=phase_id)

    updates = {
        "prepare_sampling_seconds": prepare_sampling_seconds,
        "sampling_seconds": sampling_seconds,
        "curriculum_update_seconds": curriculum_update_seconds,
        "materialization_seconds": materialization_seconds,
        "training_seconds": training_seconds,
        "export_seconds": export_seconds,
    }
    for key, value in list(updates.items()):
        if value is None:
            del updates[key]

    manifest = replace(manifest, **updates, last_updated_at=utc_now_iso())
    write_phase_metrics_manifest(path, manifest)
    return manifest


def sampling_host_statuses_exist(paths: AlternatingRunPaths, manifest: SamplingManifest) -> bool:
    """Return whether every sampling host status file exists for a phase."""
    return all(
        exists(paths.sampling_host_status_path(manifest.phase_id, assignment.host_ordinal))
        for assignment in manifest.host_assignments
    )
