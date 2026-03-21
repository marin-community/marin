# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run state, manifests, and phase enums for alternating RL.

All phase state is persisted as JSON in the run root on GCS.
The controller reads and writes these; phase processes read them.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from iris.marin_fs import url_to_fs

logger = logging.getLogger(__name__)


class RunStatus(StrEnum):
    BOOTSTRAPPING = "bootstrapping"
    SAMPLING = "sampling"
    MATERIALIZING = "materializing"
    TRAINING = "training"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AlternatingRunState:
    """Source of truth for the controller and resume logic."""

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
    current_policy_path: str
    current_levanter_checkpoint_path: str | None
    current_sampling_manifest: str | None = None
    current_materialized_manifest: str | None = None
    last_completed_phase: int = -1

    def to_json(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "AlternatingRunState":
        d = json.loads(raw)
        d["status"] = RunStatus(d["status"])
        return cls(**d)


@dataclass
class PolicyManifest:
    """Metadata for one exported HF/safetensors policy."""

    policy_version: int
    phase_id: int
    source_global_step: int
    hf_export_path: str
    levanter_checkpoint_path: str | None
    model_name: str
    created_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "PolicyManifest":
        return cls(**json.loads(raw))


@dataclass
class SamplingHostAssignment:
    """Per-host quota and seed for one sampling phase."""

    host_ordinal: int
    seed: int
    target_train_groups: int
    target_eval_groups: int


@dataclass
class SamplingManifest:
    """Written by the controller before launching sampling hosts."""

    phase_id: int
    policy_version: int
    policy_manifest_path: str
    curriculum_state_path: str
    num_hosts: int
    local_tensor_parallel_size: int
    host_assignments: list[SamplingHostAssignment]
    frozen_lesson_weights: dict[str, float]
    output_root: str

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "SamplingManifest":
        d = json.loads(raw)
        d["host_assignments"] = [SamplingHostAssignment(**ha) for ha in d["host_assignments"]]
        return cls(**d)


@dataclass
class HostStatus:
    """Written by each sampling host on completion."""

    host_ordinal: int
    phase_id: int
    policy_version: int
    eval_groups_written: int
    train_groups_written: int
    success: bool
    started_at: str
    finished_at: str
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "HostStatus":
        return cls(**json.loads(raw))


@dataclass
class MaterializationManifest:
    """Written by the materializer after producing training batches."""

    phase_id: int
    policy_version: int
    num_input_rollout_files: int
    num_rollout_groups: int
    num_individual_rollouts: int
    num_training_batches: int
    global_batch_size: int
    max_seq_len: int
    batch_paths: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "MaterializationManifest":
        return cls(**json.loads(raw))


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------


def write_json_to_path(path: str, content: str) -> None:
    """Write JSON string to a GCS or local path."""
    fs, _ = url_to_fs(path)
    parent = os.path.dirname(path)
    fs.makedirs(parent, exist_ok=True)
    with fs.open(path, "w") as f:
        f.write(content)
    logger.info("Wrote %s", path)


def read_json_from_path(path: str) -> str:
    """Read JSON string from a GCS or local path."""
    fs, _ = url_to_fs(path)
    with fs.open(path, "r") as f:
        return f.read()


def path_exists(path: str) -> bool:
    """Check whether a GCS or local path exists."""
    fs, _ = url_to_fs(path)
    return fs.exists(path)
