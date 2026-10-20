"""Trusted, opt-in capture/replay helpers for policy-forward GPU diagnostics."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, TypedDict

from omegaconf import DictConfig, OmegaConf

from skyrl_train.timing_observability import StepWallTime
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingInputBatch


SCHEMA_VERSION = 1
CAPTURE_STAGE = "pre_forward"
DIAGNOSTIC_CONFIG_KEY = "batch_replay"
_BATCH_FILE = "batch.pkl"
_MANIFEST_FILE = "manifest.json"


class TensorManifestEntry(TypedDict):
    dtype: str
    shape: list[int]


@dataclass(frozen=True)
class BatchReplayProvenance:
    """Facts that must match before a captured batch may be replayed."""

    source_revision: str
    config_fingerprint: str
    checkpoint_path: str
    checkpoint_step: int
    target_step: int


@dataclass(frozen=True)
class TrainingBatchManifest:
    schema_version: int
    stage: str
    provenance: BatchReplayProvenance
    batch_sha256: str
    tensors: dict[str, TensorManifestEntry]

    @classmethod
    def parse(cls, payload: Any) -> "TrainingBatchManifest":
        if not isinstance(payload, dict):
            raise ValueError("Training-batch manifest must be a JSON object")
        try:
            fields = dict(payload)
            provenance = BatchReplayProvenance(**fields.pop("provenance"))
            return cls(provenance=provenance, **fields)
        except (KeyError, TypeError) as error:
            raise ValueError(f"Training-batch manifest does not match schema: {error}") from error


def config_fingerprint(cfg: DictConfig) -> str:
    """Hash the resolved training config, excluding only diagnostic controls."""

    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Expected a mapping-shaped Hydra configuration")
    resolved.pop(DIAGNOSTIC_CONFIG_KEY, None)
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_manifest(batch: TrainingInputBatch) -> dict[str, TensorManifestEntry]:
    return {
        key: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for key, value in sorted(batch.items())
        if value is not None
    }


def _cpu_snapshot(batch: TrainingInputBatch) -> TrainingInputBatch:
    tensors = {
        key: None if value is None else value.detach().to(device="cpu").contiguous().clone()
        for key, value in batch.items()
    }
    snapshot = TrainingInputBatch(tensors, routed_expert_rows=copy.deepcopy(batch.routed_expert_rows))
    snapshot.metadata = copy.deepcopy(batch.metadata)
    return snapshot


def _fsync(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_destination(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite replay artifact: {destination}")


def save_training_batch_artifact(
    artifact_path: Path,
    batch: TrainingInputBatch,
    provenance: BatchReplayProvenance,
) -> None:
    """Publish a complete artifact or leave the destination absent."""

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    validate_artifact_destination(artifact_path)

    temporary = Path(tempfile.mkdtemp(prefix=f".{artifact_path.name}.tmp-", dir=artifact_path.parent))
    try:
        snapshot = _cpu_snapshot(batch)
        batch_path = temporary / _BATCH_FILE
        snapshot.save(str(batch_path))
        _fsync(batch_path)

        manifest = TrainingBatchManifest(
            schema_version=SCHEMA_VERSION,
            stage=CAPTURE_STAGE,
            provenance=provenance,
            batch_sha256=_file_sha256(batch_path),
            tensors=_tensor_manifest(snapshot),
        )
        manifest_path = temporary / _MANIFEST_FILE
        manifest_path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
        _fsync(manifest_path)
        os.replace(temporary, artifact_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_manifest(artifact_path: Path) -> TrainingBatchManifest:
    manifest_path = artifact_path / _MANIFEST_FILE
    batch_path = artifact_path / _BATCH_FILE
    if not artifact_path.is_dir() or not manifest_path.is_file() or not batch_path.is_file():
        raise ValueError(f"Incomplete training-batch artifact: {artifact_path}")
    try:
        manifest = TrainingBatchManifest.parse(json.loads(manifest_path.read_text()))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid training-batch manifest: {manifest_path}") from error
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version mismatch: expected {SCHEMA_VERSION}, got {manifest.schema_version!r}")
    if manifest.stage != CAPTURE_STAGE:
        raise ValueError(f"stage mismatch: expected {CAPTURE_STAGE!r}, got {manifest.stage!r}")
    return manifest


def load_training_batch_artifact(
    artifact_path: Path,
    *,
    expected: BatchReplayProvenance,
) -> TrainingInputBatch:
    """Validate provenance and load a trusted internal replay artifact."""

    manifest = _load_manifest(artifact_path)
    for field, expected_value in asdict(expected).items():
        actual_value = getattr(manifest.provenance, field)
        if actual_value != expected_value:
            raise ValueError(f"{field} mismatch: expected {expected_value!r}, got {actual_value!r}")

    batch_path = artifact_path / _BATCH_FILE
    actual_digest = _file_sha256(batch_path)
    if actual_digest != manifest.batch_sha256:
        raise ValueError("batch_sha256 mismatch: batch payload is corrupt or was modified")

    # This is deliberately pickle-based and must only be used with trusted,
    # access-controlled artifacts produced by this diagnostic.
    batch = TrainingInputBatch().load(str(batch_path))
    if not isinstance(batch, TrainingInputBatch):
        raise ValueError(f"Unexpected batch type: {type(batch).__name__}")
    actual_tensors = _tensor_manifest(batch)
    if actual_tensors != manifest.tensors:
        raise ValueError("tensor manifest mismatch: captured tensors do not match manifest")
    return batch


class CapturingRayPPOTrainer(RayPPOTrainer):
    """Trainer variant that captures one batch before policy forward."""

    def __init__(
        self,
        *args,
        capture_artifact_path: Path,
        capture_provenance: BatchReplayProvenance,
        **kwargs,
    ):
        self.capture_artifact_path = capture_artifact_path
        self.capture_provenance = capture_provenance
        super().__init__(*args, **kwargs)

    async def _run_training(self, training_input: TrainingInputBatch, *, step_wall: StepWallTime | None = None):
        if self.global_step == self.capture_provenance.target_step:
            save_training_batch_artifact(
                self.capture_artifact_path,
                training_input,
                self.capture_provenance,
            )
        return await super()._run_training(training_input, step_wall=step_wall)


class _PolicyForwardComplete(Exception):
    def __init__(self, result: TrainingInputBatch):
        self.result = result


async def replay_policy_forward(trainer: RayPPOTrainer, training_input: TrainingInputBatch) -> TrainingInputBatch:
    """Run the production training-step prefix through policy forward."""

    production_forward = trainer.fwd_logprobs_values_reward

    def stop_after_forward(batch: TrainingInputBatch):
        raise _PolicyForwardComplete(production_forward(batch))

    trainer.fwd_logprobs_values_reward = stop_after_forward
    try:
        await trainer._run_training(training_input)
    except _PolicyForwardComplete as complete:
        return complete.result
    finally:
        trainer.fwd_logprobs_values_reward = production_forward
    raise AssertionError("Production training step returned without invoking policy forward")
