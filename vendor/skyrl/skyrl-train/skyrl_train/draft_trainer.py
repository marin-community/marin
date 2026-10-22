"""Independent online draft training over cloud-backed rollout captures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

from loguru import logger
import ray

from marinskyrl.resource_locator import is_cloud_uri, join_resource_path
from marinskyrl.speculative_decoding import SpeculatorModelConfig, SpeculatorTrainingConfig
from skyrl_train.inference_engines.vllm.online_eagle_trainer import (
    ONLINE_EAGLE_MERGED_CAPTURE_DIRECTORY,
    OnlineEagleTrainerRuntime,
    OnlineEagleTrainingJob,
    OnlineEagleUpdateResult,
    merge_online_eagle_captures,
)
from skyrl_train.hf_model_io import HF_WEIGHT_FILENAME
from skyrl_train.io import io


LATEST_DRAFT_FILENAME = "latest.json"
COMPLETE_DRAFT_FILENAME = "complete.json"


@dataclass(frozen=True)
class DraftCheckpoint:
    """One completed draft checkpoint visible to inference workers."""

    step: int
    revision: str
    uri: str
    weights_uri: str
    weights_size: int
    completion_uri: str
    source_identity: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DraftCheckpoint":
        step = value.get("step")
        revision = value.get("revision")
        uri = value.get("uri")
        weights_uri = value.get("weights_uri")
        weights_size = value.get("weights_size")
        completion_uri = value.get("completion_uri")
        source_identity = value.get("source_identity")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("Draft checkpoint step must be a positive integer")
        if not isinstance(revision, str) or not revision:
            raise ValueError("Draft checkpoint revision must be nonempty")
        if not isinstance(uri, str) or not is_cloud_uri(uri):
            raise ValueError("Draft checkpoint URI must be cloud-backed")
        if not isinstance(weights_uri, str) or not is_cloud_uri(weights_uri):
            raise ValueError("Draft checkpoint weights URI must be cloud-backed")
        if isinstance(weights_size, bool) or not isinstance(weights_size, int) or weights_size <= 0:
            raise ValueError("Draft checkpoint weights size must be a positive integer")
        if not isinstance(completion_uri, str) or not is_cloud_uri(completion_uri):
            raise ValueError("Draft checkpoint completion URI must be cloud-backed")
        if not isinstance(source_identity, str) or not source_identity:
            raise ValueError("Draft checkpoint source identity must be nonempty")
        expected_weights_uri = join_resource_path(uri, HF_WEIGHT_FILENAME)
        expected_completion_uri = join_resource_path(uri, COMPLETE_DRAFT_FILENAME)
        if weights_uri != expected_weights_uri or completion_uri != expected_completion_uri:
            raise ValueError("Draft checkpoint object URIs do not match its immutable directory")
        return cls(
            step=step,
            revision=revision,
            uri=uri,
            weights_uri=weights_uri,
            weights_size=weights_size,
            completion_uri=completion_uri,
            source_identity=source_identity,
        )


@dataclass(frozen=True)
class _DraftCheckpointPointer:
    revision: str
    completion_uri: str
    source_identity: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "_DraftCheckpointPointer":
        revision = value.get("revision")
        completion_uri = value.get("completion_uri")
        source_identity = value.get("source_identity")
        if not isinstance(revision, str) or not revision:
            raise ValueError("Latest draft revision must be nonempty")
        if not isinstance(completion_uri, str) or not is_cloud_uri(completion_uri):
            raise ValueError("Latest draft completion URI must be cloud-backed")
        if not isinstance(source_identity, str) or not source_identity:
            raise ValueError("Latest draft source identity must be nonempty")
        return cls(
            revision=revision,
            completion_uri=completion_uri,
            source_identity=source_identity,
        )


@dataclass(frozen=True)
class DraftUpdateRequest:
    """Control-plane request for one asynchronous draft update."""

    step: int
    capture_uri: str
    target_revision: str
    num_speculative_tokens: int
    seed: int
    training: SpeculatorTrainingConfig


def latest_draft_checkpoint_uri(checkpoint_root: str) -> str:
    return join_resource_path(checkpoint_root, LATEST_DRAFT_FILENAME)


def read_latest_draft_checkpoint(
    checkpoint_root: str,
    *,
    source_identity: str | None = None,
) -> DraftCheckpoint | None:
    """Read the latest publication matching the requested source lineage."""
    latest_uri = latest_draft_checkpoint_uri(checkpoint_root)
    if not io.exists(latest_uri):
        return None
    pointer = _DraftCheckpointPointer.from_mapping(json.loads(io.read_bytes(latest_uri)))
    if source_identity is not None and pointer.source_identity != source_identity:
        return None
    checkpoint = DraftCheckpoint.from_mapping(json.loads(io.read_bytes(pointer.completion_uri)))
    if (
        checkpoint.revision != pointer.revision
        or checkpoint.source_identity != pointer.source_identity
        or checkpoint.completion_uri != pointer.completion_uri
    ):
        raise ValueError("Latest draft pointer does not match its completion manifest")
    remote_size = io.file_size(checkpoint.weights_uri)
    if remote_size != checkpoint.weights_size:
        raise ValueError(
            f"Draft checkpoint weights are incomplete: expected {checkpoint.weights_size} bytes, got {remote_size}"
        )
    return checkpoint


class DraftTrainer:
    """Long-lived one-GPU owner for draft weights, optimizer state, and publication."""

    def __init__(self, *, initial_model: SpeculatorModelConfig, checkpoint_root: str):
        if not is_cloud_uri(checkpoint_root):
            raise ValueError(f"DraftTrainer checkpoint root must be cloud-backed: {checkpoint_root}")
        self._initial_model = initial_model
        self._checkpoint_root = checkpoint_root
        try:
            self._latest = read_latest_draft_checkpoint(
                checkpoint_root,
                source_identity=initial_model.source_identity,
            )
        except Exception as error:
            logger.warning("Ignoring invalid latest draft checkpoint at {}: {}", checkpoint_root, error)
            self._latest = None
        self._accepted_revision = self._latest.revision if self._latest is not None else initial_model.source_identity
        self._runtime: OnlineEagleTrainerRuntime | None = None
        self._node_id = str(ray.get_runtime_context().get_node_id()) if ray.is_initialized() else None
        self._gpu_ids = [str(gpu_id) for gpu_id in ray.get_gpu_ids()] if ray.is_initialized() else []

    def _publish_checkpoint(
        self,
        *,
        step: int,
        revision: str,
        candidate_dir: Path,
    ) -> DraftCheckpoint:
        candidate_uri = join_resource_path(self._checkpoint_root, revision)
        io.upload_directory(str(candidate_dir), candidate_uri)
        weights_uri = join_resource_path(candidate_uri, HF_WEIGHT_FILENAME)
        weights_size = (candidate_dir / HF_WEIGHT_FILENAME).stat().st_size
        remote_weights_size = io.file_size(weights_uri)
        if remote_weights_size != weights_size:
            raise IOError(
                f"Draft checkpoint upload is incomplete: expected {weights_size} bytes, got {remote_weights_size}"
            )
        completion_uri = join_resource_path(candidate_uri, COMPLETE_DRAFT_FILENAME)
        checkpoint = DraftCheckpoint(
            step=step,
            revision=revision,
            uri=candidate_uri,
            weights_uri=weights_uri,
            weights_size=weights_size,
            completion_uri=completion_uri,
            source_identity=self._initial_model.source_identity,
        )
        io.write_bytes_atomic(
            completion_uri,
            json.dumps(asdict(checkpoint), sort_keys=True).encode(),
        )
        pointer = _DraftCheckpointPointer(
            revision=checkpoint.revision,
            completion_uri=checkpoint.completion_uri,
            source_identity=checkpoint.source_identity,
        )
        io.write_bytes_atomic(
            latest_draft_checkpoint_uri(self._checkpoint_root),
            json.dumps(asdict(pointer), sort_keys=True).encode(),
        )
        return checkpoint

    def update(self, request: DraftUpdateRequest) -> OnlineEagleUpdateResult:
        """Consume a cloud capture, train, and publish a completed candidate."""
        if not is_cloud_uri(request.capture_uri):
            raise ValueError(f"Draft capture must be cloud-backed: {request.capture_uri}")
        try:
            with tempfile.TemporaryDirectory(prefix="marinskyrl-draft-update-") as scratch:
                scratch_root = Path(scratch)
                capture_root = scratch_root / "capture"
                io.download_directory(request.capture_uri, str(capture_root))
                merged_capture = scratch_root / ONLINE_EAGLE_MERGED_CAPTURE_DIRECTORY
                merge_online_eagle_captures(
                    capture_root,
                    merged_capture,
                    expected_step=request.step,
                    max_tokens=request.training.max_tokens_per_update,
                    max_sequences_per_prompt_group=request.training.max_sequences_per_prompt_group,
                    max_window_tokens=request.training.max_window_tokens,
                )
                candidate_dir = scratch_root / "candidate"
                job = OnlineEagleTrainingJob(
                    step=request.step,
                    capture_dir=str(merged_capture),
                    initial_draft_model=self._initial_model,
                    parent_draft_revision=self._accepted_revision,
                    target_revision=request.target_revision,
                    output_dir=str(candidate_dir),
                    num_speculative_tokens=request.num_speculative_tokens,
                    seed=request.seed,
                    training=request.training,
                )
                if self._runtime is None:
                    self._runtime = OnlineEagleTrainerRuntime(job, merged_capture)
                    if self._latest is not None:
                        restored = scratch_root / "restored"
                        io.download_directory(self._latest.uri, str(restored))
                        self._runtime.restore(restored)
                result = self._runtime.update(job)
                if result.accepted:
                    assert result.draft_revision is not None
                    checkpoint = self._publish_checkpoint(
                        step=request.step,
                        revision=result.draft_revision,
                        candidate_dir=candidate_dir,
                    )
                    self._latest = checkpoint
                    self._accepted_revision = checkpoint.revision
                    result = replace(result, candidate_uri=checkpoint.uri)
                return result
        except Exception as error:
            logger.exception("DraftTrainer update failed at step {}", request.step)
            return OnlineEagleUpdateResult(
                accepted=False,
                step=request.step,
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            try:
                if io.exists(request.capture_uri):
                    io.remove(request.capture_uri)
            except Exception as error:
                logger.warning("Failed to remove consumed draft capture {}: {}", request.capture_uri, error)

    def status(self) -> dict[str, Any]:
        return {
            "accepted_revision": self._accepted_revision,
            "checkpoint": None if self._latest is None else asdict(self._latest),
            "node_id": self._node_id,
            "gpu_ids": self._gpu_ids,
        }


def create_draft_trainer(*, initial_model: SpeculatorModelConfig, checkpoint_root: str):
    """Create the dedicated GPU actor outside the inference placement group."""
    actor = ray.remote(num_gpus=1, max_restarts=-1, max_task_retries=0)(DraftTrainer)
    return actor.options(runtime_env={"env_vars": {"TORCH_COMPILE_DISABLE": "1"}}).remote(
        initial_model=initial_model,
        checkpoint_root=checkpoint_root,
    )
