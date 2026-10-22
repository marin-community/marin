"""Policy-only conversion of distributed checkpoints to Hugging Face format."""

from __future__ import annotations

import os
import hashlib
import json
from io import BytesIO
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import ray
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group
from transformers import GenerationConfig, PreTrainedTokenizerBase

from marin_hf_release import POLICY_VERSION, PrecisionMode, ReleasePolicy, generation_metadata

from marinskyrl.checkpoint_paths import POLICY_CHECKPOINT_SUBDIRECTORY, policy_export_path
from marinskyrl.resource_locator import is_hugging_face_repo_id, join_resource_path
from marinskyrl.speculative_decoding import is_hugging_face_commit
from skyrl_train import hf_model_io
from skyrl_train.hf_release_export import ReleaseExportOptions
from skyrl_train.hf_export_schema import (
    DEFAULT_HF_HUB_REVISION,
    DEFAULT_HF_UPLOAD_MODE,
    HFUploadMode,
    TRAINER_STATE_FILENAME,
)
from skyrl_train.hf_publisher import HuggingFacePublisher
from skyrl_train.tokenizer import tokenizer_from_config
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.io import io
from skyrl_train.utils.utils import (
    policy_force_cvd_mask_enabled,
    policy_per_gpu_bundles_enabled,
    policy_spread_bundles,
)
from skyrl_train.workers.worker import PPORayActorGroup


@dataclass(frozen=True)
class CheckpointExportPlan:
    """Inputs required to convert one immutable policy checkpoint."""

    step: int
    checkpoint_path: str
    export_root: str
    model_path: str
    model_revision: str | None = None
    release_precision: str = "preserve"
    release_policy_version: int = POLICY_VERSION
    eos_token_ids: list[int] | None = None
    generation_controls: dict[str, object] | None = None
    chat_template: str | None = None

    @property
    def policy_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_path, POLICY_CHECKPOINT_SUBDIRECTORY)

    @property
    def policy_export_path(self) -> str:
        return policy_export_path(self.export_root, self.step)


@dataclass(frozen=True)
class CheckpointExportResult:
    step: int
    export_path: str


class PolicyExportWorkers(Protocol):
    """Distributed policy operations required by checkpoint conversion."""

    def initialize(self, model_path: str) -> None: ...

    def load_model_checkpoint(self, checkpoint_path: str) -> None: ...

    def save_hf_model(
        self, export_path: str, tokenizer: PreTrainedTokenizerBase, release_options: ReleaseExportOptions
    ) -> None: ...

    def close(self) -> None: ...


class ExportPublisher(Protocol):
    def publish(self, export_path: str, step: int) -> None: ...


class RayPolicyExportWorkers:
    """Adapt a policy Ray actor group to the model-only export protocol."""

    def __init__(
        self,
        actor_group: PPORayActorGroup,
        placement: PlacementGroup | None = None,
        resolve: Callable[[Sequence[object]], object] = ray.get,
    ):
        self._actor_group = actor_group
        self._placement = placement
        self._resolve = resolve

    def _run(self, method_name: str, *args, **kwargs) -> None:
        refs = self._actor_group.async_run_ray_method("pass_through", method_name, *args, **kwargs)
        self._resolve(refs)

    def initialize(self, model_path: str) -> None:
        self._run("init_model_for_export", model_path)

    def load_model_checkpoint(self, checkpoint_path: str) -> None:
        self._run(
            "load_checkpoint",
            ckpt_dir=checkpoint_path,
            load_training_state=False,
        )

    def save_hf_model(
        self, export_path: str, tokenizer: PreTrainedTokenizerBase, release_options: ReleaseExportOptions
    ) -> None:
        self._run("save_hf_model", export_path, tokenizer, release_options)

    def close(self) -> None:
        self._actor_group.kill_actors()
        if self._placement is not None:
            remove_placement_group(self._placement)


class CheckpointExporter:
    """Validate, load, and convert one policy checkpoint."""

    def __init__(
        self,
        plan: CheckpointExportPlan,
        workers: PolicyExportWorkers,
        tokenizer: PreTrainedTokenizerBase,
        publisher: ExportPublisher | None = None,
    ):
        self._plan = plan
        self._workers = workers
        self._tokenizer = tokenizer
        self._publisher = publisher

    def _checkpoint_identity(self) -> str:
        """Validate the completed checkpoint and hash its trainer-state marker."""
        trainer_state_path = os.path.join(self._plan.checkpoint_path, TRAINER_STATE_FILENAME)
        if not io.exists(trainer_state_path):
            raise FileNotFoundError(f"completed checkpoint marker not found: {trainer_state_path}")
        with io.open_file(trainer_state_path, "rb") as source:
            marker = source.read()
        trainer_state = torch.load(BytesIO(marker), map_location="cpu", weights_only=False)
        saved_step = trainer_state.get("global_step")
        if saved_step != self._plan.step:
            raise ValueError(
                f"checkpoint step mismatch: requested global_step_{self._plan.step}, marker records {saved_step!r}"
            )
        if not io.exists(self._plan.policy_checkpoint_path):
            raise FileNotFoundError(f"policy checkpoint not found: {self._plan.policy_checkpoint_path}")
        return hashlib.sha256(marker).hexdigest()

    def _release_options(self, checkpoint_identity: str) -> ReleaseExportOptions:
        policy = ReleasePolicy(PrecisionMode(self._plan.release_precision), self._plan.release_policy_version)
        revision = self._plan.model_revision
        is_pinned = revision is not None and is_hugging_face_commit(revision)
        if is_hugging_face_repo_id(self._plan.model_path) and not is_pinned:
            raise ValueError("HF release source model must be pinned to a 40-character commit")
        eos_token_ids = self._plan.eos_token_ids
        controls = {}
        source_generation = None
        if io.is_cloud_path(self._plan.model_path):
            generation_path = join_resource_path(self._plan.model_path, "generation_config.json")
            if io.exists(generation_path):
                with io.open_file(generation_path, "r") as source:
                    source_generation = json.load(source)
        elif is_pinned or os.path.isfile(os.path.join(self._plan.model_path, "generation_config.json")):
            source_generation = GenerationConfig.from_pretrained(
                self._plan.model_path, revision=revision if is_pinned else None
            ).to_diff_dict()
        if source_generation is not None:
            source_eos, source_controls = generation_metadata(source_generation)
            if eos_token_ids is None:
                eos_token_ids = source_eos
            controls = source_controls
        controls.update(self._plan.generation_controls or {})
        return ReleaseExportOptions(
            precision=policy.mode,
            source_checkpoint=self._plan.checkpoint_path,
            source_step=self._plan.step,
            checkpoint_identity=checkpoint_identity,
            source_model=self._plan.model_path if is_pinned else None,
            source_model_commit=revision if is_pinned else None,
            eos_token_ids=eos_token_ids,
            generation_controls=controls,
            chat_template=self._plan.chat_template,
        )

    def run(self) -> CheckpointExportResult:
        try:
            checkpoint_identity = self._checkpoint_identity()
            release_options = self._release_options(checkpoint_identity)
            self._workers.initialize(self._plan.model_path)
            self._workers.load_model_checkpoint(self._plan.policy_checkpoint_path)
            self._workers.save_hf_model(self._plan.policy_export_path, self._tokenizer, release_options)
            hf_model_io.verify_hf_model_export(self._plan.policy_export_path, require_release_manifest=True)
            if self._publisher is not None:
                self._publisher.publish(self._plan.policy_export_path, self._plan.step)
            return CheckpointExportResult(step=self._plan.step, export_path=self._plan.policy_export_path)
        finally:
            self._workers.close()


def checkpoint_export_plan(cfg: DictConfig) -> CheckpointExportPlan:
    """Resolve and validate the dedicated checkpoint-export configuration."""
    export_cfg = cfg.checkpoint_export
    step = export_cfg.get("step")
    checkpoint_path = export_cfg.get("checkpoint_path")
    export_root = export_cfg.get("export_root")
    if not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint_export.step must be a non-negative integer")
    if not checkpoint_path:
        raise ValueError("checkpoint_export.checkpoint_path is required")
    if not export_root:
        raise ValueError("checkpoint_export.export_root is required")
    eos_token_ids = export_cfg.get("eos_token_ids")
    generation_controls = export_cfg.get("generation_controls")
    if isinstance(eos_token_ids, ListConfig):
        eos_token_ids = OmegaConf.to_container(eos_token_ids, resolve=True)
    if isinstance(generation_controls, DictConfig):
        generation_controls = OmegaConf.to_container(generation_controls, resolve=True)
    return CheckpointExportPlan(
        step=step,
        checkpoint_path=str(checkpoint_path).rstrip("/"),
        export_root=str(export_root).rstrip("/"),
        model_path=str(cfg.trainer.policy.model.path),
        model_revision=cfg.trainer.policy.model.get("revision"),
        release_precision=str(export_cfg.get("release_precision", "preserve")),
        release_policy_version=int(export_cfg.get("release_policy_version", POLICY_VERSION)),
        eos_token_ids=eos_token_ids,
        generation_controls=generation_controls,
        chat_template=export_cfg.get("chat_template"),
    )


def policy_export_workers(cfg: DictConfig) -> RayPolicyExportWorkers:
    """Create exactly one policy worker per saved policy rank."""
    if cfg.trainer.strategy == "megatron":
        from skyrl_train.workers.megatron.megatron_worker import PolicyWorker
    else:
        raise ValueError(f"checkpoint export does not support strategy {cfg.trainer.strategy!r}")

    placement_config = cfg.trainer.placement
    per_gpu_bundles = policy_per_gpu_bundles_enabled(cfg)
    policy_placement = placement_group(
        policy_spread_bundles(cfg),
        strategy="PACK" if per_gpu_bundles else "STRICT_SPREAD",
    )
    try:
        get_ray_pg_ready_with_timeout(
            policy_placement, timeout=int(cfg.trainer.distributed.placement_group_timeout_seconds)
        )
        actor_group = PPORayActorGroup(
            cfg,
            placement_config.policy_num_nodes,
            placement_config.policy_num_gpus_per_node,
            PolicyWorker,
            pg=policy_placement,
            num_gpus_per_actor=1,
            colocate_all=False,
            sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            pin_to_ray_gpu_id=per_gpu_bundles,
            force_cvd_mask=per_gpu_bundles and policy_force_cvd_mask_enabled(cfg),
        )
    except Exception:
        remove_placement_group(policy_placement)
        raise
    return RayPolicyExportWorkers(actor_group, placement=policy_placement)


def hub_publisher(cfg: DictConfig) -> HuggingFacePublisher | None:
    repo_id = cfg.checkpoint_export.get("hf_hub_repo_id")
    if not repo_id:
        return None
    return HuggingFacePublisher(
        repo_id=str(repo_id),
        private=bool(cfg.checkpoint_export.get("hf_hub_private", False)),
        revision=str(cfg.checkpoint_export.get("hf_hub_revision", DEFAULT_HF_HUB_REVISION)),
        upload_mode=HFUploadMode(cfg.checkpoint_export.get("hf_upload_mode", DEFAULT_HF_UPLOAD_MODE)),
    )


def checkpoint_exporter(cfg: DictConfig) -> CheckpointExporter:
    """Construct the standalone exporter from a resolved Hydra configuration."""
    return CheckpointExporter(
        checkpoint_export_plan(cfg),
        policy_export_workers(cfg),
        tokenizer_from_config(cfg),
        hub_publisher(cfg),
    )
