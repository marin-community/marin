from dataclasses import dataclass, replace
from enum import StrEnum

from marinskyrl.checkpoint_paths import GLOBAL_STEP_PREFIX
from marinskyrl.resource_locator import join_resource_path, validate_replayable_model_reference
from omegaconf import DictConfig

HF_EXPORT_REQUEST_SCHEMA_VERSION = 1
TRAINER_STATE_FILENAME = "trainer_state.pt"
DEFAULT_HF_EXPORT_TIMEOUT = 7200
DEFAULT_HF_HUB_REVISION = "main"


class HFUploadMode(StrEnum):
    LATEST = "latest"
    ALL = "all"


DEFAULT_HF_UPLOAD_MODE = HFUploadMode.LATEST


class HFExportStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"


@dataclass(frozen=True)
class HFExportRequest:
    step: int
    checkpoint_base_path: str
    checkpoint_path: str
    export_path: str
    model_path: str
    num_nodes: int
    gpus_per_node: int
    model_source_uri: str | None = None
    model_source_identity: str | None = None
    schema_version: int = HF_EXPORT_REQUEST_SCHEMA_VERSION
    status: HFExportStatus = HFExportStatus.PENDING
    hf_hub_repo_id: str | None = None
    hf_hub_private: bool = False
    hf_hub_revision: str = DEFAULT_HF_HUB_REVISION
    hf_upload_mode: HFUploadMode = DEFAULT_HF_UPLOAD_MODE
    attempts: int = 0
    timeout: int | None = None
    last_exit_code: int | None = None

    def __post_init__(self) -> None:
        validate_replayable_model_reference(self.model_path, self.model_source_uri, self.model_source_identity)

    @classmethod
    def from_trainer(
        cls, trainer: DictConfig, *, step: int, checkpoint_root: str, export_root: str
    ) -> "HFExportRequest":
        """Describe an export at the trainer's policy geometry."""
        model, placement = trainer.policy.model, trainer.placement
        return cls(
            step=step,
            checkpoint_base_path=checkpoint_root,
            checkpoint_path=join_resource_path(checkpoint_root, f"{GLOBAL_STEP_PREFIX}{step}"),
            export_path=export_root,
            model_path=model.path,
            num_nodes=placement.policy_num_nodes,
            gpus_per_node=placement.policy_num_gpus_per_node,
            model_source_uri=model.source_uri,
            model_source_identity=model.source_identity,
            hf_hub_repo_id=trainer.get("hf_hub_repo_id"),
            hf_hub_private=trainer.get("hf_hub_private", False),
            hf_hub_revision=trainer.get("hf_hub_revision", DEFAULT_HF_HUB_REVISION),
            hf_upload_mode=HFUploadMode(trainer.get("hf_upload_mode", DEFAULT_HF_UPLOAD_MODE)),
        )

    def with_status(
        self,
        status: HFExportStatus,
        *,
        timeout: int | None = None,
        last_exit_code: int | None = None,
        increment_attempts: bool = False,
    ) -> "HFExportRequest":
        return replace(
            self,
            status=status,
            timeout=timeout if timeout is not None else self.timeout,
            last_exit_code=last_exit_code,
            attempts=self.attempts + int(increment_attempts),
        )
