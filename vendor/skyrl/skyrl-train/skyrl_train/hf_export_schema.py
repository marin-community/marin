from dataclasses import dataclass, field, replace
from enum import StrEnum

from marin_hf_release import POLICY_VERSION, PrecisionMode, ReleasePolicy

from marinskyrl.resource_locator import is_hugging_face_repo_id, validate_replayable_model_reference
from marinskyrl.speculative_decoding import is_hugging_face_commit

HF_EXPORT_REQUEST_SCHEMA_VERSION = 2
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
    release_precision: str = "preserve"
    release_policy_version: int = POLICY_VERSION
    eos_token_ids: list[int] | None = None
    generation_controls: dict[str, object] = field(default_factory=dict)
    chat_template: str | None = None
    model_revision: str | None = None
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
        ReleasePolicy(PrecisionMode(self.release_precision), version=self.release_policy_version)
        if is_hugging_face_repo_id(self.model_path) and (
            self.model_revision is None or not is_hugging_face_commit(self.model_revision)
        ):
            raise ValueError("HF release source model must be pinned to a 40-character commit")

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
