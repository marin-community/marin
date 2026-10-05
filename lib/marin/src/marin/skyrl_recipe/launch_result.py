"""The launcher response consumed by training graph execution."""

from dataclasses import dataclass
from enum import StrEnum


class LaunchState(StrEnum):
    PREPARED = "prepared"
    SUBMITTED = "submitted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class ExportedPolicy:
    policy_export_uri: str
    global_step: int
    tokenizer_uri: str
    tokenizer_revision: str
    checkpoint_root: str
    terminal_manifest_uri: str


@dataclass(frozen=True)
class LaunchResult:
    run_id: str
    attempt_id: str
    state: LaunchState
    iris_job_id: str | None
    iris_job_state: str | None
    launcher_commit: str
    runtime_profile: str
    model: ExportedPolicy | None
    failure: str | None
