# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static no-oracle retention, explicitly separate from measured execution."""

from typing import Literal

from finestore.rl.curation import ActorCensusInput, DatasetRelease, NoOracleExecutabilityPolicy, SourceUse
from pydantic import BaseModel, ConfigDict, Field, model_validator


class NoOracleSourceContext(BaseModel):
    """Independently configured source population expected by an admission consumer."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    source_id: str = Field(min_length=1)
    source_release: DatasetRelease
    input_cohort_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    current_input: ActorCensusInput
    intended_use: SourceUse
    policy: NoOracleExecutabilityPolicy

    @model_validator(mode="after")
    def validate_current_pin(self):
        if (self.current_input.dataset_id, self.current_input.dataset_revision) != (
            self.source_release.repo,
            self.source_release.revision,
        ):
            raise ValueError("No-oracle context differs from its current input pin")
        return self


class StaticRetainedTaskReceipt(BaseModel):
    """One unchanged no-oracle task retained without an execution claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    outcome: Literal["static_retained"] = "static_retained"
    has_oracle: Literal[False] = False
    executability: Literal["unmeasured"] = "unmeasured"
    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str | None
    native_task_checksum: str | None
    input_cohort_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exclusion_reason: None = None

    @property
    def passed(self) -> bool:
        """Return static retention only; this does not represent an oracle pass."""
        return True
