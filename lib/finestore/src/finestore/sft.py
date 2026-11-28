# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-bound evidence from the Datakit SFT registration workflow."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finestore.rl.curation import ContentEvidenceReference as SftEvidenceReference
from finestore.schema import SHA256_PATTERN

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]


class SftEvalFamily(StrEnum):
    AA = "aa"
    LM_EVAL = "lm_eval"
    UNCHEATABLE = "uncheatable"
    ATLAS_REFERENCES = "atlas_references"


REQUIRED_SFT_EVAL_FAMILIES = frozenset(SftEvalFamily)


class SftRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetVariantInput(SftRecord):
    """The exact dataset variant that a preparation workflow consumed."""

    source_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    data_sha256: Sha256
    task_specs_sha256: Sha256 | None = None


class SftStage(StrEnum):
    DOWNLOAD = "download"
    CHAT = "chat_normalization"
    RENDER = "chat_rendering"
    NORMALIZE = "text_normalization"
    EVAL_BLOOM = "eval_bloom"
    DECONTAM = "decontamination"
    FILTER = "filtering_and_deduplication"
    TRAINING = "training_preparation"


class SftPreparationReceipt(SftRecord):
    """A completed registration, with output and stage identities for one variant."""

    protocol: Literal["datakit-sft-registration-v1"] = "datakit-sft-registration-v1"
    input: DatasetVariantInput
    registry_source: str = Field(min_length=1)
    implementation_sha256: Sha256
    stages: dict[SftStage, SftEvidenceReference]
    eval_manifests: dict[SftEvalFamily, SftEvidenceReference]
    eval_policy_sha256: Sha256
    output_manifest: SftEvidenceReference
    tokenizer: SftEvidenceReference
    chat_template_sha256: Sha256
    input_rows: int = Field(gt=0)
    training_rows: int = Field(gt=0)
    filtered_rows: int = Field(ge=0)
    chat_rows: int = Field(gt=0)
    normalization_counters: dict[str, dict[str, int | float]]
    stage_files: dict[SftStage, SftEvidenceReference]
    training_tokens: int = Field(gt=0)
    supervised_tokens: int = Field(gt=0)
    training_sha256: Sha256
    completed_at: datetime

    @model_validator(mode="after")
    def _complete_population(self) -> Self:
        if set(self.stages) != set(SftStage):
            raise ValueError("SFT registration requires every completed Datakit stage")
        if set(self.eval_manifests) != REQUIRED_SFT_EVAL_FAMILIES:
            raise ValueError("SFT registration requires core, Uncheatable and Atlas reference coverage")
        if self.training_rows + self.filtered_rows != self.chat_rows:
            raise ValueError("SFT registration must account for the complete input population")
        if set(self.stage_files) != {SftStage.CHAT, SftStage.NORMALIZE, SftStage.DECONTAM}:
            raise ValueError("SFT registration requires pinned chat, text and decontamination files")
        if self.supervised_tokens > self.training_tokens:
            raise ValueError("SFT supervision exceeds the tokenized population")
        return self

    def require_input(self, selected: DatasetVariantInput) -> None:
        """Reject reuse of a receipt on another variant, pin or content digest."""
        if self.input != selected:
            raise ValueError("SFT preparation receipt does not match the selected dataset variant")
