# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed source manifests for ingestion registries."""

import hashlib
import json
from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ManifestModel(BaseModel):
    """Strict base model for ingestion metadata payloads."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class UsagePolicy(StrEnum):
    """Policy gate for whether a source is allowed in training or eval."""

    TRAINING_ALLOWED = auto()
    EVAL_ONLY = auto()
    BLOCKED = auto()


class IdentityTreatment(StrEnum):
    """How person-identifying strings should be handled during ingestion."""

    PRESERVE = auto()
    PSEUDONYMIZE = auto()
    DROP = auto()


class SecretRedaction(StrEnum):
    """Whether source text requires secret redaction before downstream use."""

    NONE = auto()
    REQUIRED = auto()


class IngestionPolicy(ManifestModel):
    """Policy and risk metadata for an ingestible source."""

    usage_policy: UsagePolicy
    use_policy: str
    requires_sanitization: bool = False
    identity_treatment: IdentityTreatment = IdentityTreatment.PRESERVE
    secret_redaction: SecretRedaction = SecretRedaction.NONE
    contamination_risk: str = ""
    provenance_notes: str = ""

    @property
    def training_allowed(self) -> bool:
        return self.usage_policy == UsagePolicy.TRAINING_ALLOWED

    @property
    def eval_only(self) -> bool:
        return self.usage_policy == UsagePolicy.EVAL_ONLY


class SampleCapConfig(ManifestModel):
    """Optional source-level sampling caps used by a staging or extraction step."""

    max_bytes_per_source: int | None = None
    max_bytes_per_document: int | None = None
    max_records: int | None = None
    max_files: int | None = None
    max_members: int | None = None
    max_examples: int | None = None


class StagingMetadata(ManifestModel):
    """How a source is rendered into downstream text records.

    This block is the source -> text projection contract. Keep only fields
    that affect or meaningfully describe the emitted text surface here.
    """

    transform_name: str
    serializer_name: str | None = None
    split: str | None = None
    subset: str | None = None
    preserve_header: bool | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class IngestionSourceManifest(ManifestModel):
    """Typed source manifest for a reusable ingestion registry entry."""

    dataset_key: str
    slice_key: str
    source_label: str
    source_urls: tuple[str, ...]
    source_license: str
    source_format: str
    surface_form: str
    policy: IngestionPolicy
    staging: StagingMetadata
    epic_issue: int | None = None
    issue_numbers: tuple[int, ...] = ()
    sample_caps: SampleCapConfig = Field(default_factory=SampleCapConfig)
    compressed_size_bytes: int | None = None
    rough_tokens_b: float | None = None
    source_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    def content_fingerprint_payload(self) -> dict[str, Any]:
        """Return the subset of manifest fields that can affect staged bytes.

        This payload is for executor hashing and config validation. It excludes
        issue links, descriptive policy text, and other provenance-only fields
        so additive metadata evolution does not force unnecessary rebuilds.
        """
        return {
            "dataset_key": self.dataset_key,
            "slice_key": self.slice_key,
            "source_label": self.source_label,
            "source_urls": list(self.source_urls),
            "source_format": self.source_format,
            "surface_form": self.surface_form,
            "policy": {
                "requires_sanitization": self.policy.requires_sanitization,
                "identity_treatment": str(self.policy.identity_treatment),
                "secret_redaction": str(self.policy.secret_redaction),
            },
            "staging": self.staging.model_dump(mode="json"),
            "sample_caps": self.sample_caps.model_dump(mode="json"),
            "source_metadata": self.source_metadata,
        }

    def fingerprint(self) -> str:
        """Return the content hash used for cache keys and config validation."""
        blob = json.dumps(self.content_fingerprint_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
