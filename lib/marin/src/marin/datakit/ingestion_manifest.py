# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed source manifests for small ingestion registries and sidecar metadata."""

from __future__ import annotations

import hashlib
import json
import posixpath
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from rigging.filesystem import open_url
from zephyr.writers import atomic_rename

from marin.utils import fsspec_mkdirs

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
INGESTION_METADATA_SCHEMA_VERSION = 1


def _json_ready(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


class UsagePolicy(StrEnum):
    """Policy gate for whether a source is allowed in training or eval."""

    TRAINING_ALLOWED = "training_allowed"
    EVAL_ONLY = "eval_only"
    BLOCKED = "blocked"


class IdentityTreatment(StrEnum):
    """How person-identifying strings should be handled during ingestion."""

    PRESERVE = "preserve"
    PSEUDONYMIZE = "pseudonymize"
    DROP = "drop"


class SecretRedaction(StrEnum):
    """Whether source text requires secret redaction before downstream use."""

    NONE = "none"
    REQUIRED = "required"


@dataclass(frozen=True)
class IngestionPolicy:
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


@dataclass(frozen=True)
class SampleCapConfig:
    """Optional source-level sampling caps used by a staging or extraction step."""

    max_bytes_per_source: int | None = None
    max_bytes_per_document: int | None = None
    max_records: int | None = None
    max_files: int | None = None
    max_members: int | None = None
    max_examples: int | None = None


@dataclass(frozen=True)
class StagingMetadata:
    """How a source is rendered into downstream text records.

    This block is the source -> text projection contract. Keep only fields
    that affect or meaningfully describe the emitted text surface here.
    Filesystem plumbing and runtime output details belong in
    ``MaterializedOutputMetadata`` instead.
    """

    transform_name: str
    serializer_name: str | None = None
    split: str | None = None
    subset: str | None = None
    preserve_header: bool | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionSourceManifest:
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
    sample_caps: SampleCapConfig = field(default_factory=SampleCapConfig)
    compressed_size_bytes: int | None = None
    rough_tokens_b: float | None = None
    source_metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = _json_ready(asdict(self))
        payload["policy"]["training_allowed"] = self.policy.training_allowed
        payload["policy"]["eval_only"] = self.policy.eval_only
        return payload

    def content_fingerprint_payload(self) -> dict[str, Any]:
        """Return the subset of manifest fields that can affect staged bytes."""
        return _json_ready(
            {
                "dataset_key": self.dataset_key,
                "slice_key": self.slice_key,
                "source_label": self.source_label,
                "source_urls": self.source_urls,
                "source_format": self.source_format,
                "surface_form": self.surface_form,
                "policy": {
                    "requires_sanitization": self.policy.requires_sanitization,
                    "identity_treatment": self.policy.identity_treatment,
                    "secret_redaction": self.policy.secret_redaction,
                },
                "staging": asdict(self.staging),
                "sample_caps": asdict(self.sample_caps),
                "source_metadata": self.source_metadata,
            }
        )

    def fingerprint(self) -> str:
        """Return the content hash used for cache keys and config validation."""
        blob = json.dumps(self.content_fingerprint_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def provenance_fingerprint(self) -> str:
        """Return an exact hash over the full manifest payload."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MaterializedOutputMetadata:
    """Runtime output metadata for a concrete staging or extraction run."""

    input_path: str
    output_path: str
    output_file: str
    record_count: int
    bytes_written: int
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


def render_ingestion_metadata(
    manifest: IngestionSourceManifest,
    materialized_output: MaterializedOutputMetadata,
) -> dict[str, Any]:
    """Render the sidecar payload for a materialized source."""

    return {
        "schema_version": INGESTION_METADATA_SCHEMA_VERSION,
        "manifest_fingerprint": manifest.provenance_fingerprint(),
        "content_fingerprint": manifest.fingerprint(),
        "source_manifest": manifest.to_dict(),
        "materialized_output": materialized_output.to_dict(),
    }


def write_ingestion_metadata_json(
    *,
    manifest: IngestionSourceManifest,
    materialized_output: MaterializedOutputMetadata,
    metadata_filename: str = "metadata.json",
) -> str:
    """Write ``metadata.json`` for a materialized source and return its path."""

    fsspec_mkdirs(materialized_output.output_path, exist_ok=True)
    metadata_path = posixpath.join(materialized_output.output_path, metadata_filename)
    payload = render_ingestion_metadata(manifest, materialized_output)

    with atomic_rename(metadata_path) as temp_path:
        with open_url(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    return metadata_path
