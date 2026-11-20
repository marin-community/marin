# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project method tags from each dataset variant's own preparation evidence."""

import hashlib
from typing import Any, Literal, Self

from finestore.sft import DatasetVariantInput, SftEvidenceReference, SftPreparationReceipt, SftRecord, Sha256
from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.storage_path import StoragePath

from .source_identity import canonical_source_id


class ManifestReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    uri: str
    sha256: str


def manifest_bytes(reference: ManifestReference) -> bytes:
    payload = StoragePath(reference.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError(f"Pinned artifact hash differs: {reference.uri}")
    return payload


class VariantIdentity(SftRecord):
    source_id: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    repository: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    data_sha256: Sha256 | None = None
    task_specs_sha256: Sha256 | None = None


class ConversionLineage(SftRecord):
    parent: DatasetVariantInput
    output: DatasetVariantInput
    receipt: SftEvidenceReference


class DatasetVariant(SftRecord):
    input: VariantIdentity
    tags: frozenset[Literal["rl", "sft"]] = frozenset()
    sft_receipt: SftPreparationReceipt | None = None
    sft_receipt_reference: SftEvidenceReference | None = None
    conversion: ConversionLineage | None = None

    @model_validator(mode="after")
    def require_own_preparation(self) -> Self:
        if "sft" in self.tags:
            if self.sft_receipt is None or self.sft_receipt_reference is None:
                raise ValueError("SFT tag requires a completed Datakit preparation receipt")
            self.sft_receipt.require_input(DatasetVariantInput.model_validate(self.input.model_dump()))
        elif self.sft_receipt is not None or self.sft_receipt_reference is not None:
            raise ValueError("SFT preparation evidence requires its variant's SFT tag")
        if self.conversion is not None:
            if self.conversion.output.model_dump() != self.input.model_dump():
                raise ValueError("Conversion lineage belongs to another output variant")
            if self.conversion.parent.variant_id == self.input.variant_id:
                raise ValueError("Conversion must create a distinct variant")
        return self


def rl_variant(declaration: dict[str, Any]) -> DatasetVariant:
    census = declaration.get("actor_census_input") or {}
    return DatasetVariant(
        input=VariantIdentity(
            source_id=declaration["id"],
            variant_id="rl-environment",
            repository=census.get("dataset_id") or declaration.get("dataset_id"),
            revision=census.get("dataset_revision") or declaration.get("dataset_revision"),
            task_specs_sha256=census.get("task_specs_sha256"),
        ),
        tags=frozenset({"rl"}),
    )


def variant_projection(declaration: dict[str, Any], records: list[dict] | None) -> dict[str, Any]:
    variants = (
        [DatasetVariant.model_validate(value) for value in records] if records is not None else [rl_variant(declaration)]
    )
    ids = [variant.input.variant_id for variant in variants]
    if len(ids) != len(set(ids)) or any(
        canonical_source_id(variant.input.source_id) != canonical_source_id(declaration["id"]) for variant in variants
    ):
        raise ValueError("Dataset variants must have unique IDs and belong to their source")
    return {
        "method_tags": sorted({tag for variant in variants for tag in variant.tags}),
        "method_variants": [variant.model_dump(mode="json") for variant in variants],
    }


def require_receipt_bytes(variant: DatasetVariant, raw: bytes) -> None:
    """Bind the catalog's receipt projection to the producer's immutable bytes."""
    reference = variant.sft_receipt_reference
    if reference is None or hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError("SFT preparation receipt digest mismatch")
    if SftPreparationReceipt.model_validate_json(raw) != variant.sft_receipt:
        raise ValueError("SFT preparation receipt differs from the variant evidence")
