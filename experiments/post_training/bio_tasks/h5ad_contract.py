# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Declared identities and transformation for a sparse H5AD artifact.

The byte-independent count digest starts with BIOH5AD_COUNTS_V1 followed by a
NUL byte, then little-endian uint64 shape and nonzero count, length-prefixed
UTF-8 cell and feature IDs, and row-major uint64 (row, column, count) triples.
Optional NumPy/HDF5 decoding lives in h5ad_verifier.
"""

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

HASH_PREFIX = b"BIOH5AD_COUNTS_V1\0"
MAX_IDENTIFIER_BYTES = 1024
MAX_METADATA_STRING_BYTES = 4096
MAX_INTEGER = 2**63 - 1


class H5adNormalizationContract(BaseModel):
    """Exact biological identities/counts and the declared per-cell transform."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    cell_ids: list[str] = Field(min_length=1, max_length=100000)
    feature_ids: list[str] = Field(min_length=1, max_length=250000)
    nonzeros: int = Field(gt=0, le=100000000)
    counts_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligible_gene_reads: list[int]
    target_sum: float = Field(gt=0, allow_inf_nan=False)
    atol: float = Field(ge=0, allow_inf_nan=False)
    rtol: float = Field(ge=0, allow_inf_nan=False)
    max_bytes: int = Field(gt=0, le=1024 * 1024 * 1024)
    max_decoded_bytes: int = Field(gt=0, le=2 * 1024 * 1024 * 1024)
    obs_metadata: dict[str, list[str | int]] = Field(default_factory=dict)
    var_metadata: dict[str, list[str | int]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference(self) -> "H5adNormalizationContract":
        for identifiers in (self.cell_ids, self.feature_ids):
            validate_identifiers(identifiers)
        if self.nonzeros > len(self.cell_ids) * len(self.feature_ids):
            raise ValueError("H5AD nonzero count exceeds its shape")
        if len(self.eligible_gene_reads) != len(self.cell_ids) or any(
            not 0 < value <= MAX_INTEGER for value in self.eligible_gene_reads
        ):
            raise ValueError("H5AD normalization denominators must be positive integer cell totals")
        if not math.isfinite(self.atol + self.rtol * math.log1p(self.target_sum)):
            raise ValueError("Nonfinite H5AD tolerance")
        for metadata, identifiers in ((self.obs_metadata, self.cell_ids), (self.var_metadata, self.feature_ids)):
            if len(metadata) > 32:
                raise ValueError("Too many required H5AD metadata columns")
            for name, values in metadata.items():
                if not name or "/" in name or name in {".", ".."} or len(values) != len(identifiers):
                    raise ValueError("Invalid required H5AD metadata column")
                if not (all(type(value) is str for value in values) or all(type(value) is int for value in values)):
                    raise ValueError("H5AD metadata columns must be uniformly strings or integers")
                for value in values:
                    if isinstance(value, str) and ("\0" in value or len(value.encode()) > MAX_METADATA_STRING_BYTES):
                        raise ValueError("Oversized H5AD metadata string")
                    if isinstance(value, int) and not -MAX_INTEGER <= value <= MAX_INTEGER:
                        raise ValueError("Oversized H5AD metadata integer")
        return self


def validate_identifiers(identifiers: Sequence[str]) -> None:
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("h5ad_duplicate_or_empty_ids")
    if any(
        not identifier or "\0" in identifier or len(identifier.encode()) > MAX_IDENTIFIER_BYTES
        for identifier in identifiers
    ):
        raise ValueError("h5ad_invalid_ids")
