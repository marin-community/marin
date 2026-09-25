# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Semantic verification of complete BAM records and their coordinate index."""

import hashlib
import json
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BamRecord(Protocol):
    is_unmapped: bool
    reference_id: int
    reference_start: int
    reference_end: int | None

    def to_string(self) -> str: ...


class BamContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    references: dict[str, int]
    record_digests: dict[str, int]
    records: int = Field(gt=0, le=1_000_000)
    max_bytes: int = Field(gt=0, le=128 * 1024 * 1024)
    index_max_bytes: int = Field(gt=0, le=16 * 1024 * 1024)

    @model_validator(mode="after")
    def valid_reference(self) -> "BamContract":
        if not self.references or any(not name or length <= 0 for name, length in self.references.items()):
            raise ValueError("bam_reference_shape")
        if any(
            len(key) != 64 or any(c not in "0123456789abcdef" for c in key) or count <= 0
            for key, count in self.record_digests.items()
        ):
            raise ValueError("bam_record_digest_shape")
        if sum(self.record_digests.values()) != self.records:
            raise ValueError("bam_record_count")
        return self


def record_digest(record: BamRecord) -> str:
    """Hash every SAM alignment field, independent of BAM compression and tag order."""
    # Pysam's SAM rendering preserves field values, while explicit tag sorting
    # removes a representation detail that does not change the alignment.
    fields = record.to_string().split("\t")
    if len(fields) < 11:
        raise ValueError("bam_record_fields")
    core = fields[:11]
    tags = sorted(fields[11:])
    serialized = json.dumps((core, tags), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def summarize_bam(
    path: Path, max_bytes: int, index_max_bytes: int, max_records: int
) -> tuple[dict[str, int], Counter[str], int]:
    """Decode all records and exercise the BAI on fixed reference intervals."""
    import pysam  # noqa: PLC0415 — installed only for tasks requiring BAM decoding.

    index = path.with_name(path.name + ".bai")
    if path.is_symlink() or index.is_symlink() or not path.is_file() or not index.is_file():
        raise ValueError("bam_or_index_missing_or_nonregular")
    if path.stat().st_size > max_bytes or index.stat().st_size > index_max_bytes:
        raise ValueError("bam_or_index_too_large")
    if not 0 < max_records <= 1_000_000:
        raise ValueError("bam_record_limit")
    with pysam.AlignmentFile(path, "rb", check_sq=True) as bam:
        if not bam.has_index():
            raise ValueError("bam_index_unreadable")
        references = dict(zip(bam.references, bam.lengths, strict=True))
        if not references:
            raise ValueError("bam_missing_reference")
        digests: Counter[str] = Counter()
        interval_records: dict[tuple[str, int, int], Counter[str]] = {}
        for name, length in references.items():
            boundaries = sorted({0, length // 4, length // 2, 3 * length // 4, length})
            for start, stop in pairwise(boundaries):
                if start < stop:
                    interval_records[(name, start, stop)] = Counter()
        count = 0
        for record in bam.fetch(until_eof=True):
            count += 1
            if count > max_records:
                raise ValueError("bam_too_many_records")
            digest = record_digest(record)
            digests[digest] += 1
            if record.reference_id < 0 or record.reference_start < 0:
                continue
            name = bam.get_reference_name(record.reference_id)
            end = record.reference_end
            if end is None:
                if not record.is_unmapped:
                    raise ValueError("bam_mapped_record_without_reference_span")
                # HTSlib indexes an unmapped record retaining RNAME/POS as a
                # one-base placed interval; completely unplaced reads stay out.
                end = record.reference_start + 1
            for ref_name, start, stop in interval_records:
                if name == ref_name and record.reference_start < stop and end > start:
                    interval_records[(ref_name, start, stop)][digest] += 1
        for (name, start, stop), expected in interval_records.items():
            observed = Counter()
            for fetched, record in enumerate(bam.fetch(name, start, stop), 1):
                if fetched > max_records:
                    raise ValueError("bam_index_too_many_records")
                observed[record_digest(record)] += 1
            if observed != expected:
                raise ValueError("bam_index_fetch_mismatch")
    return references, digests, count


def check_bam(path: Path, target: BamContract) -> dict:
    """Compare every native record and ensure the BAI retrieves the same intervals."""
    try:
        references, digests, count = summarize_bam(path, target.max_bytes, target.index_max_bytes, target.records)
    except (OSError, ValueError, KeyError) as error:
        return {"passed": False, "index_passed": False, "reason": str(error)[:160]}
    return {
        "passed": references == target.references and digests == Counter(target.record_digests),
        "index_passed": True,
        "records": count,
        "references": len(references),
        "different_record_types": len(digests),
    }
