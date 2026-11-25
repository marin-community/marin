# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable reference scores and separately versioned acceptance policies."""

import hashlib
import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.buckets import filesystem_for


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ModelIdentity(Record):
    weights: str
    revision: str
    config_sha256: str
    tokenizer: str
    tokenizer_revision: str
    eos_token_id: int


class Artifact(Record):
    uri: str
    sha256: str


class ReferenceCase(Record):
    id: str
    tokens: tuple[int, ...] = Field(min_length=2)
    # Position p is the distribution after tokens[:p + 1].
    positions: tuple[int, ...] = Field(min_length=1)
    group: str = "representative"


class ScoreRow(Record):
    case_id: str
    position: int
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    history_sha256: str


class ReferenceManifest(Record):
    schema_version: int = 1
    model: ModelIdentity
    suite_id: str
    cases: Artifact
    scores: Artifact
    reference_source: str
    reference_commit: str
    reference_config: dict
    lineage: dict
    raw_artifacts: tuple[Artifact, ...] = ()


class ErrorLimits(Record):
    """Bounds on per-position maximum absolute probability error, in [0, 1]."""

    maximum: float = Field(ge=0, le=1)
    p99: float = Field(ge=0, le=1)
    p90: float = Field(ge=0, le=1)
    fraction_above_0075: float = Field(ge=0, le=1)


class AcceptancePolicy(Record):
    id: str
    provisional: bool
    rationale: str
    evidence: tuple[str, ...]
    hardware: tuple[str, ...]
    dtype: str = "bfloat16"
    quantization: str | None = None
    kv_cache_dtypes: tuple[str, ...] = ("auto", "bfloat16")
    reference: dict[str, ErrorLimits]
    same_split: ErrorLimits
    different_split: ErrorLimits
    batch: ErrorLimits
    decode: ErrorLimits
    batch_size: int = Field(default=4, ge=2)
    decode_tokens: int = Field(default=32, ge=2)
    cache_decode_tokens: int = Field(default=257, ge=2)
    boundary_lengths: tuple[int, ...] = (127, 128, 129, 255, 256, 257, 2047, 2048, 2049)


class ModelFixture(Record):
    model: ModelIdentity
    manifest: Artifact
    policy: AcceptancePolicy


class LookupStatus(StrEnum):
    FOUND = "found"
    MISSING = "missing"
    MISMATCHED = "mismatched"


class ReferenceLookup(Record):
    status: LookupStatus
    reason: str
    manifest: ReferenceManifest | None = None
    cases: tuple[ReferenceCase, ...] = ()
    scores: tuple[ScoreRow, ...] = ()


def read_bytes(uri: str) -> bytes:
    filesystem, path = filesystem_for(uri)
    with filesystem.open(path, "rb") as stream:
        return stream.read()


def verified_bytes(artifact: Artifact) -> bytes:
    raw = read_bytes(artifact.uri)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != artifact.sha256:
        raise ValueError(f"Artifact checksum differs: {artifact.uri}: {actual} != {artifact.sha256}")
    return raw


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def history_digest(tokens: tuple[int, ...]) -> str:
    # JSON fixes byte order and works without NumPy in the HTTP checker.
    return hashlib.sha256(canonical_bytes(tokens)).hexdigest()


def read_fixture(path: str) -> ModelFixture:
    return ModelFixture.model_validate_json(Path(path).read_bytes())


def lookup_reference(fixture: ModelFixture) -> ReferenceLookup:
    """Read saved scores; distinguish absent objects from contradictory evidence."""
    try:
        manifest = ReferenceManifest.model_validate_json(verified_bytes(fixture.manifest))
        if manifest.model != fixture.model:
            raise ValueError("The reference manifest identifies a different checkpoint or tokenizer")
        cases_raw = verified_bytes(manifest.cases)
        if hashlib.sha256(cases_raw).hexdigest() != manifest.suite_id:
            raise ValueError("The prompt file does not match the immutable suite identity")
        cases = tuple(ReferenceCase.model_validate(row) for row in json.loads(cases_raw))
        scores = tuple(ScoreRow.model_validate(row) for row in json.loads(verified_bytes(manifest.scores)))
        by_id = {case.id: case for case in cases}
        if len(by_id) != len(cases):
            raise ValueError("The prompt suite contains duplicate case IDs")
        expected = {(case.id, position) for case in cases for position in case.positions}
        found = set()
        for row in scores:
            key = (row.case_id, row.position)
            if key in found or key not in expected:
                raise ValueError(f"Unexpected or duplicate reference row: {key}")
            case = by_id[row.case_id]
            if not 0 <= row.position < len(case.tokens):
                raise ValueError(f"Reference position falls outside its history: {key}")
            if row.history_sha256 != history_digest(case.tokens[: row.position + 1]):
                raise ValueError(f"Reference score belongs to another token history: {key}")
            if not row.token_ids or len(row.token_ids) != len(row.logprobs):
                raise ValueError(f"Missing reference token probabilities: {key}")
            if len(set(row.token_ids)) != len(row.token_ids) or any(value > 0 for value in row.logprobs):
                raise ValueError(f"Invalid reference token probabilities: {key}")
            found.add(key)
        if not expected or found != expected:
            raise ValueError(f"Reference coverage differs: {len(found)}/{len(expected)} rows")
        if {case.group for case in cases} - fixture.policy.reference.keys():
            raise ValueError("The acceptance policy omits a reference group")
        return ReferenceLookup(
            status=LookupStatus.FOUND,
            reason="Checkpoint, prompts, scores and hashes match",
            manifest=manifest,
            cases=cases,
            scores=scores,
        )
    except FileNotFoundError as error:
        return ReferenceLookup(status=LookupStatus.MISSING, reason=str(error))
    except ValueError as error:
        return ReferenceLookup(status=LookupStatus.MISMATCHED, reason=str(error))
