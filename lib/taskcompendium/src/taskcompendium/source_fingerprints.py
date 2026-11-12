# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned source identity and advisory overlap records for catalog ingestion."""

from enum import StrEnum
from typing import Any, Literal

from finestore.rl.source_identity import SourceId, canonical_source_map, original_source_id
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceLineage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    revision: str


class FingerprintSample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    task_sha256: str
    instruction_sha256: str
    instruction: str
    instruction_characters: int
    reference_sha256: str | None
    tests_sha256: str | None


class SourceFingerprint(BaseModel):
    """A census sketch plus a deterministic, bounded sample for human review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    seed: int
    task_count: int = Field(gt=0)
    within_source_duplicates: int = Field(ge=0)
    sample_sha256: str
    sample: list[FingerprintSample]
    instruction_minhash: list[int]
    instruction_hashes: list[str]
    reference_hashes: list[str]
    test_hashes: list[str]
    task_id_hashes: list[str]
    url_hashes: list[str]
    lineage: list[SourceLineage]
    inputs: dict[str, str]


class PairSimilarity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    left: SourceId
    right: SourceId
    rank_score: float = Field(ge=0, le=1)
    instruction_similarity: float = Field(ge=0, le=1)
    instruction_overlap: float = Field(ge=0, le=1)
    reference_overlap: float = Field(ge=0, le=1)
    test_overlap: float = Field(ge=0, le=1)
    task_id_overlap: float = Field(ge=0, le=1)
    url_overlap: float = Field(ge=0, le=1)
    shared_lineage: bool


class DuplicateLabel(StrEnum):
    DUPLICATE = "duplicate"
    PARTIAL_OVERLAP = "partial_overlap"
    DISTINCT = "distinct"


class PairVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: DuplicateLabel
    overlap_fraction: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_label_fraction(self) -> "PairVerdict":
        if self.label == DuplicateLabel.DISTINCT and self.overlap_fraction != 0:
            raise ValueError("Distinct sources have zero estimated overlap")
        if self.label == DuplicateLabel.DUPLICATE and self.overlap_fraction != 1:
            raise ValueError("Duplicate populations have complete estimated overlap")
        if self.label == DuplicateLabel.PARTIAL_OVERLAP and self.overlap_fraction == 0:
            raise ValueError("Partial overlap requires a positive estimate")
        return self


class DuplicateReview(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    similarity: PairSimilarity
    verdict: PairVerdict
    prompt_sha256: str
    model: str
    completion_id: str
    prompt_tokens: int
    completion_tokens: int
    acknowledgement: str | None = None


class DuplicateGroupMember(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    overlap_with_canonical: float | None


class DuplicateGroup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    group_id: str
    canonical_source_id: SourceId
    members: list[DuplicateGroupMember]
    advisory: Literal[True] = True


class DuplicateJudgeRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    revision: str
    prompt_version: str
    prompt_sha256: str
    temperature: float = Field(ge=0.7, le=0.7, strict=True)
    top_k: int
    labelled_controls: int
    labelled_controls_passed: int
    route: Literal["hosted", "local"]
    serving_jobs: list[str]
    lease_seconds_upper_bound: float | None


class SourceIdentitySurvey(BaseModel):
    """Source identities and overlap opinions selected with one catalog snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    catalog_root: str
    snapshot_id: str
    generated_at: str
    fingerprints: dict[str, SourceFingerprint | None]
    reviews: list[DuplicateReview]
    groups: list[DuplicateGroup]
    judge: DuplicateJudgeRun | None

    def identity_payload(self) -> dict:
        """Keep survey and cached-review identities tied to the original source labels."""
        document = self.model_dump(mode="json")
        document["fingerprints"] = {original_source_id(key): value for key, value in document["fingerprints"].items()}
        for review in document["reviews"]:
            similarity = review["similarity"]
            for side in ("left", "right"):
                similarity[side] = original_source_id(similarity[side])
        for group in document["groups"]:
            group["canonical_source_id"] = original_source_id(group["canonical_source_id"])
            for member in group["members"]:
                member["source_id"] = original_source_id(member["source_id"])
        return document

    @field_validator("fingerprints", mode="before")
    @classmethod
    def _source_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        return canonical_source_map(value)

    @model_validator(mode="after")
    def validate_references(self) -> "SourceIdentitySurvey":
        sources = set(self.fingerprints)
        pairs = [(row.similarity.left, row.similarity.right) for row in self.reviews]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate source-pair reviews")
        for left, right in pairs:
            if left == right or left not in sources or right not in sources:
                raise ValueError("Pair reviews must reference distinct inventory sources")
        members = [member.source_id for group in self.groups for member in group.members]
        if len(set(members)) != len(members) or not set(members).issubset(sources):
            raise ValueError("Duplicate groups must partition inventory members")
        for group in self.groups:
            if group.canonical_source_id not in {member.source_id for member in group.members}:
                raise ValueError("Duplicate group canonical source must be a member")
        return self
