# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable data contracts for fast-track runs."""

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.tokenizers import MarinTokenizer, tokenizer_content_hash
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json, register_fingerprint
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_namespaced_name
from pydantic import BaseModel, ConfigDict, Field, field_validator

TRAIN_SPLIT = "train"
TOKENIZATION_POLICY = "fast-track-long-string-v1"
TOKENIZATION_CHUNK_CHARS = 10_000
TOKENIZATION_MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
STEP_NAME_DIGEST_CHARS = 20


def tokenization_policy_identity() -> dict[str, str | int]:
    """Return the tokenization policy fields that every tokenized artifact name includes."""
    return {
        "tokenization_policy": TOKENIZATION_POLICY,
        "tokenization_chunk_chars": TOKENIZATION_CHUNK_CHARS,
        "tokenization_max_document_bytes": TOKENIZATION_MAX_DOCUMENT_BYTES,
    }


def policy_batch_tokenizer(tokenizer: MarinTokenizer, *, text_field: str) -> BatchTokenizer:
    """Return a BOS/EOS tokenizer that splits long strings at the policy chunk size."""
    return BatchTokenizer(
        tokenizer,
        text_field=text_field,
        enforce_bos=True,
        enforce_eos=True,
        long_string_workaround=True,
        _workaround_len=TOKENIZATION_CHUNK_CHARS,
    )


@dataclass(frozen=True)
class ContentStepName:
    """A step name that a hash of its identity payload sets, and its resolved version."""

    digest: str
    name: str
    version: str


def content_step_name(
    kind: str,
    identity: object,
    version: str | None,
    *,
    digest_chars: int = STEP_NAME_DIGEST_CHARS,
) -> ContentStepName:
    """Name a step ``fast-track/<kind>/<digest>`` from the canonical JSON of ``identity``."""
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:digest_chars]
    name = f"fast-track/{kind}/{digest}"
    resolved_version = resolve_version(name, version)
    return ContentStepName(digest, user_namespaced_name(name, resolved_version), resolved_version)


@dataclass(frozen=True)
class TokenizerSpec:
    """Tokenizer location and cache identity, not a revision enforced by the loader."""

    name: str
    identity: str


def require_tokenizer_content(name: str, expected_hash: str, *, owner: str) -> None:
    """Fail when the files that the loader stages for ``name`` differ from ``expected_hash``."""
    actual_hash = tokenizer_content_hash(name)
    if actual_hash != expected_hash:
        raise ValueError(f"{owner}: tokenizer {name!r} content is {actual_hash}, expected {expected_hash}")


def adopt_artifact_path[A: Artifact](path: str, *, kind: type[A]) -> ArtifactStep[A]:
    """Adopt a completed artifact by path, with a name that the path identifies."""
    digest = hashlib.sha256(path.encode()).hexdigest()[:20]
    artifact_name = f"fast-track/adopted/{kind.__name__}/{digest}"
    version = resolve_version(artifact_name, None)
    return ArtifactStep.adopt(user_namespaced_name(artifact_name, version), version, path, kind=kind)


@dataclass(frozen=True)
class FrozenBaselineComponent:
    """One fixed cache in the baseline mixture."""

    name: str
    cache_dir: str
    weight: float


@dataclass(frozen=True)
class FrozenBaselineManifest:
    """Fixed tokenizer and cache mixture for a baseline run."""

    tokenizer: str
    components: tuple[FrozenBaselineComponent, ...]


@dataclass(frozen=True)
class ResolvedTrainingBudget:
    """Resolved batch, step, and sequence counts for a fast-track run."""

    batch_size: int
    num_steps: int
    sequence_length: int

    @property
    def token_count(self) -> int:
        return self.batch_size * self.num_steps * self.sequence_length


class DatasetPrefix(BaseModel):
    """Immutable source and bounds for the first rows of one HF dataset split."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str = Field(min_length=1)
    revision: str
    subset: str | None
    split: str = Field(min_length=1)
    text_field: str = Field(min_length=1)
    tokenizer: str = Field(min_length=1)
    tokenizer_hash: str = Field(min_length=1)
    max_rows: int = Field(ge=1)
    requested_token_cap: int = Field(ge=1)

    @field_validator("revision")
    @classmethod
    def _immutable_revision(cls, revision: str) -> str:
        if len(revision) not in {40, 64} or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("revision must be an immutable Hugging Face commit hash")
        return revision


register_fingerprint(DatasetPrefix, lambda value: value.model_dump(mode="json"))


class PreparedAddDatasetCache(Artifact):
    """A token cache with the source and measured prefix recorded in its artifact."""

    cache_dir: str
    prefix: DatasetPrefix
    actual_num_rows: int
    actual_num_tokens: int
    tokenization_policy: str


@dataclass(frozen=True)
class AddDatasetConfig:
    """Identity and exposure limits for one prepared Hugging Face cache."""

    token_cache: ArtifactStep[PreparedAddDatasetCache]
    fraction: float
    target_production_tokens: int
    available_unique_tokens: int

    def __post_init__(self) -> None:
        if not 0 < self.fraction < 1:
            raise ValueError("add-dataset fraction must be greater than 0 and less than 1")
        if self.target_production_tokens < 1 or self.available_unique_tokens < 1:
            raise ValueError("add-dataset token counts must be positive")


def add_dataset_mixture_weights(
    baseline_weights: Mapping[str, float],
    *,
    new_component: str,
    fraction: float,
) -> dict[str, float]:
    """Normalize the frozen mix, then assign it the remaining token share."""
    total_weight = sum(baseline_weights.values())
    return {name: weight / total_weight * (1 - fraction) for name, weight in baseline_weights.items()} | {
        new_component: fraction
    }


def unique_token_sample_cap(
    *,
    target_production_tokens: int,
    fast_track_budget: int,
    available_unique_tokens: int,
    fraction: float,
    sequence_length: int,
) -> int:
    """Return the loader-aligned unique-token cap for a prepared dataset sample.

    Both formula limits round down to tokens. The result rounds down to whole
    sequences and never exceeds either formula limit.
    """
    if fast_track_budget > target_production_tokens:
        raise ValueError("fast-track token budget must not exceed the target production token budget")
    share_limit = math.floor(fraction * fast_track_budget)
    scaled_availability_limit = available_unique_tokens * fast_track_budget // target_production_tokens
    raw_cap = min(share_limit, scaled_availability_limit)
    return raw_cap // sequence_length * sequence_length
