# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable data contracts for fast-track runs."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from marin.execution.artifact import Artifact
from marin.execution.fingerprint import register_fingerprint
from marin.execution.lazy import ArtifactStep
from pydantic import BaseModel, ConfigDict, model_validator

TRAIN_SPLIT = "train"
TOKENIZATION_POLICY = "fast-track-long-string-v1"
TOKENIZATION_CHUNK_CHARS = 10_000
TOKENIZATION_MAX_DOCUMENT_BYTES = 32 * 1024 * 1024


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
        """Return the number of tokens in the resolved run."""
        return self.batch_size * self.num_steps * self.sequence_length


class AddDatasetSamplingPolicy(StrEnum):
    """Sampling policy for the prepared Hugging Face token cache."""

    PREFIX = "prefix"


class DatasetPrefix(BaseModel):
    """Immutable source and bounds for one HF dataset prefix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repo: str
    revision: str
    subset: str | None
    split: str
    text_field: str
    tokenizer: str
    tokenizer_hash: str
    sampling_policy: AddDatasetSamplingPolicy
    max_rows: int
    max_overshoot_tokens: int
    requested_token_cap: int

    @model_validator(mode="after")
    def validate_prefix(self) -> "DatasetPrefix":
        if len(self.revision) not in {40, 64} or any(char not in "0123456789abcdef" for char in self.revision):
            raise ValueError("revision must be an immutable Hugging Face commit hash")
        if not self.repo or not self.split or not self.text_field or not self.tokenizer or not self.tokenizer_hash:
            raise ValueError("repo, split, text field, tokenizer, and tokenizer hash must be non-empty")
        if self.max_rows < 1 or self.max_overshoot_tokens < 0 or self.requested_token_cap < 1:
            raise ValueError("prefix row and token limits are invalid")
        return self


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
    prefix: DatasetPrefix
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
    if not 0 < fraction < 1:
        raise ValueError("add-dataset fraction must be greater than 0 and less than 1")
    if not baseline_weights or any(not math.isfinite(weight) or weight < 0 for weight in baseline_weights.values()):
        raise ValueError("baseline weights must be finite, non-negative, and non-empty")
    total_weight = sum(baseline_weights.values())
    if total_weight <= 0:
        raise ValueError("baseline weights must have a positive total")
    if new_component in baseline_weights:
        raise ValueError(f"new dataset component {new_component!r} already exists in baseline weights")
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
    if target_production_tokens < 1 or fast_track_budget < 1 or available_unique_tokens < 1 or sequence_length < 1:
        raise ValueError("token counts and sequence length must be positive")
    if fast_track_budget > target_production_tokens:
        raise ValueError("fast-track token budget must not exceed the target production token budget")
    if not 0 < fraction < 1:
        raise ValueError("add-dataset fraction must be greater than 0 and less than 1")
    share_limit = math.floor(fraction * fast_track_budget)
    scaled_availability_limit = available_unique_tokens * fast_track_budget // target_production_tokens
    raw_cap = min(share_limit, scaled_availability_limit)
    return raw_cap // sequence_length * sequence_length
