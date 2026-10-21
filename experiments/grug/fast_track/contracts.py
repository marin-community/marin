# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable data contracts for fast-track runs."""

import hashlib
from dataclasses import dataclass

from levanter.data.text._batch_tokenizer import BatchTokenizer
from levanter.tokenizers import MarinTokenizer, tokenizer_content_hash
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep
from marin.experiment.namespacing import user_namespaced_name

TOKENIZATION_POLICY = "fast-track-long-string-v1"
TOKENIZATION_CHUNK_CHARS = 10_000
TOKENIZATION_MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
STEP_NAME_DIGEST_CHARS = 20
# Frozen fast-track inputs that are not executor steps: task copies, probe tables, and fixed models.
FAST_TRACK_ROOT = "s3://marin-us-east-02a/marin/fast-track"


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
    digest = hashlib.sha256(path.encode()).hexdigest()[:STEP_NAME_DIGEST_CHARS]
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
