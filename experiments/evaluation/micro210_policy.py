# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a model to the content-pinned Micro210 evaluation protocol."""

import hashlib
import json
import re
from dataclasses import dataclass

from rigging.filesystem.storage_path import StoragePath

MODEL_IDENTITY_PREFIX = "sha256:"
POLICY_DIGEST = "e0faa32c68994e8bf01d0ab7177893aa9c29f8c4309b9183de5c3dfcb6fa67aa"
POLICY_REVISION = "2026.11.18"


@dataclass(frozen=True)
class Micro210PolicyReference:
    uri: str
    revision: str
    sha256: str
    byte_size: int


POLICY_REFERENCE = Micro210PolicyReference(
    uri=(
        "s3://marin-us-east-08a/marin/users/benfeuer/evaluation-policies/micro210/"
        f"{POLICY_REVISION}/{POLICY_DIGEST}/policy.json"
    ),
    revision=POLICY_REVISION,
    sha256=POLICY_DIGEST,
    byte_size=122243,
)


@dataclass(frozen=True)
class Micro210Model:
    """Immutable model and tokenizer identities, with no generation settings."""

    uri: str
    identity: str
    tokenizer_uri: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        if re.fullmatch(re.escape(MODEL_IDENTITY_PREFIX) + r"[0-9a-f]{64}", self.identity) is None:
            raise ValueError("model identity must pin the export manifest as sha256:<64 lowercase hex characters>")
        if re.fullmatch(r"[0-9a-f]{40}", self.tokenizer_revision) is None:
            raise ValueError("tokenizer_revision must be an immutable 40-character commit SHA")


@dataclass(frozen=True)
class Micro210Record:
    """Serializable identity of the model, protocol and evaluation outputs."""

    canonical_spec_sha256: str
    model: Micro210Model
    run_id: str
    output_uri: str
    policy_reference: Micro210PolicyReference


@dataclass(frozen=True)
class Micro210Evaluation:
    """A model binding and output location under the frozen protocol."""

    model: Micro210Model
    run_id: str
    output_uri: str

    def policy(self) -> dict:
        raw = StoragePath(POLICY_REFERENCE.uri).read_bytes()
        if len(raw) != POLICY_REFERENCE.byte_size or hashlib.sha256(raw).hexdigest() != POLICY_REFERENCE.sha256:
            raise ValueError("Canonical Micro210 specification hash mismatch")
        return json.loads(raw)

    def record(self) -> Micro210Record:
        self.policy()
        return Micro210Record(POLICY_REFERENCE.sha256, self.model, self.run_id, self.output_uri, POLICY_REFERENCE)


def micro210_evaluation(model: Micro210Model, *, run_id: str, output_uri: str, **overrides) -> Micro210Evaluation:
    """Construct the canonical evaluation; name the first forbidden override."""
    if overrides:
        raise ValueError(f"Canonical Micro210 refuses override field: {next(iter(overrides))}")
    request = Micro210Evaluation(model, run_id, output_uri)
    request.policy()
    return request
