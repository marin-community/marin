# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a model to the content-pinned Micro210 evaluation protocol."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SPEC_PATH = Path(__file__).with_name("micro210-policy.json")
SPEC_SHA256 = "37fd7d1647c7e64294205935780442b4fde8a24df9bf6e11c166826a07b660cb"


@dataclass(frozen=True)
class Micro210Model:
    """Immutable model and tokenizer identities, with no generation settings."""

    uri: str
    identity: str
    tokenizer_uri: str
    tokenizer_revision: str


@dataclass(frozen=True)
class Micro210Record:
    """Serializable identity of the model, protocol and evaluation outputs."""

    canonical_spec_sha256: str
    model: Micro210Model
    run_id: str
    output_uri: str


@dataclass(frozen=True)
class Micro210Evaluation:
    """A model binding and output location under the frozen protocol."""

    model: Micro210Model
    run_id: str
    output_uri: str

    def policy(self) -> dict:
        raw = SPEC_PATH.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SPEC_SHA256:
            raise ValueError("Canonical Micro210 specification hash mismatch")
        return json.loads(raw)

    def record(self) -> Micro210Record:
        self.policy()
        return Micro210Record(SPEC_SHA256, self.model, self.run_id, self.output_uri)


def micro210_evaluation(model: Micro210Model, *, run_id: str, output_uri: str, **overrides) -> Micro210Evaluation:
    """Construct the canonical evaluation; name the first forbidden override."""
    if overrides:
        raise ValueError(f"Canonical Micro210 refuses override field: {next(iter(overrides))}")
    request = Micro210Evaluation(model, run_id, output_uri)
    request.policy()
    return request
