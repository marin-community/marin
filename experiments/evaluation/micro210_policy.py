# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a model to the content-pinned Micro210 evaluation protocol."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SPEC_PATH = Path(__file__).with_name("micro210-policy.json")
SPEC_SHA256 = "13cebe0fc688fa906dc0490e6db269a4b2c2f311289deb10d07a2690dc9e9f0d"


@dataclass(frozen=True)
class Micro210Model:
    """Immutable model and tokenizer identities, with no generation settings."""

    uri: str
    identity: str
    tokenizer_uri: str
    tokenizer_revision: str


@dataclass(frozen=True)
class Micro210Evaluation:
    """A model binding and run address under the frozen protocol."""

    model: Micro210Model
    run_id: str
    output_uri: str

    def policy(self) -> dict:
        raw = SPEC_PATH.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SPEC_SHA256:
            raise ValueError("Canonical Micro210 specification hash mismatch")
        return json.loads(raw)

    def record(self) -> dict:
        self.policy()
        return {
            "canonical_spec_sha256": SPEC_SHA256,
            "run_id": self.run_id,
            "output_uri": self.output_uri,
            "model": {
                "uri": self.model.uri,
                "identity": self.model.identity,
                "tokenizer_uri": self.model.tokenizer_uri,
                "tokenizer_revision": self.model.tokenizer_revision,
            },
        }


def micro210_evaluation(model: Micro210Model, *, run_id: str, output_uri: str, **overrides) -> Micro210Evaluation:
    """Construct the canonical evaluation; report any attempted override by name."""
    if overrides:
        raise ValueError(f"Canonical Micro210 refuses override field: {next(iter(overrides))}")
    request = Micro210Evaluation(model, run_id, output_uri)
    request.policy()
    return request
