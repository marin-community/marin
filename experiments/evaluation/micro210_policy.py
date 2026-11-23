# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a model to the content-pinned Micro210 evaluation protocol."""

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass

from rigging.filesystem.storage_path import StoragePath

MODEL_IDENTITY_PREFIX = "sha256:"
LEGACY_POLICY_DIGEST = "e0faa32c68994e8bf01d0ab7177893aa9c29f8c4309b9183de5c3dfcb6fa67aa"
LEGACY_POLICY_REVISION = "2026.11.18"
LEGACY_POLICY_COMMIT = "6ae1211cf3aa80beb0ca364b171f149a9506c05a"

POLICY_REPO = "open-athena/micro210-policy"
POLICY_DIGEST = "8f7ef8170fbef2880214cbf03f70d80cffa3114276e024389617777bf07f53e5"
POLICY_VERSION = "2026.11.20"
POLICY_REVISION = f"{POLICY_VERSION}-together"
POLICY_COMMIT = "fa38e3eacc25137c9febe898119f08d65cc5ce88"

# The prepared table re-hosts benchmark items that must not be public, so it lives in a
# private repository that CoreWeave tasks read with the standard HF_TOKEN.
PREPARED_REPO = "open-athena/micro210-prepared-tasks"
PREPARED_COMMIT = "d8fa0e259ae63b1978388b20a17f72036642dd05"


HARBOR_RUNTIME_FIELD_RENAMES = (
    ("daytona_provisioning", "daytona_image_build"),
    ("eval_daytona_provisioning", "eval_daytona_image_build"),
)


@dataclass(frozen=True)
class Micro210RuntimeConfig:
    config: dict
    field_renames: dict[str, str]


def runtime_policy_config(policy: dict) -> Micro210RuntimeConfig:
    """Translate only the approved Harbor names, preserving the frozen policy."""
    config = deepcopy(policy["evaluation"]["config"])
    harbor = config["terminal_bench"]["harbor"]
    applied = {}
    for previous, current in HARBOR_RUNTIME_FIELD_RENAMES:
        if previous not in harbor:
            continue
        if current in harbor:
            raise ValueError(f"Canonical Micro210 has ambiguous Harbor fields: {previous}, {current}")
        harbor[current] = harbor.pop(previous)
        applied[f"terminal_bench.harbor.{previous}"] = f"terminal_bench.harbor.{current}"
    return Micro210RuntimeConfig(config, applied)


@dataclass(frozen=True)
class Micro210PolicyReference:
    uri: str
    revision: str
    sha256: str
    byte_size: int


# Human-readable copy (sha256: e0faa32c68994e8bf01d0ab7177893aa9c29f8c4309b9183de5c3dfcb6fa67aa): https://gist.github.com/penfever/b9b7be780c14969634e897621454deef/0b5add0ef2843418eaded61258a91f51f3b1b69a
LEGACY_POLICY_REFERENCE = Micro210PolicyReference(
    uri=f"hf://datasets/{POLICY_REPO}@{LEGACY_POLICY_COMMIT}/policy.json",
    revision=LEGACY_POLICY_REVISION,
    sha256=LEGACY_POLICY_DIGEST,
    byte_size=122243,
)


POLICY_REFERENCE = Micro210PolicyReference(
    uri=f"hf://datasets/{POLICY_REPO}@{POLICY_COMMIT}/policy.json",
    revision=POLICY_REVISION,
    sha256=POLICY_DIGEST,
    byte_size=121292,
)

# The staged-in-regional-storage new version is re-added with its content digest after
# the prepared table build; the default stays on the HF pin until the new policy and
# private prepared table have immutable HF commit revisions.
SPEC_POLICY_VERSION = "2026.11.23"
SPEC_POLICY_REVISION = f"{SPEC_POLICY_VERSION}-together"


# The prepared table re-hosts benchmark items that must not be public, so it lives in a
# private repository that CoreWeave tasks read with the standard HF_TOKEN.
PREPARED_REPO = "open-athena/micro210-prepared-tasks"
PREPARED_COMMIT = "d8fa0e259ae63b1978388b20a17f72036642dd05"


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
        return read_micro210_policy(POLICY_REFERENCE)

    def record(self) -> Micro210Record:
        self.policy()
        return Micro210Record(POLICY_REFERENCE.sha256, self.model, self.run_id, self.output_uri, POLICY_REFERENCE)


def micro210_evaluation(model: Micro210Model, *, run_id: str, output_uri: str, **overrides) -> Micro210Evaluation:
    """Verify the frozen policy and bind the model and run identity."""
    if overrides:
        raise ValueError(f"Canonical Micro210 refuses override field: {next(iter(overrides))}")
    request = Micro210Evaluation(model, run_id, output_uri)
    request.policy()
    return request


def read_micro210_policy(reference: Micro210PolicyReference) -> dict:
    """Read and verify a policy from a current or historical evaluation record."""
    raw = StoragePath(reference.uri).read_bytes()
    if len(raw) != reference.byte_size or hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError("Canonical Micro210 specification hash mismatch")
    return json.loads(raw)
