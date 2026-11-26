# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a model to the content-pinned Micro210 evaluation protocol."""

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass

from rigging.filesystem.storage_path import StoragePath
from rolloutengine.agents.pi.agent import PI_HARNESS

MODEL_IDENTITY_PREFIX = "sha256:"
LEGACY_POLICY_DIGEST = "e0faa32c68994e8bf01d0ab7177893aa9c29f8c4309b9183de5c3dfcb6fa67aa"
LEGACY_POLICY_REVISION = "2026.11.18"
LEGACY_POLICY_COMMIT = "6ae1211cf3aa80beb0ca364b171f149a9506c05a"

POLICY_REPO = "open-athena/micro210-policy"
POLICY_DIGEST = "9d850b9f05c13c7a56df4c0bd9db23e7c805c505018cd330b175519ddca2057e"
POLICY_VERSION = "2026.11.23"
POLICY_REVISION = f"{POLICY_VERSION}-together"
POLICY_COMMIT = "7ab57cf5e33cacc98440bfcc2dfd452b94c44344"


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
    byte_size=131430,
)


# The prepared table re-hosts benchmark items that must not be public, so it lives in a
# private repository that CoreWeave tasks read with the standard HF_TOKEN.
PREPARED_REPO = "open-athena/micro210-prepared-tasks"
PREPARED_COMMIT = "2d33d7be129e6a93bb4279af8a4998a6915771a9"


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
    native_sha256: str
    native_inputs_uri: str
    smoke_rows: tuple[int, ...] = ()


@dataclass(frozen=True)
class Micro210Evaluation:
    """A model binding and output location under the frozen protocol."""

    model: Micro210Model
    run_id: str
    output_uri: str
    smoke_rows: tuple[int, ...] = ()
    native_sha256: str | None = None
    native_inputs_uri: str | None = None

    def policy(self) -> dict:
        return read_micro210_policy(POLICY_REFERENCE)

    def record(self) -> Micro210Record:
        self.policy()
        if self.native_sha256 is None or self.native_inputs_uri is None:
            raise ValueError("Micro210 native inputs must be verified before recording")
        return Micro210Record(
            POLICY_REFERENCE.sha256,
            self.model,
            self.run_id,
            self.output_uri,
            POLICY_REFERENCE,
            self.native_sha256,
            self.native_inputs_uri,
            self.smoke_rows,
        )


def micro210_evaluation(model: Micro210Model, *, run_id: str, output_uri: str) -> Micro210Evaluation:
    """Verify the frozen policy and bind the model and run identity."""
    request = Micro210Evaluation(model, run_id, output_uri)
    request.policy()
    return request


def read_micro210_policy(reference: Micro210PolicyReference) -> dict:
    """Read and verify a policy from a current or historical evaluation record."""
    raw = StoragePath(reference.uri).read_bytes()
    if len(raw) != reference.byte_size or hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError("Canonical Micro210 specification hash mismatch")
    return json.loads(raw)


def runtime_policy_config(policy: dict) -> dict:
    """Bind the frozen Harbor protocol to the native evaluation runtime."""
    config = deepcopy(policy["evaluation"]["config"])
    config.pop("entrypoint")
    config.pop("config_groups")
    retired = config.pop("terminal_bench")
    harbor, installed = retired["harbor"], retired["installed_agent"]
    agent = installed["agent"]
    if agent["name"] != "pi":
        raise ValueError("Micro210 requires the frozen Pi agent")
    config["trainer"]["placement"]["colocate_all"] = False
    sessions = config["environment"]["task_sessions"]
    sessions["daytona"] = {
        "api_key_env": harbor["daytona_api_key_env"],
        "ttl_minutes": 180,
        "image_build": harbor["daytona_provisioning"],
    }
    sessions["agent_max_retries"] = harbor["max_retries"]
    sessions["max_concurrent_agents"] = harbor["n_concurrent_trials"]
    sessions["agent_harnesses"] = {
        PI_HARNESS: {key: agent["kwargs"][key] for key in ("version", "thinking", "thinking_format")}
    }
    sessions["agent_harnesses"][PI_HARNESS]["setup_timeout"] = installed["setup_timeout"]
    config["generator"]["error_handling"] = {
        "enable_error_classification": True,
        "default_error_treatment": "mask",
        "passthrough_exceptions": ["ContextLengthExceededError", "AgentTimeoutError"],
    }
    return config
