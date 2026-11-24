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

PRIOR_POLICY_DIGEST = "8f7ef8170fbef2880214cbf03f70d80cffa3114276e024389617777bf07f53e5"
PRIOR_POLICY_VERSION = "2026.11.20"
PRIOR_POLICY_REVISION = f"{PRIOR_POLICY_VERSION}-together"
PRIOR_POLICY_COMMIT = "fa38e3eacc25137c9febe898119f08d65cc5ce88"

POLICY_REPO = "open-athena/micro210-policy"
POLICY_DIGEST = "9d850b9f05c13c7a56df4c0bd9db23e7c805c505018cd330b175519ddca2057e"
POLICY_VERSION = "2026.11.23"
POLICY_REVISION = f"{POLICY_VERSION}-together"
POLICY_COMMIT = "7ab57cf5e33cacc98440bfcc2dfd452b94c44344"

# The prepared table re-hosts benchmark items that must not be public, so it lives in a
# private repository that CoreWeave tasks read with the standard HF_TOKEN.
PREPARED_REPO = "open-athena/micro210-prepared-tasks"
PREPARED_COMMIT = "2d33d7be129e6a93bb4279af8a4998a6915771a9"


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


PRIOR_POLICY_REFERENCE = Micro210PolicyReference(
    uri=f"hf://datasets/{POLICY_REPO}@{PRIOR_POLICY_COMMIT}/policy.json",
    revision=PRIOR_POLICY_REVISION,
    sha256=PRIOR_POLICY_DIGEST,
    byte_size=121292,
)


POLICY_REFERENCE = Micro210PolicyReference(
    uri=f"hf://datasets/{POLICY_REPO}@{POLICY_COMMIT}/policy.json",
    revision=POLICY_REVISION,
    sha256=POLICY_DIGEST,
    byte_size=131430,
)

# The 2026.11.23-together seal: every script-graded row's TaskSpec is self-contained, the
# two unbuildable harbor slots are dropped, and the broken HumanEval/130 is swapped for
# the image-free HumanEval/51. The seal's removals and swaps ride inside the policy
# (task_changes) and the launcher binds this pin by content digest.


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
class Micro210Route:
    """The per-model execution route: where and on what accelerator class the evaluation runs.

    The route belongs to the model binding, not the protocol: a 67B Snowball model runs on
    H100 clusters and Mochi needs GB200, so each binding carries the route its model's
    first evaluation used. Every protocol field in the policy — sampling, timeouts, judge,
    task selection, per-task resources, and the priority class — stays frozen; the
    launcher accepts no other execution override.
    """

    parent_cluster_config: str
    cluster_config: str
    target_cluster: str
    container_profile: int

    def __post_init__(self) -> None:
        for path in (self.parent_cluster_config, self.cluster_config):
            if not path.endswith(".yaml"):
                raise ValueError(f"Route cluster config must be a yaml config path: {path!r}")
        if not self.target_cluster:
            raise ValueError("Route target_cluster is required")
        if self.container_profile < 0:
            raise ValueError(f"Route container_profile must not be negative: {self.container_profile}")

    @classmethod
    def from_policy_execution(cls, execution: dict) -> "Micro210Route":
        """The policy's own route, used when a binding does not carry an explicit one."""
        return cls(
            parent_cluster_config=execution["parent_cluster_config"],
            cluster_config=execution["cluster_config"],
            target_cluster=execution["target_cluster"],
            container_profile=int(execution["container_profile"]),
        )


@dataclass(frozen=True)
class Micro210Record:
    """Serializable identity of the model, protocol and evaluation outputs."""

    canonical_spec_sha256: str
    model: Micro210Model
    run_id: str
    output_uri: str
    policy_reference: Micro210PolicyReference
    route: Micro210Route


@dataclass(frozen=True)
class Micro210Evaluation:
    """A model binding and output location under the frozen protocol."""

    model: Micro210Model
    run_id: str
    output_uri: str
    route: Micro210Route

    def policy(self) -> dict:
        return read_micro210_policy(POLICY_REFERENCE)

    def record(self) -> Micro210Record:
        self.policy()
        return Micro210Record(
            POLICY_REFERENCE.sha256, self.model, self.run_id, self.output_uri, POLICY_REFERENCE, self.route
        )


def micro210_evaluation(
    model: Micro210Model, *, run_id: str, output_uri: str, route: Micro210Route | None = None, **overrides
) -> Micro210Evaluation:
    """Verify the frozen policy and bind the model, run identity and execution route."""
    if overrides:
        raise ValueError(f"Canonical Micro210 refuses override field: {next(iter(overrides))}")
    resolved = route or Micro210Route.from_policy_execution(
        read_micro210_policy(POLICY_REFERENCE)["evaluation"]["execution"]
    )
    request = Micro210Evaluation(model, run_id, output_uri, resolved)
    request.policy()
    return request


def read_micro210_policy(reference: Micro210PolicyReference) -> dict:
    """Read and verify a policy from a current or historical evaluation record."""
    raw = StoragePath(reference.uri).read_bytes()
    if len(raw) != reference.byte_size or hashlib.sha256(raw).hexdigest() != reference.sha256:
        raise ValueError("Canonical Micro210 specification hash mismatch")
    return json.loads(raw)
