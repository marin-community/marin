# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wire spec for an always-on Iris service deploy.

The contract between the Pulumi component (:mod:`iac.iris.service`) and the deploy
CLI (:mod:`iac.iris.deploy`): the component serializes a :class:`ServiceSpec` into
the deploy Command's environment; the CLI validates and submits it. Every field is
JSON-native so the serialized form is deterministic and diffs cleanly as a Pulumi
input.
"""

import dataclasses
import json

from rigging.secrets import is_secret_reference

# Effectively-unlimited retry ceiling for always-on services. Iris has no infinite
# sentinel and its retry budgets are lifetime totals that never reset, so the bound
# is set too large to realistically exhaust; each retry is paced by VM boot /
# capacity, never a tight loop.
ALWAYS_ON_RETRIES = 1_000_000

# The Command environment variable carrying the serialized spec. Riding in the
# environment (never interpolated into a shell line) means `pulumi destroy` replays
# the spec the resource was created with, and no stack-controlled string needs
# shell quoting.
SPEC_ENV_VAR = "IRIS_SVC_SPEC"

# Default bound on the post-submit readiness probe (shared with IrisServiceArgs).
DEFAULT_READY_WAIT = 600


@dataclasses.dataclass(frozen=True)
class ServiceSpec:
    """One always-on, port-publishing Iris service job."""

    cluster: str
    """Cluster name, resolved via the iris config search path (lib/iris/config)."""

    name: str
    """Job name; the job identity is ``/{user}/{name}``."""

    user: str
    """Explicit identity pin — the OS-user fallthrough would tie the job to whoever
    deploys (a CI runner's job is ``/runner/...``)."""

    entrypoint: tuple[str, ...]
    """Command argv, e.g. ``("python", "-m", "ducky.server")``."""

    resources: dict
    """``iris.rpc.job_pb2.ResourceSpecProto`` as a ``json_format`` dict."""

    regions: tuple[str, ...]
    """Region pin (required — a service is never scheduled "anywhere")."""

    port: str
    """Named Iris port the service binds (workers inject ``IRIS_PORT_<NAME>``)."""

    endpoint: str
    """Cluster-global endpoint the service registers (leading slash, e.g. ``/ducky``);
    the service is reachable at ``<controller>/proxy/<endpoint>``."""

    health_path: str
    """Readiness-probe path under the service's proxy URL. Required, and it must be a
    route the service serves without app auth: the controller proxy strips
    ``authorization``/``proxy-authorization`` before forwarding, so a probe of a
    protected route waits out the full readiness window on a healthy service."""

    env: dict[str, str] = dataclasses.field(default_factory=dict)
    """Plain environment values. Secret references are rejected here."""

    secret_env: dict[str, str] = dataclasses.field(default_factory=dict)
    """Environment values resolved at submit time from secret references
    (``gcp-secret://`` / ``env:`` / ``file:``, see :mod:`rigging.secrets`)."""

    pip_packages: tuple[str, ...] = ()
    """Additional pinned packages installed by Iris before the service starts."""

    sync_packages: tuple[str, ...] = ()
    """Optional workspace packages that scope the service's ``uv sync``."""

    wait: int = DEFAULT_READY_WAIT
    """Readiness wait in seconds. Expiry warns and exits 0: the submit already
    happened and Iris's retry budgets converge once capacity frees, so a capacity
    stall is not a failed deploy."""

    deploy_generation: int = 0
    """Bump to force a redeploy with an otherwise unchanged spec — the operational
    hammer for a wedged-but-RUNNING instance."""

    build_commands: tuple[str, ...] = ()
    """Shell commands ``up`` runs (in order, from the workspace root) before anything
    touches Secret Manager or the cluster — the build step producing the
    ``extra_bundle_includes`` outputs. Running inside every ``up`` means a deploy can
    never ship a stale or missing build; a failing command aborts before the running
    instance is disturbed."""

    extra_bundle_includes: tuple[str, ...] = ()
    """Gitignored globs (build outputs) shipped in the workspace bundle. ``up``
    fails when a glob matches nothing after the build, so a broken build stops the
    deploy."""

    max_retries_preemption: int = ALWAYS_ON_RETRIES
    max_retries_failure: int = ALWAYS_ON_RETRIES
    max_task_failures: int = ALWAYS_ON_RETRIES

    def validate(self) -> None:
        """Reject malformed specs before any cluster interaction."""
        if not self.cluster:
            raise ValueError("cluster must be set")
        if not self.name or "/" in self.name:
            raise ValueError(f"name {self.name!r} must be non-empty and contain no '/'")
        if not self.user:
            raise ValueError("user must be set (job identity is /{user}/{name})")
        if not self.entrypoint:
            raise ValueError("entrypoint must be a non-empty argv")
        if not self.regions:
            raise ValueError("regions must be non-empty; a service is pinned, never scheduled anywhere")
        if not self.port:
            raise ValueError("port must be set (the named Iris port the service binds)")
        if not self.endpoint.startswith("/"):
            raise ValueError(f"endpoint {self.endpoint!r} must start with '/' (a cluster-global endpoint)")
        if not self.health_path.startswith("/"):
            raise ValueError(f"health_path {self.health_path!r} must start with '/'")
        for key, value in self.env.items():
            if is_secret_reference(value):
                raise ValueError(f"env[{key!r}] is a secret reference; move it to secret_env")
        for key, value in self.secret_env.items():
            if not is_secret_reference(value):
                raise ValueError(
                    f"secret_env[{key!r}] is not a secret reference (expected env: / file: / gcp-secret://)"
                )
        if self.wait < 0:
            raise ValueError("wait must be >= 0")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ServiceSpec":
        raw = json.loads(text)
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown spec fields: {sorted(unknown)}")
        for field_name in (
            "entrypoint",
            "regions",
            "pip_packages",
            "sync_packages",
            "build_commands",
            "extra_bundle_includes",
        ):
            if field_name in raw:
                raw[field_name] = tuple(raw[field_name])
        return cls(**raw)
