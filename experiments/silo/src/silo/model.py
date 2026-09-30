# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sandbox and snapshot value types.

The naming scheme here is deliberately byte-compatible with the pipeline's own
(``capability_pipeline/daytona_resources.py:149-154``): a snapshot is identified by
``sha256(recipe_bytes + profile.cache_bytes())``. Keeping that identical means a
snapshot resolved by the new provider carries the same name the controller
already recorded, so nothing downstream has to be re-keyed.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

from silo.errors import SiloRecipeError

# host/path@sha256:<64 hex>. Mirrors image_runtime_metadata._REFERENCE, which is
# what the pipeline already enforces on every task image reference.
PINNED_REFERENCE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class ResourceProfile:
    """A capacity request in the units the pipeline uses: cores, GB, GB."""

    cpu: int
    memory_gb: int
    disk_gb: int

    def __post_init__(self) -> None:
        for name in ("cpu", "memory_gb", "disk_gb"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"resource profile {name} must be a positive int")

    def cache_bytes(self) -> bytes:
        """Stable request identity.

        Byte-identical to ``DaytonaResourceProfile.cache_bytes`` so that a name
        computed either side of the swap is the same string.
        """
        return json.dumps(
            {"cpu": self.cpu, "memory_gb": self.memory_gb, "disk_gb": self.disk_gb},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    @property
    def memory_bytes(self) -> int:
        return self.memory_gb * 1024**3

    @property
    def disk_bytes(self) -> int:
        return self.disk_gb * 1024**3


CANDIDATE_DEFAULT = ResourceProfile(cpu=4, memory_gb=8, disk_gb=10)
VERIFIER_DEFAULT = ResourceProfile(cpu=2, memory_gb=1, disk_gb=10)


def snapshot_name(prefix: str, recipe: str, profile: ResourceProfile) -> str:
    """Bind a cache entry to both immutable recipe bytes and requested capacity."""
    identity = hashlib.sha256(recipe.encode() + b"\0" + profile.cache_bytes()).hexdigest()
    return f"{prefix}-{identity[:20]}"


# --------------------------------------------------------------------------- #
# Recipe parsing
# --------------------------------------------------------------------------- #

# The pipeline generates exactly two recipe shapes today:
#   derive_daytona_recipe()      -> FROM <pinned>  [+ ENTRYPOINT] [+ CMD]
#   verifier_snapshot_recipe()   -> the above + USER root + one RUN pip install
# We support that set plus ENV/WORKDIR, and refuse everything else loudly.
_SUPPORTED = {"FROM", "ENTRYPOINT", "CMD", "USER", "RUN", "ENV", "WORKDIR"}


@dataclass(frozen=True)
class ImagePlan:
    """What a recipe means, once parsed.

    ``requires_build`` is the interesting field: a FROM-only recipe resolves
    straight to its pinned digest and needs no build at all, which is what makes
    snapshot creation near-instant and removes the whole capture/migrate/
    cold-pull conversion the Daytona format forced.
    """

    base_ref: str
    entrypoint: tuple[str, ...] | None = None
    cmd: tuple[str, ...] | None = None
    user: str | None = None
    workdir: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    run_steps: tuple[str, ...] = ()

    @property
    def requires_build(self) -> bool:
        return bool(self.run_steps)

    @property
    def base_pinned(self) -> bool:
        """Whether FROM names a digest.

        The pipeline's own recipes always do (``derive_daytona_recipe`` enforces
        it). Builder-authored Dockerfiles often use a tag, which Daytona accepted
        too; that is allowed here but recorded, because a tag can move and a
        snapshot built from one is not reproducible from its recipe alone.
        """
        return PINNED_REFERENCE.fullmatch(self.base_ref) is not None

    def receipt(self) -> dict[str, object]:
        return {
            "base_ref": self.base_ref,
            "entrypoint": list(self.entrypoint) if self.entrypoint else None,
            "cmd": list(self.cmd) if self.cmd else None,
            "user": self.user,
            "workdir": self.workdir,
            "env": dict(self.env),
            "base_pinned": self.base_pinned,
            "run_step_count": len(self.run_steps),
            "requires_build": self.requires_build,
        }


def _json_or_shell(argument: str, instruction: str) -> tuple[str, ...]:
    """Parse an ENTRYPOINT/CMD argument in either exec (JSON) or shell form."""
    stripped = argument.strip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise SiloRecipeError(f"{instruction} is not valid JSON: {error}") from error
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            raise SiloRecipeError(f"{instruction} JSON form must be a list of strings")
        return tuple(parsed)
    # Shell form runs through /bin/sh -c, exactly as a container runtime would.
    return ("/bin/sh", "-c", stripped)


def parse_recipe(recipe: str) -> ImagePlan:
    """Parse the Dockerfile subset this provider executes faithfully.

    Raises ``SiloRecipeError`` on anything outside that subset rather than
    guessing. An environment that silently differs from the recipe is how a task
    passes a gate it should fail.
    """
    base_ref: str | None = None
    entrypoint: tuple[str, ...] | None = None
    cmd: tuple[str, ...] | None = None
    user: str | None = None
    workdir: str | None = None
    env: dict[str, str] = {}
    run_steps: list[str] = []

    # Join continuations before looking at instructions.
    logical: list[str] = []
    pending = ""
    for raw in recipe.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.endswith("\\"):
            pending += line[:-1] + " "
            continue
        logical.append(pending + line)
        pending = ""
    if pending.strip():
        logical.append(pending)

    for line in logical:
        parts = line.strip().split(None, 1)
        instruction = parts[0].upper()
        argument = parts[1] if len(parts) > 1 else ""
        if instruction not in _SUPPORTED:
            raise SiloRecipeError(
                f"unsupported Dockerfile instruction {instruction!r}; "
                f"this provider executes only {sorted(_SUPPORTED)}"
            )
        if instruction == "FROM":
            if base_ref is not None:
                raise SiloRecipeError("multi-stage recipes are not supported")
            reference = argument.strip()
            if not reference or " " in reference:
                # "FROM image AS stage" is multi-stage syntax; bare FROM is empty.
                raise SiloRecipeError(f"FROM must name exactly one image, got {reference!r}")
            base_ref = reference
        elif base_ref is None:
            raise SiloRecipeError(f"{instruction} appears before FROM")
        elif instruction == "ENTRYPOINT":
            entrypoint = _json_or_shell(argument, "ENTRYPOINT")
        elif instruction == "CMD":
            cmd = _json_or_shell(argument, "CMD")
        elif instruction == "USER":
            user = argument.strip()
        elif instruction == "WORKDIR":
            workdir = argument.strip()
        elif instruction == "ENV":
            key, _, value = argument.partition("=")
            if not _:
                key, _, value = argument.strip().partition(" ")
            if not key.strip():
                raise SiloRecipeError(f"ENV has no name: {argument!r}")
            env[key.strip()] = value.strip().strip('"')
        elif instruction == "RUN":
            run_steps.append(argument.strip())

    if base_ref is None:
        raise SiloRecipeError("recipe has no FROM instruction")

    return ImagePlan(
        base_ref=base_ref,
        entrypoint=entrypoint,
        cmd=cmd,
        user=user,
        workdir=workdir,
        env=env,
        run_steps=tuple(run_steps),
    )


def build_dockerfile(plan: ImagePlan) -> str:
    """Render a plan back to a Dockerfile, for the build path.

    Only used when ``plan.requires_build``; the FROM-only path never builds.
    """
    lines = [f"FROM {plan.base_ref}"]
    if plan.user:
        lines.append(f"USER {plan.user}")
    if plan.workdir:
        lines.append(f"WORKDIR {plan.workdir}")
    for key, value in plan.env.items():
        lines.append(f"ENV {key}={shlex.quote(value)}")
    for step in plan.run_steps:
        lines.append(f"RUN {step}")
    if plan.entrypoint is not None:
        lines.append("ENTRYPOINT " + json.dumps(list(plan.entrypoint), separators=(",", ":")))
    if plan.cmd is not None:
        lines.append("CMD " + json.dumps(list(plan.cmd), separators=(",", ":")))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #

# The vocabulary wait_for_snapshot_active() accepts: "active" is terminal-good,
# these four are retryable, anything else is a hard failure upstream
# (capability_pipeline/daytona_snapshot.py:169-180).
SNAPSHOT_PENDING_STATES = ("pending", "building", "creating", "queued")
SNAPSHOT_ACTIVE = "active"
SNAPSHOT_ERROR = "error"


@dataclass
class BuildInfo:
    """Echoes the exact recipe bytes back.

    ``validate_snapshot_recipe`` compares the full submitted Dockerfile text
    against this field and fails closed on any difference, so name reuse with
    different content cannot slip through. It is the single hardest requirement
    the provider has to meet.
    """

    dockerfile_content: str

    def to_dict(self) -> dict[str, object]:
        return {"dockerfile_content": self.dockerfile_content}


@dataclass
class SnapshotRecord:
    name: str
    id: str
    state: str
    dockerfile_content: str
    plan: ImagePlan
    profile: ResourceProfile
    created_at: str
    resolved_ref: str = ""
    error_message: str = ""
    size: int = 0
    organization_id: str = "silo"
    general: bool = False

    @property
    def build_info(self) -> BuildInfo:
        return BuildInfo(dockerfile_content=self.dockerfile_content)

    @property
    def image_name(self) -> str:
        return self.resolved_ref or self.plan.base_ref

    @property
    def ref(self) -> str:
        return self.resolved_ref or self.plan.base_ref

    @property
    def cpu(self) -> int:
        return self.profile.cpu

    @property
    def mem(self) -> int:
        return self.profile.memory_gb

    @property
    def disk(self) -> int:
        return self.profile.disk_gb

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "id": self.id,
            "state": self.state,
            "image_name": self.image_name,
            "ref": self.ref,
            "size": self.size,
            "cpu": self.cpu,
            "mem": self.mem,
            "disk": self.disk,
            "created_at": self.created_at,
            "organization_id": self.organization_id,
            "general": self.general,
            "error_message": self.error_message,
            "build_info": self.build_info.to_dict(),
            "plan": self.plan.receipt(),
        }


@dataclass
class SandboxRecord:
    id: str
    snapshot: str
    state: str
    profile: ResourceProfile
    network_block_all: bool
    host_id: str
    created_at: str
    labels: Mapping[str, str] = field(default_factory=dict)
    ttl_minutes: int = 180
    deleted_at: float | None = None

    @property
    def cpu(self) -> int:
        return self.profile.cpu

    @property
    def memory(self) -> int:
        return self.profile.memory_gb

    @property
    def disk(self) -> int:
        return self.profile.disk_gb

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "snapshot": self.snapshot,
            "state": self.state,
            "cpu": self.cpu,
            "memory": self.memory,
            "disk": self.disk,
            "network_block_all": self.network_block_all,
            "host_id": self.host_id,
            "created_at": self.created_at,
            "labels": dict(self.labels),
            "ttl_minutes": self.ttl_minutes,
        }


def new_sandbox_id() -> str:
    """Unique and non-recycled.

    The repeated-diagnostics gate asserts ``reused_sandbox_ids == []`` across
    3 attempts x N controls, so ids must never be handed out twice -- including
    after a host restarts and forgets what it issued. A random uuid4 is what
    makes that true without shared state.
    """
    return f"slb{uuid.uuid4().hex}"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
