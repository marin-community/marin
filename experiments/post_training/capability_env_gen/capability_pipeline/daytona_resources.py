"""Explicit Daytona resource requests and cache identities.

Daytona's snapshot API accepts CPU, memory, and disk values in the units used
by the maintained ``dt.py`` helper: CPU cores and GB for memory and disk.  A
request is not evidence that the provider enforced it, so receipts deliberately
describe effective limits as unverified until an in-sandbox measurement exists.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DaytonaResourceProfile:
    """Provider request using the documented Daytona helper units."""

    cpu: int
    memory_gb: int
    disk_gb: int
    source: str

    def __post_init__(self) -> None:
        if (
            type(self.cpu) is not int
            or type(self.memory_gb) is not int
            or type(self.disk_gb) is not int
            or self.cpu <= 0
            or self.memory_gb <= 0
            or self.disk_gb <= 0
            or not isinstance(self.source, str)
            or not self.source.strip()
        ):
            raise ValueError(
                "Daytona resource profile values must be positive integers"
            )

    def receipt(self) -> dict[str, object]:
        return {
            "cpu": self.cpu,
            "memory_gb": self.memory_gb,
            "disk_gb": self.disk_gb,
            "provider_units": {
                "cpu": "cores",
                "memory": "GB",
                "disk": "GB",
            },
            "source": self.source,
            "effective_limits": "unverified",
        }

    def cache_bytes(self) -> bytes:
        """Stable request identity; source is provenance, not capacity."""
        return json.dumps(
            {
                "cpu": self.cpu,
                "memory_gb": self.memory_gb,
                "disk_gb": self.disk_gb,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


CANDIDATE_DEFAULT = DaytonaResourceProfile(
    cpu=4,
    memory_gb=8,
    disk_gb=10,
    source="compatible_default_no_pinned_profile",
)
VERIFIER_DEFAULT = DaytonaResourceProfile(
    cpu=2,
    memory_gb=1,
    disk_gb=10,
    source="compatible_default_no_pinned_profile",
)


def resolve_profile(
    value: DaytonaResourceProfile | Mapping[str, object] | None,
    *,
    default: DaytonaResourceProfile,
) -> DaytonaResourceProfile:
    """Accept only the adapter's explicit, unit-bearing profile shape.

    Pinned TaskSpec, DockerEnvironment, and ContainerRuntime schemas currently
    have no resource fields.  Callers therefore must pass this adapter-specific
    shape deliberately; unknown or unit-less fields fail closed rather than
    being guessed from a task's semantic requirements.
    """
    if value is None:
        return default
    if isinstance(value, DaytonaResourceProfile):
        return value
    if not isinstance(value, Mapping) or set(value) != {
        "cpu",
        "memory_gb",
        "disk_gb",
    }:
        raise ValueError(
            "Daytona resource_profile must contain only cpu, memory_gb, and disk_gb"
        )
    return DaytonaResourceProfile(
        cpu=value["cpu"],
        memory_gb=value["memory_gb"],
        disk_gb=value["disk_gb"],
        source="adapter_kwargs",
    )


def profile_from_receipt(value: object) -> DaytonaResourceProfile:
    """Parse the exact resource receipt emitted by the Daytona adapters.

    Receipt parsing is deliberately distinct from :func:`resolve_profile`: a
    receipt attests the requested units and provenance already used to create a
    snapshot.  A present but malformed receipt must not fall back to the
    recipe-only legacy cache name.
    """
    if not isinstance(value, Mapping) or set(value) != {
        "cpu",
        "memory_gb",
        "disk_gb",
        "provider_units",
        "source",
        "effective_limits",
    }:
        raise ValueError("Daytona resource receipt has an invalid shape")
    if value["provider_units"] != {
        "cpu": "cores",
        "memory": "GB",
        "disk": "GB",
    }:
        raise ValueError("Daytona resource receipt has invalid provider units")
    if value["effective_limits"] != "unverified":
        raise ValueError("Daytona resource receipt has invalid effective-limits state")
    if not isinstance(value["source"], str) or not value["source"]:
        raise ValueError("Daytona resource receipt has invalid source")
    return DaytonaResourceProfile(
        cpu=value["cpu"],
        memory_gb=value["memory_gb"],
        disk_gb=value["disk_gb"],
        source=value["source"],
    )


def snapshot_name(prefix: str, recipe: str, profile: DaytonaResourceProfile) -> str:
    """Bind a cache entry to both immutable recipe bytes and requested capacity."""
    identity = hashlib.sha256(
        recipe.encode() + b"\0" + profile.cache_bytes()
    ).hexdigest()
    return f"{prefix}-{identity[:20]}"
