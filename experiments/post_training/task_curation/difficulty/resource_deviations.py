# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One explicitly authorized native SWE resource deviation, separate from frozen tasks."""

import hashlib
import json
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from rigging.filesystem.storage_path import StoragePath

SWE_SOURCE = "SWE-rebench:ibragim-bad/SWE-rebench-V2.1@repaired-20261119"
DEVIATION_VERSION = "swe-rebench-daytona-cap-20261125-v1"
AUTHORITY_URI = (
    "gs://marin-us-central2/rl-data/atlas/difficulty-tpu-v4/d7cb5dc7f21b1584dfc849aa646aaac7bb1b37ba8d9dbd6aa45f530e974fb579/"
    "resource-deviations/swe-rebench-daytona-cap-20261125-v1/authority.json"
)
AUTHORITY_SHA256 = "c514fbd6c9b5c89de0b84d05fa965c22107ef9a904e8bb624e7812315d87e9d9"


@dataclass(frozen=True)
class ResourceAllocation:
    cpus: int
    memory_mb: int
    storage_mb: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in (self.cpus, self.memory_mb, self.storage_mb)):
            raise ValueError("Resource allocation requires positive integer limits")


DECLARED = ResourceAllocation(6, 24576, 32768)
ACTUAL = ResourceAllocation(4, 8192, 10240)


@dataclass(frozen=True)
class ResourceDeviation:
    version: Literal["swe-rebench-daytona-cap-20261125-v1"]
    source_id: str
    task_id: str
    task_sha256: str
    runtime_sha256: str
    declared: ResourceAllocation
    actual: ResourceAllocation
    cause: Literal["daytona_per_sandbox_cap"]
    authority_uri: str
    authority_sha256: str

    def __post_init__(self) -> None:
        if (
            self.version != DEVIATION_VERSION
            or self.source_id != SWE_SOURCE
            or self.declared != DECLARED
            or self.actual != ACTUAL
            or self.cause != "daytona_per_sandbox_cap"
            or self.authority_uri != AUTHORITY_URI
            or self.authority_sha256 != AUTHORITY_SHA256
        ):
            raise ValueError("Resource deviation differs from the authorized SWE Daytona condition")
        for digest in (self.task_sha256, self.runtime_sha256):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("Resource deviation requires frozen task/runtime digests")

    def environment_overrides(self, path: Path, *, backend: str, workload: str) -> dict[str, int]:
        """Verify the native bytes before passing explicit limits to Harbor's Daytona adapter."""
        if backend != "daytona" or workload != "native":
            raise ValueError("The SWE resource deviation applies only to native Daytona tasks")
        if native_allocation(path) != self.declared:
            raise ValueError("Native resource declaration changed after the deviation was prepared")
        return {
            "override_cpus": self.actual.cpus,
            "override_memory_mb": self.actual.memory_mb,
            "override_storage_mb": self.actual.storage_mb,
        }


def native_allocation(path: Path) -> ResourceAllocation:
    config = tomllib.loads((path / "task.toml").read_text())["environment"]
    return ResourceAllocation(config["cpus"], config["memory_mb"], config["storage_mb"])


@cache
def authority() -> dict:
    data = StoragePath(AUTHORITY_URI).read_bytes()
    if hashlib.sha256(data).hexdigest() != AUTHORITY_SHA256:
        raise ValueError("SWE resource-deviation authority bytes changed")
    record = json.loads(data)
    if record["version"] != DEVIATION_VERSION or record["source_id"] != SWE_SOURCE:
        raise ValueError("Foreign SWE resource-deviation authority")
    return record


def prepare_resource_deviation(
    source_id: str, registry_revision: str, task_id: str, task_sha256: str, runtime_sha256: str, path: Path
) -> ResourceDeviation | None:
    """Apply the user ruling only to its pinned native tasks and original 6/24/32 declaration."""
    if source_id != SWE_SOURCE:
        return None
    scope = authority()
    if registry_revision != scope["registry_revision"]:
        return None
    if scope["tasks"].get(task_id) != task_sha256:
        raise ValueError("SWE task differs from the frozen authorized resource-deviation scope")
    if native_allocation(path) != DECLARED:
        return None
    return ResourceDeviation(
        DEVIATION_VERSION,
        source_id,
        task_id,
        task_sha256,
        runtime_sha256,
        DECLARED,
        ACTUAL,
        "daytona_per_sandbox_cap",
        AUTHORITY_URI,
        AUTHORITY_SHA256,
    )
