# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit expected-runtime bindings for protected production Trial observations."""

from enum import StrEnum
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec

RUNTIME_BINDING_FILE = "expected-runtime.json"
RUNTIME_OBSERVATION_TIMEOUT = 5
RUNTIME_OBSERVATION_SCRIPT = """
import hashlib
import importlib
import json
import stat
import sys
from pathlib import Path

paths = json.loads(sys.argv[1])
modules = {name: importlib.import_module(name).__file__ for name in ("verifyit", "verifyit.grade")}
files = {}
for value in paths:
    path = Path(value)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    files[value] = {"sha256": digest, "mode": stat.S_IMODE(path.stat().st_mode)}
lock = Path(sys.prefix).parent / "requirements.lock"
with lock.open("rb") as stream:
    lock_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
print(json.dumps({"executable": sys.executable, "version": list(sys.version_info[:3]),
    "prefix": sys.prefix, "import_path": sys.path, "module_origins": modules,
    "files": files, "lock_path": str(lock), "lock_sha256": lock_sha256}))
"""


class RuntimeRole(StrEnum):
    ORACLE_CONTROL = "oracle_control"
    SOURCE_GRADER = "source_grader"
    HOST_FRAMEWORK = "host_framework"


class RuntimeFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: int


class RuntimeObservation(BaseModel):
    """Interpreter facts read before a bound source machine executes commands."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    executable: str
    version: tuple[int, int, int]
    prefix: str
    import_path: list[str]
    module_origins: dict[str, str]
    files: dict[str, RuntimeFile]
    lock_path: str
    lock_sha256: str


class ExpectedRuntime(BaseModel):
    """Expected guest interpreter and immutable managed-runtime files."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    python_version: tuple[int, int]
    verifyit_root: str
    files: dict[str, RuntimeFile]
    lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def require_observation(self, observation: RuntimeObservation) -> None:
        if observation.version[:2] != self.python_version:
            raise ValueError("Protected runtime Python version differs from the frozen declaration")
        root = PurePosixPath(self.verifyit_root)
        if observation.module_origins != {
            "verifyit": str(root / "__init__.py"),
            "verifyit.grade": str(root / "grade.py"),
        }:
            raise ValueError("Protected grader imports do not use the frozen runtime")
        if observation.files != self.files:
            raise ValueError("Protected runtime file hashes or modes differ from the frozen manifest")
        if observation.lock_sha256 != self.lock_sha256:
            raise ValueError("Protected interpreter package lock differs from the frozen environment")


class ProtectedRuntimeBinding(BaseModel):
    """Request observations only for explicitly bound source-machine roles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roles: dict[RuntimeRole, ExpectedRuntime]

    @model_validator(mode="after")
    def source_roles_only(self):
        if RuntimeRole.HOST_FRAMEWORK in self.roles:
            raise ValueError("A source-runtime projection cannot change the trusted host judge framework")
        return self

    def require_task(self, task: TaskSpec) -> None:
        if self.task_sha256 != canonical_sha256(task.model_dump(mode="json")):
            raise ValueError("Protected runtime binding belongs to another TaskSpec")


def read_runtime_binding(directory: Path) -> ProtectedRuntimeBinding | None:
    path = directory / RUNTIME_BINDING_FILE
    return ProtectedRuntimeBinding.model_validate_json(path.read_text()) if path.is_file() else None


def write_runtime_binding(
    task: TaskSpec, runtime: dict, roles: dict[RuntimeRole, ExpectedRuntime], directory: Path
) -> None:
    """Freeze the selected source roles and expected runtime before task checksums are measured."""
    binding = ProtectedRuntimeBinding(
        task_sha256=canonical_sha256(task.model_dump(mode="json")), runtime_sha256=canonical_sha256(runtime), roles=roles
    )
    with (directory / RUNTIME_BINDING_FILE).open("x") as stream:
        stream.write(binding.model_dump_json(indent=2) + "\n")
