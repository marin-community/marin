# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile a native SHARED task from a separately captured trusted grading reference."""

import hashlib
import io
import shlex
import tarfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, Self, cast

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tomlkit.items import Table
from verifyit.spec import PytestSpec

from taskcompendium.harbor.shared_package import (
    SharedExecutionPolicy,
    SharedPytestFormat,
    SourceSetupPolicy,
    apply_shared_task_config,
    render_shared_pytest_spec,
    shared_grader_script,
    shared_private_image_layer,
    shared_source_spec,
    shared_user_layer,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.integrity import PRIVATE_CONTEXT, PRIVATE_DIRECTORY, SharedIntegrityContract, shared_preamble
from taskcompendium.runtime import shared_pytest, shared_reference

REFERENCE_INPUT = f"{PRIVATE_CONTEXT}/reference.json"
REFERENCE_BUNDLE = f"{PRIVATE_CONTEXT}/reference.tar.gz"
REFERENCE_INITIALIZER = f"{PRIVATE_CONTEXT}/reference.py"

ACTOR_PREFIX = "/opt/taskcompendium-agent"
DEFAULT_SOURCE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class ActorEnvironment(BaseModel):
    """The agent-owned Python environment declared by a trusted reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    prefix: str = Field(min_length=1)
    python: str = Field(min_length=1)
    bin: str = Field(min_length=1)
    uid: int = Field(ge=0)
    gid: int = Field(ge=0)
    group: str = Field(min_length=1)
    venv_flags: tuple[str, ...]
    directory_mode: int
    file_mode: int
    executable_mode: int
    pip_shim: str = Field(min_length=1)
    path: str = Field(min_length=1)
    declared_changes: tuple[str, ...]

    @model_validator(mode="after")
    def validate_complete(self) -> Self:
        if self.bin != f"{self.python}/bin":
            raise ValueError("The actor executable directory must be the environment's bin")
        if not self.python.startswith(self.prefix + "/"):
            raise ValueError("The actor Python environment must live inside its prefix")
        if not self.path.startswith(self.bin + ":"):
            raise ValueError("The actor bin must be the first PATH entry")
        if "--copies" not in self.venv_flags or "--without-pip" not in self.venv_flags:
            raise ValueError("The actor environment must copy the exact source interpreter without vendored pip")
        if "--system-site-packages" in self.venv_flags:
            raise ValueError("The captured baseline must be exposed through the declared source-baseline path file")
        if not self.pip_shim.startswith("#!/bin/sh\nexec "):
            raise ValueError("The actor pip must be a declared shim over the source pip module")
        if not self.declared_changes:
            raise ValueError("The actor environment must declare every difference from the original")
        return self


def actor_environment_for(
    contract: SharedIntegrityContract, *, prefix: str = ACTOR_PREFIX, source_path: str = DEFAULT_SOURCE_PATH
) -> ActorEnvironment:
    """Declare the approved repair environment: exact interpreter, read-only baseline, agent-owned installs."""
    python = f"{prefix}/python"
    bin_directory = f"{python}/bin"
    path = f"{bin_directory}:{source_path}"
    return ActorEnvironment(
        prefix=prefix,
        python=python,
        bin=bin_directory,
        uid=contract.agent_uid,
        gid=contract.agent_uid,
        group="taskcompendium-agent",
        venv_flags=("--copies", "--without-pip"),
        directory_mode=0o755,
        file_mode=0o644,
        executable_mode=0o755,
        pip_shim=f'#!/bin/sh\nexec {bin_directory}/python -m pip "$@"\n',
        path=path,
        declared_changes=(
            f"The agent runs as UID {contract.agent_uid} instead of root",
            f"An agent-owned Python environment at {python} copies the exact source interpreter and exposes the "
            "captured baseline site directories read-only through a declared source-baseline path file; "
            "no package is upgraded",
            f"A pip shim at {bin_directory}/pip executes the source pip module from the captured baseline against "
            "the agent environment, so task-time installs and console entrypoints land in agent-owned space",
            f"The agent environment PATH is {path}",
            f"{contract.workspace} is owned by UID {contract.agent_uid} and stays empty at handoff",
            "The source image site-packages stay root-owned and unmodified; task-time dependency changes live only "
            "in the agent environment",
            f"The grading runtime under {contract.protected_root} stays root-owned and frozen",
        ),
    )


def shared_actor_preamble(actor: ActorEnvironment) -> str:
    """Render the agent-facing card so the published instruction declares the environment."""
    return (
        f"Agent environment (declared v{actor.version}): you run as UID {actor.uid} ({actor.group}) and own "
        f"{actor.prefix}. python and pip resolve to {actor.bin} first (PATH: {actor.path}). The captured baseline "
        f"packages are exposed read-only and unchanged; your installs land in {actor.python}. "
        "This environment replaces the original root-owned task installation. Follow the source setup "
        "instructions in this environment.\n\n"
    )


class FrozenSharedReference(BaseModel):
    """Immutable provenance for a trusted capture, never an agent-produced reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    original_files_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_cli_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    capture_trial_uri: str = Field(min_length=1)
    capture_trial_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract: SharedIntegrityContract
    actor_environment: ActorEnvironment

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if self.actor_environment.uid != self.contract.agent_uid:
            raise ValueError("The actor environment belongs to another agent identity")
        actor = Path(self.actor_environment.prefix)
        protected = Path(self.contract.protected_root)
        if actor.is_relative_to(protected) or protected.is_relative_to(actor):
            raise ValueError("The actor prefix and the protected grading root must be disjoint")
        return self


def harden_shared_swe_reference(
    files: dict[str, bytes],
    *,
    spec: PytestSpec,
    reference: FrozenSharedReference,
    bundle: bytes,
    pytest_format: SharedPytestFormat,
    host_init_timeout: float,
) -> SharedIntegrityContract:
    """Install a frozen grader without moving the source's public setup or oracle.

    The caller must authenticate the capture trial and original TaskSpec/runtime
    against its source inventory before compilation. This function verifies the
    supplied native bytes, grader, image context and captured bundle. The init
    timeout is the shared calibrator's host-only bound; source clocks are retained.
    Only an originally empty workspace is supported by this mode.
    """
    contract = reference.contract
    controller = Path(shared_pytest.__file__).read_bytes()
    if contract.native is not None or hashlib.sha256(controller).hexdigest() != contract.harness_sha256:
        raise ValueError("Frozen pytest reference requires the exact maintained C83 controller")
    if contract.workspace != spec.workspace:
        raise ValueError("Frozen reference belongs to another source workspace")
    if canonical_sha256({name: data.hex() for name, data in files.items()}) != reference.original_files_sha256:
        raise ValueError("Frozen reference belongs to another native source package")
    if canonical_sha256(asdict(spec)) != reference.source_spec_sha256:
        raise ValueError("Frozen reference belongs to another original verifier")
    context = {name: data.hex() for name, data in files.items() if name.startswith("environment/")}
    if canonical_sha256(context) != reference.image_context_sha256:
        raise ValueError("Frozen reference belongs to another source image context")
    if hashlib.sha256(bundle).hexdigest() != reference.bundle_sha256:
        raise ValueError("Frozen grading reference bytes changed")
    normalized = shared_source_spec(spec)
    assert isinstance(normalized, PytestSpec)
    expected_spec = render_shared_pytest_spec(
        replace(normalized, setup="", python=f"{contract.protected_root}/pytest-python"), pytest_format
    )
    with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
        contract_file = archive.extractfile("contract.json")
        verifier_file = archive.extractfile("tests/verifier.toml")
        if contract_file is None or verifier_file is None:
            raise ValueError("Frozen grading reference lacks its contract or verifier")
        if SharedIntegrityContract.model_validate_json(contract_file.read()) != contract:
            raise ValueError("Frozen grading contract differs from its provenance")
        if verifier_file.read().decode() != expected_spec:
            raise ValueError("Frozen protected verifier differs from the source CLI contract")
    if any(name in files for name in (REFERENCE_INPUT, REFERENCE_BUNDLE, REFERENCE_INITIALIZER)):
        raise ValueError("Source package occupies protected reference resources")
    config = tomlkit.parse(files["task.toml"].decode())
    actor = reference.actor_environment
    files[REFERENCE_INPUT] = reference.model_dump_json().encode()
    files[REFERENCE_BUNDLE] = bundle
    files[REFERENCE_INITIALIZER] = Path(shared_reference.__file__).read_bytes()
    setup_command = (
        "set -euo pipefail\n"
        'verifyit_path="$(readlink -f "$(command -v verifyit)")"\n'
        'control_python="$(head -n 1 "$verifyit_path" | cut -c 3-)"\n'
        '"$control_python" -I -S '
        + shlex.quote(f"{PRIVATE_DIRECTORY}/reference.py")
        + " "
        + shlex.quote(f"{PRIVATE_DIRECTORY}/reference.json")
        + " "
        + shlex.quote(f"{PRIVATE_DIRECTORY}/reference.tar.gz")
        + "\nrm "
        + shlex.quote(f"{PRIVATE_DIRECTORY}/reference.py")
        + "\n"
    )
    apply_shared_task_config(
        config,
        contract,
        setup_command,
        setup_timeout=host_init_timeout,
        source_setup_policy=SourceSetupPolicy.REJECT,
        execution_policy=SharedExecutionPolicy.ACTOR_ENVIRONMENT,
        actor_path=actor.path,
    )
    metadata = cast(Table, config["metadata"])
    metadata["actor_environment"] = actor.model_dump(mode="json", exclude_none=True)
    files["task.toml"] = tomlkit.dumps(config).encode()
    files["instruction.md"] = (shared_preamble(contract) + shared_actor_preamble(actor)).encode() + files[
        "instruction.md"
    ]
    files["tests/test.sh"] = shared_grader_script(contract)
    files["environment/Dockerfile"] += (shared_user_layer(contract) + shared_private_image_layer()).encode()
    return contract
