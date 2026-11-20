# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingestion-time SHARED SWE protection for newly versioned Harbor task images."""

import hashlib
import json
import shlex
import zlib
from dataclasses import asdict, replace
from enum import StrEnum
from pathlib import Path
from typing import cast

import tomlkit
from pydantic import BaseModel, ConfigDict, JsonValue
from tomlkit.items import Table
from verifyit.spec import PytestSpec, ScriptSpec, render_spec

from taskcompendium.harbor.native_shared import native_runner
from taskcompendium.integrity import (
    AGENT_HOME,
    BOOTSTRAP_CONTROLLER,
    BOOTSTRAP_DIRECTORY,
    BOOTSTRAP_TIMEOUT,
    CONTROLLER_CONTEXT,
    DEPENDENCY_AUDIT_PROBES,
    PRIVATE_CONTEXT,
    PRIVATE_DIRECTORY,
    SHARED_CONTROLLER_CONTEXT,
    SharedDependencyState,
    SharedIntegrityContract,
    shared_preamble,
)
from taskcompendium.native_integrity import TEST_PATTERNS
from taskcompendium.runtime import shared_native, shared_pytest

BOOTSTRAP_INPUT = "/tmp/taskcompendium-shared-input.json"
BOOTSTRAP_ARCHIVE = f"{PRIVATE_CONTEXT}/input.zlib"
BOOTSTRAP_PAYLOAD = f"{PRIVATE_DIRECTORY}/input.zlib"
SETUP_PREFIX = "## Environment Setup (complete these steps first)\n\n```bash\n"


class SharedPytestFormat(StrEnum):
    CURRENT = "current"
    SINGLE_PROTECTED_MANIFEST = "single_protected_manifest"


class SourceSetupPolicy(StrEnum):
    PREPEND = "prepend"
    INCLUDED = "included"
    REJECT = "reject"


class SharedExecutionPolicy(StrEnum):
    MUTABLE = "shared_agent_mutable_v2"
    ACTOR_ENVIRONMENT = "shared_actor_environment_v1"


def apply_shared_task_config(
    config: tomlkit.TOMLDocument,
    contract: SharedIntegrityContract,
    setup_command: str,
    *,
    setup_timeout: float,
    source_setup_policy: SourceSetupPolicy,
    execution_policy: SharedExecutionPolicy,
    actor_path: str | None = None,
    verifier_timeout: float | None = None,
) -> None:
    """Apply the UID boundary, safe actor HOME, and explicit initialization policy."""
    environment = cast(Table, config.setdefault("environment", tomlkit.table()))
    existing = environment.get("setup")
    if source_setup_policy is SourceSetupPolicy.REJECT and existing is not None:
        raise ValueError("Reference mode cannot reorder an existing source initialization command")
    if source_setup_policy is SourceSetupPolicy.PREPEND and existing is not None:
        setup_command = "set -euo pipefail\n" + str(existing["command"]) + "\n" + setup_command
    source_environment = dict(environment.get("env", {}))
    source_home = str(source_environment.get("HOME") or "/root")
    environment["setup"] = {
        "command": f"export HOME={shlex.quote(source_home)}\n" + setup_command,
        "timeout_sec": setup_timeout,
    }
    if source_environment.get("HOME") in {None, "/root"}:
        source_environment["HOME"] = AGENT_HOME
    if actor_path is not None:
        source_environment["PATH"] = actor_path
    environment["env"] = source_environment
    cast(Table, config.setdefault("agent", tomlkit.table()))["user"] = str(contract.agent_uid)
    verifier = cast(Table, config.setdefault("verifier", tomlkit.table()))
    verifier["user"] = "root"
    verifier["restart_environment"] = False
    if verifier_timeout is not None:
        verifier["timeout_sec"] = verifier_timeout
    metadata = cast(Table, config.setdefault("metadata", tomlkit.table()))
    metadata["integrity_contract"] = contract.model_dump(mode="json", exclude_none=True)
    metadata["execution_policy"] = execution_policy.value
    config["schema_version"] = "1.3"


def render_shared_pytest_spec(spec: PytestSpec, pytest_format: SharedPytestFormat) -> str:
    """Render the explicitly selected source CLI's protected-manifest syntax."""
    if pytest_format is SharedPytestFormat.CURRENT:
        return render_spec(spec)
    if len(spec.protected_paths_files) != 1 or spec.candidate_modules:
        raise ValueError("Single-manifest source CLI requires one manifest and no candidate modules")
    document = tomlkit.parse(render_spec(spec))
    del document["protected_paths_files"]
    del document["candidate_modules"]
    document["protected_paths_file"] = spec.protected_paths_files[0]
    return tomlkit.dumps(document)


class SharedBootstrapPayload(BaseModel):
    """Private input to the published guest controller's bootstrap operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract: SharedIntegrityContract
    spec: dict[str, JsonValue]
    verifier_spec: str
    test_files: dict[str, str]


def shared_source_spec(spec: PytestSpec | ScriptSpec) -> PytestSpec | ScriptSpec:
    """Resolve private source manifests beside the protected verifier specification."""
    if not isinstance(spec, PytestSpec):
        return spec
    return replace(
        spec,
        protected_paths_files=tuple(
            Path(path).relative_to("/tests").as_posix() if Path(path).is_absolute() else path
            for path in spec.protected_paths_files
        ),
    )


def split_source_setup(instruction: str) -> tuple[str, str]:
    """Separate the explicit public setup block without rewriting its command bytes."""
    if not instruction.startswith(SETUP_PREFIX):
        return "", instruction
    setup, boundary, body = instruction[len(SETUP_PREFIX) :].partition("\n```\n")
    if not boundary:
        raise ValueError("Incomplete source environment setup block")
    return setup, body.removeprefix("\n---\n\n")


def shared_task_instruction(body: str, contract: SharedIntegrityContract) -> str:
    return (
        shared_preamble(contract)
        + "The task workspace and initial dependencies are prepared before you start.\n\n"
        + body
    )


def shared_grader_script(contract: SharedIntegrityContract) -> bytes:
    """Launch the pinned controller with the image's protected interpreter."""
    return (
        "#!/bin/bash\nset -euo pipefail\n"
        f'control_python="{contract.protected_root}-bootstrap/python"\n'
        f'exec "$control_python" -I -S {contract.harness} grade --contract {contract.protected_root}/contract.json '
        f"--verifyit {BOOTSTRAP_DIRECTORY}/verifyit\n"
    ).encode()


def harden_shared_swe(
    files: dict[str, bytes],
    *,
    spec: PytestSpec | ScriptSpec,
    dependency_state: SharedDependencyState,
    pytest_format: SharedPytestFormat = SharedPytestFormat.CURRENT,
) -> SharedIntegrityContract:
    """Rewrite one new native package with a complete, versioned SHARED contract.

    The caller must publish this package under a new dataset version and qualify
    it; this function never changes an already-published task at runtime.
    """
    spec = shared_source_spec(spec)
    native = native_runner(files) if isinstance(spec, ScriptSpec) else None
    source = Path(shared_native.__file__ if native else shared_pytest.__file__).read_bytes()
    test_files = {name.removeprefix("tests/"): data.hex() for name, data in files.items() if name.startswith("tests/")}
    reference = hashlib.sha256(json.dumps(test_files, sort_keys=True).encode()).hexdigest()
    contract = SharedIntegrityContract(
        dependency_state=dependency_state,
        reference_sha256=reference,
        harness_sha256=hashlib.sha256(source).hexdigest(),
    )
    if native is not None:
        contract = SharedIntegrityContract.model_validate(
            {
                **contract.model_dump(mode="json"),
                "id": "shared-native",
                "native": native,
                "protected_test_patterns": TEST_PATTERNS,
                "audit_probes": DEPENDENCY_AUDIT_PROBES,
                "known_open_gaps": ("Candidate-produced test-output injection", "Mutable dependency startup hooks"),
            }
        )
    root = contract.protected_root
    protected_spec = replace(spec, setup="", python=f"{root}/pytest-python") if isinstance(spec, PytestSpec) else spec
    payload = {
        "contract": contract.model_dump(mode="json"),
        "spec": {**asdict(spec), "mode": "script" if native else "pytest"},
        "verifier_spec": (
            render_shared_pytest_spec(protected_spec, pytest_format)
            if isinstance(protected_spec, PytestSpec)
            else render_spec(protected_spec)
        ),
        "test_files": test_files,
    }
    if BOOTSTRAP_ARCHIVE in files:
        raise ValueError(f"Source task occupies protected bootstrap resource: {BOOTSTRAP_ARCHIVE}")
    encoded = json.dumps(payload).encode()
    SharedBootstrapPayload.model_validate_json(encoded)
    files[BOOTSTRAP_ARCHIVE] = zlib.compress(encoded)
    config = tomlkit.parse(files["task.toml"].decode())
    setup, instruction = split_source_setup(files["instruction.md"].decode())
    # Payloads are root-only in the task image even when a runner skips initialization.
    unpack = (
        "import pathlib,zlib; "
        f"pathlib.Path({BOOTSTRAP_INPUT!r}).write_bytes("
        f"zlib.decompress(pathlib.Path({BOOTSTRAP_PAYLOAD!r}).read_bytes()))"
    )
    control_python = f"{BOOTSTRAP_DIRECTORY}/python"
    bootstrap_command = (
        "set -euo pipefail\n"
        + setup
        + "\n"
        + f'control_python="{control_python}"\n'
        + f'"$control_python" -I -S -c {shlex.quote(unpack)}\n'
        + f'"$control_python" -I -S {BOOTSTRAP_CONTROLLER} bootstrap {BOOTSTRAP_INPUT}\n'
        + f"rm {BOOTSTRAP_INPUT} {BOOTSTRAP_PAYLOAD}\n"
    )
    apply_shared_task_config(
        config,
        contract,
        bootstrap_command,
        setup_timeout=BOOTSTRAP_TIMEOUT,
        source_setup_policy=SourceSetupPolicy.PREPEND,
        execution_policy=SharedExecutionPolicy.MUTABLE,
    )
    files["task.toml"] = tomlkit.dumps(config).encode()
    files["instruction.md"] = shared_task_instruction(instruction, contract).encode()
    files[CONTROLLER_CONTEXT] = source
    if native:
        files[SHARED_CONTROLLER_CONTEXT] = Path(shared_pytest.__file__).read_bytes()
    files["environment/Dockerfile"] += shared_image_layer(contract).encode()
    files["tests/test.sh"] = shared_grader_script(contract)
    return contract


def shared_user_layer(contract: SharedIntegrityContract) -> str:
    """Create the actor and grading identities before installing private image inputs."""
    return (
        "\nUSER 0:0\n"
        f"RUN getent passwd {contract.agent_uid} >/dev/null || "
        f"useradd --create-home --uid {contract.agent_uid} --user-group taskcompendium-agent\n"
        f"RUN getent passwd {contract.grader_uid} >/dev/null || "
        f"useradd --create-home --uid {contract.grader_uid} --user-group taskcompendium-grade\n"
        f"RUN mkdir -p {AGENT_HOME} && "
        f"chown {contract.agent_uid}:{contract.agent_uid} {AGENT_HOME}\n"
    )


def shared_private_image_layer() -> str:
    """Keep bootstrap inputs private even for runners that ignore the setup hook."""
    return (
        f"COPY taskcompendium-shared-private/ {PRIVATE_DIRECTORY}/\n"
        f"RUN chown -R 0:0 {PRIVATE_DIRECTORY} && chmod 700 {PRIVATE_DIRECTORY} && "
        f"chmod 600 {PRIVATE_DIRECTORY}/*\n"
        "USER 0:0\n"
    )


def shared_image_layer(contract: SharedIntegrityContract) -> str:
    """Install the maintained controller without replacing the source image context."""
    native = contract.native is not None
    return (
        shared_user_layer(contract)
        + f"COPY {CONTROLLER_CONTEXT.removeprefix('environment/')} {BOOTSTRAP_CONTROLLER}\n"
        + (
            f"COPY {SHARED_CONTROLLER_CONTEXT.removeprefix('environment/')} {BOOTSTRAP_DIRECTORY}/shared_pytest.py\n"
            if native
            else ""
        )
        + 'RUN verifyit_path="$(readlink -f "$(command -v verifyit)")"; '
        'python_path="$(head -n 1 "$verifyit_path" | cut -c 3-)"; '
        f'ln -s "$python_path" {BOOTSTRAP_DIRECTORY}/python; '
        f'ln -s "$verifyit_path" {BOOTSTRAP_DIRECTORY}/verifyit\n'
        f"RUN chmod -R go-w {BOOTSTRAP_DIRECTORY}\n" + shared_private_image_layer()
    )
