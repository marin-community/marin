# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingestion-time SHARED SWE protection, independent of population-specific image builds."""

import hashlib
import json
import shlex
import zlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import tomlkit
from pydantic import BaseModel, ConfigDict, JsonValue
from tomlkit.items import Table
from verifyit.spec import PytestSpec, ScriptSpec, render_spec

from taskcompendium.harbor.native_shared import native_runner
from taskcompendium.integrity import SharedDependencyState, SharedIntegrityContract, shared_preamble
from taskcompendium.native_integrity import TEST_PATTERNS
from taskcompendium.runtime import shared_native, shared_pytest

BOOTSTRAP = "/opt/taskcompendium-shared-bootstrap/control.py"
BOOTSTRAP_INPUT = "/tmp/taskcompendium-shared-input.json"
BOOTSTRAP_ARCHIVE = "setup_files/taskcompendium-shared-input.zlib"
SETUP_PREFIX = "## Environment Setup (complete these steps first)\n\n```bash\n"


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
        'control_python="/opt/taskcompendium-shared-bootstrap/python"\n'
        f'exec "$control_python" -I -S {contract.harness} grade --contract {contract.protected_root}/contract.json '
        "--verifyit /opt/taskcompendium-shared-bootstrap/verifyit\n"
    ).encode()


def harden_shared_swe(
    files: dict[str, bytes], *, spec: PytestSpec | ScriptSpec, dependency_state: SharedDependencyState
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
                "audit_probes": ("dependencies-diff.json", "startup-hooks-diff.json"),
                "known_open_gaps": ("Candidate-produced test-output injection", "Mutable dependency startup hooks"),
            }
        )
    root = contract.protected_root
    protected_spec = replace(spec, setup="", python=f"{root}/pytest-python") if isinstance(spec, PytestSpec) else spec
    payload = {
        "contract": contract.model_dump(mode="json"),
        "spec": {**asdict(spec), "mode": "script" if native else "pytest"},
        "verifier_spec": render_spec(protected_spec),
        "test_files": test_files,
    }
    if BOOTSTRAP_ARCHIVE in files:
        raise ValueError(f"Source task occupies protected bootstrap resource: {BOOTSTRAP_ARCHIVE}")
    encoded = json.dumps(payload).encode()
    SharedBootstrapPayload.model_validate_json(encoded)
    files[BOOTSTRAP_ARCHIVE] = zlib.compress(encoded)
    config = tomlkit.parse(files["task.toml"].decode())
    setup, instruction = split_source_setup(files["instruction.md"].decode())
    # Harbor uploads setup files before initialization and agent handoff. Keep
    # per-task payloads out of both command arguments and the common image.
    unpack = (
        "import pathlib,zlib; "
        f"pathlib.Path({BOOTSTRAP_INPUT!r}).write_bytes("
        f"zlib.decompress(pathlib.Path('/{BOOTSTRAP_ARCHIVE}').read_bytes()))"
    )
    control_python = "/opt/taskcompendium-shared-bootstrap/python"
    bootstrap_command = (
        "set -euo pipefail\n"
        + setup
        + "\n"
        + f'control_python="{control_python}"\n'
        + f'"$control_python" -I -S -c {shlex.quote(unpack)}\n'
        + f'"$control_python" -I -S {BOOTSTRAP} bootstrap {BOOTSTRAP_INPUT}\n'
        + f"rm {BOOTSTRAP_INPUT} /{BOOTSTRAP_ARCHIVE}\n"
    )
    environment = cast(Table, config["environment"])
    agent = cast(Table, config["agent"])
    verifier = cast(Table, config["verifier"])
    metadata = cast(Table, config["metadata"])
    environment["setup"] = {"command": bootstrap_command, "timeout_sec": 1800.0}
    source_user = str(agent.get("user", "root"))
    source_environment = dict(environment.get("env", {}))
    if source_user in {str(contract.agent_uid), "taskcompendium-agent"}:
        source_environment.setdefault("HOME", "/home/taskcompendium-agent")
    environment["env"] = source_environment
    agent["user"] = str(contract.agent_uid)
    verifier["user"] = "root"
    verifier["restart_environment"] = False
    metadata["integrity_contract"] = contract.model_dump(mode="json", exclude_none=True)
    metadata["execution_policy"] = f"shared_agent_mutable_v{contract.version}"
    files["task.toml"] = tomlkit.dumps(config).encode()
    files["instruction.md"] = shared_task_instruction(instruction, contract).encode()
    files["environment/taskcompendium-shared/control.py"] = source
    if native:
        files["environment/taskcompendium-shared/shared_pytest.py"] = Path(shared_pytest.__file__).read_bytes()
    files["environment/Dockerfile"] += shared_image_layer(contract).encode()
    files["tests/test.sh"] = shared_grader_script(contract)
    return contract


def shared_image_layer(contract: SharedIntegrityContract) -> str:
    """Install the maintained controller without replacing the source image context."""
    native = contract.native is not None
    return (
        "\nUSER 0:0\n"
        f"RUN getent passwd {contract.agent_uid} >/dev/null || "
        f"useradd --create-home --uid {contract.agent_uid} --user-group taskcompendium-agent\n"
        f"RUN getent passwd {contract.grader_uid} >/dev/null || "
        f"useradd --create-home --uid {contract.grader_uid} --user-group taskcompendium-grade\n"
        f"RUN mkdir -p /home/taskcompendium-agent && "
        f"chown {contract.agent_uid}:{contract.agent_uid} /home/taskcompendium-agent\n"
        f"COPY taskcompendium-shared/control.py {BOOTSTRAP}\n"
        + (
            "COPY taskcompendium-shared/shared_pytest.py /opt/taskcompendium-shared-bootstrap/shared_pytest.py\n"
            if native
            else ""
        )
        + 'RUN verifyit_path="$(readlink -f "$(command -v verifyit)")"; '
        'python_path="$(head -n 1 "$verifyit_path" | cut -c 3-)"; '
        'ln -s "$python_path" /opt/taskcompendium-shared-bootstrap/python; '
        'ln -s "$verifyit_path" /opt/taskcompendium-shared-bootstrap/verifyit\n'
        "RUN chmod -R go-w /opt/taskcompendium-shared-bootstrap\n"
    )
