# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingestion-time SHARED SWE protection, independent of population-specific image builds."""

import base64
import hashlib
import json
import shlex
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import tomlkit
from taskcompendium.integrity import SharedDependencyState, SharedIntegrityContract, shared_preamble
from taskcompendium.runtime import shared_pytest
from verifyit.spec import PytestSpec, render_spec

BOOTSTRAP = "/opt/taskcompendium-shared-bootstrap/control.py"
SETUP_PREFIX = "## Environment Setup (complete these steps first)\n\n```bash\n"


def harden_shared_swe(
    files: dict[str, bytes], *, spec: PytestSpec, dependency_state: SharedDependencyState
) -> SharedIntegrityContract:
    """Rewrite one new native package with a complete, versioned SHARED contract.

    The caller must publish this package under a new dataset version and qualify
    it; this function never changes an already-published task at runtime.
    """
    source = Path(shared_pytest.__file__).read_bytes()
    test_files = {name.removeprefix("tests/"): data.hex() for name, data in files.items() if name.startswith("tests/")}
    reference = hashlib.sha256(json.dumps(test_files, sort_keys=True).encode()).hexdigest()
    contract = SharedIntegrityContract(
        dependency_state=dependency_state, reference_sha256=reference, harness_sha256=hashlib.sha256(source).hexdigest()
    )
    root = contract.protected_root
    protected_spec = replace(spec, setup="", python=f"{root}/pytest-python")
    payload = {
        "contract": contract.model_dump(mode="json"),
        "spec": {**asdict(spec), "mode": "pytest"},
        "verifier_spec": render_spec(protected_spec),
        "test_files": test_files,
    }
    packed = base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    config = tomlkit.parse(files["task.toml"].decode())
    instruction = files["instruction.md"].decode()
    setup = ""
    if instruction.startswith(SETUP_PREFIX):
        setup, boundary, instruction = instruction[len(SETUP_PREFIX) :].partition("\n```\n")
        if not boundary:
            raise ValueError("Incomplete source environment setup block")
        instruction = instruction.removeprefix("\n---\n\n")
    # The reference payload lives in task.toml's bounded setup, never the common
    # image context: distinct task setup does not produce thousands of snapshots.
    unpack = (
        "import base64,pathlib,zlib; "
        f"pathlib.Path('/tmp/taskcompendium-shared-input.json').write_bytes(zlib.decompress(base64.b64decode({packed!r})))"
    )
    control_python = "/opt/taskcompendium-shared-bootstrap/python"
    bootstrap_command = (
        "set -euo pipefail\n"
        + setup
        + "\n"
        + f'control_python="{control_python}"\n'
        + f'"$control_python" -I -S -c {shlex.quote(unpack)}\n'
        + f'"$control_python" -I -S {BOOTSTRAP} bootstrap /tmp/taskcompendium-shared-input.json\n'
        + "rm /tmp/taskcompendium-shared-input.json\n"
    )
    config["environment"]["setup"] = {"command": bootstrap_command, "timeout_sec": 1800.0}
    config["environment"]["env"] = {**dict(config["environment"].get("env", {})), "HOME": "/home/taskcompendium-agent"}
    config["agent"]["user"] = str(contract.agent_uid)
    config["verifier"]["user"] = "root"
    config["verifier"]["restart_environment"] = False
    config["metadata"]["integrity_contract"] = contract.model_dump(mode="json")
    config["metadata"]["execution_policy"] = f"shared_agent_mutable_v{contract.version}"
    files["task.toml"] = tomlkit.dumps(config).encode()
    files["instruction.md"] = (
        shared_preamble(contract)
        + "The task workspace and initial dependencies are prepared before you start.\n\n"
        + instruction
    ).encode()
    files["environment/taskcompendium-shared/control.py"] = source
    dockerfile = files["environment/Dockerfile"].decode()
    dockerfile += (
        "\nUSER 0:0\n"
        "RUN getent passwd 10001 >/dev/null || useradd --create-home --uid 10001 --user-group taskcompendium-agent\n"
        "RUN getent passwd 10002 >/dev/null || useradd --create-home --uid 10002 --user-group taskcompendium-grade\n"
        "RUN mkdir -p /home/taskcompendium-agent && chown 10001:10001 /home/taskcompendium-agent\n"
        f"COPY taskcompendium-shared/control.py {BOOTSTRAP}\n"
        'RUN verifyit_path="$(readlink -f "$(command -v verifyit)")"; '
        'python_path="$(head -n 1 "$verifyit_path" | cut -c 3-)"; '
        'ln -s "$python_path" /opt/taskcompendium-shared-bootstrap/python; '
        'ln -s "$verifyit_path" /opt/taskcompendium-shared-bootstrap/verifyit\n'
        "RUN chmod -R go-w /opt/taskcompendium-shared-bootstrap\n"
    )
    files["environment/Dockerfile"] = dockerfile.encode()
    files["tests/test.sh"] = (
        "#!/bin/bash\nset -euo pipefail\n"
        f'control_python="{control_python}"\n'
        f'exec "$control_python" -I -S {contract.harness} grade --contract {root}/contract.json '
        "--verifyit /opt/taskcompendium-shared-bootstrap/verifyit\n"
    ).encode()
    return contract
