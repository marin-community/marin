# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingestion-time SHARED SWE protection, independent of population-specific image builds."""

import hashlib
import json
import shlex
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import tomlkit
from taskcompendium.integrity import SharedDependencyState, SharedIntegrityContract, shared_preamble
from taskcompendium.native_integrity import TEST_PATTERNS
from taskcompendium.runtime import shared_native, shared_pytest
from verifyit.spec import PytestSpec, ScriptSpec, render_spec

from experiments.post_training.task_curation.tasktrove.native_shared_swe import native_runner

BOOTSTRAP = "/opt/taskcompendium-shared-bootstrap/control.py"
BOOTSTRAP_INPUT = "/tmp/taskcompendium-shared-input.json"
BOOTSTRAP_ARCHIVE = "setup_files/taskcompendium-shared-input.zlib"
SETUP_PREFIX = "## Environment Setup (complete these steps first)\n\n```bash\n"


def harden_shared_swe(
    files: dict[str, bytes], *, spec: PytestSpec | ScriptSpec, dependency_state: SharedDependencyState
) -> SharedIntegrityContract:
    """Rewrite one new native package with a complete, versioned SHARED contract.

    The caller must publish this package under a new dataset version and qualify
    it; this function never changes an already-published task at runtime.
    """
    if isinstance(spec, PytestSpec):
        spec = replace(
            spec,
            protected_paths_files=tuple(
                Path(path).relative_to("/tests").as_posix() if Path(path).is_absolute() else path
                for path in spec.protected_paths_files
            ),
        )
    native = native_runner(files) if isinstance(spec, ScriptSpec) else None
    source = Path(shared_native.__file__ if native else shared_pytest.__file__).read_bytes()
    test_files = {name.removeprefix("tests/"): data.hex() for name, data in files.items() if name.startswith("tests/")}
    reference = hashlib.sha256(json.dumps(test_files, sort_keys=True).encode()).hexdigest()
    contract = SharedIntegrityContract(
        dependency_state=dependency_state,
        reference_sha256=reference,
        harness_sha256=hashlib.sha256(source).hexdigest(),
        **(
            {
                "id": "shared-native",
                "native": native,
                "protected_test_patterns": TEST_PATTERNS,
                "audit_probes": ("dependencies-diff.json", "startup-hooks-diff.json"),
                "known_open_gaps": ("Candidate-produced test-output injection", "Mutable dependency startup hooks"),
            }
            if native
            else {}
        ),
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
    files[BOOTSTRAP_ARCHIVE] = zlib.compress(json.dumps(payload).encode())
    config = tomlkit.parse(files["task.toml"].decode())
    instruction = files["instruction.md"].decode()
    setup = ""
    if instruction.startswith(SETUP_PREFIX):
        setup, boundary, instruction = instruction[len(SETUP_PREFIX) :].partition("\n```\n")
        if not boundary:
            raise ValueError("Incomplete source environment setup block")
        instruction = instruction.removeprefix("\n---\n\n")
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
    config["environment"]["setup"] = {"command": bootstrap_command, "timeout_sec": 1800.0}
    source_user = str(config["agent"].get("user", "root"))
    source_environment = dict(config["environment"].get("env", {}))
    if source_user in {str(contract.agent_uid), "taskcompendium-agent"}:
        source_environment.setdefault("HOME", "/home/taskcompendium-agent")
    config["environment"]["env"] = source_environment
    config["agent"]["user"] = str(contract.agent_uid)
    config["verifier"]["user"] = "root"
    config["verifier"]["restart_environment"] = False
    config["metadata"]["integrity_contract"] = contract.model_dump(mode="json", exclude_none=True)
    config["metadata"]["execution_policy"] = f"shared_agent_mutable_v{contract.version}"
    files["task.toml"] = tomlkit.dumps(config).encode()
    files["instruction.md"] = (
        shared_preamble(contract)
        + "The task workspace and initial dependencies are prepared before you start.\n\n"
        + instruction
    ).encode()
    files["environment/taskcompendium-shared/control.py"] = source
    if native:
        files["environment/taskcompendium-shared/shared_pytest.py"] = Path(shared_pytest.__file__).read_bytes()
    dockerfile = files["environment/Dockerfile"].decode()
    dockerfile += (
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
    files["environment/Dockerfile"] = dockerfile.encode()
    files["tests/test.sh"] = (
        "#!/bin/bash\nset -euo pipefail\n"
        f'control_python="{control_python}"\n'
        f'exec "$control_python" -I -S {contract.harness} grade --contract {root}/contract.json '
        "--verifyit /opt/taskcompendium-shared-bootstrap/verifyit\n"
    ).encode()
    return contract
