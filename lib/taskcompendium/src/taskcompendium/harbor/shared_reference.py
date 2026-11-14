# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile a native SHARED task from a separately captured trusted grading reference."""

import hashlib
import io
import json
import shlex
import tarfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, cast

import tomlkit
from pydantic import BaseModel, ConfigDict, Field
from tomlkit.items import Table
from verifyit.spec import PytestSpec

from taskcompendium.harbor.shared_package import (
    SharedPytestFormat,
    render_shared_pytest_spec,
    shared_grader_script,
    shared_source_spec,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.integrity import SharedIntegrityContract, shared_preamble
from taskcompendium.runtime import shared_pytest, shared_reference

REFERENCE_INPUT = "setup_files/taskcompendium-frozen-reference.json"
REFERENCE_BUNDLE = "setup_files/taskcompendium-frozen-reference.tar.gz"
REFERENCE_INITIALIZER = "setup_files/taskcompendium-frozen-reference.py"


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
    environment = cast(Table, config["environment"])
    if environment.get("setup") is not None:
        raise ValueError("Reference mode cannot reorder an existing source initialization command")
    files[REFERENCE_INPUT] = reference.model_dump_json().encode()
    files[REFERENCE_BUNDLE] = bundle
    files[REFERENCE_INITIALIZER] = Path(shared_reference.__file__).read_bytes()
    environment["setup"] = {
        "command": (
            "set -euo pipefail\n"
            'verifyit_path="$(readlink -f "$(command -v verifyit)")"\n'
            'control_python="$(head -n 1 "$verifyit_path" | cut -c 3-)"\n'
            '"$control_python" -I -S '
            + shlex.quote("/" + REFERENCE_INITIALIZER)
            + " "
            + shlex.quote("/" + REFERENCE_INPUT)
            + " "
            + shlex.quote("/" + REFERENCE_BUNDLE)
            + "\nrm "
            + shlex.quote("/" + REFERENCE_INITIALIZER)
            + "\n"
        ),
        "timeout_sec": host_init_timeout,
    }
    cast(Table, config["agent"])["user"] = str(contract.agent_uid)
    verifier = cast(Table, config["verifier"])
    verifier["user"] = "root"
    verifier["restart_environment"] = False
    metadata = cast(Table, config["metadata"])
    metadata["integrity_contract"] = contract.model_dump(mode="json", exclude_none=True)
    metadata["execution_policy"] = "shared_frozen_reference_v1"
    files["task.toml"] = tomlkit.dumps(config).encode()
    files["instruction.md"] = shared_preamble(contract).encode() + files["instruction.md"]
    files["tests/test.sh"] = shared_grader_script(contract)
    return contract
