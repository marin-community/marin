# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingestion-time SHARED protection for original Prime pytest-family source scorers.

Scale-SWE and SWE-Lego grade with their own hidden scorers and private patch
material instead of the SWE-rebench parser, so ``harden_shared_swe`` does not
apply. This module repackages one converted native package so the original
scorer runs unchanged behind the shared UID boundary:

- every ``tests/`` file becomes a root-only image payload, installed before the agent and
  unpacked into the protected root; the shipped ``tests/test.sh`` only launches
  the controller;
- bootstrap materializes the family's original test-restore scope, freezes it
  as the grading reference, pins a pristine ``.git``, and preserves the exact
  pre-bootstrap workspace state for the agent;
- grading restores that frozen state and the pinned ``.git`` (a declared
  versioned security change: committed agent history never reaches grading),
  then executes the original ``tests/test.sh`` bytes as the grading UID.
"""

import hashlib
import json
import re
import shlex
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import tomlkit
from pydantic import BaseModel, ConfigDict, JsonValue
from tomlkit.items import Table

from taskcompendium.harbor.shared_package import (
    SharedExecutionPolicy,
    SourceSetupPolicy,
    apply_shared_task_config,
    shared_private_image_layer,
    shared_task_instruction,
    shared_user_layer,
)
from taskcompendium.integrity import (
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
    SharedSourceFamily,
    SourceSharedRunner,
)
from taskcompendium.runtime import shared_pytest, shared_source
from taskcompendium.runtime.shared_source import patch_paths

BOOTSTRAP_INPUT = "/tmp/taskcompendium-source-input.json"
SOURCE_ARCHIVE = f"{PRIVATE_CONTEXT}/source-input.zlib"
SOURCE_PAYLOAD = f"{PRIVATE_DIRECTORY}/source-input.zlib"

COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")


class SourceBootstrapPayload(BaseModel):
    """Private input to the published guest controller's bootstrap operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract: SharedIntegrityContract
    tests: dict[str, str]
    spec: dict[str, JsonValue]


@dataclass(frozen=True)
class SourceGraderProfile:
    """The declared conversion contract for one original source family.

    ``removal_policy`` discloses, in the published contract, where this adapter
    removes agent-created files beyond the source's own restore scope. Scale's
    original ``RESTORE`` already removes test-pattern files absent from the
    pinned base commit, so its disclosure records source semantics; Lego's
    original ``REVERT_TEST_FILES`` touches only ``test_patch`` paths, so its
    disclosure records an adapter security policy.

    ``path_config_keys`` names private config entries holding explicit
    workspace-relative path lists that join the capture scope when they exist
    (a source that enumerates its trusted test files instead of following
    capture patterns).
    """

    family: SharedSourceFamily
    config_keys: frozenset[str]
    private_files: tuple[str, ...]
    capture_dirs: tuple[str, ...]
    capture_patterns: tuple[str, ...]
    patch_config_keys: tuple[str, ...]
    dependency_paths: tuple[str, ...]
    removal_policy: str
    path_config_keys: tuple[str, ...] = ()


SCALE_SWE_PROFILE = SourceGraderProfile(
    family=SharedSourceFamily.SCALE_SWE,
    config_keys=frozenset({"workdir", "base_commit", "f2p_patch", "f2p_script", "test_ids"}),
    private_files=("grade.py", "native_scripts.py", "scale_score.py", "config.json", "test.sh"),
    capture_dirs=("tests", "test", "Test", "Tests"),
    capture_patterns=("test_*.py", "*/test_*.py", "*_test.py", "*/*_test.py", "conftest.py", "*/conftest.py"),
    patch_config_keys=(),
    dependency_paths=("/opt/miniconda3/envs/testbed", "/opt/conda/envs/testbed"),
    removal_policy=(
        "The source RESTORE itself removes test-pattern files absent from the pinned base commit; "
        "the frozen reference additionally restores the same scope before grading."
    ),
)

SWE_LEGO_PROFILE = SourceGraderProfile(
    family=SharedSourceFamily.SWE_LEGO,
    config_keys=frozenset({"base_commit", "test_patch", "test_cmd", "test_ids"}),
    private_files=("grade.py", "native_scripts.py", "config.json", "test.sh"),
    capture_dirs=(),
    capture_patterns=(),
    patch_config_keys=("test_patch",),
    dependency_paths=("/opt/miniconda3/envs/testbed", "/opt/conda/envs/testbed"),
    removal_policy=(
        "The source REVERT restores only test_patch paths; agent-created files outside those paths are "
        "left untouched, matching source semantics. Agent-created files at captured patch paths are "
        "removed by the frozen reference restore (adapter security policy)."
    ),
)

SWE_REBENCH_PROFILE = SourceGraderProfile(
    family=SharedSourceFamily.SWE_REBENCH,
    config_keys=frozenset(
        {
            "workdir",
            "base_commit",
            "test_patch",
            "test_cmd",
            "test_files",
            "log_parser",
            "language",
            "test_ids",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        }
    ),
    private_files=("grade.py", "config.json", "test.sh", "test.patch"),
    capture_dirs=(),
    capture_patterns=(),
    patch_config_keys=("test_patch",),
    dependency_paths=(),
    removal_policy=(
        "The source's own test.sh restores test_files (git restore for base-tracked entries, rm -f otherwise) "
        "before applying test.patch - Scale-style source-semantics disclosure. The frozen reference additionally "
        "restores the captured grading-material scope (test_patch paths plus the row test_files list) before "
        "grading; agent-created files at captured paths are removed by the frozen reference restore (adapter "
        "security policy)."
    ),
    path_config_keys=("test_files",),
)

PROFILES = {
    SharedSourceFamily.SCALE_SWE: SCALE_SWE_PROFILE,
    SharedSourceFamily.SWE_LEGO: SWE_LEGO_PROFILE,
    SharedSourceFamily.SWE_REBENCH: SWE_REBENCH_PROFILE,
}


def source_contract(
    profile: SourceGraderProfile,
    config: dict,
    tests: dict[str, bytes],
    *,
    dependency_state: SharedDependencyState,
) -> SharedIntegrityContract:
    """Bind the family profile and this row's config into a versioned contract.

    ``tests`` holds the package's private ``tests/`` material keyed by
    tests-relative name; its canonical hash becomes the contract's reference.
    """
    workspace = (
        config.get("workdir", "/testbed")
        if profile.family in (SharedSourceFamily.SCALE_SWE, SharedSourceFamily.SWE_REBENCH)
        else "/testbed"
    )
    patterns: tuple[str, ...] = profile.capture_patterns
    if profile.family in (SharedSourceFamily.SWE_LEGO, SharedSourceFamily.SWE_REBENCH):
        patterns = tuple(dict.fromkeys((*patterns, *patch_paths(config["test_patch"]))))
    runner = SourceSharedRunner(
        family=profile.family,
        capture_dirs=profile.capture_dirs,
        capture_patterns=patterns,
        patch_config_keys=profile.patch_config_keys,
        path_config_keys=profile.path_config_keys,
        dependency_paths=profile.dependency_paths,
        shared_controller_sha256=hashlib.sha256(Path(shared_pytest.__file__).read_bytes()).hexdigest(),
    )
    protected_test_patterns = (
        *profile.capture_patterns,
        *(f"{directory}/**" for directory in profile.capture_dirs),
    )
    if profile.family is SharedSourceFamily.SWE_REBENCH:
        # Publish the row's grading-material scope: the test patch paths plus
        # the explicit test_files list (both are per-row config data).
        protected_test_patterns = tuple(dict.fromkeys((*patterns, *(config.get("test_files") or []))))
    return SharedIntegrityContract(
        id="shared-source",
        dependency_state=dependency_state,
        workspace=workspace,
        protected_test_patterns=protected_test_patterns,
        audit_probes=DEPENDENCY_AUDIT_PROBES,
        known_open_gaps=(
            "Agent-installed pytest plugins and imports in the shared interpreter environments",
            "Candidate-produced test-output injection",
        ),
        reference_sha256=hashlib.sha256(
            json.dumps({name: data.hex() for name, data in tests.items()}, sort_keys=True).encode()
        ).hexdigest(),
        harness_sha256=hashlib.sha256(Path(shared_source.__file__).read_bytes()).hexdigest(),
        native=runner,
    )


def parse_source_config(files: dict[str, bytes], profile: SourceGraderProfile) -> dict:
    """Validate the private scorer configuration against the family schema."""
    config = json.loads(files["tests/config.json"])
    if not isinstance(config, dict) or set(config) != profile.config_keys:
        raise ValueError(f"{profile.family.value} config keys differ from the declared family schema")
    if COMMIT_PATTERN.fullmatch(config["base_commit"]) is None:
        raise ValueError("Source grading requires a pinned base commit")
    if not config["test_ids"]:
        raise ValueError("The source scorer requires nonempty F2P/P2P test ids")
    for key in profile.patch_config_keys:
        if not isinstance(config[key], str):
            raise ValueError(f"Source patch key {key} must be a string")
    for key in profile.path_config_keys:
        value = config[key]
        if not isinstance(value, list) or any(not isinstance(entry, str) for entry in value):
            raise ValueError(f"Source path key {key} must be a list of workspace-relative paths")
        if any(PurePosixPath(entry).is_absolute() or ".." in PurePosixPath(entry).parts for entry in value):
            raise ValueError(f"Source path key {key} entries must be relative workspace paths")
    if profile.family is SharedSourceFamily.SCALE_SWE and not config["workdir"].startswith("/"):
        raise ValueError("Scale-SWE workdir must be an absolute workspace path")
    if profile.family is SharedSourceFamily.SWE_REBENCH and not config["workdir"].startswith("/"):
        raise ValueError("SWE-rebench workdir must be an absolute workspace path")
    return config


def source_image_layer(contract: SharedIntegrityContract) -> str:
    """Install the controller and grading identities without touching the source image."""
    agent = contract.agent_uid
    return (
        shared_user_layer(contract)
        # UID handover at build time: the build daemon has the full capability
        # set, so workspace files whose owner the sandbox fs refuses to change
        # at runtime are converted here (image files are agent-owned before the
        # agent phase; setup-created files stay runtime-chowned). mkdir -p
        # covers families whose workspace is created by the per-task setup
        # (e.g. a repository cloned at task start on a shared base image) - the
        # bootstrap hands the cloned tree to the agent afterwards.
        + f"RUN mkdir -p {contract.workspace} && chown -R {agent}:{agent} {contract.workspace}\n"
        f"COPY {CONTROLLER_CONTEXT.removeprefix('environment/')} {BOOTSTRAP_CONTROLLER}\n"
        f"COPY {SHARED_CONTROLLER_CONTEXT.removeprefix('environment/')} {BOOTSTRAP_DIRECTORY}/shared_pytest.py\n"
        f"RUN chmod -R go-w {BOOTSTRAP_DIRECTORY}\n" + shared_private_image_layer()
    )


def source_bootstrap_command(setup: str, workspace: str) -> str:
    """Run the source setup, then unpack and bootstrap the protected controller.

    Python resolution happens here, with the task environment's PATH, because the
    source image is not guaranteed to keep an interpreter on its build PATH. The
    controller requires 3.9+ and takes the first PATH interpreter that qualifies;
    the source's own scorer keeps running on the task environment's python.
    """
    unpack = (
        "import pathlib,zlib; "
        f"pathlib.Path({BOOTSTRAP_INPUT!r}).write_bytes("
        f"zlib.decompress(pathlib.Path({SOURCE_PAYLOAD!r}).read_bytes()))"
    )
    probe = "import sys; assert sys.version_info >= (3, 9); print(sys.executable)"
    lines = [
        "set -euo pipefail",
        f"git config --global --add safe.directory {shlex.quote(workspace)}",
        setup,
        'control_python=""',
        'for control_dir in $(printf "%s\n" "$PATH" | tr ":" " "); do',
        "  for control_name in python3 python; do",
        '    [ -x "$control_dir/$control_name" ] || continue',
        f'    if control_python="$("$control_dir/$control_name" -I -S -B -c {shlex.quote(probe)} 2>/dev/null)"; then',
        "      break 2",
        "    fi",
        "  done",
        "done",
        'if [ -z "$control_python" ]; then',
        '  echo "shared-source adapter requires a python >= 3.9 on PATH" >&2',
        "  exit 1",
        "fi",
        'control_python="$(readlink -f "$control_python")"',
        f'ln -sfn "$control_python" {BOOTSTRAP_DIRECTORY}/python',
        f'"$control_python" -I -S -B -c {shlex.quote(unpack)}',
        f'"$control_python" -I -S -B {BOOTSTRAP_CONTROLLER} bootstrap {BOOTSTRAP_INPUT}',
        f"rm {BOOTSTRAP_INPUT} {SOURCE_PAYLOAD}",
    ]
    return "\n".join(lines) + "\n"


def source_grader_script(contract: SharedIntegrityContract) -> bytes:
    """Run the pinned controller's grader on the protected contract."""
    return (
        "#!/bin/bash\nset -euo pipefail\n"
        f'exec "{BOOTSTRAP_DIRECTORY}/python" -I -S -B {contract.harness} grade '
        f"--contract {contract.protected_root}/contract.json\n"
    ).encode()


def harden_shared_source(
    files: dict[str, bytes],
    *,
    profile: SourceGraderProfile,
    dependency_state: SharedDependencyState,
) -> SharedIntegrityContract:
    """Rewrite one converted native package with a complete, versioned SHARED contract.

    The verifier deadline is the source's own declaration: a declared
    ``verifier.timeout_sec`` passes through verbatim to the controller spec, and
    an undeclared deadline is preserved as absent (the controller waits
    unbounded under the same outer enforcement the source runs under). The
    caller must publish the package under a new dataset version and qualify it
    with fresh oracle/NoP QA; this function never changes an already-published
    task at runtime.
    """
    missing = [f"tests/{name}" for name in profile.private_files if f"tests/{name}" not in files]
    missing += [name for name in ("task.toml", "instruction.md", "environment/Dockerfile") if name not in files]
    if missing:
        raise ValueError(f"Source package lacks protected grading files: {missing}")
    unexpected = [
        name for name in files if name.startswith("tests/") and name.removeprefix("tests/") not in profile.private_files
    ]
    if unexpected:
        raise ValueError(f"Source package declares undeclared tests files: {sorted(unexpected)}")
    task_config = tomlkit.parse(files["task.toml"].decode())
    verifier_table = task_config.get("verifier")
    declared_timeout = (
        verifier_table.get("timeout_sec").value
        if verifier_table is not None and "timeout_sec" in verifier_table
        else None
    )
    config = parse_source_config(files, profile)
    tests = {name.removeprefix("tests/"): data for name, data in files.items() if name.startswith("tests/")}
    contract = source_contract(profile, config, tests, dependency_state=dependency_state)
    payload = SourceBootstrapPayload(
        contract=contract,
        tests={name: data.hex() for name, data in tests.items()},
        spec={"timeout": declared_timeout},
    )
    if SOURCE_ARCHIVE in files:
        raise ValueError(f"Source task occupies protected bootstrap resource: {SOURCE_ARCHIVE}")
    files[SOURCE_ARCHIVE] = zlib.compress(payload.model_dump_json().encode())
    for name in profile.private_files:
        del files[f"tests/{name}"]
    files["tests/test.sh"] = source_grader_script(contract)

    environment = cast(Table, task_config.setdefault("environment", tomlkit.table()))
    setup_table = cast(Table, environment.setdefault("setup", tomlkit.table()))
    source_setup = str(setup_table.get("command", ""))
    apply_shared_task_config(
        task_config,
        contract,
        source_bootstrap_command(source_setup, contract.workspace),
        setup_timeout=BOOTSTRAP_TIMEOUT,
        source_setup_policy=SourceSetupPolicy.INCLUDED,
        execution_policy=SharedExecutionPolicy.MUTABLE,
        verifier_timeout=declared_timeout,
    )
    files["task.toml"] = tomlkit.dumps(task_config).encode()
    instruction = files["instruction.md"].decode()
    if instruction.startswith("## Environment Setup (complete these steps first)\n\n```bash\n"):
        raise ValueError("Source family declares its setup inside the instruction, unsupported here")
    files["instruction.md"] = shared_task_instruction(instruction, contract).encode()
    files[CONTROLLER_CONTEXT] = Path(shared_source.__file__).read_bytes()
    files[SHARED_CONTROLLER_CONTEXT] = Path(shared_pytest.__file__).read_bytes()
    files["environment/Dockerfile"] += source_image_layer(contract).encode()
    return contract


def published_contract(files: dict[str, bytes]) -> SharedIntegrityContract:
    """Read back the contract a hardened package publishes in its metadata."""
    config = tomlkit.parse(files["task.toml"].decode())
    document = config["metadata"].unwrap()
    return SharedIntegrityContract.model_validate(document["integrity_contract"])
