# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import tarfile
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import tomlkit
from verifyit.spec import PytestSpec

from taskcompendium.harbor.shared_package import SharedPytestFormat, render_shared_pytest_spec, shared_grader_script
from taskcompendium.harbor.shared_reference import (
    ACTOR_PREFIX,
    REFERENCE_INPUT,
    ActorEnvironment,
    FrozenSharedReference,
    actor_environment_for,
    harden_shared_swe_reference,
    shared_actor_preamble,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.integrity import SharedDependencyState, SharedIntegrityContract, shared_preamble
from taskcompendium.runtime import shared_pytest


@pytest.fixture
def source_reference():
    spec = PytestSpec(
        paths=("tests/test_value.py",),
        protected_paths_files=("trusted.txt",),
        must_pass=("tests/test_value.py::test_value",),
        setup="restore-source-tests",
        timeout=600.0,
        workspace="/testbed",
    )
    files = {
        "task.toml": (
            b"[metadata]\n[environment]\n[agent]\ntimeout_sec=900\n"
            b'[verifier]\ntimeout_sec=600\nenvironment_mode="shared"\n'
        ),
        "instruction.md": (
            b"## Environment Setup (complete these steps first)\n\n```bash\ngit clone source .\n```\nRepair.\n"
        ),
        "environment/Dockerfile": b"FROM original@sha256:" + b"a" * 64 + b"\n",
        "solution/solve.sh": b"#!/bin/bash\ngit clone original-fixed .\n",
        "tests/test.sh": b"#!/bin/bash\nexec verifyit /tests/verifier.toml\n",
        "tests/trusted.txt": b"tests/test_value.py\n",
    }
    controller = Path(shared_pytest.__file__).read_bytes()
    contract = SharedIntegrityContract(
        dependency_state=SharedDependencyState.AGENT_MUTABLE,
        reference_sha256=hashlib.sha256(
            json.dumps(
                {k.removeprefix("tests/"): v.hex() for k, v in files.items() if k.startswith("tests/")}, sort_keys=True
            ).encode()
        ).hexdigest(),
        harness_sha256=hashlib.sha256(controller).hexdigest(),
    )
    # This fixture exercises compilation. Full runtime bundles are covered by Linux UID tests.
    captured = {
        "contract.json": json.dumps(contract.model_dump(mode="json")).encode(),
        "tests/verifier.toml": (
            render_shared_pytest_spec(
                replace(spec, setup="", python=f"{contract.protected_root}/pytest-python"),
                SharedPytestFormat.SINGLE_PROTECTED_MANIFEST,
            ).encode()
        ),
        "control.py": controller,
    }
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content in captured.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    bundle = stream.getvalue()
    reference = FrozenSharedReference(
        original_files_sha256=canonical_sha256({k: v.hex() for k, v in files.items()}),
        original_task_sha256="1" * 64,
        original_runtime_sha256="2" * 64,
        source_spec_sha256=canonical_sha256(asdict(spec)),
        image_context_sha256=canonical_sha256({k: v.hex() for k, v in files.items() if k.startswith("environment/")}),
        source_cli_revision="a" * 40,
        capture_trial_uri="s3://fixture/original-trial.tar.zst",
        capture_trial_sha256="3" * 64,
        bundle_sha256=hashlib.sha256(bundle).hexdigest(),
        contract=contract,
        actor_environment=actor_environment_for(contract),
    )
    return files, spec, reference, bundle


def test_reference_compilation_preserves_original_setup_oracle_image_and_source_clocks(source_reference):
    files, spec, reference, bundle = source_reference
    original = dict(files)
    harden_shared_swe_reference(
        files,
        spec=spec,
        reference=reference,
        bundle=bundle,
        pytest_format=SharedPytestFormat.SINGLE_PROTECTED_MANIFEST,
        host_init_timeout=259200,
    )
    config = tomlkit.parse(files["task.toml"].decode()).unwrap()
    assert config["agent"]["timeout_sec"] == 900
    assert config["verifier"]["timeout_sec"] == 600
    assert config["environment"]["setup"]["timeout_sec"] == 259200
    assert config["agent"]["user"] == str(reference.contract.agent_uid)
    assert config["verifier"]["user"] == "root"
    assert config["verifier"]["restart_environment"] is False
    actor = reference.actor_environment
    assert config["environment"]["env"]["PATH"] == actor.path
    assert ActorEnvironment.model_validate(config["metadata"]["actor_environment"]) == actor
    assert config["metadata"]["execution_policy"] == "shared_actor_environment_v1"
    # The public setup stays at task time: the initializer command must not absorb it.
    assert "git clone" not in config["environment"]["setup"]["command"]
    assert files["instruction.md"].startswith(
        (shared_preamble(reference.contract) + shared_actor_preamble(actor)).encode()
    )
    assert files["instruction.md"].endswith(original["instruction.md"])
    assert files["solution/solve.sh"] == original["solution/solve.sh"]
    assert files["tests/test.sh"] == shared_grader_script(reference.contract)
    assert {k: v for k, v in files.items() if k.startswith("environment/")} == {
        k: v for k, v in original.items() if k.startswith("environment/")
    }
    assert FrozenSharedReference.model_validate_json(files[REFERENCE_INPUT]) == reference


def test_declared_actor_environment_lists_every_difference(source_reference):
    _, _, reference, _ = source_reference
    actor = reference.actor_environment
    assert actor.prefix == ACTOR_PREFIX
    assert actor.uid == actor.gid == reference.contract.agent_uid
    assert actor.path.startswith(f"{actor.bin}:")
    changes = "\n".join(actor.declared_changes)
    assert "runs as UID 10001 instead of root" in changes
    assert f"agent-owned Python environment at {actor.python}" in changes
    assert "copies the exact source interpreter" in changes
    assert "source-baseline path file" in changes
    assert "no package is upgraded" in changes
    assert f"pip shim at {actor.bin}/pip" in changes
    assert "source pip module from the captured baseline" in changes
    assert f"The agent environment PATH is {actor.path}" in changes
    assert "owned by UID 10001" in changes
    assert "stay root-owned and unmodified" in changes
    assert "grading runtime under /opt/taskcompendium-shared stays root-owned and frozen" in changes


def test_actor_prefix_and_protected_root_must_be_disjoint(source_reference):
    _, _, reference, _ = source_reference
    displaced = reference.model_copy(
        update={
            "actor_environment": actor_environment_for(
                reference.contract, prefix=str(Path(reference.contract.protected_root) / "agent")
            )
        }
    )
    with pytest.raises(ValueError, match="disjoint"):
        displaced.validate_environment()


def test_actor_environment_belongs_to_the_contract_identity(source_reference):
    _, _, reference, _ = source_reference
    actor = dict(reference.actor_environment.model_dump(mode="json"))
    actor["uid"] = 19999
    tampered = reference.model_copy(update={"actor_environment": ActorEnvironment.model_validate(actor)})
    with pytest.raises(ValueError, match="another agent identity"):
        tampered.validate_environment()


def test_reference_from_another_native_source_package_cannot_be_installed(source_reference):
    files, spec, reference, bundle = source_reference
    files["environment/Dockerfile"] += b"RUN substitute-different-dependencies\n"
    with pytest.raises(ValueError, match="another native source package"):
        harden_shared_swe_reference(
            files,
            spec=spec,
            reference=reference,
            bundle=bundle,
            pytest_format=SharedPytestFormat.SINGLE_PROTECTED_MANIFEST,
            host_init_timeout=259200,
        )
