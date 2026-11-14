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

from taskcompendium.harbor.shared_package import SharedPytestFormat, render_shared_pytest_spec
from taskcompendium.harbor.shared_reference import (
    REFERENCE_INPUT,
    FrozenSharedReference,
    harden_shared_swe_reference,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.integrity import SharedDependencyState, SharedIntegrityContract
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
        "instruction.md": b"## Environment Setup (complete these steps first)\n\n```bash\ngit clone source .\n```\nRepair.\n",
        "environment/Dockerfile": b"FROM original@sha256:" + b"a" * 64 + b"\n",
        "solution/solve.sh": b"#!/bin/bash\ngit clone original-fixed .\n",
        "tests/test.sh": b"#!/bin/bash\nexec verifyit /tests/verifier.toml\n",
        "tests/trusted.txt": b"tests/test_value.py\n",
    }
    controller = Path(shared_pytest.__file__).read_bytes()
    contract = SharedIntegrityContract(
        dependency_state=SharedDependencyState.AGENT_MUTABLE,
        reference_sha256=hashlib.sha256(
            json.dumps({k.removeprefix("tests/"): v.hex() for k, v in files.items() if k.startswith("tests/")}, sort_keys=True).encode()
        ).hexdigest(),
        harness_sha256=hashlib.sha256(controller).hexdigest(),
    )
    # This fixture exercises compilation. Full runtime bundles are covered by Linux UID tests.
    captured = {
        "contract.json": json.dumps(contract.model_dump(mode="json")).encode(),
        "tests/verifier.toml": render_shared_pytest_spec(
            replace(spec, setup="", python=f"{contract.protected_root}/pytest-python"),
            SharedPytestFormat.SINGLE_PROTECTED_MANIFEST,
        ).encode(),
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
    assert files["instruction.md"].endswith(original["instruction.md"])
    assert files["solution/solve.sh"] == original["solution/solve.sh"]
    assert {k: v for k, v in files.items() if k.startswith("environment/")} == {
        k: v for k, v in original.items() if k.startswith("environment/")
    }
    assert FrozenSharedReference.model_validate_json(files[REFERENCE_INPUT]) == reference


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
