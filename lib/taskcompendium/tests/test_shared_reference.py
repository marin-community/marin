# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import shlex
import subprocess
import tarfile
import zlib
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import tomlkit
from verifyit.spec import PytestSpec

from taskcompendium.harbor.shared_package import (
    BOOTSTRAP_ARCHIVE,
    SharedExecutionPolicy,
    SharedPytestFormat,
    SourceSetupPolicy,
    apply_shared_task_config,
    harden_shared_swe,
    render_shared_pytest_spec,
    shared_grader_script,
)
from taskcompendium.harbor.shared_reference import (
    REFERENCE_INPUT,
    ActorEnvironment,
    FrozenSharedReference,
    actor_environment_for,
    harden_shared_swe_reference,
    shared_actor_preamble,
)
from taskcompendium.identity import canonical_sha256
from taskcompendium.integrity import (
    AGENT_HOME,
    PRIVATE_CONTEXT,
    PRIVATE_DIRECTORY,
    SharedDependencyState,
    SharedIntegrityContract,
    shared_preamble,
)
from taskcompendium.runtime import shared_pytest


@pytest.fixture(params=[b"FROM original@sha256:" + b"a" * 64 + b"\n"])
def source_reference(request):
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
        "environment/Dockerfile": request.param,
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


def test_shared_config_preserves_source_initialization_and_replaces_root_home(tmp_path, source_reference):
    _, _, reference, _ = source_reference
    output = tmp_path / "initialization.txt"
    config = tomlkit.document()
    config["environment"] = {
        "env": {"HOME": "/root"},
        "setup": {"command": f"printf 'source:%s\\n' \"$HOME\" >> {shlex.quote(str(output))}"},
    }
    apply_shared_task_config(
        config,
        reference.contract,
        f"printf 'bootstrap\\n' >> {shlex.quote(str(output))}",
        setup_timeout=180,
        source_setup_policy=SourceSetupPolicy.PREPEND,
        execution_policy=SharedExecutionPolicy.MUTABLE,
    )
    environment = config["environment"]
    subprocess.run(["/bin/bash", "-c", environment["setup"]["command"]], env=environment["env"].unwrap(), check=True)
    assert output.read_text().splitlines() == ["source:/root", "bootstrap"]
    actor = subprocess.run(
        ["/bin/bash", "-c", 'printf "%s" "$HOME"'],
        env=environment["env"].unwrap(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert actor.stdout == AGENT_HOME


@pytest.mark.docker
@pytest.mark.parametrize("source_reference", [b"FROM ghcr.io/astral-sh/uv:python3.12-bookworm\n"], indirect=True)
def test_built_image_keeps_private_reference_unreadable_without_initialization(tmp_path, source_reference):
    files, spec, reference, bundle = source_reference
    harden_shared_swe_reference(
        files,
        spec=spec,
        reference=reference,
        bundle=bundle,
        pytest_format=SharedPytestFormat.SINGLE_PROTECTED_MANIFEST,
        host_init_timeout=180,
    )
    for name, content in files.items():
        if name.startswith("environment/"):
            target = tmp_path / name.removeprefix("environment/")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    built = subprocess.run(["docker", "build", "--quiet", str(tmp_path)], check=True, capture_output=True, text=True)
    image = built.stdout.strip()
    actor_user = str(tomlkit.parse(files["task.toml"].decode())["agent"]["user"])
    paths = [f"{PRIVATE_DIRECTORY}/{Path(name).name}" for name in files if name.startswith(PRIVATE_CONTEXT + "/")]
    try:
        root_probe = "import pathlib,sys,json;assert all(pathlib.Path(p).read_bytes() for p in json.loads(sys.argv[1]))"
        subprocess.run(
            ["docker", "run", "--rm", "--user", "0:0", image, "python", "-I", "-c", root_probe, json.dumps(paths)],
            check=True,
        )
        actor_probe = (
            "import pathlib,sys,json,os\n"
            f"assert os.getuid() == {reference.contract.agent_uid}\n"
            "for name in json.loads(sys.argv[1]):\n"
            "    try:\n"
            "        pathlib.Path(name).read_bytes()\n"
            "    except PermissionError:\n"
            "        continue\n"
            "    raise AssertionError('actor read private image input')\n"
        )
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                actor_user,
                image,
                "python",
                "-I",
                "-c",
                actor_probe,
                json.dumps(paths),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    finally:
        subprocess.run(["docker", "image", "rm", image], check=True, capture_output=True)


@pytest.mark.parametrize("custom_actor", [False, True])
def test_reference_compilation_preserves_original_setup_oracle_image_and_source_clocks(source_reference, custom_actor):
    files, spec, reference, bundle = source_reference
    if custom_actor:
        reference = reference.model_copy(
            update={
                "actor_environment": actor_environment_for(
                    reference.contract, prefix="/opt/custom-actor", source_path="/opt/source/bin:/usr/bin"
                )
            }
        )
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
    assert files["environment/Dockerfile"].startswith(original["environment/Dockerfile"])
    assert FrozenSharedReference.model_validate_json(files[REFERENCE_INPUT]) == reference


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


def test_compiled_package_keeps_source_manifests_in_private_bootstrap_payload(source_reference):
    files, spec, reference, _ = source_reference
    original_verifier = files["tests/test.sh"]
    files["tests/verifier.toml"] = b"source verifier contract"
    contract = harden_shared_swe(
        files,
        spec=replace(spec, protected_paths_files=("/tests/trusted.txt",)),
        dependency_state=SharedDependencyState.AGENT_MUTABLE,
        pytest_format=SharedPytestFormat.SINGLE_PROTECTED_MANIFEST,
    )
    payload = json.loads(zlib.decompress(files[BOOTSTRAP_ARCHIVE]))
    assert bytes.fromhex(payload["test_files"]["verifier.toml"]) == b"source verifier contract"
    assert bytes.fromhex(payload["test_files"]["test.sh"]) == original_verifier
    verifier = tomlkit.parse(payload["verifier_spec"])
    assert verifier["protected_paths_file"] == "trusted.txt"
    assert "protected_paths_files" not in verifier
    assert files["tests/test.sh"] == shared_grader_script(contract)
    published = tomlkit.parse(files["task.toml"].decode())
    assert published["agent"]["user"] == str(reference.contract.agent_uid)
    assert published["verifier"]["user"] == "root"
