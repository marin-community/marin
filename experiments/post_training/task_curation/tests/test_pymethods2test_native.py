# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import io
import tarfile
import tomllib

import pytest
from taskcompendium.integrity import SharedDependencyState
from taskcompendium.models import ScriptGrader, Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.pymethods2test.tasks import (
    convert_pymethods,
    decode_pymethods,
)


def native_files():
    config = (
        b'version = "1.0"\n\n[agent]\ntimeout_sec = 900.0\n\n[metadata]\ncategory = "sandbox"\n\n'
        b'[verifier]\nenvironment_mode = "shared"\ntimeout_sec = 720.0\n'
    )
    return {
        "instruction.md": b"Implement the requested function in /app/solution.py.\n",
        "metadata.json": b'{"source": "pymethods2test"}\n',
        "task.toml": config,
        "environment/Dockerfile": b"FROM python:3.10-slim\nWORKDIR /app\nRUN pip install --no-cache-dir pytest==9.1.1\n",
        "solution/solution.py": b"def solve():\n    return 1\n",
        "solution/solve.sh": b"#!/bin/bash\ncp /solution/solution.py /app/solution.py\n",
        "tests/test.sh": b"#!/bin/bash\nset -e\ncd /app\npytest /tests/test_solution.py -v\n",
        "tests/test_solution.py": b"from solution import *\n\ndef test_solve():\n    assert solve() == 1\n",
    }


def native_row(workdir=None, verifier_overrides=None, environment_limits=None):
    files = native_files()
    config = tomllib.loads(files["task.toml"].decode())
    if workdir is not None or environment_limits is not None:
        config["environment"] = {"workdir": workdir, **(environment_limits or {})}
    for key, value in (verifier_overrides or {}).items():
        if value is None:
            del config["verifier"][key]
        else:
            config["verifier"][key] = value
    lines = ['version = "1.0"', "", "[agent]", f'timeout_sec = {config["agent"]["timeout_sec"]}', ""]
    if "environment" in config:
        lines += ["[environment]"]
        if config["environment"].get("workdir") is not None:
            lines.append(f'workdir = "{config["environment"]["workdir"]}"')
        for key in ("cpus", "memory_mb", "storage_mb"):
            if key in config["environment"]:
                lines.append(f'{key} = {config["environment"][key]}')
        lines.append("")
    lines += ["[metadata]", f'category = "{config["metadata"]["category"]}"', "", "[verifier]"]
    if "environment_mode" in config["verifier"]:
        lines.append(f'environment_mode = "{config["verifier"]["environment_mode"]}"')
    if "timeout_sec" in config["verifier"]:
        lines.append(f'timeout_sec = {config["verifier"]["timeout_sec"]}')
    files["task.toml"] = ("\n".join(lines) + "\n").encode()
    return {
        "archive_links": {},
        "archive_sha256": "a" * 64,
        "files": {name: base64.b64encode(raw).decode() for name, raw in files.items()},
        "file_metadata": {
            name: {"mode": "0755" if name.endswith(".sh") else "0644", "mtime_ns": 1700000000123456789} for name in files
        },
        "directory_metadata": {
            name.rstrip("/"): {"mode": "0755", "mtime_ns": 1700000000123456789}
            for name in ("environment/", "solution/", "tests/")
        },
        "pymethods_environment": {
            "bundle": "b" * 64,
            "environment_sha256": "c" * 64,
            "runtime": "apptainer",
            "bundle_prefix": "pymethods2test_v1/cached_images/bundles-v1/" + "b" * 64,
        },
    }


def convert_row(data):
    return convert_pymethods(
        RawRow(
            id="pymethods2test-0000",
            source=Source(dataset="FWeindel/validated-tasks", revision="pin", row="native", importer_revision="1"),
            data=data,
        ),
        ConversionContext(inputs={}, grader_environment=None),
    )


def test_native_converter_binds_grader_and_contract_and_keeps_the_dockerfile_build():
    task = convert_row(native_row())
    assert isinstance(task, TaskSpec)
    files = native_files()
    assert task.context.events[0].content == files["instruction.md"].decode()
    requirements = task.environment_requirements
    assert requirements.docker_image is None
    assert requirements.docker_build is not None
    build_files = {item.path: resource_bytes(item) for item in requirements.docker_build.files}
    assert build_files == {"Dockerfile": files["environment/Dockerfile"]}
    assert requirements.working_directory is None
    assert requirements.setup_commands == ()
    assert task.execution_requirements is None
    assert task.verifier_execution_requirements is None
    assert isinstance(task.grader, ScriptGrader)
    assert task.grader.timeout == 720.0
    execute = task.grader.collect[-1]
    assert "/tests/test.sh" in execute.argv[2] and execute.cwd is None
    assert {item.path: resource_bytes(item) for item in task.resources.worker} == {}
    assert {item.path: resource_bytes(item) for item in task.resources.oracle} == {
        "solution/solution.py": files["solution/solution.py"],
        "solution/solve.sh": files["solution/solve.sh"],
    }
    archived = {item.path.removeprefix("source_archive/"): item for item in task.resources.verifier}
    assert {name: resource_bytes(item) for name, item in archived.items()} == {
        name: raw for name, raw in files.items() if not name.startswith("solution/")
    }
    assert all(item.mtime_ns == 1700000000123456789 for item in archived.values())
    contract = task.integrity_contract
    assert contract.dependency_state == SharedDependencyState.AGENT_MUTABLE
    assert contract.agent_user is None and contract.undeclared_deadline is False
    metadata = native_row()["file_metadata"]
    assert {item.name: (item.mode, item.sha256) for item in contract.verifier_files} == {
        name.removeprefix("tests/"): (int(metadata[name]["mode"], 8), hashlib.sha256(raw).hexdigest())
        for name, raw in files.items()
        if name.startswith("tests/")
    }
    assert task.grader.collect[0].argv[:2] == ("bash", "-c")
    assert task.labels["pymethods2test_environment_bundle"] == "b" * 64
    assert task.labels["pymethods2test_environment_sha256"] == "c" * 64
    assert task.labels["native_archive_sha256"] == "a" * 64
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task


def test_native_converter_declared_workdir_and_limits_pass_through():
    task = convert_row(native_row(workdir="/app"))
    assert task.environment_requirements.working_directory == "/app"

    data = native_row(environment_limits={"cpus": 2, "memory_mb": 2048, "storage_mb": 10240})
    task = convert_row(data)
    assert task.execution_requirements is not None
    assert task.execution_requirements.cpus == 2
    assert task.execution_requirements.memory_mb == 2048
    assert task.execution_requirements.storage_mb == 10240
    assert task.execution_requirements.agent_timeout_sec == 900.0
    assert task.execution_requirements.verifier_timeout_sec == 720.0
    assert task.verifier_execution_requirements == task.execution_requirements

    task = convert_row(native_row())
    assert task.execution_requirements is None
    assert task.verifier_execution_requirements is None


@pytest.mark.parametrize("field", ["environment_mode", "timeout_sec"])
def test_native_converter_fails_closed_without_the_shared_declaration(field):
    data = native_row(verifier_overrides={field: None})
    with pytest.raises(ValueError, match="shared verifier deadline"):
        convert_row(data)


def archive_row():
    files = native_files()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, raw in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            info.pax_headers = {"mtime": "1700000000.123456789"}
            archive.addfile(info, io.BytesIO(raw))
    return {
        "path": "pymethods2test-0000",
        "task_binary": buffer.getvalue(),
        "pymethods_environment": native_row()["pymethods_environment"],
    }


def test_native_decoder_preserves_modes_and_nanosecond_times():
    decoded = decode_pymethods(archive_row(), ConversionContext(inputs={}, grader_environment=None))
    assert {name: base64.b64decode(raw) for name, raw in decoded["files"].items()} == native_files()
    assert decoded["file_metadata"]["tests/test.sh"]["mode"] == "0755"
    assert decoded["file_metadata"]["tests/test.sh"]["mtime_ns"] == 1700000000123456789
    assert decoded["archive_links"] == {}
    assert decoded["pymethods_environment"] == archive_row()["pymethods_environment"]


def test_native_decoder_refuses_links():
    row = archive_row()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo("environment/evil")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
        for name, raw in native_files().items():
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))
    row["task_binary"] = buffer.getvalue()
    with pytest.raises(ValueError, match="links"):
        decode_pymethods(row, ConversionContext(inputs={}, grader_environment=None))
