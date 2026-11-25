# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import io
import tarfile

import pytest
from taskcompendium.integrity import SharedDependencyState
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.endless.tasks import convert_endless, decode_endless


def native_files():
    config = (
        b"[agent]\ntimeout_sec=300\n[verifier]\ntimeout_sec=300\n"
        b"[environment]\ncpus=1\nmemory_mb=2048\nstorage_mb=10240\n"
    )
    return {
        "instruction.md": b"Create the requested file in /home/user.\n",
        "task.toml": config,
        "environment/Dockerfile": b"FROM mutable:tag\nRUN never-rebuild\n",
        "setup_files/setup.sh": b"#!/bin/bash\nprintf initial > /home/user/input\n",
        "tests/test.sh": b"#!/bin/bash\npytest /tests/test_final_state.py\nprintf 1 > /logs/verifier/reward.txt\n",
        "tests/test_final_state.py": b"def test_result():\n    assert open('/home/user/result').read() == 'answer'\n",
        "solution/solve.sh": b"#!/bin/bash\nbash /setup_files/setup.sh\nprintf answer > /home/user/result\n",
    }


def native_data(workdir):
    files = native_files()
    return {
        "archive_links": {},
        "archive_sha256": "a" * 64,
        "files": {name: base64.b64encode(raw).decode() for name, raw in files.items()},
        "file_metadata": {
            name: {"mode": "0755" if name.endswith(".sh") else "0644", "mtime_ns": 1700000000123456789} for name in files
        },
        "directory_metadata": {},
        "endless_image": {
            "image": "ghcr.io/example/endless@sha256:" + "b" * 64,
            "preserved_image": {
                "sif_sha256": "c" * 64,
                "provenance_uri": "s3://fixture/preservation.json",
                "provenance_sha256": "d" * 64,
            },
        },
        "endless_contract": {"image_runtime": {"user": "", "working_directory": workdir}},
    }


@pytest.mark.parametrize("workdir", ["/", "/home/user/project"])
def test_native_converter_preserves_roles_deadlines_files_and_image_cwd(workdir):
    data = native_data(workdir)
    task = convert_endless(
        RawRow(
            id="endless-native",
            source=Source(dataset="FWeindel/validated-tasks", revision="pin", row="native", importer_revision="1"),
            data=data,
        ),
        ConversionContext(inputs={}, grader_environment=None),
    )
    assert isinstance(task, TaskSpec)
    files = native_files()
    assert task.context.events[0].content == files["instruction.md"].decode()
    assert task.environment_requirements.docker_image == data["endless_image"]["image"]
    assert task.environment_requirements.docker_build is None
    assert task.environment_requirements.working_directory == workdir
    assert task.environment_requirements.setup_commands == ()
    assert task.execution_requirements.model_dump() == {
        "cpus": 1,
        "memory_mb": 2048,
        "storage_mb": 10240,
        "agent_timeout_sec": 300,
        "verifier_timeout_sec": 300,
    }
    assert task.verifier_execution_requirements == task.execution_requirements
    assert task.grader.timeout == 300 and task.grader.collect[-1].cwd == workdir
    assert {item.path: resource_bytes(item) for item in task.resources.worker} == {
        "setup_files/setup.sh": files["setup_files/setup.sh"]
    }
    assert {item.path: resource_bytes(item) for item in task.resources.oracle} == {
        "solution/solve.sh": files["solution/solve.sh"]
    }
    archived = {item.path.removeprefix("source_archive/"): item for item in task.resources.verifier}
    assert {name: resource_bytes(item) for name, item in archived.items()} == {
        name: raw for name, raw in files.items() if not name.startswith("solution/")
    }
    assert all(item.mtime_ns == 1700000000123456789 for item in archived.values())
    assert task.integrity_contract.dependency_state == SharedDependencyState.AGENT_MUTABLE
    assert {item.name: item.sha256 for item in task.integrity_contract.verifier_files} == {
        name.removeprefix("tests/"): hashlib.sha256(raw).hexdigest()
        for name, raw in files.items()
        if name.startswith("tests/")
    }
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task


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
    blob = buffer.getvalue()
    return {
        "path": "native",
        "task_binary": blob,
        "endless_image": {},
        "endless_contract": {
            "archive_sha256": hashlib.sha256(blob).hexdigest(),
            "files": {
                name: {
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                    "mode": "0755" if name.endswith(".sh") else "0644",
                    "mtime": "1700000000.123456789",
                }
                for name, raw in files.items()
            },
        },
    }


def test_native_decoder_verifies_complete_census_and_retains_nanosecond_times():
    row = archive_row()
    decoded = decode_endless(row, ConversionContext(inputs={}, grader_environment=None))
    assert {name: base64.b64decode(raw) for name, raw in decoded["files"].items()} == native_files()
    assert decoded["file_metadata"]["solution/solve.sh"]["mtime_ns"] == 1700000000123456789


@pytest.mark.parametrize("edit", ["archive", "file", "timestamp"])
def test_native_decoder_refuses_changes_against_sealed_contract(edit):
    row = archive_row()
    if edit == "archive":
        row["endless_contract"]["archive_sha256"] = "0" * 64
    elif edit == "file":
        row["endless_contract"]["files"]["tests/test.sh"]["sha256"] = "0" * 64
    else:
        row["endless_contract"]["files"]["solution/solve.sh"]["mtime"] = "1700000001"
    with pytest.raises(ValueError, match="changed"):
        decode_endless(row, ConversionContext(inputs={}, grader_environment=None))
