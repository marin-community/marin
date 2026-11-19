# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import os
import subprocess
import tarfile

import pytest
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.harbor.adapter import declared_build_timeout
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.task_curation.datasets.fineenvs.artifacts import GroupedInputImage
from experiments.post_training.task_curation.datasets.fineenvs.image_inputs import NativeInputFile, input_signature
from experiments.post_training.task_curation.datasets.fineenvs.images import BASE_IMAGE
from experiments.post_training.task_curation.datasets.fineenvs.tasks import VERSION, convert_fineenvs, decode_fineenvs

CONTEXT = ConversionContext(inputs={}, grader_environment=None)


def native_row(tmp_path, private_setup=b'printf "private\\n" >> "$TRIAL_TRACE"\n'):
    payload = b"original coordinate\n" * 100_000
    public = (
        b'#!/bin/bash\nset -eu\ntest ! -e "$(dirname "$0")/../tests/setup.sh"\n'
        b'test -f "$(dirname "$0")/inputs/coordinates.csv"\nprintf "public\\n" >> "$TRIAL_TRACE"\n'
    )
    verifier = (
        b'#!/bin/bash\nset -eu\nprintf "test\\n" >> "$TRIAL_TRACE"\n'
        b'test "$EXPECTED_ANSWER" = original-private-answer\nprintf 1 > "$TEST_REWARD_DIR/reward.txt"\n'
    )
    config = (
        "[environment]\ncpus=1\nmemory_mb=1024\nstorage_mb=5120\nallow_internet=true\n"
        "build_timeout_sec=600\n[environment.env]\n"
        f"TRIAL_TRACE='{tmp_path}/trace'\n[agent]\ntimeout_sec=900\n"
        f"[verifier]\ntimeout_sec=120\n[verifier.env]\nEXPECTED_ANSWER='original-private-answer'\nTEST_REWARD_DIR='{tmp_path}/logs/verifier'\n"
    ).encode()
    files = {
        "task.toml": config,
        "instruction.md": b"Read the coordinates and write the result.\n",
        "environment/Dockerfile": ("FROM " + BASE_IMAGE + "\nWORKDIR /workdir\n").encode(),
        "setup_files/setup.sh": public,
        "setup_files/inputs/coordinates.csv": payload,
        "tests/setup.sh": private_setup,
        "tests/test.sh": verifier,
        "tests/__pycache__/grader.cpython-312.pyc": b"\x00native-private-cache\xff",
        "solution/solve.sh": b"printf original > result\n",
    }
    archive = io.BytesIO()
    inventory = []
    with tarfile.open(fileobj=archive, mode="w:gz", format=tarfile.PAX_FORMAT) as writer:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o755 if name.endswith(".sh") else 0o640
            member.pax_headers = {"mtime": "1700000000.123456789"}
            writer.addfile(member, io.BytesIO(data))
            inventory.append(
                {
                    "path": name,
                    "bytes": len(data),
                    "mode": member.mode,
                    "mtime": member.pax_headers["mtime"],
                    "type": "0",
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    input_file = NativeInputFile.model_validate(
        next(file for file in inventory if file["path"].startswith("setup_files/inputs/"))
    )
    signature = input_signature((input_file,))
    record = GroupedInputImage(
        path="s3://fixture/group",
        data_restriction=DataRestriction.OPEN,
        group=0,
        base_image=BASE_IMAGE,
        input_signatures=(signature,),
        manifest_digest="sha256:" + "a" * 64,
        tree_sha256="b" * 64,
        provenance_uri="s3://fixture/group/provenance.json",
        provenance_sha256="c" * 64,
    )
    raw = archive.getvalue()
    return {
        "path": "native-coordinate-task",
        "task_binary": raw,
        "fineenvs_receipt": {"archive_sha256": hashlib.sha256(raw).hexdigest(), "files": inventory},
        "fineenvs_contract": {"input_signature": signature},
        "fineenvs_image": {
            "image": "ghcr.io/example/fineenvs@" + record.manifest_digest,
            "grouped_input_image": record.model_dump(mode="json"),
            "provenance": {"uri": record.provenance_uri, "sha256": record.provenance_sha256},
        },
    }, files


def converted(row):
    data = decode_fineenvs(row, CONTEXT)
    task = convert_fineenvs(
        RawRow(
            id="native-coordinate-task",
            source=Source(
                dataset="FWeindel/validated-tasks",
                revision="d" * 40,
                row="native-coordinate-task",
                importer_revision=VERSION,
            ),
            data=data,
        ),
        CONTEXT,
    )
    assert isinstance(task, TaskSpec)
    return task


def test_large_input_binds_to_image_without_changing_the_native_private_contract(tmp_path):
    row, files = native_row(tmp_path)
    task = converted(row)

    assert task.context.events[0].content == files["instruction.md"].decode()
    assert task.labels["native_archive_sha256"] == row["fineenvs_receipt"]["archive_sha256"]
    assert task.environment_requirements.docker_image == row["fineenvs_image"]["image"]
    assert task.environment_requirements.docker_build is None
    assert task.environment_requirements.working_directory == "/workdir"
    assert task.environment_requirements.capabilities == ("shell",)
    assert task.execution_requirements.model_dump() == {
        "cpus": 1,
        "memory_mb": 1024,
        "storage_mb": 5120,
        "agent_timeout_sec": 900,
        "verifier_timeout_sec": 120,
    }
    assert task.verifier_execution_requirements == task.execution_requirements
    assert declared_build_timeout(task) == 600
    assert task.grader.timeout == 120
    assert task.grader.collect[-1].env["EXPECTED_ANSWER"] == "original-private-answer"
    assert "EXPECTED_ANSWER" not in task.environment_requirements.environment_variables
    assert {r.path: resource_bytes(r) for r in task.resources.worker} == {
        "setup_files/setup.sh": files["setup_files/setup.sh"]
    }
    assert {r.path: resource_bytes(r) for r in task.resources.oracle} == {
        "solution/solve.sh": files["solution/solve.sh"]
    }
    original = {r.path.removeprefix("source_archive/"): resource_bytes(r) for r in task.resources.verifier}
    assert original == {
        name: data for name, data in files.items() if not name.startswith(("solution/", "setup_files/inputs/"))
    }
    assert (
        sum(
            len(resource_bytes(resource))
            for resource in (*task.resources.worker, *task.resources.verifier, *task.resources.oracle)
        )
        < 1_000_000
    )
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task


def test_input_file_corruption_stops_conversion_even_with_a_matching_archive_hash(tmp_path):
    row, _ = native_row(tmp_path)
    inventory = row["fineenvs_receipt"]["files"]
    inventory[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="native file checksum changed"):
        decode_fineenvs(row, CONTEXT)


@pytest.mark.parametrize(
    "private_setup,expected_trace,expected_reward",
    [
        (b'printf "private\\n" >> "$TRIAL_TRACE"\n', "public\nactor\nprivate\ntest\n", "1"),
        (
            b'printf "private\\n" >> "$TRIAL_TRACE"\nprintf 0 > "$TEST_REWARD_DIR/reward.txt"\nexit 7\n',
            "public\nactor\nprivate\n",
            "0",
        ),
        (b'printf "private\\n" >> "$TRIAL_TRACE"\nexit 7\n', "public\nactor\nprivate\n", None),
    ],
)
def test_native_setup_runs_in_its_own_phase_and_source_zero_skips_checks(
    tmp_path, private_setup, expected_trace, expected_reward
):
    row, files = native_row(tmp_path, private_setup)
    task = converted(row)
    for resource in task.resources.worker:
        target = tmp_path / resource.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(resource_bytes(resource))
        target.chmod(int(resource.mode, 8))
    signature = row["fineenvs_contract"]["input_signature"]
    inert = tmp_path / "opt/fineenvs-inputs" / signature
    inert.mkdir(parents=True)
    (inert / "coordinates.csv").write_bytes(files["setup_files/inputs/coordinates.csv"])
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    environment = {**os.environ, **task.environment_requirements.environment_variables}

    def run(argv, extra_environment=None):
        # Run the original fixture scripts and delivered archive in a temporary
        # guest tree; never run the process-quiescence command on the test host.
        for prefix in ("/opt/fineenvs-inputs", "/setup_files", "/tests", "/logs/verifier"):
            argv = tuple(value.replace(prefix, str(tmp_path) + prefix) for value in argv)
        subprocess.run(argv, cwd=workdir, env={**environment, **(extra_environment or {})}, check=True)

    run(("bash", "-c", task.environment_requirements.setup_commands[0]))
    assert not (tmp_path / "tests/setup.sh").exists()
    with (tmp_path / "trace").open("a") as stream:
        stream.write("actor\n")
    actor_directory = tmp_path / "actor-cache"
    actor_directory.mkdir()
    sentinel = actor_directory / "sentinel"
    sentinel.write_text("actor-owned file")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/__pycache__").symlink_to(actor_directory, target_is_directory=True)
    if expected_reward is None:
        with pytest.raises(subprocess.CalledProcessError) as error:
            for command in task.grader.collect[1:]:
                run(command.argv, command.env)
        assert error.value.returncode == 7
        assert not (tmp_path / "logs/verifier/reward.txt").exists()
    else:
        for command in task.grader.collect[1:]:
            run(command.argv, command.env)
        assert (tmp_path / "logs/verifier/reward.txt").read_text() == expected_reward

    assert (tmp_path / "trace").read_text() == expected_trace
    assert (tmp_path / "tests/__pycache__/grader.cpython-312.pyc").read_bytes() == files[
        "tests/__pycache__/grader.cpython-312.pyc"
    ]
    assert not (tmp_path / "tests/__pycache__").is_symlink()
    assert sentinel.read_text() == "actor-owned file"
