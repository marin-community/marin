# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest
import tomlkit
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.harbor.shared_package import harden_shared_swe
from taskcompendium.harbor.snapshots import file_map_snapshot, task_snapshot
from taskcompendium.integrity import SharedDependencyState, SharedIntegrityContract, shared_preamble
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, ResourceGroups, Source, TaskSpec
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import PytestSpec

from experiments.post_training.task_curation.tasktrove.harbor_export import (
    UnsupportedHarborTask,
    VerifierPayloadIdentity,
    archive_bytes,
    archive_file_mode,
    harbor_payload,
    harbor_record,
)


@pytest.mark.parametrize(
    "path,content,mode",
    [
        ("tests/test.sh", b"new wrapper", "755"),
        ("tests/Dockerfile", b"FROM different:tag", "644"),
        ("tests/helper.py", b"private checker", "755"),
        ("task.toml", b"new dispatch", "644"),
    ],
)
def test_verifier_identity_tracks_payload_recipe_and_modes(path, content, mode):
    files = {
        "tests/test.sh": b"original wrapper",
        "tests/Dockerfile": b"FROM source:tag",
        "tests/helper.py": b"private checker",
        "task.toml": b"dispatch",
        "instruction.md": b"public prompt",
    }
    original = VerifierPayloadIdentity()
    original.add(archive_bytes(files, {}))
    changed = VerifierPayloadIdentity()
    changed.add(archive_bytes({**files, path: content}, {path: mode}))
    assert changed.ref != original.ref
    equivalent = VerifierPayloadIdentity()
    equivalent.add(archive_bytes({**files, "instruction.md": b"different public prompt"}, {}))
    equivalent.add(archive_bytes(files, {}))
    assert equivalent.ref == original.ref


@pytest.mark.parametrize(
    "prompt,reference,leaked",
    [
        ("The answer is ALPHA\n  BETA GAMMA.", "alpha beta gamma", True),
        ("Name the first three letters.", "alpha beta gamma", False),
        ("Explain why red is a color.", "red", False),
    ],
)
def test_harbor_filters_disclosed_long_references_without_dropping_short_answers(prompt, reference, leaked):
    source = Source(dataset="fixture", revision="pinned", row="fixture/tasks.parquet:0", importer_revision="1")
    task = exact_answer_task(RawRow("fixture", source, {}), prompt=prompt, answers=(reference,), ignore_case=False)
    task = cast(TaskSpec, task).model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "b" * 64
            )
        }
    )
    row = {"task_json": task.model_dump_json(), "source_row": source.row, "original_path": "fixture.tar.gz"}
    options = {
        "grader_image": "example.test/grader@sha256:" + "a" * 64,
        "family": "fixture",
    }
    if leaked:
        with pytest.raises(UnsupportedHarborTask, match="gold_leak: expected value appears in instruction"):
            harbor_payload(row, **options)
    else:
        payload = harbor_payload(row, **options)
        assert payload.files["instruction.md"].startswith(prompt.encode())


def test_file_map_snapshot_matches_actual_export_archives():
    source = Source(dataset="fixture", revision="pinned", row="fixture/tasks.parquet:0", importer_revision="1")
    task = exact_answer_task(RawRow("fixture", source, {}), prompt="Name a color", answers=("red",), ignore_case=False)
    task = cast(TaskSpec, task).model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "b" * 64
            )
        }
    )
    task = task.model_copy(
        update={
            "resources": ResourceGroups(
                worker=(inline_resource("app/input.txt", b"public input").model_copy(update={"mode": "600"}),),
                verifier=(inline_resource("helper.sh", b"private helper").model_copy(update={"mode": "700"}),),
                oracle=(inline_resource("solution/solve.sh", b"echo red").model_copy(update={"mode": "750"}),),
            )
        }
    )
    row = {"task_json": task.model_dump_json(), "source_row": source.row, "original_path": "fixture.tar.gz"}
    options = {
        "grader_image": "example.test/grader@sha256:" + "a" * 64,
        "family": "fixture",
    }
    payload = harbor_payload(row, **options)
    record = harbor_record(row, **options)
    direct = {
        **file_map_snapshot(
            payload.files, {name: archive_file_mode(name, payload.modes) for name in payload.files}, "task"
        ),
        **file_map_snapshot(
            payload.solution,
            {name: archive_file_mode(name, payload.solution_modes) for name in payload.solution},
            "oracle",
        ),
    }
    archived = task_snapshot(
        record.source,
        record.path,
        "converted",
        task_binary=gzip.compress(gzip.decompress(record.task_binary), compresslevel=9, mtime=0),
        solution_binary=record.solution_binary,
    )
    assert direct == archived.files
    assert direct["task/environment/files/app/input.txt"].mode == 0o600
    assert direct["task/tests/helper.sh"].mode == 0o700
    assert direct["oracle/solution/solve.sh"].mode == 0o750
    assert direct["task/tests/test.sh"].mode == 0o755


def test_harbor_export_rejects_a_task_without_its_own_environment():
    source = Source(dataset="fixture", revision="pinned", row="fixture/tasks.parquet:0", importer_revision="1")
    task = exact_answer_task(RawRow("missing", source, {}), prompt="Name a color", answers=("red",), ignore_case=False)
    row = {"task_json": task.model_dump_json(), "source_row": source.row, "original_path": "missing"}
    with pytest.raises(UnsupportedHarborTask, match="declares no environment"):
        harbor_payload(row, grader_image="grader@sha256:" + "a" * 64, family="fixture")


def test_shared_ingestion_preamble_and_harness_share_the_published_contract():
    files = {
        "instruction.md": b"Repair the broken function.\n",
        "environment/Dockerfile": b"FROM example.test/python@sha256:" + b"a" * 64 + b"\n",
        "task.toml": b'[metadata]\n[environment]\n[agent]\n[verifier]\nenvironment_mode="shared"\n',
        "tests/test.sh": b"true\n",
    }
    contract = harden_shared_swe(
        files, spec=PytestSpec(paths=("tests/test_case.py",)), dependency_state=SharedDependencyState.AGENT_MUTABLE
    )
    config = tomlkit.parse(files["task.toml"].decode())
    published = SharedIntegrityContract.model_validate(config["metadata"]["integrity_contract"].unwrap())
    assert files["instruction.md"].decode().startswith(shared_preamble(published))
    assert published == contract
    assert files["instruction.md"].decode().endswith("Repair the broken function.\n")
    assert all(gap not in files["instruction.md"].decode() for gap in contract.known_open_gaps)
    assert config["agent"]["user"] == str(contract.agent_uid)
    assert config["verifier"]["environment_mode"] == "shared"
    assert config["verifier"]["user"] == "root"


@pytest.mark.skipif(
    sys.platform != "linux" or os.geteuid() != 0, reason="Generated bootstrap requires isolated Linux root"
)
@pytest.mark.parametrize("declared_home", [False, True])
def test_shared_root_setup_keeps_source_compiler_home(tmp_path, declared_home):
    workspace = Path("/testbed")
    bootstrap = Path("/opt/taskcompendium-shared-bootstrap")
    protected = Path("/opt/taskcompendium-shared")
    assert not any(path.exists() for path in (workspace, bootstrap, protected))
    source_home = tmp_path / "source-home"
    source_home.mkdir()
    compiler = source_home / ".cargo/bin/source-compiler"
    marker = tmp_path / "compiler-result"
    setup = (
        'mkdir -p "$HOME/.cargo/bin"\n'
        "printf '#!/bin/sh\\necho source-toolchain\\n' > \"$HOME/.cargo/bin/source-compiler\"\n"
        'chmod +x "$HOME/.cargo/bin/source-compiler"\n'
        f"export PATH={shlex.quote(str(compiler.parent))}:$PATH\n"
        f"source-compiler > {shlex.quote(str(marker))}\n"
    )
    home_config = f'[environment.env]\nHOME="{source_home}"\n' if declared_home else ""
    files = {
        "instruction.md": (
            ("## Environment Setup (complete these steps first)\n\n```bash\n" + setup + "\n```\nRepair.\n").encode()
        ),
        "environment/Dockerfile": b"FROM source-python\nUSER root\n",
        "task.toml": ("[metadata]\n[environment]\n" + home_config + '[agent]\nuser="root"\n[verifier]\n').encode(),
        "tests/test.sh": b"true\n",
    }
    harden_shared_swe(
        files, spec=PytestSpec(paths=("test_case.py",)), dependency_state=SharedDependencyState.AGENT_MUTABLE
    )
    config = tomlkit.parse(files["task.toml"].decode()).unwrap()
    try:
        workspace.mkdir()
        (workspace / "test_case.py").write_text("def test_source(): pass\n")
        bootstrap.mkdir()
        (bootstrap / "control.py").write_bytes(files["environment/taskcompendium-shared/control.py"])
        (bootstrap / "python").symlink_to(sys.executable)
        environment = {**os.environ, "HOME": str(source_home), **config["environment"]["env"]}
        subprocess.run(["bash", "-c", config["environment"]["setup"]["command"]], env=environment, check=True)
        assert marker.read_text() == "source-toolchain\n"
        assert (protected / "contract.json").is_file()
    finally:
        for path in (workspace, bootstrap, protected):
            if path.exists():
                shutil.rmtree(path)
