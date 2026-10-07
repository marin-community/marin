# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native archive curation keeps original controls private and readiness explicit."""

import hashlib
import io
import tarfile

import pytest
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.datasets.native_harbor import policy
from taskcompendium.datasets.source_definitions import unpack_task_binary
from taskcompendium.grader import grader_config
from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import CheckStatus, ImportFailureKind, ImportRejection, RawRow
from taskcompendium.runtime.resources import resource_bytes


def test_native_harbor_preserves_archive_files_and_unbound_grading_contract(tmp_path):
    files = {
        "instruction.md": b"Repair the repository and run its native tests.",
        "task.toml": b'[environment]\ndocker_image = "mutable-native-tag"\n',
        "setup_files/input.bin": b"\x00\xffpublic",
        "tests/test.sh": b"#!/bin/bash\n/opt/native/grade /request /result\n",
        "tests/reference.bin": b"\x00\xffprivate",
        "environment/Dockerfile": b"FROM native/environment:mutable\n",
        "solution/solve.sh": b"#!/bin/bash\nprivate oracle\n",
    }
    source = Source(dataset="fixture/native", revision="pinned", row="tasks.parquet:3", importer_revision="1")
    archive_data = io.BytesIO()
    with tarfile.open(fileobj=archive_data, mode="w") as archive:
        for path, content in files.items():
            member = tarfile.TarInfo(path)
            member.size = len(content)
            member.mode = 0o755 if path.endswith(".sh") else 0o644
            member.pax_headers = {"mtime": "1.000000001"}
            archive.addfile(member, io.BytesIO(content))
    decoded = unpack_task_binary(
        {"task_binary": archive_data.getvalue(), "path": "native/task-3"}, StoragePath(str(tmp_path))
    )
    row = RawRow("native-3", source, decoded)
    recipe = policy("fixture", "repository repair", ("Native private request/result mount adapter",))
    task = recipe.normalize(row)
    assert isinstance(task, TaskSpec)
    assert task.context.events[0].content == files["instruction.md"].decode()
    worker = {item.path: resource_bytes(item) for item in task.resources.worker}
    private = {item.path: resource_bytes(item) for item in task.resources.verifier + task.resources.oracle}
    assert worker == {"setup_files/input.bin": files["setup_files/input.bin"]}
    assert all(private[path.removeprefix("tests/")] == content for path, content in files.items() if path not in worker)
    assert "tests/reference.bin" not in worker and "solution/solve.sh" not in worker
    assert next(item for item in task.resources.verifier if item.path == "test.sh").mode == "0755"
    assert all(item.mtime_ns == 1000000001 for item in task.resources.worker + task.resources.oracle)
    config = grader_config(task)
    assert (
        config["contract"]["source_file_sha256"]["tests/test.sh"] == hashlib.sha256(files["tests/test.sh"]).hexdigest()
    )
    assert recipe.check_suite is not None
    checks = recipe.check_suite.run(task).checks
    assert checks[0].status == CheckStatus.UNSUPPORTED
    assert config["contract"]["binding_status"] == "unbound"


@pytest.mark.parametrize("link_kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_arc_graderhive_links_are_explicitly_unsupported(tmp_path, link_kind):
    archive_data = io.BytesIO()
    with tarfile.open(fileobj=archive_data, mode="w") as archive:
        for path, content in {"instruction.md": b"Run the native grader.", "tests/test.sh": b"#!/bin/bash\n"}.items():
            member = tarfile.TarInfo(path)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        link = tarfile.TarInfo("tests/native_grader")
        link.type = link_kind
        link.linkname = "test.sh"
        archive.addfile(link)
    decoded = unpack_task_binary(
        {"task_binary": archive_data.getvalue(), "path": "native/linked"}, StoragePath(str(tmp_path))
    )
    source = Source(dataset="fixture/native", revision="pinned", row="tasks.parquet:3", importer_revision="1")
    result = policy("fixture", "shell", ("Native grader",)).normalize(RawRow("linked-3", source, decoded))
    assert isinstance(result, ImportRejection)
    assert result.kind == ImportFailureKind.UNSUPPORTED
    assert result.reason == "native_harbor_archive_links"
    assert decoded["archive_links"]["tests/native_grader"]["target"] == "test.sh"
