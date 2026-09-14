# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import io
import tarfile

import pytest
from tasktrove_verify.spec import Mode

from taskcompendium.execution import (
    ChatWithTools,
    HarborExecutionConfig,
    HarnessToolBinding,
    environment_for_requirements,
)
from taskcompendium.harbor.snapshot import MAX_SNAPSHOT_BYTES, extract_snapshot
from taskcompendium.lowering import lower_to_harbor
from taskcompendium.models import (
    AnswerRequirements,
    Capability,
    ContainerRuntime,
    FinalState,
    ImageOverlay,
    Rendering,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpecification,
    TaskTroveVerifier,
    WorkspaceState,
)
from taskcompendium.serialization import from_json, read_parquet, specification_hash, to_json, write_parquet

IMAGE = "sha256:" + "1" * 64


def _specification():
    return TaskSpecification(
        id="overlay",
        requirements=TaskRequirements(
            (Capability.FILESYSTEM, Capability.SHELL, Capability.PROCESS), WorkspaceState(IMAGE)
        ),
        resources=(),
        metadata=TaskMetadata(Source("test", "1", "0", "1")),
        steps=(
            StepSpecification(
                instructions="Modify main.py",
                verifier=TaskTroveVerifier(
                    Mode.STDIO,
                    {"command": ".venv/bin/python main.py"},
                    runtime=ContainerRuntime(
                        IMAGE,
                        timeout=300,
                        workspace=ImageOverlay((".venv",)),
                        supervisor_python="/opt/python/bin/python",
                    ),
                ),
                answer_requirements=AnswerRequirements("final_state"),
            ),
        ),
    )


def test_overlay_parquet_roundtrip_preserves_runtime_contract(tmp_path):
    spec = _specification()
    path = str(tmp_path / "tasks.parquet")
    write_parquet([spec], path)
    restored = list(read_parquet(path))
    assert restored == [spec]
    assert specification_hash(restored[0]) == specification_hash(spec)
    assert to_json(from_json(to_json(spec))) == to_json(spec)


@pytest.mark.parametrize("excluded", [(), (".venv/bin",), ("other",)])
def test_overlay_export_rejects_incomplete_dependency_exclusion(tmp_path, excluded):
    spec = _specification()
    destination = tmp_path / "task"
    with pytest.raises(ValueError, match="exclusions must cover"):
        lower_to_harbor(
            spec,
            (Rendering("overlay", FinalState((".",), excluded_paths=excluded)),),
            HarborExecutionConfig(
                "replay",
                environment_for_requirements(spec.requirements),
                interaction=(ChatWithTools((HarnessToolBinding("replay", "docker"),))),
            ),
            destination,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escaped", tarfile.REGTYPE),
        ("/escaped", tarfile.REGTYPE),
        ("link", tarfile.SYMTYPE),
        ("hardlink", tarfile.LNKTYPE),
        (".venv/leak", tarfile.REGTYPE),
    ],
)
def test_snapshot_rejects_escaping_links_and_excluded_members(tmp_path, name, kind):
    archive_path = tmp_path / "input.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "/etc/passwd" if kind != tarfile.REGTYPE else ""
        archive.addfile(member)
    with pytest.raises(ValueError, match="Unsafe workspace archive member"):
        extract_snapshot(archive_path, tmp_path / "snapshot", (".venv",))
    assert not (tmp_path / "escaped").exists()


def test_snapshot_extracts_content_and_executable_mode(tmp_path):
    archive_path = tmp_path / "input.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("./src/program")
        member.size = 3
        member.mode = 0o755
        archive.addfile(member, io.BytesIO(b"run"))
    target = tmp_path / "snapshot"
    extract_snapshot(archive_path, target, ())
    assert (target / "src/program").read_bytes() == b"run"
    assert (target / "src/program").stat().st_mode & 0o111


def test_snapshot_rejects_oversized_member_before_creating_file(tmp_path):
    member = tarfile.TarInfo("oversized")
    member.size = MAX_SNAPSHOT_BYTES + 1
    archive = tmp_path / "input.tar"
    archive.write_bytes(member.tobuf() + b"\0" * 1024)
    target = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="byte budget"):
        extract_snapshot(archive, target, ())
    assert not (target / "oversized").exists()
