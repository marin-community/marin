# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local filesystem transfers and host path containment."""

import asyncio
import io
import os
import tarfile
import uuid
from pathlib import Path

import pytest
from shellbox.backends.local.runtime import LocalMachine
from shellbox.backends.local.sandbox import _archive
from shellbox.backends.local.transfers import unpack_local_archive
from shellbox.machine import (
    Command,
    HostImage,
    MachineSpec,
    Result,
    UnsupportedMachineSpec,
)
from shellbox.transfer import download_files

from lib.shellbox.tests.local_machine_support import (
    SPEC,
    _member,
    _stream,
    on_machine,
    run_command,
)


def test_recursive_download_preserves_hardlinks_as_files_and_literal_names(factory, tmp_path):
    async def check():
        machine = await factory.create(SPEC)
        try:
            source = tmp_path / "source"
            source.write_bytes(b"answer\x00")
            name = "/app/submission/nested/colon:\nλ\\file"
            await machine.upload(source, name)
            linked = await machine.run(Command(("ln", name, "/app/submission/alias")))
            assert linked.exit_code == 0
            symlink = await machine.run(Command(("ln", "-s", "/etc/passwd", "/app/submission/private")))
            assert symlink.exit_code == 0
            files = await download_files(
                machine,
                ("/missing", "/app/submission", name),
                timeout=10,
                max_files=2,
                max_bytes=1024,
                max_file_bytes=1024,
            )
            assert files == {name: b"answer\x00", "/app/submission/alias": b"answer\x00"}
            with pytest.raises(RuntimeError, match="links"):
                await download_files(
                    machine,
                    ("/app/submission/private",),
                    timeout=10,
                    max_files=2,
                    max_bytes=1024,
                    max_file_bytes=1024,
                )
        finally:
            await machine.close()

    asyncio.run(check())


def test_files_round_trip_between_host_paths_and_commands(tmp_path, factory):
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/data.txt").write_text("from directory")
    single = tmp_path / "single.txt"
    single.write_text("from file")
    downloaded = tmp_path / "downloaded"

    async def scenario(machine: LocalMachine) -> Result:
        await machine.upload(single, "/tests/inputs/single.txt")
        await machine.upload(source, "/tmp/tree")
        read = await machine.run(Command(("cat", "inputs/single.txt", "/tmp/tree/nested/data.txt"), cwd="/tests"))
        await machine.run(Command(("sh", "-c", "mkdir -p /logs/verifier && echo 0.5 > /logs/verifier/reward.txt")))
        await machine.download("/logs/verifier/reward.txt", downloaded / "reward.txt")
        await machine.download("/etc/passwd", downloaded / "passwd")
        with pytest.raises(RuntimeError):
            await machine.download("/logs/verifier/missing.txt", downloaded / "missing.txt")
        with pytest.raises(UnsupportedMachineSpec):
            await machine.upload(single, "/usr/local/single.txt")
        return read

    assert on_machine(factory, SPEC, scenario).stdout == b"from filefrom directory"
    assert (downloaded / "reward.txt").read_text() == "0.5\n"
    assert (downloaded / "passwd").read_bytes() == Path("/etc/passwd").read_bytes()


def test_machines_run_side_by_side_on_private_filesystems_that_close_removes(factory):
    # Paths that the host may use itself, such as an Iris task's /app, as well as /tmp, HOME and a new directory.
    marker = f"shellbox-test-{uuid.uuid4().hex}"
    paths = (f"/app/{marker}", f"/tmp/{marker}", f"$HOME/{marker}", f"/srv/{marker}")
    write = "mkdir /srv && " + " && ".join(f'echo "$1" > {path}' for path in paths)
    read = "cat " + " ".join(paths)

    async def scenario() -> tuple[list[bytes], list[Path]]:
        machines = await asyncio.gather(factory.create(SPEC), factory.create(SPEC))
        try:
            for machine, name in zip(machines, "ab", strict=True):
                await machine.run(Command(("sh", "-c", write, "sh", name)))
            results = await asyncio.gather(*(machine.run(Command(("sh", "-c", read))) for machine in machines))
            return [result.stdout for result in results], [machine.root for machine in machines]
        finally:
            await asyncio.gather(*(machine.close() for machine in machines))

    outputs, roots = asyncio.run(scenario())
    assert outputs == [b"a\n" * len(paths), b"b\n" * len(paths)]
    assert not any(Path(path.replace("$HOME", "/home/shellbox")).exists() for path in paths)
    assert not any(root.exists() for root in roots)


def test_an_archive_extracted_at_the_root_stays_in_its_machine(tmp_path, factory):
    # The grading runtime unpacks its inputs with ``tar -xf ... -C /``, which writes into the workspace.
    payload = tmp_path / "payload"
    payload.write_text("7")
    archive = tmp_path / "inputs.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(payload, arcname="app/solution.py")

    async def scenario(machine: LocalMachine) -> Result:
        await machine.upload(archive, "/tmp/inputs.tar")
        return await machine.run(Command(("sh", "-c", "tar -xf /tmp/inputs.tar -C / && cat /app/solution.py")))

    extracted = on_machine(factory, MachineSpec(HostImage(), workdir="/"), scenario)
    assert (extracted.exit_code, extracted.stdout) == (0, b"7"), extracted.stderr
    later = run_command(factory, Command(("ls", "-A", "/app")))
    assert (later.exit_code, later.stdout) == (0, b"")


def test_a_downloaded_archive_keeps_inner_symlinks_as_links_and_refuses_host_reads(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("host only")
    tree = _stream(
        (_member(".", tarfile.DIRTYPE), b""),
        (_member("data.txt"), b"data"),
        (_member("link", tarfile.SYMTYPE, str(secret)), b""),
    )
    unpack_local_archive(tree, tmp_path / "tree", source_name="tree")
    assert (tmp_path / "tree/data.txt").read_text() == "data"
    assert os.readlink(tmp_path / "tree/link") == str(secret)

    refused = [
        _stream((_member("secret", tarfile.SYMTYPE, str(secret)), b"")),
        _stream((_member(".", tarfile.DIRTYPE), b""), (_member("copy", tarfile.LNKTYPE, str(secret)), b"")),
        _stream((_member(".", tarfile.DIRTYPE), b""), (_member("disk", tarfile.BLKTYPE), b"")),
        _stream((_member(str(secret)), b"guest")),
        _stream(),
    ]
    for stream in refused:
        with pytest.raises((ValueError, tarfile.FilterError)):
            unpack_local_archive(stream, tmp_path / "refused", source_name="source")
    assert not (tmp_path / "refused").exists()


def test_a_download_cannot_choose_an_unrelated_host_filename(tmp_path):
    destination = tmp_path / "results"
    destination.mkdir()
    protected = destination / "configuration.json"
    protected.write_bytes(b"host configuration")
    forged = _stream((_member("configuration.json"), b"guest replacement"))

    with pytest.raises(ValueError, match="requested source"):
        unpack_local_archive(forged, destination, source_name="result.json")

    assert protected.read_bytes() == b"host configuration"
    assert not (destination / "result.json").exists()

    valid = _stream((_member("result.json"), b"task result"))
    unpack_local_archive(valid, destination, source_name="result.json")
    assert (destination / "result.json").read_bytes() == b"task result"
    assert protected.read_bytes() == b"host configuration"


def test_an_uploaded_archive_holds_what_a_symlinked_source_leads_to(tmp_path):
    (tmp_path / "directory").mkdir()
    (tmp_path / "directory/data.txt").write_text("data")
    (tmp_path / "directory/inner").symlink_to("data.txt")
    (tmp_path / "linked-directory").symlink_to(tmp_path / "directory")
    (tmp_path / "linked-file").symlink_to(tmp_path / "directory/data.txt")

    with tarfile.open(fileobj=io.BytesIO(_archive(tmp_path / "linked-directory", "."))) as archive:
        members = {member.name: member for member in archive.getmembers()}
    assert members["."].isdir() and members["./data.txt"].isfile()
    assert members["./inner"].issym() and members["./inner"].linkname == "data.txt"
    with tarfile.open(fileobj=io.BytesIO(_archive(tmp_path / "linked-file", "copy.txt"))) as archive:
        assert [(member.name, member.isfile()) for member in archive.getmembers()] == [("copy.txt", True)]


def test_a_transfer_does_not_follow_a_symlink_out_of_the_machine_root(tmp_path, factory, outside_dir):
    (outside_dir / "secret.txt").write_text("host")
    source = tmp_path / "payload"
    source.write_text("x")

    async def scenario() -> None:
        machine = await factory.create(MachineSpec(HostImage(), workdir="/app"))
        try:
            await machine.run(Command(("ln", "-s", str(outside_dir), "/app/escape")))
            with pytest.raises(RuntimeError):
                await machine.upload(source, "/app/escape/planted.txt")
            with pytest.raises(RuntimeError):
                await machine.download("/app/escape/secret.txt", tmp_path / "stolen.txt")
        finally:
            await machine.close()

    asyncio.run(scenario())
    assert not (outside_dir / "planted.txt").exists() and not (tmp_path / "stolen.txt").exists()


def test_absolute_guest_symlinks_resolve_inside_the_private_root(factory, tmp_path):
    source = tmp_path / "payload"
    source.write_text("updated")
    target = tmp_path / "downloaded"

    async def scenario(machine):
        prepared = await machine.run(
            Command(("sh", "-c", "mkdir -p /app/actual && echo guest >/app/actual/file && ln -s /app/actual /app/link"))
        )
        assert prepared.exit_code == 0
        await machine.upload(source, "/app/link/file")
        await machine.download("/app/link/file", target)

    on_machine(factory, SPEC, scenario)
    assert target.read_text() == "updated"


def test_repeated_download_cannot_follow_an_artifact_link_into_the_host(factory, tmp_path, outside_dir):
    host_marker = outside_dir / "marker"
    host_marker.write_text("host-private")
    downloaded = tmp_path / "artifacts"

    async def scenario(machine):
        prepared = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    'mkdir /app/artifacts && ln -s "$1" /app/artifacts/nested '
                    '&& ln -s "$1/marker" /app/artifacts/linked-file',
                    "sh",
                    str(outside_dir),
                )
            )
        )
        assert prepared.exit_code == 0
        await machine.download("/app/artifacts", downloaded)
        (downloaded / "unrelated").write_text("retained")
        replaced = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    "rm /app/artifacts/nested /app/artifacts/linked-file && mkdir /app/artifacts/nested "
                    "&& echo changed >/app/artifacts/nested/marker && echo replaced >/app/artifacts/linked-file",
                )
            )
        )
        assert replaced.exit_code == 0
        await machine.download("/app/artifacts", downloaded)
        assert (downloaded / "nested/marker").read_text() == "changed\n"
        assert (downloaded / "linked-file").read_text() == "replaced\n"
        assert (downloaded / "unrelated").read_text() == "retained"
        target = tmp_path / "file-link"
        target.symlink_to(host_marker)
        with pytest.raises(ValueError):
            await machine.download("/app/artifacts/nested/marker", target)
        target.unlink()
        target.symlink_to(outside_dir, target_is_directory=True)
        with pytest.raises(OSError):
            await machine.download("/app/artifacts/nested/marker", target / "marker")
        target.unlink()
        target = outside_dir / "file-hardlink"
        target.hardlink_to(host_marker)
        await machine.download("/app/artifacts/nested/marker", target)
        assert target.read_text() == "changed\n"

    on_machine(factory, SPEC, scenario)
    assert host_marker.read_text() == "host-private"


@pytest.mark.parametrize("first_mode,second_mode,new_mode", [(0o500, 0o777, 0o700), (0o777, 0o500, 0o755)])
def test_repeated_download_preserves_existing_host_directory_permissions(
    factory, tmp_path, first_mode, second_mode, new_mode
):
    downloaded = tmp_path / "artifacts"
    existing = downloaded / "existing"
    existing.mkdir(parents=True)
    downloaded.chmod(0o750)
    existing.chmod(0o710)
    (downloaded / "unrelated").write_text("host-owned")

    async def scenario(machine):
        first = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    "mkdir -p /app/artifacts/existing /app/artifacts/new; "
                    "printf first >/app/artifacts/existing/state; printf first >/app/artifacts/new/state; "
                    f"chmod {first_mode:o} /app/artifacts /app/artifacts/existing /app/artifacts/new",
                )
            )
        )
        assert first.exit_code == 0, first.stderr
        await machine.download("/app/artifacts", downloaded)
        assert downloaded.stat().st_mode & 0o777 == 0o750
        assert existing.stat().st_mode & 0o777 == 0o710
        assert (downloaded / "new").stat().st_mode & 0o777 == new_mode
        # The caller's permissions also survive later guest metadata changes.
        (downloaded / "new").chmod(0o750)
        second = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    "chmod 700 /app/artifacts /app/artifacts/existing /app/artifacts/new; "
                    "printf second >/app/artifacts/existing/state; printf second >/app/artifacts/new/state; "
                    f"chmod {second_mode:o} /app/artifacts /app/artifacts/existing /app/artifacts/new; "
                    "mkdir /app/locked; printf private >/app/locked/state; chmod 000 /app/locked; "
                    'ln -s "$1" /app/host-download',
                    "sh",
                    str(downloaded),
                )
            )
        )
        assert second.exit_code == 0, second.stderr
        await machine.download("/app/artifacts", downloaded)
        return machine.root

    root = on_machine(factory, SPEC, scenario)
    assert not root.exists()
    assert downloaded.stat().st_mode & 0o777 == 0o750
    assert existing.stat().st_mode & 0o777 == 0o710
    assert (downloaded / "new").stat().st_mode & 0o777 == 0o750
    assert (existing / "state").read_text() == "second"
    assert (downloaded / "new/state").read_text() == "second"
    assert (downloaded / "unrelated").read_text() == "host-owned"
