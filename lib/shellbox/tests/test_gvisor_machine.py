# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise file transfer without Docker's view of the container overlay."""

import asyncio
import io
import os
import subprocess
import tarfile

import pytest
from shellbox.backends.docker.machine import DockerCommandResult
from shellbox.backends.gvisor.machine import GvisorMachineFactory
from shellbox.machine import Command, DockerImage, MachineSpec

TEST_IMAGE = "ghcr.io/marin-community/iris-task@sha256:66cba7cb3eb682f9a53e444876ef2468670336a71e03559de85b5b2b5d4cdde6"


def test_gvisor_transfers_binary_files_and_directory_contents_through_exec(tmp_path, monkeypatch):
    async def docker(*args, stdin=b"", **kwargs):
        if args[0] in {"run", "rm"}:
            return DockerCommandResult(0, b"", b"")
        # The provider's writable filesystem is accessible only through exec.
        # docker cp would use a different overlay, as on a runsc host.
        assert args[0] == "exec"
        index = next(i for i, value in enumerate(args) if value.startswith("harbor-machine-"))
        command = args[index + 1 :]
        # Private fixtures are unreadable to the image's default unprivileged user.
        # Command lookup does not access private files.
        options = args[1:index]
        lookup = command[:2] == ("/bin/sh", "-c") and command[2].startswith("command -v ")
        if not lookup and ("--user" not in options or options[options.index("--user") + 1] != "0"):
            return DockerCommandResult(1, b"", b"Permission denied")
        if stdin:
            with tarfile.open(fileobj=io.BytesIO(stdin), mode="r:") as archive:
                assert all((member.uid, member.gid) == (0, 0) for member in archive)
        result = subprocess.run(command, input=stdin, capture_output=True, timeout=30)
        return DockerCommandResult(result.returncode, result.stdout, result.stderr)

    monkeypatch.setattr("shellbox.backends.docker.machine.docker", docker)
    monkeypatch.setattr("shellbox.backends.gvisor.machine.docker", docker)
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer").write_bytes(b"\x00\xffpayload")
    (source / "answer").chmod(0o755)
    if os.geteuid() == 0:
        os.chown(source / "answer", 12345, 12345)
    remote = tmp_path / "remote"
    target = tmp_path / "download"
    workdir = tmp_path / "workdir"

    async def scenario():
        machine = await GvisorMachineFactory().create(MachineSpec(DockerImage("fixture"), workdir=str(workdir)))
        try:
            await machine.upload(source, str(remote))
            await machine.download(str(remote), target)
            await machine.upload(source / "answer", str(remote / "nested/copy"))
            await machine.download(str(remote / "nested/copy"), tmp_path / "file")
        finally:
            await machine.close()

    asyncio.run(scenario())
    assert workdir.is_dir()
    assert (target / "answer").read_bytes() == b"\x00\xffpayload"
    assert (target / "answer").stat().st_mode & 0o777 == 0o755
    assert sorted(path.name for path in target.iterdir()) == ["answer"]
    assert (tmp_path / "file").read_bytes() == b"\x00\xffpayload"
    assert (tmp_path / "file").stat().st_mode & 0o777 == 0o755


@pytest.mark.docker
def test_gvisor_upload_preserves_guest_root_ownership_and_executable_mode(tmp_path):
    source = tmp_path / "seeds"
    source.mkdir()
    script = source / "script.sh"
    script.write_bytes(b"#!/bin/sh\nprintf 'ready\\n'\n")
    script.chmod(0o755)
    if os.geteuid() == 0:
        os.chown(script, 12345, 12345)

    async def scenario():
        machine = await GvisorMachineFactory().create(MachineSpec(DockerImage(TEST_IMAGE)))
        try:
            await machine.upload(source, "/setup_files/seeds")
            copied = await machine.run(
                Command(
                    ("sh", "-c", "cp -a /setup_files/seeds/. /workspace/ && stat -c '%u:%g %a' /workspace/script.sh")
                )
            )
            assert (copied.exit_code, copied.stdout) == (0, b"0:0 755\n")
            matched = await machine.run(Command(("find", "/workspace", "-type", "f", "-user", "root", "-name", "*.sh")))
            assert (matched.exit_code, matched.stdout) == (0, b"/workspace/script.sh\n")
            executed = await machine.run(Command(("/workspace/script.sh",), user="65534"))
            assert (executed.exit_code, executed.stdout) == (0, b"ready\n")
        finally:
            await machine.close()

    asyncio.run(scenario())
