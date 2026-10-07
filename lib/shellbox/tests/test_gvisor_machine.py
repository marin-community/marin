# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise file transfer without Docker's view of the container overlay."""

import asyncio
import subprocess

from shellbox.backends.docker.machine import DockerCommandResult
from shellbox.backends.gvisor.machine import GvisorMachineFactory
from shellbox.machine import DockerImage, MachineSpec


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
        lookup = command[:2] == ("sh", "-c") and command[2].startswith("command -v ")
        if not lookup and ("--user" not in options or options[options.index("--user") + 1] != "0"):
            return DockerCommandResult(1, b"", b"Permission denied")
        result = subprocess.run(command, input=stdin, capture_output=True, timeout=30)
        return DockerCommandResult(result.returncode, result.stdout, result.stderr)

    monkeypatch.setattr("shellbox.backends.docker.machine.docker", docker)
    monkeypatch.setattr("shellbox.backends.gvisor.machine.docker", docker)
    source = tmp_path / "source"
    source.mkdir()
    (source / "answer").write_bytes(b"\x00\xffpayload")
    (source / "answer").chmod(0o755)
    remote = tmp_path / "remote"
    target = tmp_path / "download"

    async def scenario():
        machine = await GvisorMachineFactory().create(MachineSpec(DockerImage("fixture")))
        try:
            await machine.upload(source, str(remote))
            await machine.download(str(remote), target)
            await machine.upload(source / "answer", str(remote / "nested/copy"))
            await machine.download(str(remote / "nested/copy"), tmp_path / "file")
        finally:
            await machine.close()

    asyncio.run(scenario())
    assert (target / "answer").read_bytes() == b"\x00\xffpayload"
    assert (target / "answer").stat().st_mode & 0o777 == 0o755
    assert sorted(path.name for path in target.iterdir()) == ["answer"]
    assert (tmp_path / "file").read_bytes() == b"\x00\xffpayload"
    assert (tmp_path / "file").stat().st_mode & 0o777 == 0o755
