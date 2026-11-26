# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local host mounts and copy-on-write filesystems."""

import asyncio
import uuid
from pathlib import Path, PurePosixPath

import pytest
from local_machine_support import (
    SPEC,
    make_factory,
    on_machine,
    root_commands,
    run_command,
)
from shellbox.backends.local.runtime import LocalMachine
from shellbox.backends.local.sandbox import SystemDirectories, _submounts
from shellbox.machine import (
    Command,
    HostImage,
    MachineSpec,
    Mount,
    Result,
    UnsupportedMachineSpec,
)


@pytest.mark.parametrize("system_directories", list(SystemDirectories))
def test_host_configuration_is_hidden_from_commands_and_downloads(tmp_path, system_directories):
    factory = make_factory(system_directories=system_directories)
    # These host paths must not become visible even when commands run as the host's root user.
    hidden = ("/etc/profile", "/etc/environment", "/etc/shadow", "/etc/gshadow", "/etc/ssl/private", "/sys/kernel")

    async def scenario(machine: LocalMachine):
        result = await machine.run(Command(("sh", "-c", 'for path do test ! -e "$path" || exit 1; done', "sh", *hidden)))
        assert result.exit_code == 0, result.stderr
        for path in hidden:
            with pytest.raises(RuntimeError, match="No such file"):
                await machine.download(path, tmp_path / "download")
        configuration = await machine.run(Command(("cat", "/etc/resolv.conf")))
        assert (configuration.exit_code, configuration.stdout) == (0, Path("/etc/resolv.conf").read_bytes())
        await machine.download("/etc", tmp_path / "etc")

    on_machine(factory, SPEC, scenario)
    assert not (tmp_path / "etc/profile").exists()
    assert (tmp_path / "etc/passwd").read_bytes() == Path("/etc/passwd").read_bytes()


def test_host_image_adds_read_only_mount_and_program_path(tmp_path, factory):
    bin_dir = tmp_path / "venv-bin"
    bin_dir.mkdir()
    command = bin_dir / "grader-python"
    command.write_text("#!/bin/sh\necho mounted-runtime\n")
    command.chmod(0o755)
    spec = MachineSpec(HostImage(read_only=(bin_dir,), bin_dirs=(bin_dir,)), workdir="/app")

    result = run_command(factory, Command(("grader-python",)), spec)

    assert (result.exit_code, result.stdout) == (0, b"mounted-runtime\n")


def test_host_image_mounts_a_directory_read_only_at_another_guest_path(tmp_path, factory):
    data = tmp_path / "english"
    data.mkdir()
    (data / "abbrev.tab").write_text("etc\n")
    guest = PurePosixPath("/srv/fixture/data")
    spec = MachineSpec(HostImage(mounts=(Mount(data, guest),)), workdir="/app")
    script = f"cat {guest}/abbrev.tab; echo changed > {guest}/abbrev.tab 2>/dev/null || echo refused"

    result = run_command(factory, Command(("sh", "-c", script)), spec)

    assert (result.exit_code, result.stdout) == (0, b"etc\nrefused\n"), result.stderr
    assert (data / "abbrev.tab").read_text() == "etc\n"


def test_a_mount_inside_a_host_system_directory_is_refused(tmp_path, factory):
    spec = MachineSpec(HostImage(mounts=(Mount(tmp_path, PurePosixPath("/usr/local/share/fixture")),)), workdir="/app")
    with pytest.raises(UnsupportedMachineSpec, match="system directory"):
        asyncio.run(factory.create(spec))


def test_host_system_directories_are_read_only(factory):
    script = "touch /usr/shellbox-test 2>/dev/null || echo refused; touch /etc/shellbox-test 2>/dev/null || echo refused"
    assert run_command(factory, Command(("sh", "-c", script))).stdout == b"refused\nrefused\n"


@root_commands
def test_copy_on_write_system_directories_keep_each_machines_writes_to_itself(copy_on_write_factory):
    name = f"shellbox-test-{uuid.uuid4().hex}"
    paths = (f"/opt/{name}/file", f"/usr/local/bin/{name}", f"/etc/{name}")
    write = f"mkdir /opt/{name} && " + " && ".join(f'echo "$1" > {path}' for path in paths)
    read = "cat " + " ".join(paths)

    async def scenario() -> tuple[list[bytes], list[Path]]:
        machines = await asyncio.gather(copy_on_write_factory.create(SPEC), copy_on_write_factory.create(SPEC))
        try:
            for machine, label in zip(machines, "ab", strict=True):
                written = await machine.run(Command(("sh", "-c", write, "sh", label)))
                assert written.exit_code == 0, written.stderr
            results = await asyncio.gather(*(machine.run(Command(("sh", "-c", read))) for machine in machines))
            return [result.stdout for result in results], [machine.layers for machine in machines]
        finally:
            await asyncio.gather(*(machine.close() for machine in machines))

    outputs, layers = asyncio.run(scenario())
    assert outputs == [b"a\n" * len(paths), b"b\n" * len(paths)]
    assert not any(Path(path).exists() for path in (f"/opt/{name}", *paths[1:]))
    assert not any(layer.exists() for layer in layers)


@root_commands
def test_copy_on_write_transfers_see_the_machines_own_system_directories(tmp_path, copy_on_write_factory):
    name = f"shellbox-test-{uuid.uuid4().hex}"
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "nested/data.txt").write_text("from directory")
    single = tmp_path / "single.txt"
    single.write_text("from file")
    downloaded = tmp_path / "downloaded"

    async def scenario(machine: LocalMachine) -> Result:
        await machine.upload(source, f"/opt/{name}/tree")
        await machine.upload(single, f"/usr/local/share/{name}/single.txt")
        read = await machine.run(
            Command(("cat", "tree/nested/data.txt", f"/usr/local/share/{name}/single.txt"), cwd=f"/opt/{name}")
        )
        await machine.run(Command(("sh", "-c", f"echo changed > /opt/{name}/tree/nested/data.txt")))
        await machine.download(f"/opt/{name}/tree", downloaded / "tree")
        await machine.download(f"/usr/local/share/{name}/single.txt", downloaded / "single.txt")
        await machine.download("/etc/passwd", downloaded / "passwd")
        with pytest.raises(RuntimeError, match="No such file"):
            await machine.download(f"/opt/{name}/missing", downloaded / "missing")
        with pytest.raises(UnsupportedMachineSpec):
            await machine.upload(single, f"/sys/{name}")
        # A link to a path that exists only on the host does not resolve in the sandbox.
        await machine.run(Command(("ln", "-s", str(tmp_path), f"/usr/local/share/{name}/host")))
        with pytest.raises(RuntimeError, match="No such file"):
            await machine.download(f"/usr/local/share/{name}/host", downloaded / "host")
        return read

    read = on_machine(copy_on_write_factory, SPEC, scenario)
    assert (read.exit_code, read.stdout) == (0, b"from directoryfrom file"), read.stderr
    assert (downloaded / "tree/nested/data.txt").read_text() == "changed\n"
    assert (downloaded / "single.txt").read_text() == "from file"
    assert (downloaded / "passwd").read_bytes() == Path("/etc/passwd").read_bytes()
    assert not Path(f"/opt/{name}").exists() and not Path(f"/usr/local/share/{name}").exists()


@root_commands
def test_copy_on_write_uploads_to_the_root_reach_the_layer_whatever_the_machines_environment(
    tmp_path, copy_on_write_factory
):
    name = f"shellbox-test-{uuid.uuid4().hex}"
    tree = tmp_path / "tree"
    (tree / "opt" / name).mkdir(parents=True)
    (tree / "opt" / name / "data.txt").write_text("in the layer")
    (tree / "app").mkdir()
    (tree / "app" / "data.txt").write_text("in the root")
    # The backend's own transfer commands ignore the PATH and tar options meant for the task's commands.
    spec = MachineSpec(HostImage(), workdir="/app", env={"PATH": "/nonexistent", "TAR_OPTIONS": "--no-such-option"})

    async def scenario(machine: LocalMachine) -> Result:
        await machine.upload(tree, "/")
        with pytest.raises(UnsupportedMachineSpec, match="contains its system directories"):
            await machine.download("/", tmp_path / "everything")
        return await machine.run(Command(("/bin/cat", f"/opt/{name}/data.txt", "/app/data.txt")))

    read = on_machine(copy_on_write_factory, spec, scenario)
    assert (read.exit_code, read.stdout) == (0, b"in the layerin the root"), read.stderr
    assert not Path(f"/opt/{name}").exists()


@root_commands
def test_a_copy_on_write_machine_may_work_in_a_system_directory(copy_on_write_factory):
    workdir = f"/opt/shellbox-test-{uuid.uuid4().hex}/work"
    spec = MachineSpec(HostImage(), workdir=workdir)
    result = run_command(copy_on_write_factory, Command(("sh", "-c", "pwd && touch made && ls")), spec)
    assert (result.exit_code, result.stdout) == (0, f"{workdir}\nmade\n".encode()), result.stderr
    assert not Path(workdir).parent.exists()


def test_files_mounted_inside_overlaid_directories_are_found_parents_first():
    # A container runtime mounts files such as /etc/resolv.conf over the image; an overlay of /etc hides them.
    mountinfo = "\n".join(
        [
            "21 1 0:31 / / rw,relatime - overlay overlay rw",
            "22 21 9:127 /pods/resolv.conf /etc/resolv.conf rw - xfs /dev/md127 rw",
            "23 21 9:127 /pods/host /etc/host\\040name rw - xfs /dev/md127 rw",
            "24 21 0:40 / /usr/lib/modules ro - tmpfs tmpfs rw",
            "25 21 0:41 / /usr ro - tmpfs tmpfs rw",
            "26 21 0:42 / /usrlocal/data rw - tmpfs tmpfs rw",
            "27 21 9:127 /pods/app /app rw - xfs /dev/md127 rw",
        ]
    )
    assert _submounts(mountinfo, ("/etc", "/usr")) == ("/etc/host name", "/etc/resolv.conf", "/usr/lib/modules")
