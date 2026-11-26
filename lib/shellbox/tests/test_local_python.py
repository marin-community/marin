# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained Python environments for local commands."""

import asyncio
import io
import os
import sys
import tarfile
from pathlib import Path

import pytest
from local_machine_support import (
    PYTHON_VERSION,
    SPEC,
    make_factory,
    on_machine,
)
from shellbox.backends.local.python_environment import (
    UvError,
    resolve_python_environment,
)
from shellbox.machine import (
    Command,
    MachineSpec,
    PackageSpec,
)


@pytest.mark.skipif(sys.platform != "linux", reason="The local bubblewrap backend requires Linux")
def test_a_python_environment_runs_its_own_interpreter_and_locked_packages(python_environment, monkeypatch, tmp_path):
    script = (
        "import sys, six; from pathlib import Path; print(sys.prefix); print(sys.base_prefix); "
        "Path('answer').write_text(six.text_type(42))"
    )

    async def scenario(machine):
        result = await machine.run(Command(("python3", "-c", script)))
        assert result.exit_code == 0, result.stderr.decode()
        prefix, base_prefix = map(Path, result.stdout.decode().split())
        assert prefix == python_environment.bin_dir.parent
        assert base_prefix.is_relative_to(python_environment.root)
        await machine.download("/app/answer", tmp_path / "answer")
        assert (tmp_path / "answer").read_text() == "42"

    with monkeypatch.context() as host:
        host.setenv("PATH", "/usr/bin:/bin")
        factory = make_factory(read_only=(python_environment.root,), bin_dirs=(python_environment.bin_dir,))
        on_machine(factory, SPEC, scenario)

    packages = PackageSpec(pypi_packages=("six==1.17.0",), tools=("python3",))
    factory = make_factory(python_version=PYTHON_VERSION, hash_seed="23")
    spec = MachineSpec(packages, workdir="/app")

    async def check():
        first = await factory.create(spec)
        try:
            result = await first.run(
                Command(
                    (
                        "python3",
                        "-c",
                        "import six,sys,os; print(sys.prefix); print(sys.base_prefix); print(six.text_type(42)); "
                        "print(os.environ['PYTHONHASHSEED'])",
                    )
                )
            )
            assert result.exit_code == 0, result.stderr.decode()
            prefix, base_prefix, answer, seed = result.stdout.decode().splitlines()
            assert prefix != base_prefix
            assert Path(base_prefix).is_relative_to(factory.package_cache)
            assert (answer, seed) == ("42", "23")
            assert (await first.run(Command(("touch", "private")))).exit_code == 0
        finally:
            await first.close()
        # A fresh factory must reuse the resolved installation without index access.
        monkeypatch.setenv("UV_OFFLINE", "1")
        second = await make_factory(python_version=PYTHON_VERSION).create(spec)
        try:
            result = await second.run(Command(("python3", "-c", "import six,sys; print(sys.prefix)")))
            assert (result.exit_code, result.stdout.decode().strip()) == (0, prefix)
            assert (await second.run(Command(("test", "-e", "private")))).exit_code == 1
        finally:
            await second.close()

    asyncio.run(check())


def test_package_resolution_cannot_run_build_hooks_on_the_host(tmp_path, monkeypatch):
    marker = tmp_path / "host-hook"
    files = {
        "pyproject.toml": b'[build-system]\nrequires=[]\nbuild-backend="backend"\nbackend-path=["."]\n',
        "backend.py": (
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\nraise RuntimeError('hook')\n".encode()
        ),
    }
    source = tmp_path / "unsafe-1.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        for name, body in files.items():
            entry = tarfile.TarInfo("unsafe-1/" + name)
            entry.size = len(body)
            archive.addfile(entry, io.BytesIO(body))
    monkeypatch.setenv("UV_OFFLINE", "1")
    with pytest.raises(UvError):
        resolve_python_environment(tmp_path / "cache", (f"unsafe @ {source.as_uri()}",), PYTHON_VERSION)
    assert not marker.exists()


def test_every_symlink_in_a_python_environment_resolves_inside_its_root(python_environment):
    root = python_environment.root
    links = [
        Path(directory, name)
        for directory, directories, files in os.walk(root)
        for name in (*directories, *files)
        if os.path.islink(Path(directory, name))
    ]
    assert python_environment.python in links
    assert [link for link in links if not Path(os.path.realpath(link)).is_relative_to(root)] == []
