# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host requirements and offline OCI images for Shellbox tests."""

import hashlib
import io
import json
import shutil
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from itertools import count
from pathlib import Path

import pytest
from shellbox.backends.local.machine import LocalMachineFactory
from shellbox.backends.local.python_environment import PythonEnvironment, UvError, build_python_environment
from shellbox.backends.local.sandbox import SystemDirectories
from shellbox.image import PreparedImage, RegistryImage

from lib.shellbox.tests.local_machine_support import LOCK, PYTHON_VERSION, make_factory

HOST_REQUIREMENTS = {
    "requires_host_setsid": "The local Linux guest fake needs the host setsid executable",
    "requires_linux_procfs": "The reset test reads Linux /proc process records",
    "requires_linux_process_api": "The local process fake needs Linux /proc and waitid with WNOWAIT",
}


def pytest_configure(config: pytest.Config) -> None:
    for marker, reason in HOST_REQUIREMENTS.items():
        config.addinivalue_line("markers", f"{marker}: {reason} (skipped on macOS)")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if sys.platform != "darwin":
        return

    for item in items:
        for marker, reason in HOST_REQUIREMENTS.items():
            if item.get_closest_marker(marker) is not None:
                item.add_marker(pytest.mark.skip(reason=reason))


def pytest_addoption(parser):
    parser.addoption("--local-cgroup-parent", help="Writable delegated cgroup v2 parent for manual local-limit tests")


@pytest.fixture
def oci_image(tmp_path: Path) -> Callable[[tuple[bytes, ...], dict[str, object]], PreparedImage]:
    sequence = count()

    def build(layers: tuple[bytes, ...], process: dict[str, object]) -> PreparedImage:
        layout = tmp_path / f"oci-{next(sequence)}"
        blobs = layout / "blobs/sha256"
        blobs.mkdir(parents=True)

        def blob(data: bytes, media_type: str) -> dict[str, object]:
            digest = hashlib.sha256(data).hexdigest()
            (blobs / digest).write_bytes(data)
            return {"digest": f"sha256:{digest}", "size": len(data), "mediaType": media_type}

        descriptors = tuple(blob(layer, "application/vnd.oci.image.layer.v1.tar") for layer in layers)
        config = {
            "os": "linux",
            "architecture": "amd64",
            "config": process,
            "rootfs": {"type": "layers", "diff_ids": [layer["digest"] for layer in descriptors]},
        }
        descriptor = blob(json.dumps(config).encode(), "application/vnd.oci.image.config.v1+json")
        manifest = blob(
            json.dumps({"schemaVersion": 2, "config": descriptor, "layers": descriptors}).encode(),
            "application/vnd.oci.image.manifest.v1+json",
        )
        manifest["annotations"] = {"org.opencontainers.image.ref.name": "image"}
        (layout / "index.json").write_text(json.dumps({"schemaVersion": 2, "manifests": [manifest]}))
        (layout / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}))
        return PreparedImage(layout, str(manifest["digest"]), RegistryImage("offline-fixture"))

    return build


@pytest.fixture
def executable_oci_image(oci_image):
    busybox = shutil.which("busybox")
    if busybox is None:
        pytest.skip("The offline executable image requires busybox-static; the Shellbox CI lane installs it")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as output:
        binary = Path(busybox).read_bytes()
        entry = tarfile.TarInfo("bin/busybox")
        entry.mode, entry.size = 0o755, len(binary)
        output.addfile(entry, io.BytesIO(binary))
        for applet in (
            "sh",
            "sleep",
            "mkdir",
            "cat",
            "tar",
            "httpd",
            "wget",
            "dirname",
            "basename",
            "id",
            "dd",
            "rm",
            "true",
        ):
            entry = tarfile.TarInfo(f"bin/{applet}")
            entry.type, entry.linkname = tarfile.SYMTYPE, "busybox"
            output.addfile(entry)
        entry = tarfile.TarInfo("etc")
        entry.type, entry.mode = tarfile.DIRTYPE, 0o755
        output.addfile(entry)
    return oci_image(
        (archive.getvalue(),),
        {
            "Env": ["PATH=/bin", "IMAGE_VALUE=from-image"],
            "WorkingDir": "/app",
            "Entrypoint": ["/bin/sh", "-c"],
            "Cmd": ['printf "%s" "$IMAGE_VALUE" >index.html; exec httpd -f -p 8123 -h .'],
            "User": "0:0",
        },
    )


@pytest.fixture
def factory() -> LocalMachineFactory:
    return make_factory()


@pytest.fixture
def undelegated_factory(monkeypatch) -> LocalMachineFactory:
    monkeypatch.setattr("shellbox.backends.local.machine.delegated_cgroup_parent", lambda: None)
    return make_factory()


@pytest.fixture(scope="session")
def python_environment(tmp_path_factory) -> PythonEnvironment:
    lock = tmp_path_factory.mktemp("lock") / "requirements.lock"
    lock.write_text(LOCK)
    try:
        return build_python_environment(
            Path.home() / ".cache/shellbox/python/test-locked-six" / PYTHON_VERSION, lock, PYTHON_VERSION
        )
    except UvError as error:
        # uv reports an unreachable or failed fetch this way; any other failure is a bug in the builder.
        if not str(error).startswith("uv python install") or "Failed to download" not in str(error):
            raise
        pytest.skip(f"uv could not download a managed CPython {PYTHON_VERSION}: {error}")


@pytest.fixture
def python_factory(python_environment) -> LocalMachineFactory:
    """A factory whose commands find python3 in a self-contained environment that it mounts read-only."""
    return make_factory(read_only=(python_environment.root,), bin_dirs=(python_environment.bin_dir,))


@pytest.fixture
def outside_dir() -> Iterator[Path]:
    """A host directory the test user may write, which local commands cannot see."""
    path = Path(tempfile.mkdtemp(dir="/var/tmp"))
    yield path
    shutil.rmtree(path)


@pytest.fixture
def copy_on_write_factory() -> LocalMachineFactory:
    return make_factory(system_directories=SystemDirectories.COPY_ON_WRITE)
