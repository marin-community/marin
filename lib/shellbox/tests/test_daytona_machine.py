# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Daytona adapter against a local process and filesystem fake."""

import asyncio
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from daytona.v2 import (
    DISABLED,
    ClientTimeoutError,
    ConflictError,
    DaytonaError,
    LimitExceededError,
    NotFoundError,
    RateLimitError,
    Resources,
)
from daytona.v2 import ConnectionError as ProviderConnectionError
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.errors import DaytonaCapacityExhausted, DaytonaProvisioningError
from shellbox.backends.daytona.images import ImageRequest, PruneAction, ensure_image, image_name, prune_images
from shellbox.backends.daytona.machine import (
    PROVIDER_DIAGNOSTIC_CHAR_LIMIT,
    DaytonaCommandStopped,
    DaytonaMachine,
    DaytonaMachineFactory,
    DaytonaNetworkMode,
    DaytonaNetworkPolicy,
    DaytonaOutputUnavailable,
    _create_sandbox,
    _delete_sandbox,
    _SandboxClaim,
    sandbox_resources,
)
from shellbox.backends.docker.machine import DockerMachineFactory, docker
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import (
    Command,
    CommandOutputLost,
    CommandSemantics,
    DockerImage,
    ExitReason,
    MachineDiskFull,
    MachineSpec,
    MachineTerminated,
    MachineUnavailable,
    UnsupportedMachineSpec,
)


def control_script(argv) -> str:
    """The script the backend runs through ``/bin/sh -c`` after fixing its control PATH."""
    return argv[2].partition("export PATH; ")[2]


def command_script(argv) -> str:
    """The control script, or the staged command script that it runs."""
    script = control_script(argv)
    words = shlex.split(script)
    if len(words) == 2 and words[0] == "/bin/sh" and words[1].endswith("/command.sh"):
        return Path(words[1]).read_text()
    return script


def exited(exit_code: int, output: str = "") -> SimpleNamespace:
    return SimpleNamespace(exit_code=exit_code, output=output, reason="exited", signal=None)


class LocalFiles:
    async def write_bytes(self, path: str, data: bytes) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def upload_file(self, local_path, remote_path: str) -> None:
        Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, remote_path)

    async def download_file(self, remote_path: str, local_path) -> None:
        shutil.copyfile(remote_path, local_path)


class LocalProcess:
    """Run each provider command as a local process group, as the sandbox runs it."""

    def __init__(self):
        self.processes = []

    async def run(self, *, argv, cwd=None, env=None, execution_timeout=None):
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self.processes.append(process)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), timeout=execution_timeout)
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            return SimpleNamespace(exit_code=None, output="", reason="timed_out", signal="KILL")
        return exited(process.returncode, output.decode(errors="replace"))

    async def close(self):
        for process in self.processes:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await process.wait()


class LocalImages:
    def __init__(self):
        self.images = {}
        self.submitted = []
        self.in_use: set[str] = set()

    def add(self, name, source, *, status="ready", error=None, pending_polls=0, created_at=None):
        image = SimpleNamespace(
            id=f"img_{len(self.images)}",
            name=name,
            status=status,
            error=error,
            source=source,
            polls=pending_polls,
            created_at=created_at or datetime(2026, 11, 1, tzinfo=UTC),
        )
        self.images[name] = image
        return image

    async def delete(self, image_id):
        if image_id in self.in_use:
            raise ConflictError("Image is used by a sandbox", status_code=409)
        image = self._find(image_id)
        del self.images[image.name]

    def _find(self, name_or_id):
        for image in self.images.values():
            if name_or_id in (image.name, image.id):
                return image
        raise NotFoundError("Image does not exist", status_code=404)

    async def get(self, name_or_id, *, wait=None):
        image = self._find(name_or_id)
        if wait == "ready" and image.status == "pending":
            # Stands in for the server's long poll.
            await asyncio.sleep(0.001)
            image.polls -= 1
            if image.polls == 0:
                image.status = "ready"
        return image

    async def pull(self, reference, *, name, wait):
        assert wait is False
        self.submitted.append(name)
        return self.add(name, reference)

    async def build_from_context(self, context, *, dockerfile, name):
        self.submitted.append(name)
        return self.add(name, (Path(context), dockerfile))

    async def iter(self):
        for image in self.images.values():
            yield image


def local_sandbox() -> SimpleNamespace:
    async def reset_idle_timer():
        return None

    async def refresh_data():
        return None

    return SimpleNamespace(
        id="sbx_fixture",
        state="started",
        error=None,
        fs=LocalFiles(),
        process=LocalProcess(),
        reset_idle_timer=reset_idle_timer,
        refresh_data=refresh_data,
    )


class LocalDaytona:
    def __init__(self):
        self.sandbox = local_sandbox()
        self.images = LocalImages()
        self.deleted = False
        self.closed = False
        self.params = None

    async def create(self, image, *, name, wait_seconds, create_timeout_seconds, **options):
        await self.images.get(image)
        self.params = SimpleNamespace(
            image=image, name=name, wait_seconds=wait_seconds, create_timeout_seconds=create_timeout_seconds, **options
        )
        return self.sandbox

    async def find_by_name(self, name):
        return (self.sandbox,) if self.params is not None and self.params.name == name else ()

    async def delete(self, sandbox, *, timeout):
        assert sandbox is self.sandbox
        await sandbox.process.close()
        self.deleted = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        self.closed = True


class FailedDeletion(LocalDaytona):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure

    async def delete(self, sandbox, *, timeout):
        await sandbox.process.close()
        raise self.failure


@pytest.mark.parametrize("failure", ["throttled", "lost", "rejected", "limit"])
def test_daytona_creation_repeats_the_same_named_request(monkeypatch, failure):
    errors = {
        "throttled": RateLimitError("rate limited", status_code=429, retry_after_seconds=17),
        "lost": ProviderConnectionError("Connection reset by peer"),
        "rejected": DaytonaError("invalid create request", status_code=400),
        # A full sandbox limit is a 429 too, but waiting out the create deadline would hide it.
        "limit": LimitExceededError("concurrent sandbox limit reached", status_code=429),
    }
    delays = []

    async def wait(delay):
        delays.append(delay)

    monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", wait)

    class FlakyCreate:
        def __init__(self):
            self.names = []

        async def create(self, image, *, name, **options):
            self.names.append(name)
            if len(self.names) < 3:
                raise errors[failure]
            return SimpleNamespace(id="sbx_created_once", state="started")

    client = FlakyCreate()
    request = _create_sandbox(
        client,
        "image",
        name="fixture",
        timeout=60,
        env={},
        labels={},
        resources=Resources(),
        lifecycle_policy={"timeout_seconds": 60, "on_timeout": "delete"},
        network_policy={"block_all": True},
    )
    if failure in ("rejected", "limit"):
        with pytest.raises(DaytonaError) as caught:
            asyncio.run(request)
        assert caught.value is errors[failure]
        assert client.names == ["fixture"] and not delays
        return
    sandbox = asyncio.run(request)
    # A named create is get-or-create, so every repeat names the same sandbox.
    assert sandbox.id == "sbx_created_once"
    assert client.names == ["fixture"] * 3
    assert len(delays) == 2
    if failure == "throttled":
        assert all(delay >= 17 for delay in delays)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux execve per-argument limit regression")
@pytest.mark.parametrize("shell", ["/bin/bash", "/bin/sh"])
def test_daytona_large_argument_preserves_process_contract(tmp_path: Path, shell: str) -> None:
    class RecordedFiles(LocalFiles):
        def __init__(self):
            self.staging = set()

        async def write_bytes(self, path: str, data: bytes) -> None:
            self.staging.add(Path(path).parent)
            await super().write_bytes(path, data)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.fs = RecordedFiles()
        workdir = tmp_path / "work with spaces"
        workdir.mkdir()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(source=RegistryImage("fixture@sha256:" + "a" * 64), workdir=str(workdir))
        )
        body = '/bin/cat; printf "|%s|%s|%s|%s" "$VALUE" "$PWD" "$PATH" "$0"; printf "error" >&2; exit 23'
        # The reported setup fits one Linux argument; nested control quoting did not.
        padding = 130712 - len(body.encode()) - 2
        source = "#" + "x" * padding + "\n" + body
        try:
            result = await machine.run(
                Command(
                    (shell, "-c", source),
                    stdin=b"input\x00\xff",
                    env={"VALUE": "quoted '\" value", "PATH": "/nonexistent"},
                    timeout=10,
                )
            )
            assert result.exit_code == 23
            assert result.stdout == b"input\x00\xff" + f"|quoted '\" value|{workdir}|/nonexistent|{shell}".encode()
            assert result.stderr == b"error"
            assert not result.stdout_truncated and not result.stderr_truncated
            assert all(not path.exists() for path in client.sandbox.fs.staging)
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.fixture
def lifecycle_client():
    client = LocalDaytona()

    async def prepared(**_kwargs):
        return exited(0, "/work\n")

    client.sandbox.process.run = prepared
    return client


@pytest.mark.parametrize(
    "idle_timeout_minutes, idle_policy",
    [
        (None, {"idle_expiry_seconds": 1800, "on_idle_timeout": "delete"}),
        (9, {"idle_expiry_seconds": 540, "on_idle_timeout": "delete"}),
        (0, {"idle_expiry_seconds": DISABLED}),
    ],
)
def test_daytona_creation_sets_orphan_cleanup_and_caller_labels(lifecycle_client, idle_timeout_minutes, idle_policy):
    async def scenario():
        labels = {"owner": "agent-eval", "lane": "eval", "job": "swe-rebench"}
        options = {} if idle_timeout_minutes is None else {"idle_timeout_minutes": idle_timeout_minutes}
        factory = DaytonaMachineFactory(lambda: lifecycle_client, ttl_minutes=75, labels=labels, **options)
        labels["job"] = "another-job"
        machine = await factory.create(MachineSpec(source=RegistryImage("fixture:latest")))
        try:
            params = lifecycle_client.params
            assert params.lifecycle_policy == {"timeout_seconds": 4500, "on_timeout": "delete", **idle_policy}
            assert params.labels == {"owner": "agent-eval", "lane": "eval", "job": "swe-rebench"}
        finally:
            await machine.close()
        assert lifecycle_client.deleted and lifecycle_client.closed

    asyncio.run(scenario())


def test_daytona_open_machine_renews_activity_after_provider_failure_and_stops_before_deletion(
    lifecycle_client, monkeypatch
):
    async def scenario():
        ticks = asyncio.Queue()
        events = []
        refreshing = asyncio.Event()

        async def wait(interval):
            tick = asyncio.get_running_loop().create_future()
            await ticks.put((interval, tick))
            await tick

        async def reset_idle_timer():
            events.append("refresh")
            if events.count("refresh") == 1:
                raise DaytonaError("temporary gateway failure", status_code=502)
            if events.count("refresh") == 2:
                return
            refreshing.set()
            try:
                await asyncio.Future()
            finally:
                events.append("refresh stopped")

        delete = lifecycle_client.delete

        async def record_deletion(sandbox, *, timeout):
            events.append("delete")
            await delete(sandbox, timeout=timeout)

        lifecycle_client.sandbox.reset_idle_timer = reset_idle_timer
        lifecycle_client.delete = record_deletion
        monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", wait)
        machine = await DaytonaMachineFactory(lambda: lifecycle_client, idle_timeout_minutes=9).create(
            MachineSpec(source=RegistryImage("fixture:latest"))
        )
        try:
            for _ in range(3):
                interval, tick = await ticks.get()
                assert interval == 180
                tick.set_result(None)
            await refreshing.wait()
        finally:
            await machine.close()
        assert events == ["refresh", "refresh", "refresh", "refresh stopped", "delete"]
        assert lifecycle_client.deleted and lifecycle_client.closed

    asyncio.run(scenario())


def test_daytona_failed_setup_cancels_activity_renewal(lifecycle_client, monkeypatch):
    async def scenario():
        sleeping = asyncio.Event()
        stopped = asyncio.Event()

        async def wait(_interval):
            sleeping.set()
            try:
                await asyncio.Future()
            finally:
                stopped.set()

        async def failed_setup(**_kwargs):
            await sleeping.wait()
            raise DaytonaError("setup failed", status_code=500)

        lifecycle_client.sandbox.process.run = failed_setup
        monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", wait)
        with pytest.raises(DaytonaProvisioningError):
            await DaytonaMachineFactory(lambda: lifecycle_client).create(
                MachineSpec(source=RegistryImage("fixture:latest"))
            )
        assert stopped.is_set()
        assert lifecycle_client.deleted and lifecycle_client.closed
        assert lifecycle_client.params.labels == {"owner": "shellbox"}

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["error", "cancelled", "not_started"])
def test_daytona_failed_creation_deletes_an_accepted_sandbox(no_retry_wait, failure):
    class AcceptedThenFailed(LocalDaytona):
        searches = 0

        async def find_by_name(self, name):
            # Search is eventually consistent: the accepted sandbox appears on the third read.
            self.searches += 1
            return await super().find_by_name(name) if self.searches >= 3 else ()

        async def create(self, image, **kwargs):
            sandbox = await super().create(image, **kwargs)
            if failure == "not_started":
                # A v2 create returns the sandbox in whatever state its wait reached.
                sandbox.state, sandbox.error = "error", "boot failed"
                return sandbox
            # The provider accepted the sandbox, but no handle came back.
            raise (RuntimeError if failure == "error" else asyncio.CancelledError)("SDK failed after acceptance")

    async def scenario():
        client = AcceptedThenFailed()
        factory = DaytonaMachineFactory(lambda: client)
        expected = asyncio.CancelledError if failure == "cancelled" else RuntimeError
        with pytest.raises(expected):
            await factory.create(MachineSpec(source=RegistryImage("fixture@sha256:" + "a" * 64), cpus=2))
        assert client.deleted
        assert client.closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("provider_error", "expected"),
    [
        (LimitExceededError("image limit reached", status_code=429), DaytonaCapacityExhausted),
        (RateLimitError("too many requests", status_code=429), DaytonaCapacityExhausted),
        (ProviderConnectionError("Connection reset by peer"), MachineUnavailable),
        (ClientTimeoutError("Request timed out", status_code=408), MachineUnavailable),
        (DaytonaError("invalid image reference", status_code=400), DaytonaProvisioningError),
    ],
    ids=["limit", "throttled", "lost-connection", "timed-out", "refused"],
)
def test_daytona_factory_raises_provider_failures_under_shellbox_types(no_retry_wait, provider_error, expected):
    # Callers classify failures by name, and the SDK's names (ConnectionError, RateLimitError) are shared.
    class FailingImages(LocalImages):
        async def get(self, name_or_id, *, wait=None):
            raise provider_error

    client = LocalDaytona()
    client.images = FailingImages()
    factory = DaytonaMachineFactory(lambda: client)
    spec = MachineSpec(source=RegistryImage("fixture:tag"))
    for operation in (factory.prepare, factory.create):
        with pytest.raises(expected) as caught:
            asyncio.run(operation(spec))
        assert type(caught.value) is expected and caught.value.__cause__ is provider_error


@pytest.mark.requires_host_setsid
def test_daytona_machine_close_raises_a_refused_deletion_under_a_shellbox_type(tmp_path):
    refusal = DaytonaError("Sandbox cannot be deleted", status_code=400)

    async def scenario():
        client = FailedDeletion(refusal)
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        with pytest.raises(DaytonaProvisioningError) as caught:
            await machine.close()
        assert caught.value.__cause__ is refusal

    asyncio.run(scenario())


@pytest.mark.parametrize("refusal", ["throttle", "limit"])
def test_image_preparation_waits_out_a_throttle_but_not_a_full_limit(no_retry_wait, refusal):
    class ThrottledOnce(LocalImages):
        refusals = 0

        async def get(self, name_or_id, *, wait=None):
            if not self.refusals:
                self.refusals += 1
                if refusal == "limit":
                    raise LimitExceededError("image limit reached", status_code=429)
                raise RateLimitError("too many requests", status_code=429, retry_after_seconds=1)
            return await super().get(name_or_id, wait=wait)

    client = LocalDaytona()
    client.images = ThrottledOnce()
    client.images.add(image_name(RegistryImage("fixture:tag")), "fixture:tag")
    prepare = DaytonaMachineFactory(lambda: client).prepare(MachineSpec(source=RegistryImage("fixture:tag")))
    if refusal == "limit":
        with pytest.raises(DaytonaCapacityExhausted):
            asyncio.run(prepare)
    else:
        assert asyncio.run(prepare) == image_name(RegistryImage("fixture:tag"))


def test_daytona_refused_creation_skips_the_name_search():
    class RefusedCreate(LocalDaytona):
        searches = 0

        async def create(self, image, **kwargs):
            raise LimitExceededError("concurrent sandbox limit reached", status_code=429)

        async def find_by_name(self, name):
            self.searches += 1
            return ()

    client = RefusedCreate()
    with pytest.raises(DaytonaCapacityExhausted) as caught:
        asyncio.run(DaytonaMachineFactory(lambda: client).create(MachineSpec(source=RegistryImage("fixture:tag"))))
    assert isinstance(caught.value.__cause__, LimitExceededError)
    # A refused create made no sandbox, so cleanup does not wait out eventually consistent search.
    assert client.searches == 0
    assert client.closed


def test_image_preflight_builds_without_starting_sandbox_and_ignores_resources():
    client = LocalDaytona()
    factory = DaytonaMachineFactory(lambda: client)
    spec = MachineSpec(source=RegistryImage("fixture@sha256:" + "a" * 64), cpus=2, memory_mb=2049)

    async def scenario():
        first = await factory.prepare(spec)
        second = await factory.prepare(replace(spec, cpus=4, memory_mb=8192, storage_mb=20480))
        assert first == second
        assert client.images.submitted == [first]
        assert (await client.images.get(first)).source == spec.source.reference
        assert client.params is None

    asyncio.run(scenario())


@pytest.mark.parametrize("policy", [None, DaytonaNetworkPolicy(DaytonaNetworkMode.DOMAIN_ALLOW_LIST, "example.org")])
@pytest.mark.requires_host_setsid
def test_daytona_binary_command_and_files(tmp_path: Path, policy) -> None:
    async def scenario() -> None:
        client = LocalDaytona()
        workdir = tmp_path / "work"
        machine = await DaytonaMachineFactory(lambda: client, network_policy=policy).create(
            MachineSpec(
                source=RegistryImage("ubuntu:24.04"),
                workdir=str(workdir),
                cpus=2,
                memory_mb=1536,
                storage_mb=1025,
                env={"TASK_VALUE": "daytona"},
                startup_timeout=900,
            )
        )
        if policy is None:
            assert client.params.network_policy == {"block_all": True}
        else:
            assert client.params.network_policy == {"block_all": True, "allow_domains": ["example.org"]}
        assert client.params.lifecycle_policy["timeout_seconds"] == 360 * 60
        assert client.params.resources == Resources(cpu=1, cpu_max=1, memory=1.5, memory_max=1.5, disk=1025 / 1024)
        assert client.params.user == "root"
        assert client.params.env == {"TASK_VALUE": "daytona"}
        assert client.params.wait_seconds == 900
        try:
            environment = await machine.run(Command(("sh", "-c", 'printf "%s" "$TASK_VALUE"'), user="0"))
            assert environment.stdout == b"daytona"
            result = await machine.run(
                Command(
                    ("/bin/sh", "-c", "cat; printf '\\000\\377' >&2"),
                    stdin=b"abc\x00\xff",
                    output_limit_bytes=4,
                )
            )
            assert result.exit_code == 0
            assert result.stdout == b"abc\x00"
            assert result.stdout_truncated
            assert result.stderr == b"\x00\xff"

            source = tmp_path / "input.bin"
            source.write_bytes(b"\x00\xffpayload")
            await machine.upload(source, str(workdir / "data.bin"))
            downloaded = tmp_path / "downloaded.bin"
            await machine.download(str(workdir / "data.bin"), downloaded)
            assert downloaded.read_bytes() == source.read_bytes()
        finally:
            await machine.close()
        assert client.deleted
        assert client.closed

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
@pytest.mark.parametrize("inherit_image_workdir", [False, True])
def test_daytona_recreated_workdir_reopens_path_for_next_command(tmp_path: Path, inherit_image_workdir: bool) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    original_inode = workdir.stat().st_ino

    class StaleWorkdirProcess(LocalProcess):
        def __init__(self):
            super().__init__()
            self.workdir_replaced = False

        async def run(self, *, argv, cwd=None, **kwargs):
            if cwd is None:
                if self.workdir_replaced:
                    raise OSError("Provider process retained the deleted working directory")
                cwd = str(workdir)
            result = await super().run(argv=argv, cwd=cwd, **kwargs)
            self.workdir_replaced = workdir.stat().st_ino != original_inode
            return result

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = StaleWorkdirProcess()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="" if inherit_image_workdir else str(workdir))
        )
        try:
            replaced = await machine.run(Command(("sh", "-c", 'rm -rf "$PWD" && mkdir -p "$PWD" && printf recreated')))
            assert (replaced.exit_code, replaced.stdout) == (0, b"recreated")
            following = await machine.run(Command((sys.executable, "-c", "import os; print(os.getcwd())")))
            assert (following.exit_code, following.stdout) == (0, f"{workdir}\n".encode())
            assert workdir.stat().st_ino != original_inode
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_output_count_ignores_provider_diagnostics(tmp_path: Path) -> None:
    class NoisyCountProcess(LocalProcess):
        async def run(self, **kwargs):
            result = await super().run(**kwargs)
            result.output = "shell-init: getcwd failed\n" + result.output
            return result

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = NoisyCountProcess()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            result = await machine.run(Command(("printf", "output"), output_limit_bytes=3))
            assert (result.exit_code, result.stdout, result.stdout_truncated) == (0, b"out", True)
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_output_removed_by_the_guest_is_reported_as_lost(tmp_path: Path) -> None:
    class ClearedTmpProcess(LocalProcess):
        armed = False

        async def run(self, *, argv, **kwargs):
            if "wc -c" in argv[2] and self.armed:
                # Stands in for a guest process (e.g. systemd starting) that clears /tmp once.
                prefix = re.search(r"/tmp/\.shellbox-[0-9a-f]+", argv[2])
                assert prefix is not None
                shutil.rmtree(prefix.group(0), ignore_errors=True)
                self.armed = False
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = LocalDaytona()
        process = ClearedTmpProcess()
        client.sandbox.process = process
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        process.armed = True
        try:
            with pytest.raises(CommandOutputLost):
                await machine.run(Command(("printf", "output")))
            following = await machine.run(Command(("printf", "ready")))
            assert (following.exit_code, following.stdout) == (0, b"ready")
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.skipif(shutil.which("setsid") is None, reason="Daytona command execution needs setsid")
@pytest.mark.parametrize("collection_timeout, succeeds", [(0.01, False), (1.0, True)])
def test_daytona_output_collection_uses_its_own_deadline(collection_timeout: float, succeeds: bool):
    class SlowFiles(LocalFiles):
        async def read_bytes(self, path: str) -> bytes:
            await asyncio.sleep(0.1)
            return await super().read_bytes(path)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.fs = SlowFiles()
        machine = await DaytonaMachineFactory(lambda: client, output_collection_timeout=collection_timeout).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        try:
            command = Command(("sh", "-c", "printf out; printf err >&2"))
            if succeeds:
                result = await machine.run(command)
                assert (result.stdout, result.stderr) == (b"out", b"err")
            else:
                with pytest.raises(DaytonaOutputUnavailable) as caught:
                    await machine.run(command)
                # Callers retry a lost provider connection.
                assert isinstance(caught.value, ConnectionError)
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("oversized", [False, True])
def test_daytona_output_capture_preserves_provider_startup_failure(tmp_path: Path, oversized: bool) -> None:
    diagnostic = "Provider rejected the command before opening its output files"
    if oversized:
        diagnostic += "x" * (2 * PROVIDER_DIAGNOSTIC_CHAR_LIMIT)

    class RejectedCommandProcess(LocalProcess):
        async def run(self, *, env=None, **kwargs):
            if env is not None:
                return exited(126, diagnostic)
            return await super().run(env=env, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = RejectedCommandProcess()
        machine = DaytonaMachine(
            client.sandbox, MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path)), AsyncExitStack()
        )
        try:
            with pytest.raises(RuntimeError, match="Failed to measure command output") as caught:
                await machine.run(Command(("printf", "not executed")))
            note = next(note for note in caught.value.__notes__ if note.startswith("Daytona exec response: "))
            response = json.loads(note.removeprefix("Daytona exec response: "))
            assert response["exit_code"] == 126
            assert response["diagnostic"].startswith("Provider rejected the command")
            assert len(response["diagnostic"]) <= PROVIDER_DIAGNOSTIC_CHAR_LIMIT
            assert response["diagnostic_truncated"] is oversized
            if not oversized:
                assert response["diagnostic"] == diagnostic
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_command_the_provider_stopped_is_not_a_result(tmp_path: Path) -> None:
    class StoppedRequest(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "candidate-stopped" in command_script(argv):
                # v2 reports a command its execution timeout killed without an exit code.
                return SimpleNamespace(exit_code=None, output="", reason="timed_out", signal="KILL")
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = StoppedRequest()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            with pytest.raises(DaytonaCommandStopped, match="timed_out"):
                await machine.run(Command(("printf", "candidate-stopped")))
            following = await machine.run(Command(("printf", "ready")))
            assert (following.exit_code, following.stdout) == (0, b"ready")
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.skipif(shutil.which("setsid") is None, reason="The command boundary needs a host setsid executable")
@pytest.mark.parametrize("interruption", ["timeout", "cancel", "cancel_twice", "cancel_during_timeout_stop"])
def test_daytona_command_interruption_stops_descendants_and_preserves_next_command(tmp_path, interruption):
    async def scenario():
        stopping, release_stop = asyncio.Event(), asyncio.Event()

        class HeldStop(LocalProcess):
            def __init__(self):
                super().__init__()
                self.cancellations = []

            async def run(self, *, argv, **kwargs):
                try:
                    if "stop-command" in argv[2] and interruption in {"cancel_twice", "cancel_during_timeout_stop"}:
                        stopping.set()
                        await release_stop.wait()
                    return await super().run(argv=argv, **kwargs)
                except asyncio.CancelledError as cancellation:
                    self.cancellations.append(cancellation)
                    raise

        client = LocalDaytona()
        client.sandbox.process = HeldStop()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        child = None
        try:
            await machine.run(Command(("sh", "-c", "echo 12 > answer")))
            pending = asyncio.create_task(
                machine.run(
                    Command(
                        ("sh", "-c", "sleep 3600 & echo $! > child.pid; wait"),
                        timeout=0.5 if interruption in {"timeout", "cancel_during_timeout_stop"} else None,
                    )
                )
            )
            async with asyncio.timeout(5):
                while not (tmp_path / "child.pid").exists() or not (tmp_path / "child.pid").read_text().strip():
                    # Wait for the real child to publish its PID before cancelling its parent.
                    await asyncio.sleep(0)
            child = int((tmp_path / "child.pid").read_text())
            if interruption == "timeout":
                result = await pending
                assert result.reason is ExitReason.TIMED_OUT
            else:
                if interruption == "cancel_during_timeout_stop":
                    await asyncio.wait_for(stopping.wait(), timeout=5)
                pending.cancel("agent deadline")
                if interruption == "cancel_twice":
                    await asyncio.wait_for(stopping.wait(), timeout=5)
                    pending.cancel("second cancellation")
                release_stop.set()
                with pytest.raises(asyncio.CancelledError) as caught:
                    await pending
                assert caught.value.args == ("agent deadline",)
                assert caught.value is client.sandbox.process.cancellations[-1]
            stopped = await machine.run(
                Command(
                    (
                        "sh",
                        "-c",
                        'if [ -f "/proc/$1/stat" ]; then read -r pid comm state rest < "/proc/$1/stat"; '
                        'test "$state" = Z; fi',
                        "child-state",
                        str(child),
                    )
                )
            )
            assert stopped.exit_code == 0
            graded = await machine.run(Command(("sh", "-c", 'test "$(cat answer)" = 12 && printf 1.0')))
            assert (graded.exit_code, graded.stdout) == (0, b"1.0")
            assert not client.deleted and not client.closed
        finally:
            release_stop.set()
            await machine.close()
            if child is not None:
                try:
                    os.kill(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        assert client.deleted and client.closed

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_unanswered_command_cleanup_reports_a_lost_machine(tmp_path, monkeypatch):
    monkeypatch.setattr("shellbox.backends.daytona.machine.INTERRUPT_TIMEOUT", 0.2)

    class HungCleanup(LocalProcess):
        armed = False

        async def run(self, *, argv, **kwargs):
            if self.armed and "rm -rf /tmp/.shellbox-" in argv[2]:
                await asyncio.Future()
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = LocalDaytona()
        process = HungCleanup()
        client.sandbox.process = process
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        process.armed = True
        with pytest.raises(MachineTerminated):
            await machine.run(Command(("true",)))
        assert client.closed

    asyncio.run(scenario())


@pytest.mark.parametrize("interrupt_failure", ["exit", "timeout", "files"])
@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
@pytest.mark.requires_host_setsid
def test_daytona_failed_interruption_cleanup_closes_machine(tmp_path, interrupt_failure, interruption):
    entered = asyncio.Event()

    class FailedStop(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "candidate-block" in command_script(argv):
                entered.set()
                await asyncio.Future()
            if "stop-command" in argv[2]:
                if interrupt_failure == "timeout":
                    raise TimeoutError("Provider interruption timed out")
                if interrupt_failure == "files":
                    return exited(0)
                return exited(1, "Cannot stop the process group")
            if interrupt_failure == "files" and entered.is_set() and "rm -rf /tmp/.shellbox-" in argv[2]:
                return exited(1, "Cannot remove command files")
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = FailedStop()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        pending = asyncio.create_task(
            machine.run(Command(("candidate-block",), timeout=0.5 if interruption == "timeout" else None))
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        if interruption == "cancel":
            pending.cancel("agent deadline")
            with pytest.raises(asyncio.CancelledError) as failure:
                await pending
            assert failure.value.args == ("agent deadline",)
        else:
            # The failed cleanup closes the machine, which callers retry as lost infrastructure.
            with pytest.raises(MachineTerminated) as failure:
                await pending
            assert isinstance(failure.value.__cause__, TimeoutError if interrupt_failure == "timeout" else RuntimeError)
        assert client.deleted and client.closed
        # Grading after the agent stops still uses the machine; callers retry its loss as infrastructure.
        with pytest.raises(MachineTerminated, match="was closed after"):
            await machine.run(Command(("true",)))

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_machine_its_owner_closed_rejects_commands_as_misuse(tmp_path):
    async def scenario():
        client = LocalDaytona()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        await machine.close()
        with pytest.raises(RuntimeError, match="Machine is closed") as failure:
            await machine.run(Command(("true",)))
        assert not isinstance(failure.value, MachineTerminated)

    asyncio.run(scenario())


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
@pytest.mark.requires_host_setsid
def test_daytona_user_probe_interruption_preserves_the_machine(tmp_path, interruption):
    entered = asyncio.Event()

    class BlockedProbe(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "su --help" in argv[2]:
                entered.set()
                await asyncio.Future()
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = BlockedProbe()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            pending = asyncio.create_task(
                machine.run(
                    Command(
                        ("printf", "not-started"),
                        user="nobody",
                        timeout=0.05 if interruption == "timeout" else None,
                    )
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            if interruption == "cancel":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                result = await asyncio.wait_for(pending, timeout=0.5)
                assert result.reason is ExitReason.TIMED_OUT
            following = await machine.run(Command(("printf", "ready")))
            assert (following.exit_code, following.stdout) == (0, b"ready")
            assert not client.closed and not client.deleted
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_cancellation_during_upload_removes_staged_command_and_preserves_machine(tmp_path):
    async def scenario():
        uploading = asyncio.Event()

        class BlockedUpload(LocalFiles):
            staged: Path | None = None

            async def write_bytes(self, path: str, data: bytes) -> None:
                await super().write_bytes(path, data)
                if self.staged is None:
                    self.staged = Path(path).parent
                    uploading.set()
                    await asyncio.Future()

        client = LocalDaytona()
        files = BlockedUpload()
        client.sandbox.fs = files
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            pending = asyncio.create_task(machine.run(Command(("sh", "-c", "touch cancelled-command"))))
            await asyncio.wait_for(uploading.wait(), timeout=5)
            assert files.staged is not None and (files.staged / "command.sh").is_file()
            pending.cancel("agent deadline")
            with pytest.raises(asyncio.CancelledError) as caught:
                await pending
            assert caught.value.args == ("agent deadline",)
            assert not files.staged.exists()
            graded = await machine.run(Command(("sh", "-c", "test ! -e cancelled-command && printf 1.0")))
            assert (graded.exit_code, graded.stdout) == (0, b"1.0")
            assert not client.closed and not client.deleted
        finally:
            await machine.close()
            if files.staged is not None and files.staged.exists():
                shutil.rmtree(files.staged)

    asyncio.run(scenario())


@pytest.mark.parametrize("command_timeout", [None, 30])
@pytest.mark.requires_host_setsid
def test_daytona_provider_timeout_remains_an_infrastructure_error(tmp_path, command_timeout):
    provider_error = TimeoutError("Provider request timed out")

    class FailedRequest(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "candidate-request" in command_script(argv):
                raise provider_error
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = FailedRequest()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            with pytest.raises(TimeoutError) as caught:
                await machine.run(Command(("candidate-request",), timeout=command_timeout))
            assert caught.value is provider_error
        finally:
            await machine.close()
        assert client.closed and client.deleted

    asyncio.run(scenario())


CONNECTION_RESET = ProviderConnectionError("Connection reset by peer")
BAD_GATEWAY = DaytonaError("502 Bad Gateway", status_code=502)


class LostConnection(LocalProcess):
    """Fail each request whose script contains ``marker`` in transit, at most ``losses`` times."""

    def __init__(self, marker: str, *, losses: int, runs_first: bool, error: DaytonaError = CONNECTION_RESET):
        super().__init__()
        self.marker = marker
        self.losses = losses
        self.runs_first = runs_first
        self.error = error

    async def run(self, *, argv, **kwargs):
        if self.marker not in command_script(argv) or not self.losses:
            return await super().run(argv=argv, **kwargs)
        self.losses -= 1
        if self.runs_first:
            await super().run(argv=argv, **kwargs)
        raise self.error


@pytest.mark.requires_host_setsid
@pytest.mark.parametrize(
    "provider_error,state,expected",
    [
        (ProviderConnectionError("Connection reset by peer"), "started", ConnectionError),
        (ClientTimeoutError("Request timed out", status_code=408), "started", ConnectionError),
        (DaytonaError("502 Bad Gateway", status_code=502), "started", ConnectionError),
        (DaytonaError("Sandbox is not started", status_code=409), "stopped", MachineTerminated),
        (DaytonaError("Sandbox is not started", status_code=409), None, MachineTerminated),
        (DaytonaError("Invalid request", status_code=400), "started", DaytonaError),
        (DaytonaError("Invalid request", status_code=400), "unreachable", ConnectionError),
    ],
    ids=["lost-connection", "timed-out", "bad-gateway", "stopped", "deleted", "started", "probe-unreachable"],
)
def test_daytona_provider_failures_report_a_lost_machine(provider_error, state, expected):
    class FailedRequest(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "candidate-request" in command_script(argv):
                raise provider_error
            return await super().run(argv=argv, **kwargs)

    async def refresh_data():
        if state is None:
            raise NotFoundError("Sandbox not found", status_code=404)
        if state == "unreachable":
            raise DaytonaError("502 Bad Gateway", status_code=502)
        client.sandbox.state = state

    client = LocalDaytona()
    client.sandbox.process = FailedRequest()
    client.sandbox.refresh_data = refresh_data

    async def scenario():
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        try:
            with pytest.raises(expected) as caught:
                await machine.run(Command(("candidate-request",)))
            assert caught.value is provider_error or caught.value.__cause__ is provider_error
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_full_sandbox_reports_a_full_machine():
    class FullDisk(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "mkdir -p /tmp/.shellbox-" in argv[2]:
                return exited(1, "mkdir: cannot create directory: No space left on device")
            return await super().run(argv=argv, **kwargs)

    client = LocalDaytona()
    client.sandbox.process = FullDisk()

    async def scenario():
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        try:
            with pytest.raises(MachineDiskFull):
                await machine.run(Command(("true",)))
        finally:
            await machine.close()

    asyncio.run(scenario())


class LostDownload(LocalFiles):
    lost = False

    async def read_bytes(self, path: str) -> bytes:
        if not self.lost:
            self.lost = True
            raise ProviderConnectionError("Connection reset by peer")
        return await super().read_bytes(path)


# Retry backoffs stay below this; the keepalive interval is minutes.
LONGEST_RETRY_WAIT = 60


@pytest.fixture
def no_retry_wait(monkeypatch):
    """Skip the retry waits; the keepalive wait keeps its length so that it does not spin."""

    async def wait(delay):
        if delay > LONGEST_RETRY_WAIT:
            await asyncio.sleep(delay)

    monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", wait)
    monkeypatch.setattr("shellbox.backends.daytona.transient.sleep", wait)
    monkeypatch.setattr("shellbox.backends.daytona.images.sleep", wait)


@pytest.mark.parametrize("error", [CONNECTION_RESET, BAD_GATEWAY], ids=["reset", "502"])
@pytest.mark.parametrize("command_ran", [False, True])
@pytest.mark.requires_host_setsid
def test_daytona_command_runs_once_when_its_request_fails_in_transit(tmp_path, no_retry_wait, command_ran, error):
    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = LostConnection("candidate-run", losses=1, runs_first=command_ran, error=error)
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            result = await machine.run(Command(("sh", "-c", "echo candidate-run >> runs; printf out; exit 7")))
        finally:
            await machine.close()
        assert (result.exit_code, result.stdout, result.reason) == (7, b"out", ExitReason.EXITED)
        assert (tmp_path / "runs").read_text() == "candidate-run\n"

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_command_fails_when_its_request_keeps_losing_the_connection(tmp_path, no_retry_wait):
    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = LostConnection("candidate-run", losses=math.inf, runs_first=False)
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            with pytest.raises(ConnectionError, match="Lost the connection") as failure:
                await machine.run(Command(("sh", "-c", "echo candidate-run >> runs")))
            assert isinstance(failure.value.__cause__, ProviderConnectionError)
        finally:
            await machine.close()
        assert not (tmp_path / "runs").exists()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_repeated_command_request_keeps_its_provider_failure(tmp_path, no_retry_wait):
    class FailedStart(LostConnection):
        async def run(self, *, argv, **kwargs):
            if self.marker in command_script(argv) and not self.losses:
                return exited(-1, "fork/exec /usr/bin/zsh: no such file or directory")
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = FailedStart("candidate-run", losses=1, runs_first=False)
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            with pytest.raises(RuntimeError, match="Failed to measure command output") as caught:
                await machine.run(Command(("sh", "-c", "echo candidate-run >> runs")))
        finally:
            await machine.close()
        assert any("fork/exec /usr/bin/zsh" in note for note in caught.value.__notes__)

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_command_fails_when_its_shell_dies_after_a_lost_connection(tmp_path, no_retry_wait):
    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = LostConnection("candidate-run", losses=1, runs_first=True)
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            # The command kills the shell that holds the exit trap, so no exit code is recorded.
            with pytest.raises(RuntimeError, match="without an exit code"):
                await machine.run(Command(("sh", "-c", "echo candidate-run >> runs; kill -9 $PPID")))
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("lost", ["umask 077", "measure_and_cap", "tar xzf", "download"])
@pytest.mark.requires_host_setsid
def test_daytona_repeats_a_repeatable_request_that_lost_its_connection(tmp_path, no_retry_wait, lost):
    async def scenario():
        client = LocalDaytona()
        if lost == "download":
            client.sandbox.fs = LostDownload()
        else:
            client.sandbox.process = LostConnection(lost, losses=1, runs_first=True)
        workdir = tmp_path / "work"
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(workdir))
        )
        source = tmp_path / "source"
        source.mkdir()
        (source / "data").write_text("payload")
        try:
            await machine.upload(source, str(workdir / "uploaded"))
            result = await machine.run(Command(("cat", "uploaded/data")))
        finally:
            await machine.close()
        assert (result.exit_code, result.stdout) == (0, b"payload")

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
@pytest.mark.parametrize("deletion_failure", [False, True])
def test_daytona_cancellation_during_failed_command_cleanup_remains_cancellation(deletion_failure):
    entered = asyncio.Event()
    deletion_error = DaytonaError("Failed to remove sandbox", status_code=500)

    class BlockedCleanup(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "candidate-fail" in command_script(argv):
                raise OSError("Provider command failed")
            if "rm -rf /tmp/.shellbox-" in argv[2]:
                entered.set()
                await asyncio.Future()
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = FailedDeletion(deletion_error) if deletion_failure else LocalDaytona()
        client.sandbox.process = BlockedCleanup()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        pending = asyncio.create_task(machine.run(Command(("candidate-fail",))))
        await asyncio.wait_for(entered.wait(), timeout=5)
        pending.cancel("grading deadline")
        with pytest.raises(asyncio.CancelledError) as caught:
            await pending
        assert caught.value.args == ("grading deadline",)
        if deletion_failure:
            assert any(str(deletion_error) in note for note in caught.value.__notes__)
        assert client.closed and client.deleted == (not deletion_failure)

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_output_failure_retains_cleanup_note_on_the_primary_error():
    output_error = TimeoutError("Provider output request timed out")

    class FailedOutputAndCleanup(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "wc -c" in argv[2]:
                raise output_error
            if "rm -rf /tmp/.shellbox-" in argv[2]:
                raise OSError("Provider cleanup failed")
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = FailedOutputAndCleanup()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        with pytest.raises(ConnectionError) as caught:
            await machine.run(Command(("printf", "ready")))
        assert caught.value.__cause__ is output_error
        assert len(caught.value.__notes__) == 1
        assert client.closed and client.deleted

    asyncio.run(scenario())


@pytest.mark.parametrize("path_source", ["image", "task"])
@pytest.mark.requires_host_setsid
def test_daytona_root_cleanup_does_not_execute_candidate_path_programs(tmp_path, path_source):
    binary = tmp_path / "bin"
    binary.mkdir()
    marker = tmp_path / "root-cleanup-executed"
    for name in ("touch", "rm", "wc", "head", "mkdir", "setsid"):
        program = binary / name
        program.write_text(f'#!/bin/sh\nprintf injected > "{marker}"\nexit 99\n')
        program.chmod(0o755)
    agent_program = binary / "agent-program"
    agent_program.write_text("#!/bin/sh\nprintf agent-path\n")
    agent_program.chmod(0o755)
    candidate_path = f"{binary}:{os.environ['PATH']}"

    class ImageProcess(LocalProcess):
        async def run(self, *, env=None, **kwargs):
            if path_source == "image":
                env = {"PATH": candidate_path, **(env or {})}
            return await super().run(env=env, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = ImageProcess()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(
                RegistryImage("ubuntu:24.04"),
                workdir=str(tmp_path),
                env={"PATH": candidate_path} if path_source == "task" else {},
            )
        )
        try:
            result = await machine.run(Command(("agent-program",), output_limit_bytes=4))
            assert (result.exit_code, result.stdout, result.stdout_truncated) == (0, b"agen", True)
            assert not marker.exists()
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_completion_between_probe_and_stop_preserves_the_machine(tmp_path):
    paths = []

    class CompletionDuringStop(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "candidate-finished" in command_script(argv):
                words = shlex.split(command_script(argv))
                index = words.index("shellbox-command")
                paths.extend(Path(path) for path in words[index + 1 : index + 3])
                await super().run(argv=argv, **kwargs)
                paths[1].unlink()
                paths[0].write_text("99999999\n")
                await asyncio.Future()
            if "stop-command" in argv[2]:
                # Complete after the first marker check and before PID consumption.
                interleave = f'read() {{ : > "{paths[1]}"; /bin/rm -f "{paths[0]}"; return 1; }}; '
                control, separator, script = argv[2].partition("export PATH; ")
                words = shlex.split(script)
                words[2] = interleave + words[2]
                argv = (*argv[:2], control + separator + shlex.join(words))
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = CompletionDuringStop()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            result = await machine.run(Command(("printf", "candidate-finished"), timeout=1))
            assert result.reason is ExitReason.TIMED_OUT
            following = await machine.run(Command(("printf", "ready")))
            assert (following.exit_code, following.stdout) == (0, b"ready")
            assert not client.deleted
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("primary_failure", [False, True])
@pytest.mark.parametrize("cleanup_failure", ["timeout", "exit"])
@pytest.mark.parametrize("deletion_failure", [False, True])
@pytest.mark.requires_host_setsid
def test_daytona_file_cleanup_failure_closes_machine_and_preserves_primary_error(
    primary_failure, cleanup_failure, deletion_failure
):
    provider_error = OSError("Provider command failed")
    deletion_error = DaytonaError("Failed to remove sandbox", status_code=500)

    class FailedCleanup(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "candidate-fail" in command_script(argv) and primary_failure:
                raise provider_error
            if "rm -rf /tmp/.shellbox-" in argv[2]:
                if cleanup_failure == "timeout":
                    raise TimeoutError("Provider cleanup timed out")
                return exited(1, "Cannot remove command files")
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = FailedDeletion(deletion_error) if deletion_failure else LocalDaytona()
        client.sandbox.process = FailedCleanup()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir="")
        )
        error_type = OSError if primary_failure else RuntimeError
        with pytest.raises(error_type) as caught:
            await machine.run(Command(("printf", "candidate-fail")))
        if primary_failure:
            assert caught.value is provider_error
            assert "Daytona command cleanup failed" in caught.value.__notes__[0]
        if deletion_failure:
            assert any(str(deletion_error) in note for note in caught.value.__notes__)
        assert client.closed and client.deleted == (not deletion_failure)

    asyncio.run(scenario())


@pytest.mark.docker
@pytest.mark.parametrize("backend", ["docker", "daytona"])
@pytest.mark.parametrize("tampering", ["delete", "forge", "path"])
def test_nonroot_tampering_cannot_authorize_root_commands(backend, tampering):
    # The Daytona SDK boundary uses Docker only to test real guest UIDs and processes.
    # This is not a live Daytona service test.
    async def scenario():
        guest = await DockerMachineFactory().create(MachineSpec(DockerImage("ubuntu:24.04"), workdir="/tmp"))

        class GuestProcess:
            async def run(self, *, argv, cwd=None, env=None, execution_timeout=None):
                args = ["exec", "--user", "0"]
                if cwd:
                    args.extend(("-w", cwd))
                for key, value in (env or {}).items():
                    args.extend(("-e", f"{key}={value}"))
                result = await docker(*args, guest.name, *argv, timeout=execution_timeout)
                return exited(result.exit_code, result.stdout.decode())

        class GuestFiles:
            async def write_bytes(self, path, data):
                result = await guest.run(
                    Command(("/bin/sh", "-c", '/bin/cat > "$1"', "upload", path), stdin=data, user="root")
                )
                assert result.exit_code == 0

            async def read_bytes(self, path):
                result = await docker("exec", "--user", "0", guest.name, "cat", path)
                assert result.exit_code == 0
                return result.stdout

        resources = AsyncExitStack()
        resources.push_async_callback(guest.close)
        machine = (
            guest
            if backend == "docker"
            else DaytonaMachine(SimpleNamespace(process=GuestProcess(), fs=GuestFiles()), guest.spec, resources)
        )
        try:
            await guest.run(Command(("sh", "-c", "setsid sleep 7200 >/dev/null 2>&1 & echo $! > /tmp/victim.pid")))
            if tampering == "path":
                prepared = await guest.run(Command(("sh", "-c", "mkdir -m 777 /tmp/candidate-bin")))
                assert prepared.exit_code == 0
                planted = await guest.run(
                    Command(
                        (
                            "sh",
                            "-c",
                            "for name in touch rm; do "
                            'printf \'#!/bin/sh\\nid -u > /tmp/control-user\\n/bin/%s "$@"\\n\' "$name" '
                            '> "/tmp/candidate-bin/$name"; chmod 755 "/tmp/candidate-bin/$name"; done',
                        ),
                        user="nobody",
                    )
                )
                assert planted.exit_code == 0
            mutation = (
                'rm -f "$file"'
                if tampering == "delete"
                else 'printf "%s\\n" "$victim" > "$file"' if tampering == "forge" else ":"
            )
            environment = {"PATH": "/tmp/candidate-bin:/usr/bin:/bin"} if tampering == "path" else {}
            # The deadline includes provider probes before the guest command starts.
            candidate = Command(
                (
                    "sh",
                    "-c",
                    "victim=$(cat /tmp/victim.pid); "
                    "for file in /tmp/.shellbox-*/pid; do "
                    f'[ ! -w "$file" ] || {mutation}; done; '
                    "sleep 3600 & echo $! > /tmp/candidate-child.pid; wait",
                ),
                user="nobody",
                timeout=5,
                env=environment,
            )
            result = await machine.run(candidate)
            assert result.reason is ExitReason.TIMED_OUT
            result = await guest.run(Command(("sh", "-c", 'kill -0 "$(cat /tmp/victim.pid)"')))
            assert result.exit_code == 0
            followup = await machine.run(Command(("printf", "ready"), user="nobody", env=environment))
            assert (followup.exit_code, followup.stdout) == (0, b"ready")
            if tampering == "path":
                control_user = await guest.run(Command(("test", "-f", "/tmp/control-user")))
                assert control_user.exit_code != 0
            child = await guest.run(
                Command(
                    (
                        "sh",
                        "-c",
                        'path="/proc/$(cat /tmp/candidate-child.pid)/stat"; '
                        'if [ -f "$path" ]; then read pid comm state rest < "$path"; test "$state" = Z; fi',
                    )
                )
            )
            assert child.exit_code == 0
        finally:
            await machine.close()
        disposed = await docker("inspect", guest.name)
        assert disposed.exit_code != 0

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_rejects_nonroot_execution_without_session_preserving_su(tmp_path):
    class MissingSessionOption(LocalProcess):
        async def run(self, *, argv, **kwargs):
            if "su --help" in argv[2]:
                return exited(0, "su -c command")
            return await super().run(argv=argv, **kwargs)

    async def scenario():
        client = LocalDaytona()
        client.sandbox.process = MissingSessionOption()
        machine = await DaytonaMachineFactory(lambda: client).create(
            MachineSpec(RegistryImage("ubuntu:24.04"), workdir=str(tmp_path))
        )
        try:
            with pytest.raises(UnsupportedMachineSpec, match="--session-command"):
                await machine.run(Command(("sh", "-c", "touch candidate-ran"), user="nobody"))
            assert not (tmp_path / "candidate-ran").exists()
            following = await machine.run(Command(("printf", "still-ready")))
            assert (following.exit_code, following.stdout) == (0, b"still-ready")
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("mode, exit_code", [(0o755, 0), (0o640, 126)])
@pytest.mark.requires_host_setsid
def test_daytona_upload_preserves_script_permissions(tmp_path: Path, mode: int, exit_code: int) -> None:
    async def scenario() -> None:
        source = tmp_path / "grader.sh"
        source.write_text("#!/bin/sh\nprintf '1.0\\n'\n")
        source.chmod(mode)
        target = tmp_path / "remote" / "private grader.sh"
        machine = await DaytonaMachineFactory(LocalDaytona).create(
            MachineSpec(source=RegistryImage("ubuntu:24.04"), workdir=str(target.parent))
        )
        try:
            await machine.upload(source, str(target))
            assert stat.S_IMODE(target.stat().st_mode) == mode
            result = await machine.run(Command((str(target),)))
            assert result.exit_code == exit_code
            assert result.stdout == (b"1.0\n" if exit_code == 0 else b"")
        finally:
            await machine.close()

    asyncio.run(scenario())


@pytest.mark.requires_host_setsid
def test_daytona_factory_owns_clients_across_worker_event_loops(tmp_path: Path) -> None:
    clients = []

    class LoopClient(LocalDaytona):
        def __init__(self):
            super().__init__()
            self.loop = asyncio.get_running_loop()
            clients.append(self)

        async def create(self, image, **kwargs):
            assert asyncio.get_running_loop() is self.loop
            if kwargs["env"].get("FAIL_CREATE"):
                raise ConnectionError("Sandbox creation failed")
            return await super().create(image, **kwargs)

    factory = DaytonaMachineFactory(LoopClient)

    async def scenario(index):
        spec = MachineSpec(
            source=RegistryImage("ubuntu:24.04"),
            workdir=str(tmp_path / str(index)),
            env={"FAIL_CREATE": "1"} if index == 2 else {},
        )
        if index == 2:
            with pytest.raises(ConnectionError):
                await factory.create(spec)
            return
        machine = await factory.create(spec)
        try:
            result = await machine.run(Command(("printf", str(index))))
            assert result.stdout == str(index).encode()
        finally:
            await machine.close()

    with ThreadPoolExecutor(max_workers=3) as executor:
        pending = [executor.submit(asyncio.run, scenario(index)) for index in range(3)]
        for result in pending:
            result.result()
    assert all(client.closed for client in clients)
    assert sum(client.deleted for client in clients) == 2


@pytest.mark.requires_host_setsid
def test_daytona_reuses_an_image_until_its_build_inputs_change(tmp_path: Path, monkeypatch) -> None:
    images = LocalImages()
    clients = []

    def client_factory():
        client = LocalDaytona()
        client.images = images
        clients.append(client)
        return client

    context = tmp_path / "context"
    context.mkdir()
    dockerfile = context / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:24.04\nCOPY input /input\n")
    source = context / "input"
    source.write_text("first")
    monkeypatch.chdir(context)
    spec = MachineSpec(
        source=DockerfileSource(Path("."), dockerfile), workdir=str(tmp_path / "work"), cpus=1, memory_mb=1024
    )
    factory = DaytonaMachineFactory(client_factory)

    async def scenario():
        for current in (spec, spec, replace(spec, cpus=4, memory_mb=2048)):
            machine = await factory.create(current)
            await machine.close()
        source.write_text("second")
        machine = await factory.create(spec)
        await machine.close()

    asyncio.run(scenario())
    names = [client.params.image for client in clients]
    # The sandbox shape is not part of the image identity; the build context is.
    assert names[0] == names[1] == names[2] != names[3]
    assert images.submitted == [names[0], names[3]]
    assert images.images[names[0]].source == (context.resolve(), "Dockerfile")
    assert all(client.deleted and client.closed for client in clients)


def test_image_prune_lists_then_deletes_only_unneeded_shellbox_images():
    images = LocalImages()
    old, recent = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC)
    kept = images.add(image_name(RegistryImage("kept:tag")), "kept:tag", created_at=old)
    stale = images.add(image_name(RegistryImage("stale:tag")), "stale:tag", created_at=old)
    images.add(image_name(RegistryImage("recent:tag")), "recent:tag", created_at=recent)
    busy = images.add(image_name(RegistryImage("busy:tag")), "busy:tag", created_at=old)
    # An image Shellbox did not name is never selected, whatever its age.
    images.add("team-image", "team:tag", created_at=old)
    images.in_use.add(busy.id)
    client = SimpleNamespace(images=images)

    def prune(delete):
        return asyncio.run(
            prune_images(
                client, keep=frozenset({kept.name}), created_before=datetime(2026, 6, 1, tzinfo=UTC), delete=delete
            )
        )

    listed = prune(delete=False)
    assert {(image.name, image.action) for image in listed} == {
        (stale.name, PruneAction.WOULD_DELETE),
        (busy.name, PruneAction.WOULD_DELETE),
    }
    assert len(images.images) == 5
    pruned = prune(delete=True)
    assert {(image.name, image.action) for image in pruned} == {
        (stale.name, PruneAction.DELETED),
        (busy.name, PruneAction.IN_USE),
    }
    assert stale.name not in images.images and {kept.name, busy.name, "team-image"} <= images.images.keys()


@dataclass(frozen=True)
class LinuxRequirements:
    command_semantics: CommandSemantics = CommandSemantics.LINUX_PROCESS
    capabilities: tuple[str, ...] = ()
    docker_image: str | None = None


@pytest.mark.parametrize("location", ["outside", "nested"])
def test_image_preflight_requires_the_dockerfile_at_the_context_root(tmp_path, location):
    context = tmp_path / "context"
    (context / "docker").mkdir(parents=True)
    dockerfile = (tmp_path if location == "outside" else context / "docker") / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:24.04\n")
    client = LocalDaytona()
    factory = DaytonaMachineFactory(lambda: client)
    spec = MachineSpec(source=DockerfileSource(context, dockerfile))
    with pytest.raises(UnsupportedMachineSpec, match="build context root"):
        factory.check_environment(LinuxRequirements(), spec)
    with pytest.raises(UnsupportedMachineSpec, match="build context root"):
        asyncio.run(factory.prepare(spec))
    assert not client.images.submitted
    assert client.params is None


@pytest.mark.parametrize("startup_timeout", [None, 0.01])
def test_daytona_pending_image_times_out_and_closes_client(startup_timeout) -> None:
    class PendingImages(LocalImages):
        async def pull(self, reference, *, name, wait):
            self.submitted.append(name)
            return self.add(name, reference, status="pending", pending_polls=math.inf)

    client = LocalDaytona()
    client.images = PendingImages()
    factory = DaytonaMachineFactory(lambda: client, create_timeout=0.01 if startup_timeout is None else 60)
    with pytest.raises(TimeoutError):
        asyncio.run(factory.create(MachineSpec(RegistryImage("ubuntu:24.04"), startup_timeout=startup_timeout)))
    assert client.images.submitted
    assert client.params is None
    assert client.closed


class FailingImages(LocalImages):
    """Finish the first ``failures`` submissions as failed images."""

    def __init__(self, failures):
        super().__init__()
        self.failures = failures

    def _finish(self, name, source):
        self.submitted.append(name)
        failed = len(self.submitted) <= self.failures
        return self.add(
            name, source, status="failed" if failed else "ready", error="manifest unknown" if failed else None
        )

    async def pull(self, reference, *, name, wait):
        return self._finish(name, reference)

    async def build_from_context(self, context, *, dockerfile, name):
        return self._finish(name, (Path(context), dockerfile))


def test_daytona_failed_image_closes_client_without_starting_sandbox() -> None:
    client = LocalDaytona()
    client.images = FailingImages(failures=2)
    with pytest.raises(DaytonaProvisioningError, match="manifest unknown"):
        asyncio.run(DaytonaMachineFactory(lambda: client).create(MachineSpec(source=RegistryImage("ubuntu:24.04"))))
    # A failed image is submitted once more before the failure stands.
    assert len(client.images.submitted) == 2
    assert client.params is None
    assert client.closed


@pytest.mark.parametrize("source_kind", ["registry", "dockerfile"])
def test_ensure_image_submits_a_failed_image_once_more(tmp_path, source_kind) -> None:
    # Concurrent identical builds can report a transient failure, which a second submission clears.
    (tmp_path / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    source = (
        RegistryImage("fixture:tag")
        if source_kind == "registry"
        else DockerfileSource(tmp_path, tmp_path / "Dockerfile")
    )
    request = ImageRequest(source)
    images = FailingImages(failures=1)

    assert asyncio.run(ensure_image(SimpleNamespace(images=images), request)) == request.name
    assert images.submitted == [request.name, request.name]


def test_ensure_image_awaits_an_image_another_rollout_submitted() -> None:
    request = ImageRequest(RegistryImage("fixture:tag"))
    images = LocalImages()
    images.add(request.name, "fixture:tag", status="pending", pending_polls=2)

    name = asyncio.run(ensure_image(SimpleNamespace(images=images), request))

    assert name == request.name
    assert images.images[name].status == "ready"
    assert not images.submitted


@pytest.mark.parametrize(
    "failure",
    [ConflictError("image name is taken", status_code=409), ProviderConnectionError("Connection reset by peer")],
    ids=["conflict", "lost"],
)
def test_ensure_image_reads_back_a_submission_that_did_not_answer(no_retry_wait, failure) -> None:
    class UnansweredImages(LocalImages):
        async def pull(self, reference, *, name, wait):
            # The name is registered (by another rollout, or by this request before its answer was lost).
            self.submitted.append(name)
            self.add(name, reference, status="pending", pending_polls=1)
            raise failure

    request = ImageRequest(RegistryImage("fixture:tag"))
    images = UnansweredImages()

    name = asyncio.run(ensure_image(SimpleNamespace(images=images), request))

    assert name == request.name
    assert images.images[name].status == "ready"
    assert images.submitted == [name]


@pytest.mark.parametrize(
    "lost",
    [ProviderConnectionError("Connection reset by peer"), ClientTimeoutError("operation timed out")],
    ids=["reset", "timeout"],
)
def test_ensure_image_reads_back_a_resubmission_that_did_not_answer(no_retry_wait, lost) -> None:
    class LostResubmission(LocalImages):
        async def pull(self, reference, *, name, wait):
            self.submitted.append(name)
            if len(self.submitted) == 1:
                self.add(name, reference, status="failed", error="the image build could not be prepared")
                return
            # The provider accepted the second submission, but its answer was lost.
            self.add(name, reference, status="pending", pending_polls=1)
            raise lost

    request = ImageRequest(RegistryImage("fixture:tag"))
    images = LostResubmission()

    assert asyncio.run(ensure_image(SimpleNamespace(images=images), request)) == request.name
    assert images.submitted == [request.name, request.name]


def test_image_identity_depends_on_the_source_only(tmp_path) -> None:
    registry = RegistryImage("fixture@sha256:" + "a" * 64)
    assert ImageRequest(registry).name == image_name(RegistryImage(registry.reference))
    assert image_name(registry) != image_name(RegistryImage("fixture@sha256:" + "b" * 64))

    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY input /input\n")
    (context / "input").write_text("first")
    built = DockerfileSource(context, context / "Dockerfile")
    first = image_name(built)
    (context / "input").write_text("second")
    assert image_name(built) != first


def test_image_request_interchange_round_trips_and_rejects_a_mismatched_name(tmp_path) -> None:
    registry = ImageRequest(RegistryImage("fixture:tag"))
    assert ImageRequest.from_dict(registry.as_dict()) == registry

    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    request = ImageRequest(DockerfileSource(context, context / "Dockerfile"))
    spec = request.as_dict()
    spec["name"] = "shellbox-" + "0" * 40
    with pytest.raises(ValueError, match="does not match the derived name"):
        ImageRequest.from_dict(spec)

    del spec["name"]
    assert ImageRequest.from_dict(spec) == request


def test_sandbox_resources_request_a_fixed_shape_in_cores() -> None:
    spec = MachineSpec(source=RegistryImage("fixture:tag"), cpus=3, memory_mb=1536, storage_mb=8192)
    # One core is two vCPUs; disk is only a request, since v2 raises it to a platform minimum.
    assert sandbox_resources(spec) == Resources(cpu=1.5, cpu_max=1.5, memory=1.5, memory_max=1.5, disk=8)
    # An unset request takes v0's default shape, so the guest never sees v2's burst ceiling.
    assert sandbox_resources(MachineSpec(source=RegistryImage("fixture:tag"))) == Resources(
        cpu=0.5, cpu_max=0.5, memory=1, memory_max=1
    )


def test_daytona_refuses_gpus_before_preparing_anything() -> None:
    client = LocalDaytona()
    with pytest.raises(UnsupportedMachineSpec, match="GPU"):
        asyncio.run(DaytonaMachineFactory(lambda: client).create(MachineSpec(RegistryImage("fixture:tag"), gpus=1)))
    assert not client.images.submitted
    assert client.params is None


def test_daytona_network_policy_parses_declarative_fields() -> None:
    def parameters(policy):
        return DaytonaNetworkPolicy.from_config(policy).parameters()

    assert parameters({"mode": "block_all"}) == {"block_all": True}
    assert parameters({"mode": "unrestricted"}) == {"block_all": False}
    # An allow list restricts egress only when everything else is blocked.
    assert parameters({"mode": "domain_allow_list", "value": "example.org, pypi.org"}) == {
        "block_all": True,
        "allow_domains": ["example.org", "pypi.org"],
    }
    assert parameters({"mode": "network_allow_list", "value": "10.0.0.0/8"}) == {
        "block_all": True,
        "allow_cidrs": ["10.0.0.0/8"],
    }
    with pytest.raises(ValueError, match="Unknown Daytona network policy fields"):
        DaytonaNetworkPolicy.from_config({"mode": "block_all", "extra": "value"})
    with pytest.raises(ValueError, match="requires an allow list"):
        parameters({"mode": "domain_allow_list"})


def test_connection_hides_the_api_key_from_every_representation() -> None:
    connection = DaytonaConnection(api_key="synthetic-secret-key")
    factory = connection.client_factory()

    assert "synthetic-secret-key" not in repr(connection)
    assert "synthetic-secret-key" not in repr(factory)
    with pytest.raises(ValueError, match="explicit api_key"):
        DaytonaConnection().client_factory()


def test_connection_module_imports_without_the_daytona_sdk() -> None:
    script = "\n".join(
        (
            "import sys",
            "sys.modules['daytona'] = None  # the optional extra is absent",
            "import shellbox.backends.daytona.connection",
            "connection = shellbox.backends.daytona.connection.DaytonaConnection(api_key='synthetic-secret-key')",
            "factory = connection.client_factory()",
            "assert callable(factory)",
            "print('resolved')",
        )
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "resolved" in result.stdout


class DeletionClient:
    """Find an accepted sandbox by name and delete it, failing the first deletions with ``errors``."""

    def __init__(self, *errors):
        self.errors = list(errors)
        self.deleted = []

    async def find_by_name(self, name):
        return (SimpleNamespace(name=name),)

    async def delete(self, sandbox, *, timeout):
        if self.errors:
            raise self.errors.pop(0)
        self.deleted.append(sandbox.name)


@pytest.mark.parametrize(
    "transient",
    [
        DaytonaError("502 Bad Gateway", status_code=502),
        ConflictError("Sandbox state change in progress", status_code=409),
    ],
)
def test_sandbox_deletion_retries_a_transient_error(no_retry_wait, transient) -> None:
    client = DeletionClient(transient)

    asyncio.run(_delete_sandbox(client, _SandboxClaim("fixture")))

    assert client.deleted == ["fixture"]


def test_sandbox_deletion_reports_a_recovered_sandbox_whose_deletion_hangs(monkeypatch) -> None:
    class HangingDeletion(DeletionClient):
        async def delete(self, sandbox, *, timeout):
            await asyncio.Future()

    monkeypatch.setattr("shellbox.backends.daytona.machine.INTERRUPT_TIMEOUT", 0.1)

    # The sandbox was found by name, so a deletion that never finishes is a failure, not "nothing appeared".
    with pytest.raises(TimeoutError):
        asyncio.run(_delete_sandbox(HangingDeletion(), _SandboxClaim("fixture")))


def test_sandbox_deletion_reports_a_conflict_that_outlasts_the_deadline(monkeypatch) -> None:
    conflict = ConflictError("Sandbox state change in progress", status_code=409)

    async def short_wait(delay):
        await asyncio.sleep(0.01)

    monkeypatch.setattr("shellbox.backends.daytona.machine.INTERRUPT_TIMEOUT", 0.1)
    monkeypatch.setattr("shellbox.backends.daytona.machine.sleep", short_wait)

    with pytest.raises(ConflictError) as caught:
        asyncio.run(_delete_sandbox(DeletionClient(*([conflict] * 1000)), _SandboxClaim("fixture")))

    assert caught.value is conflict
