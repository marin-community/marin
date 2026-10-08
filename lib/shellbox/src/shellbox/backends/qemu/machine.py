# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent QEMU microvm implementation of the machine contract."""

import asyncio
import base64
import gzip
import io
import json
import logging
import os
import re
import shlex
import tarfile
import tempfile
from contextlib import aclosing
from dataclasses import replace
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from shellbox.backends.qemu.image import QemuAssets, stage_qemu_image
from shellbox.file_transfer import file_chunks, write_download
from shellbox.image import DockerfileSource, PreparedImage, RegistryImage, process_image_cache
from shellbox.machine import (
    DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    Backend,
    Command,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    QemuBundle,
    Result,
    ShellStatus,
    ShellUpdate,
    UnsupportedMachineSpec,
)

REQUEST_CHUNK_BYTES = 2048
UPLOAD_CHUNK_BYTES = 16 * 1024
TRANSFER_LIMIT_BYTES = 128 * 1024 * 1024
BOOT_IDLE_TIMEOUT = 60
DEFAULT_COMMAND_TIMEOUT = 120
BOOT_OUTPUT_TAIL_BYTES = 4096
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
logger = logging.getLogger(__name__)


class Acceleration(StrEnum):
    AUTO = "auto"
    KVM = "kvm"
    TCG = "tcg"


class QemuShellSession:
    """One interactive Bash process in the guest, shared across commands."""

    interactive = True

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self._active_id: str | None = None
        self._next_id = 1
        self._read_lock = asyncio.Lock()

    async def execute(
        self,
        command: str,
        *,
        wait: float = DEFAULT_COMMAND_TIMEOUT,
        output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    ) -> ShellUpdate:
        if self._active_id is not None:
            raise RuntimeError("Previous Bash command is still running")
        command_id = str(self._next_id)
        self._next_id += 1
        self._active_id = command_id
        self.writer.write(f"EXEC|{command_id}|{command.encode().hex()}\n".encode())
        await self.writer.drain()
        return await self.read(wait=wait, output_limit_bytes=output_limit_bytes)

    async def read(
        self, *, wait: float = 0, output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES
    ) -> ShellUpdate:
        if self._active_id is None:
            raise RuntimeError("No Bash command is running")
        async with self._read_lock:
            output = bytearray()
            truncated = False
            deadline = asyncio.get_running_loop().time() + wait
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0 and output:
                    return ShellUpdate(bytes(output), ShellStatus.RUNNING, truncated=truncated)
                try:
                    line = await asyncio.wait_for(self.reader.readline(), timeout=max(remaining, 0.001))
                except TimeoutError:
                    return ShellUpdate(bytes(output), ShellStatus.RUNNING, truncated=truncated)
                if not line:
                    self._active_id = None
                    return ShellUpdate(bytes(output), ShellStatus.RESET, truncated=truncated)
                if line.startswith(b"OUT|"):
                    chunk = bytes.fromhex(line[4:].strip().decode())
                    available = max(0, output_limit_bytes - len(output))
                    output.extend(chunk[:available])
                    truncated |= len(chunk) > available
                elif line.startswith(b"DONE|"):
                    _, command_id, code = line.strip().decode().split("|", 2)
                    if command_id != self._active_id:
                        raise RuntimeError(f"Unexpected Bash completion id: {command_id}")
                    self._active_id = None
                    return ShellUpdate(bytes(output), ShellStatus.COMPLETED, int(code), truncated)
                elif line.startswith(b"RESET|"):
                    self._active_id = None
                    return ShellUpdate(bytes(output), ShellStatus.RESET, truncated=truncated)
                elif line.startswith(b"ERROR|"):
                    self._active_id = None
                    raise RuntimeError(line.decode().strip())

    async def write(self, data: bytes) -> None:
        if self._active_id is None:
            raise RuntimeError("No Bash command is running")
        self.writer.write(b"INPUT|" + data.hex().encode() + b"\n")
        await self.writer.drain()

    async def interrupt(self) -> None:
        if self._active_id is None:
            raise RuntimeError("No Bash command is running")
        self.writer.write(b"SIGNAL|INT\n")
        await self.writer.drain()

    async def close(self) -> None:
        self.writer.close()
        await self.writer.wait_closed()


async def query_acceleration(socket: Path) -> Acceleration:
    """Read the accelerator QEMU selected after its fallback sequence."""
    reader, writer = await asyncio.open_unix_connection(socket)
    try:
        greeting = json.loads(await reader.readline())
        if "QMP" not in greeting:
            raise RuntimeError(f"Invalid QEMU monitor greeting: {greeting}")
        for command in ("qmp_capabilities", "query-kvm"):
            writer.write(json.dumps({"execute": command}).encode() + b"\n")
            await writer.drain()
            while True:
                message = json.loads(await reader.readline())
                if "error" in message:
                    raise RuntimeError(f"QEMU monitor {command} failed: {message['error']}")
                if "return" in message:
                    break
        return Acceleration.KVM if message["return"]["enabled"] else Acceleration.TCG
    finally:
        writer.close()
        await writer.wait_closed()


class QemuMachine:
    """One microvm and its private writable disk."""

    def __init__(self, spec: MachineSpec, acceleration: Acceleration):
        assert isinstance(spec.source, QemuBundle)
        self.spec = spec
        self.bundle = spec.source.path.resolve()
        self.acceleration = acceleration
        self.active_acceleration: Acceleration | None = None
        self.process: asyncio.subprocess.Process | None = None
        self._runtime_dir: tempfile.TemporaryDirectory[str] | None = None
        self._pty_socket: Path | None = None
        self._shell: QemuShellSession | None = None
        self._lock = asyncio.Lock()
        metadata_path = self.bundle / "image.json"
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        self.image_env = dict(item.split("=", 1) for item in self.metadata.get("env", []))
        self.image_workdir = self.metadata.get("cwd", "/workspace")

    async def start(self) -> None:
        self._runtime_dir = tempfile.TemporaryDirectory(prefix="marin-shellbox-")
        try:
            await self._start_qemu()
        except BaseException:
            await self.close()
            raise

    async def _start_qemu(self) -> None:
        assert self._runtime_dir is not None
        runtime_path = Path(self._runtime_dir.name)
        runtime_env = os.environ.copy()
        lib = self.bundle / "lib"
        if lib.is_dir():
            runtime_env["LD_LIBRARY_PATH"] = str(lib)
            modules = lib / "qemu"
            if modules.is_dir():
                runtime_env["QEMU_MODULE_DIR"] = str(modules)
        disk_args: list[str] = []
        if (self.bundle / "rootfs.ext4").is_file():
            # QEMU keeps the base read-only and discards this guest's temporary
            # overlay at exit, avoiding a full rootfs copy for each trial.
            runtime_env["TMPDIR"] = str(runtime_path)
            disk_path = str(self.bundle / "rootfs.ext4").replace(",", ",,")
            disk_args = [
                "-drive",
                f"file={disk_path},if=none,format=raw,id=rootdisk,snapshot=on",
                "-device",
                "virtio-blk-device,drive=rootdisk",
            ]
        if self.acceleration is Acceleration.AUTO:
            accelerators = ["kvm", "tcg"] if os.access("/dev/kvm", os.R_OK | os.W_OK) else ["tcg"]
        else:
            accelerators = [self.acceleration.value]
        accelerator_args = [value for name in accelerators for value in ("-accel", name)]
        qmp_socket = runtime_path / "qmp.sock"
        pty_socket = runtime_path / "pty.sock"
        self._pty_socket = pty_socket
        clock = asyncio.get_running_loop().time
        boot_started = clock()
        # Hardware-reduced ACPI disables Linux's PIT timer fallback. Retain it
        # for failed TSC calibration under TCG, keeping the existing SeaBIOS.
        self.process = await asyncio.create_subprocess_exec(
            str(self.bundle / "qemu-system-x86_64"),
            "-L",
            str(self.bundle / "firmware"),
            "-machine",
            "microvm,acpi=off",
            "-bios",
            str(self.bundle / "firmware" / "bios-microvm.bin"),
            *accelerator_args,
            "-cpu",
            "max",
            "-m",
            f"{self.spec.memory_mb or 512}M",
            "-smp",
            str(self.spec.cpus or 1),
            "-nodefaults",
            "-no-reboot",
            "-display",
            "none",
            "-serial",
            "stdio",
            "-monitor",
            "none",
            "-qmp",
            f"unix:{qmp_socket},server=on,wait=off",
            "-chardev",
            f"socket,id=harbor-shell,path={pty_socket},server=on,wait=off",
            "-device",
            "virtio-serial-device",
            "-device",
            "virtserialport,chardev=harbor-shell,name=harbor.shell",
            "-kernel",
            str(self.bundle / "vmlinuz"),
            "-initrd",
            str(self.bundle / "initramfs.cpio.gz"),
            "-append",
            "console=ttyS0 quiet rdinit=/init",
            *disk_args,
            "-nic",
            "none",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=runtime_env,
        )
        assert self.process.stdout is not None
        output_tail = bytearray()
        last_output = boot_started
        while True:
            try:
                line = await asyncio.wait_for(self.process.stdout.readline(), timeout=BOOT_IDLE_TIMEOUT)
            except TimeoutError as error:
                now = clock()
                detail = (
                    f"QEMU guest did not become ready within {BOOT_IDLE_TIMEOUT} seconds of its last output; "
                    f"boot_elapsed={now - boot_started:.3f}s, idle_elapsed={now - last_output:.3f}s, "
                    f"accelerators={accelerators}, returncode={self.process.returncode}, "
                    f"guest_output_tail={bytes(output_tail)!r}"
                )
                logger.warning("%s", detail)
                raise TimeoutError(detail) from error
            if line:
                last_output = clock()
                output_tail.extend(line)
                del output_tail[:-BOOT_OUTPUT_TAIL_BYTES]
            if b"READY" in line:
                break
            if not line:
                assert self.process.stderr is not None
                raise RuntimeError(f"QEMU exited before guest startup: {await self.process.stderr.read()!r}")
        self.active_acceleration = await query_acceleration(qmp_socket)

    async def open_shell(self) -> QemuShellSession:
        if self._shell is not None:
            return self._shell
        check = await self.run(Command(("/bin/sh", "-c", "test -x /bin/bash && test -x /harbor/pty-agent")))
        if check.exit_code != 0:
            raise UnsupportedMachineSpec("Persistent Bash requires /bin/bash and a PTY-enabled guest bundle")
        if self._pty_socket is None:
            raise RuntimeError("QEMU machine is not running")
        reader, writer = await asyncio.open_unix_connection(self._pty_socket)
        try:
            launch = await self.run(
                Command(("/bin/sh", "-c", "/harbor/pty-agent > /tmp/harbor-pty.log 2>&1 < /dev/null &"))
            )
            if launch.exit_code != 0:
                raise RuntimeError(f"Guest Bash service could not start: {launch.stderr!r}")
            writer.write(b"START\n")
            await writer.drain()
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                if line == b"READY\n":
                    break
                if not line:
                    raise RuntimeError("Guest Bash service exited before readiness")
        except BaseException:
            writer.close()
            await writer.wait_closed()
            raise
        self._shell = QemuShellSession(reader, writer)
        return self._shell

    async def run(self, command: Command) -> Result:
        if not command.argv:
            raise ValueError("Command argv is empty")
        if command.user not in (None, "0", "root"):
            raise UnsupportedMachineSpec("QEMU does not provide separate execution users")
        if command.stdin:
            raise ValueError("QEMU serial protocol does not support command stdin")
        workdir = command.cwd or self.spec.workdir or self.image_workdir
        exports = self.image_env | self.spec.env | command.env
        for key in exports:
            if ENV_NAME.fullmatch(key) is None:
                raise ValueError(f"Invalid environment variable name: {key!r}")
        script = f"cd {shlex.quote(workdir)} && "
        if exports:
            script += "export " + " ".join(f"{key}={shlex.quote(value)}" for key, value in exports.items()) + " && "
        script += shlex.join(command.argv)
        return await self._run_script(script, command.timeout or DEFAULT_COMMAND_TIMEOUT, command.output_limit_bytes)

    async def _run_script(self, script: str, timeout: float, output_limit_bytes: int) -> Result:
        process = self.process
        if process is None or process.stdin is None or process.stdout is None or process.returncode is not None:
            raise RuntimeError("QEMU machine is not running")
        async with self._lock:
            encoded = base64.b64encode(script.encode())
            process.stdin.write(b"BEGIN\n")
            for offset in range(0, len(encoded), REQUEST_CHUNK_BYTES):
                process.stdin.write(b"DATA|" + encoded[offset : offset + REQUEST_CHUNK_BYTES] + b"\n")
            process.stdin.write(b"END\n")
            await process.stdin.drain()
            try:
                exit_code: int | None = None
                stdout = bytearray()
                stderr = bytearray()
                stdout_truncated = False
                stderr_truncated = False
                deadline = asyncio.get_running_loop().time() + timeout
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=max(remaining, 0.001))
                    if not line:
                        raise RuntimeError("QEMU guest stopped during command")
                    if line.startswith(b"RESULT|"):
                        exit_code = int(line.strip().split(b"|", 1)[1])
                    elif line.startswith((b"OUT|", b"ERR|")):
                        chunk = base64.b64decode(line[4:].strip())
                        output = stdout if line.startswith(b"OUT|") else stderr
                        available = max(0, output_limit_bytes - len(output))
                        output.extend(chunk[:available])
                        if line.startswith(b"OUT|"):
                            stdout_truncated |= len(chunk) > available
                        else:
                            stderr_truncated |= len(chunk) > available
                    elif line.strip() == b"ENDRESULT":
                        if exit_code is None:
                            raise RuntimeError("QEMU guest ended result without exit code")
                        return Result(
                            exit_code,
                            bytes(stdout),
                            bytes(stderr),
                            stdout_truncated,
                            stderr_truncated,
                            ExitReason.EXITED,
                        )
            except TimeoutError:
                await self.close()
                return Result(None, b"", b"", False, False, ExitReason.TIMED_OUT)
            except asyncio.CancelledError:
                await self.close()
                raise

    async def upload(self, source: Path, target: str) -> None:
        staged = shlex.quote(f"/tmp/harbor-upload-{uuid4().hex}.gz")
        if source.is_dir():
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                archive.add(source, arcname=".")
            compressed = stream.getvalue()
            unpack = (
                f"/harbor/busybox mkdir -p {shlex.quote(target)} && "
                f"/harbor/busybox gzip -d < {staged} | "
                f"/harbor/busybox tar xf - -C {shlex.quote(target)}"
            )
        else:
            compressed = gzip.compress(source.read_bytes(), compresslevel=1, mtime=0)
            unpack = (
                f"/harbor/busybox mkdir -p {shlex.quote(str(Path(target).parent))} && "
                f"/harbor/busybox gzip -d < {staged} > {shlex.quote(target)}"
            )
        workdir = self.spec.workdir or self.image_workdir
        # A completion acknowledgement between bounded scripts prevents large
        # serial bursts from corrupting fixture uploads in the guest console.
        # The deadline includes stdin drain and every chunk, not just replies.
        try:
            async with asyncio.timeout(DEFAULT_COMMAND_TIMEOUT):
                for offset in range(0, len(compressed), UPLOAD_CHUNK_BYTES):
                    encoded = base64.b64encode(compressed[offset : offset + UPLOAD_CHUNK_BYTES]).decode()
                    redirect = ">" if offset == 0 else ">>"
                    script = f"printf '%s' '{encoded}' | /harbor/busybox base64 -d {redirect} {staged}"
                    result = await self._run_script(script, DEFAULT_COMMAND_TIMEOUT, TRANSFER_LIMIT_BYTES)
                    if result.exit_code != 0:
                        raise RuntimeError(f"QEMU upload {result.reason}: {result.stderr.decode(errors='replace')}")
                result = await self._run_script(
                    f"cd {shlex.quote(workdir)} && {unpack} && /harbor/busybox rm -f {staged}",
                    DEFAULT_COMMAND_TIMEOUT,
                    TRANSFER_LIMIT_BYTES,
                )
                if result.exit_code != 0:
                    raise RuntimeError(f"QEMU upload {result.reason}: {result.stderr.decode(errors='replace')}")
        except TimeoutError as error:
            await self.close()
            raise RuntimeError("QEMU upload timed_out") from error

    async def download(self, source: str, target: Path, *, max_bytes: int | None = None) -> None:
        if max_bytes is not None:
            async with aclosing(file_chunks(self, source, max_bytes, busybox="/harbor/busybox")) as chunks:
                await write_download(chunks, target, max_bytes)
            return
        check = await self.run(Command(("/bin/sh", "-c", f"test -d {shlex.quote(source)}")))
        if check.exit_code == 0:
            script = (
                f"/harbor/busybox tar cf - -C {shlex.quote(source)} . | "
                "/harbor/busybox gzip -c | /harbor/busybox base64 -w 0"
            )
            result = await self.run(Command(("/bin/sh", "-c", script), output_limit_bytes=TRANSFER_LIMIT_BYTES))
            if result.exit_code != 0:
                raise RuntimeError(f"QEMU download {result.reason}: {result.stderr.decode(errors='replace')}")
            if result.stdout_truncated:
                raise RuntimeError("QEMU directory transfer exceeded limit")
            target.mkdir(parents=True, exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(result.stdout)), mode="r:gz") as archive:
                archive.extractall(target, filter="data")
            return
        script = f"/harbor/busybox base64 -w 0 < {shlex.quote(source)}"
        result = await self.run(Command(("/bin/sh", "-c", script), output_limit_bytes=TRANSFER_LIMIT_BYTES))
        if result.exit_code != 0:
            raise RuntimeError(f"QEMU download {result.reason}: {result.stderr.decode(errors='replace')}")
        if result.stdout_truncated:
            raise RuntimeError("QEMU file transfer exceeded limit")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(result.stdout))

    async def close(self) -> None:
        if self._shell is not None:
            await self._shell.close()
            self._shell = None
        if self.process is not None:
            if self.process.returncode is None:
                self.process.terminate()
            await self.process.wait()
        self.process = None
        self.active_acceleration = None
        if self._runtime_dir is not None:
            self._runtime_dir.cleanup()
            self._runtime_dir = None


class QemuMachineFactory:
    """Create a private microvm from a bundle or OCI image source."""

    backend: Backend = Backend.QEMU

    def __init__(
        self,
        acceleration: Acceleration = Acceleration.AUTO,
        *,
        assets: QemuAssets | None = None,
        bundle_cache: Path | None = None,
        image_cache: Path | None = None,
        skopeo: Path | None = None,
        authfile: Path | None = None,
        policy: Path | None = None,
    ):
        self.acceleration = Acceleration(acceleration)
        self.assets = assets
        self.bundle_cache = bundle_cache
        self.image_cache = image_cache
        self.skopeo = skopeo
        self.authfile = authfile
        self.policy = policy

    async def create(self, spec: MachineSpec) -> QemuMachine:
        if spec.storage_mb is not None or spec.gpus:
            raise UnsupportedMachineSpec("QEMU does not provide storage resizing or GPU allocation")
        if spec.network is not NetworkPolicy.DENY:
            raise UnsupportedMachineSpec("QEMU guest networking is unsupported")
        if spec.memory_mb is not None and spec.memory_mb <= 0:
            raise ValueError("memory_mb must be positive")
        if isinstance(spec.source, (RegistryImage, DockerfileSource)):
            if self.image_cache is None or self.skopeo is None:
                raise UnsupportedMachineSpec("Registry images and Dockerfiles require an image cache and Skopeo")
            cache = process_image_cache(self.image_cache, self.skopeo, self.authfile, self.policy)
            image = await asyncio.to_thread(cache.prepare, spec.source)
            spec = replace(spec, source=image)
        if isinstance(spec.source, PreparedImage):
            if self.assets is None or self.bundle_cache is None:
                raise UnsupportedMachineSpec("Prepared OCI images require QEMU assets and a bundle cache")
            bundle = await asyncio.to_thread(stage_qemu_image, spec.source, self.assets, self.bundle_cache)
            spec = replace(spec, source=QemuBundle(bundle))
        if not isinstance(spec.source, QemuBundle):
            raise UnsupportedMachineSpec("QEMU requires a QemuBundle source")
        bundle = spec.source.path
        for name in ("qemu-system-x86_64", "vmlinuz", "initramfs.cpio.gz"):
            if not (bundle / name).is_file():
                raise FileNotFoundError(bundle / name)
        if (bundle / "image.json").exists() != (bundle / "rootfs.ext4").exists():
            raise ValueError("OCI guest bundle requires both image.json and rootfs.ext4")
        machine = QemuMachine(spec, self.acceleration)
        await machine.start()
        return machine
