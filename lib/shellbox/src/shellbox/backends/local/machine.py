# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct persistent local sandboxes and coordinate task lifetimes."""

import asyncio
import logging
import os
import re
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

from shellbox.backends.local import nix, sandbox
from shellbox.backends.local.cgroup import check_cgroup_parent, delegated_cgroup_parent
from shellbox.backends.local.disk import LocalDisk, verified_local_disk_pool
from shellbox.backends.local.guest_setup import host_environment, host_mounts, start_host_machine, start_image_machine
from shellbox.backends.local.images import PreparedLocalImage, cached_local_image, prepare_local_image
from shellbox.backends.local.python_environment import DEFAULT_PYTHON_VERSION, resolve_python_environment
from shellbox.backends.local.runtime import LocalMachine, _log_task_failure
from shellbox.backends.local.sandbox_probe import _working_bwrap
from shellbox.image import PreparedImage, RegistryImage, process_image_cache
from shellbox.machine import (
    LINUX_PROCESS_CAPABILITIES,
    Backend,
    Command,
    CommandSemantics,
    ExecutionRequirements,
    HostImage,
    MachineSpec,
    NetworkPolicy,
    PackageSpec,
    Result,
    UnsupportedMachineSpec,
    check_execution_requirements,
)
from shellbox.process_session import finish_owned, own_cleanup
from shellbox.task import Task, TaskSpec, single_container_task

logger = logging.getLogger(__name__)


def _check_spec(
    spec: MachineSpec, disk_root: Path | None = None
) -> HostImage | PackageSpec | PreparedImage | RegistryImage:
    if not isinstance(spec.source, (HostImage, PackageSpec, PreparedImage, RegistryImage)):
        raise UnsupportedMachineSpec(
            "The local backend requires HostImage, PackageSpec, or a prepared/pinned OCI image, "
            f"not {type(spec.source).__name__}"
        )
    if spec.storage_mb is not None and disk_root is None:
        raise UnsupportedMachineSpec("LOCAL storage_mb requires an explicitly configured, host-verified disk_root")
    if spec.gpus:
        raise UnsupportedMachineSpec("The local backend does not provide GPU allocations")
    if spec.memory_mb is not None and spec.memory_mb <= 0:
        raise ValueError("Local memory_mb must be positive")
    if (
        isinstance(spec.source, RegistryImage)
        and re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", spec.source.reference) is None
    ):
        raise UnsupportedMachineSpec("Local registry images must be pinned by sha256 digest")
    return spec.source


def check_local_environment(
    requirements: ExecutionRequirements,
    spec: MachineSpec,
    *,
    cgroup_parent: Path | None = None,
    disk_root: Path | None = None,
) -> None:
    """Check local execution support without acquiring a bubblewrap sandbox."""
    source = _check_spec(spec, disk_root)
    origin = source.source if isinstance(source, PreparedImage) else source
    reference = origin.reference if isinstance(origin, RegistryImage) else None
    capabilities = LINUX_PROCESS_CAPABILITIES
    if cgroup_parent is not None:
        check_cgroup_parent(cgroup_parent)
        capabilities = capabilities | {"retained_command_cancellation", "process_scope"}
    check_execution_requirements(
        requirements,
        spec,
        semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=capabilities,
        image_reference=reference,
    )


class LocalMachineFactory:
    """Run trusted commands on this host in bubblewrap sandboxes, any number of machines at a time.

    A machine's commands see the host's programs and private runtime configuration as
    ``system_directories`` says: read-only by default, or writable per machine. They see each directory
    in ``read_only`` at its own path, read-only, and each ``Mount`` of a ``HostImage`` at its guest path,
    read-only; a guest path inside a system directory is refused. Every other path, including ``/tmp``
    and ``HOME``, lies in a root directory of the machine's own that is removed by
    ``close``; no other host file is visible. The caller keeps each ``read_only`` directory self-contained:
    a symlink that leads out of the mounted directories does not resolve in the sandbox.
    ``build_python_environment`` builds such a directory for a Python interpreter and its packages.

    Commands never inherit the host's environment: they get ``bin_dirs`` ahead of a standard ``PATH``,
    ``HOME``, ``LANG``, ``PYTHONHASHSEED`` when the factory has a ``hash_seed``, and the spec's and
    command's variables. ``bin_dirs`` only sets ``PATH``; a directory outside the mounted ones is not
    visible to commands. The factory discovers a writable cgroup delegation inside the worker's own
    unit or container scope for command containment and attempts aggregate CPU and memory limits
    there. When the scope directly contains the worker's processes, discovery moves them into a child
    leaf group so the scope can enable controllers for children; sandbox groups become siblings of
    that leaf and never reach a parent above the container. Unavailable limits produce warnings.
    ``cgroup_parent`` overrides discovery. ``disk_root`` enables host-probed ext4
    loop filesystems for declared storage_mb, including all writable layers.

    ``bwrap`` names the executable to use. By default the factory takes the first of the bundled bwrap
    and any bwrap on ``PATH`` that can build a sandbox on this host, and raises ``SandboxUnavailable``
    when none can.

    Prepared images use private rootless umoci bundles, without host program mounts. Registry inputs
    must be digest pinned. Only numeric root image users are supported. ``ttl`` bounds the lifetime of
    a task. With delegated cgroups, command interruption preserves the task;
    without delegation it closes every container in the task.
    """

    backend: Backend = Backend.LOCAL

    def __init__(
        self,
        *,
        read_only: tuple[Path, ...] = (),
        bin_dirs: tuple[Path, ...] = (),
        bwrap: Path | None = None,
        hash_seed: str | None = None,
        package_cache: Path | None = None,
        python_version: str = DEFAULT_PYTHON_VERSION,
        system_directories: sandbox.SystemDirectories = sandbox.SystemDirectories.READ_ONLY,
        ttl: float | None = None,
        image_cache: Path | None = None,
        authfile: Path | None = None,
        policy: Path | None = None,
        umoci: Path = Path("umoci"),
        cgroup_parent: Path | None = None,
        disk_root: Path | None = None,
    ):
        if ttl is not None and ttl <= 0:
            raise ValueError("Local machine TTL must be positive")
        self.ttl = ttl
        self.disk_root = disk_root
        self.cgroup_parent = delegated_cgroup_parent() if cgroup_parent is None else cgroup_parent.resolve()
        if self.cgroup_parent is not None:
            check_cgroup_parent(self.cgroup_parent)
        self.image_cache = process_image_cache(image_cache or Path.home() / ".cache/shellbox/images", authfile, policy)
        self.umoci = umoci
        self.read_only = tuple(directory.absolute() for directory in read_only)
        missing = [str(directory) for directory in self.read_only if not directory.is_dir()]
        if missing:
            raise ValueError(f"Read-only directories do not exist: {', '.join(missing)}")
        self.bin_dirs = tuple(directory.absolute() for directory in bin_dirs)
        self.hash_seed = hash_seed
        self.package_cache = package_cache if package_cache is not None else Path.home() / ".cache/shellbox/python"
        self.python_version = python_version
        self.system_directories = system_directories
        candidates = sandbox._bwrap_candidates(bwrap)
        self._system_mounts, self._system_bin_dirs = nix.system_programs()
        environment = {
            **sandbox.CONTROL_ENVIRONMENT,
            "PATH": os.pathsep.join((sandbox.DEFAULT_PATH, *map(str, self._system_bin_dirs))),
        }
        self.bwrap = _working_bwrap(candidates, system_directories, self._system_mounts, environment)
        logger.info("Local backend sandboxes commands with %s, system directories %s", self.bwrap, system_directories)

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        check_local_environment(requirements, spec, cgroup_parent=self.cgroup_parent, disk_root=self.disk_root)

    async def create(self, spec: MachineSpec) -> LocalMachine:
        task = await self.create_task(single_container_task(spec))
        return cast(LocalMachine, task.primary)

    async def create_task(self, spec: TaskSpec) -> Task:
        """Start named containers with private filesystems and one network."""
        for container in spec.containers:
            _check_spec(container.machine, self.disk_root)
        if len(spec.containers) > 1 and spec.network is not NetworkPolicy.DENY:
            raise UnsupportedMachineSpec("Local multi-container tasks require denied external networking")
        if any(re.fullmatch(r"[a-z][a-z0-9_-]{0,62}", container.name) is None for container in spec.containers):
            raise UnsupportedMachineSpec("Local container names must match [a-z][a-z0-9_-]{0,62}")
        primary = spec.primary_container
        if primary.entrypoint is not None or primary.arguments is not None:
            raise UnsupportedMachineSpec("The local task's primary container does not support startup overrides")
        members: list[LocalMachine] = []
        monitors: list[asyncio.Task[Result]] = []
        expiry: asyncio.Task[None] | None = None
        closing_tasks: set[asyncio.Task[None]] = set()

        async def close() -> None:
            if expiry is not None:
                expiry.cancel()
            await asyncio.gather(*(member._close_local() for member in members))
            for monitor in monitors:
                if not monitor.done():
                    monitor.cancel()
            await asyncio.gather(*monitors, return_exceptions=True)

        try:
            aliases = "127.0.0.1 localhost " + " ".join(container.name for container in spec.containers) + "\n"
            primary_machine = await self._create(primary.machine, aliases=aliases)
            members.append(primary_machine)
            containers = {primary.name: primary_machine}
            for container in spec.containers:
                if container.name == primary.name:
                    continue
                member = await self._create(container.machine, peer=primary_machine, aliases=aliases)
                members.append(member)
                containers[container.name] = member
            task = Task(containers, spec, close)
            for member in members:
                member._task_close = task.close
            for container in spec.containers:
                if container.name == primary.name:
                    continue
                member = containers[container.name]
                image = member.image
                entrypoint = (
                    container.entrypoint
                    if container.entrypoint is not None
                    else (() if image is None else image.entrypoint)
                )
                arguments = container.arguments
                if arguments is None:
                    arguments = image.arguments if container.entrypoint is None and image is not None else ()
                if not entrypoint and not arguments:
                    raise UnsupportedMachineSpec(f"Container {container.name} has no startup command")
                monitor = asyncio.create_task(member.run(Command((*entrypoint, *arguments))))
                monitor.add_done_callback(lambda finished: _companion_finished(task, finished, closing_tasks))
                monitors.append(monitor)
            if self.ttl is not None:
                expiry = asyncio.create_task(_expire(task.close, self.ttl))
            return task
        except BaseException:
            await asyncio.shield(asyncio.gather(*(member._close_local() for member in members)))
            raise

    async def _host_image(self, source: HostImage | PackageSpec) -> HostImage:
        host = source if isinstance(source, HostImage) else HostImage()
        if isinstance(source, PackageSpec) and source.pypi_packages is not None:
            installed = await asyncio.to_thread(
                resolve_python_environment, self.package_cache, source.pypi_packages, self.python_version
            )
            host = HostImage(read_only=(installed.root,), bin_dirs=(installed.bin_dir,))
        return host

    async def _create(
        self, spec: MachineSpec, peer: LocalMachine | None = None, aliases: str | None = None
    ) -> LocalMachine:
        source = _check_spec(spec, self.disk_root)
        if isinstance(source, (PreparedImage, RegistryImage)):
            return await self._create_image(spec, source, peer, aliases)

        host = await self._host_image(source)
        mounts = host_mounts(host, self.read_only, self._system_mounts)
        workdir = sandbox._absolute_path(spec.workdir or "/")
        copy_on_write = self.system_directories == sandbox.SystemDirectories.COPY_ON_WRITE
        overlaid_workdir = copy_on_write and sandbox._within(workdir, sandbox.OVERLAID_DIRECTORIES)
        if (sandbox._within(workdir, sandbox.SYSTEM_DIRECTORIES) and not overlaid_workdir) or any(
            workdir.is_relative_to(mount.guest) for mount in mounts
        ):
            raise UnsupportedMachineSpec(f"The local backend's workdir {spec.workdir} is a read-only host path")
        # A workdir in an overlaid directory is made by a command once the machine runs, so it lands in the layer.
        disk = await self._allocate_disk(spec)
        try:
            root, layers = sandbox._machine_directories(
                None if overlaid_workdir else workdir, system_directories=self.system_directories, disk=disk
            )
        except BaseException:
            if disk is not None:
                await finish_owned(own_cleanup(asyncio.to_thread(disk.close)))
            raise
        try:
            if aliases is not None:
                # The copied host file can be read-only, as on NixOS.
                (root / "etc/hosts").unlink(missing_ok=True)
                (root / "etc/hosts").write_text(aliases)
            environment = host_environment(host, self.bin_dirs, self._system_bin_dirs, self.hash_seed)
            machine = LocalMachine(
                spec,
                bwrap=self.bwrap,
                root=root,
                layers=layers,
                mounts=mounts,
                environment=environment,
                cgroup_parent=self.cgroup_parent,
                disk=disk,
            )
        except BaseException:
            if disk is not None:
                await finish_owned(own_cleanup(asyncio.to_thread(disk.close)))
            else:
                sandbox._remove(root)
                if layers is not None:
                    sandbox._remove(layers)
            raise
        tools = source.tools if isinstance(source, PackageSpec) else ()
        try:
            await machine._start_keeper(peer)
            await start_host_machine(machine, workdir if overlaid_workdir else None, tools)
        except BaseException:
            await machine.close()
            raise
        return machine

    async def _prepare_image(
        self, source: PreparedImage | RegistryImage, disk: LocalDisk | None, bundle: Path
    ) -> PreparedLocalImage:
        prepared = (
            source if isinstance(source, PreparedImage) else await asyncio.to_thread(self.image_cache.prepare, source)
        )
        if disk is None:
            return await asyncio.to_thread(prepare_local_image, prepared, bundle, umoci=self.umoci)
        image = await asyncio.to_thread(
            cached_local_image, prepared, self.image_cache.directory / "local-lowers", umoci=self.umoci
        )
        root = await asyncio.to_thread(disk.overlay, image.root)
        return replace(image, root=root)

    async def _create_image(
        self, spec: MachineSpec, source: PreparedImage | RegistryImage, peer: LocalMachine | None, aliases: str | None
    ) -> LocalMachine:
        if self.read_only or self.bin_dirs or self.system_directories is not sandbox.SystemDirectories.READ_ONLY:
            raise UnsupportedMachineSpec("Prepared local images cannot use host program mounts or overlays")
        disk = await self._allocate_disk(spec)
        parent = disk.directory if disk is not None else Path(tempfile.mkdtemp(prefix="shellbox-image-"))
        bundle = parent / "bundle"
        machine = None

        preparation = asyncio.create_task(self._prepare_image(source, disk, bundle))
        try:
            image = await asyncio.shield(preparation)
            if image.user not in (None, "", "0", "0:0"):
                raise UnsupportedMachineSpec("Prepared local images currently support only root USER")
            effective = replace(spec, workdir=spec.workdir or image.workdir)
            machine = LocalMachine(
                effective,
                bwrap=self.bwrap,
                root=image.root,
                layers=None,
                mounts=(),
                environment={"PATH": sandbox.DEFAULT_PATH, "HOME": "/root", **image.environment},
                image=image,
                bundle=parent,
                cgroup_parent=self.cgroup_parent,
                disk=disk,
            )
            await machine._start_keeper(peer)
            await start_image_machine(machine, aliases)
            return machine
        except BaseException as error:
            # A cancelled to_thread still writes its bundle. Reclaim it only
            # after the trusted unpacker has stopped.
            async def cleanup() -> None:
                try:
                    await preparation
                finally:
                    if machine is not None:
                        await machine._close_local()
                    else:
                        if disk is not None:
                            await asyncio.to_thread(disk.close)
                        else:
                            await asyncio.to_thread(sandbox._remove, parent)

            try:
                await finish_owned(own_cleanup(cleanup()))
            except BaseException as cleanup_error:
                if cleanup_error is not error:
                    error.add_note(f"Local image cleanup failed ({type(cleanup_error).__name__})")
            raise

    async def _allocate_disk(self, spec: MachineSpec) -> LocalDisk | None:
        if spec.storage_mb is None:
            return None
        assert self.disk_root is not None
        pool = await verified_local_disk_pool(self.disk_root)
        allocation = asyncio.create_task(asyncio.to_thread(pool.create, spec.storage_mb))
        try:
            return await asyncio.shield(allocation)
        except BaseException:

            async def cleanup() -> None:
                disk = await allocation
                await asyncio.to_thread(disk.close)

            await finish_owned(own_cleanup(cleanup()))
            raise


async def _expire(close: Callable[[], Awaitable[None]], ttl: float) -> None:
    await asyncio.sleep(ttl)
    await close()


def _companion_finished(task: Task, monitor: asyncio.Task[Result], closing_tasks: set[asyncio.Task[None]]) -> None:
    _log_task_failure(monitor)
    reclaim = asyncio.create_task(task.close())
    closing_tasks.add(reclaim)
    reclaim.add_done_callback(closing_tasks.discard)
    reclaim.add_done_callback(_log_task_failure)
