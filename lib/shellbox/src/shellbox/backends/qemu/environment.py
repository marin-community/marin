# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor adapter for the QEMU machine backend."""

import hashlib
import json
import shlex
from enum import StrEnum
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from shellbox.backends.qemu.image import QemuAssets
from shellbox.backends.qemu.machine import Acceleration, QemuMachine, QemuMachineFactory
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import (
    HARBOR_EXEC_OUTPUT_LIMIT_BYTES,
    Command,
    ExitReason,
    MachineSpec,
    QemuBundle,
    ShellSession,
)
from shellbox.machine import NetworkPolicy as MachineNetworkPolicy


class NetworkPolicy(StrEnum):
    REQUIRE_OFFLINE_TASK = "require-offline-task"
    DENY = "deny"


class QemuEnvironment(BaseEnvironment):
    """Run Harbor commands in one QEMU guest per trial."""

    def __init__(
        self,
        *args,
        guest_bundle: str | None = None,
        image_cache: str | None = None,
        skopeo: str | None = None,
        qemu_assets: dict[str, str | int] | None = None,
        registry_authfile: str | None = None,
        signature_policy: str | None = None,
        network_policy: NetworkPolicy = NetworkPolicy.REQUIRE_OFFLINE_TASK,
        guest_memory_mb: int = 512,
        acceleration: Acceleration = Acceleration.AUTO,
        **kwargs,
    ):
        self.guest_bundle = Path(guest_bundle).resolve() if guest_bundle is not None else None
        self.image_cache = Path(image_cache).resolve() if image_cache is not None else None
        self.skopeo = Path(skopeo).resolve() if skopeo is not None else None
        self.qemu_assets = qemu_assets
        self.registry_authfile = Path(registry_authfile).resolve() if registry_authfile is not None else None
        self.signature_policy = Path(signature_policy).resolve() if signature_policy is not None else None
        self.network_policy = NetworkPolicy(network_policy)
        self.guest_memory_mb = guest_memory_mb
        self.acceleration = Acceleration(acceleration)
        self.machine: QemuMachine | None = None
        self._source: RegistryImage | DockerfileSource | None = None
        metadata_path = self.guest_bundle / "image.json" if self.guest_bundle is not None else None
        self._image_metadata = (
            json.loads(metadata_path.read_text()) if metadata_path is not None and metadata_path.is_file() else {}
        )
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "qemu"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self):
        if self.task_env_config.allow_internet and self.network_policy is NetworkPolicy.REQUIRE_OFFLINE_TASK:
            raise ValueError("QEMU guest has no network; set network_policy='deny' to run this task offline")
        if any((self.environment_dir / name).exists() for name in ("docker-compose.yaml", "docker-compose.yml")):
            raise ValueError("QEMU prototype does not support Compose tasks")
        dockerfile = self.environment_dir / "Dockerfile"
        image_reference = self.task_env_config.docker_image
        if self.guest_bundle is None:
            if dockerfile.exists() and image_reference is not None:
                raise ValueError("QEMU task cannot specify both a Dockerfile and docker_image")
            if dockerfile.exists():
                self._source = DockerfileSource(self.environment_dir, dockerfile)
            elif image_reference is not None:
                self._source = RegistryImage(image_reference)
            else:
                raise ValueError("QEMU task needs a Dockerfile, docker_image, or guest_bundle")
            if self.image_cache is None or self.skopeo is None or self.qemu_assets is None:
                raise ValueError("QEMU image preparation requires image_cache, skopeo, and qemu_assets")
            return
        self._validate_image_metadata(self._image_metadata)

    def _validate_image_metadata(self, metadata: dict) -> None:
        dockerfile = self.environment_dir / "Dockerfile"
        if dockerfile.exists():
            digest = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
            if metadata.get("dockerfile_sha256") != digest:
                raise ValueError("Prebuilt OCI bundle does not match the task Dockerfile")
        image_reference = self.task_env_config.docker_image
        if image_reference is not None and metadata.get("image_reference") != image_reference:
            raise ValueError("Prebuilt OCI bundle does not match the task image reference")

    async def start(self, force_build: bool) -> None:
        if force_build and self.guest_bundle is not None:
            raise ValueError("QEMU guest bundles are prebuilt; force_build is unsupported")
        assets = None
        if self.qemu_assets is not None:
            config = self.qemu_assets
            assets = QemuAssets(
                qemu=Path(str(config["qemu"])),
                kernel=Path(str(config["kernel"])),
                busybox=Path(str(config["busybox"])),
                firmware=Path(str(config["firmware"])),
                libraries=Path(str(config["libraries"])),
                umoci=Path(str(config["umoci"])),
                disk_size_mb=int(config["disk_size_mb"]),
                runtime_id=str(config["runtime_id"]),
            )
        if self.guest_bundle is None:
            assert self._source is not None
            source = self._source
        else:
            source = QemuBundle(self.guest_bundle)
        spec = MachineSpec(
            source=source,
            workdir=self.task_env_config.workdir
            or self._image_metadata.get("cwd")
            or ("/workspace" if self.guest_bundle is not None else ""),
            memory_mb=self.guest_memory_mb,
            network=MachineNetworkPolicy.DENY,
        )
        factory = QemuMachineFactory(
            self.acceleration,
            assets=assets,
            image_cache=self.image_cache / "oci" if self.image_cache is not None else None,
            bundle_cache=self.image_cache / "bundles" if self.image_cache is not None else None,
            skopeo=self.skopeo,
            authfile=self.registry_authfile,
            policy=self.signature_policy,
        )
        self.machine = await factory.create(spec)
        try:
            self._validate_image_metadata(self.machine.metadata)
            self.logger.info("QEMU accelerator: %s", self.machine.active_acceleration)
            await self._upload_environment_dir_after_start()
        except BaseException:
            await self.stop(True)
            raise

    async def stop(self, delete: bool):
        if self.machine is not None:
            await self.machine.close()
            self.machine = None

    async def open_bash_session(self) -> ShellSession:
        """Open the agent's persistent Bash without affecting verifier commands."""
        if self.machine is None:
            raise RuntimeError("QEMU environment is not running")
        return await self.machine.open_shell()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if user not in (None, "root", 0):
            raise ValueError("QEMU prototype supports only the root user")
        if self.machine is None:
            raise RuntimeError("QEMU environment is not running")
        shell_command = (
            f"if [ -x /bin/bash ]; then exec /bin/bash -c {shlex.quote(command)}; "
            f"else exec /bin/sh -c {shlex.quote(command)}; fi"
        )
        result = await self.machine.run(
            Command(
                argv=("/bin/sh", "-c", shell_command),
                cwd=cwd,
                env=self._merge_env(env) or {},
                timeout=timeout_sec,
                output_limit_bytes=HARBOR_EXEC_OUTPUT_LIMIT_BYTES,
            )
        )
        if result.reason is ExitReason.TIMED_OUT:
            raise TimeoutError("QEMU command timed out")
        return ExecResult(
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            return_code=result.exit_code,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )

    async def upload_file(self, source_path: Path | str, target_path: str):
        if self.machine is None:
            raise RuntimeError("QEMU environment is not running")
        await self.machine.upload(Path(source_path), target_path)

    async def download_file(self, source_path: str, target_path: Path | str):
        if self.machine is None:
            raise RuntimeError("QEMU environment is not running")
        await self.machine.download(source_path, Path(target_path))

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        if self.machine is None:
            raise RuntimeError("QEMU environment is not running")
        await self.machine.upload(Path(source_dir), target_dir)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        if self.machine is None:
            raise RuntimeError("QEMU environment is not running")
        await self.machine.download(source_dir, Path(target_dir))
