"""Pinned Harbor BaseEnvironment backed by a fresh Daytona sandbox."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shlex
import tempfile
import time
import types
import uuid
from importlib.metadata import version
from pathlib import Path, PurePosixPath

import msgspec
from daytona import (
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    Image,
    Resources,
)
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from taskcompendium.execution import DockerEnvironment, HarborTaskBinding
from taskcompendium.models import image_digest

from . import sandbox_provider
from .daytona_resources import CANDIDATE_DEFAULT, resolve_profile, snapshot_name
from .daytona_snapshot import (
    provision_with_snapshot_recovery,
    snapshot_conflict,
    snapshot_not_found,
    wait_for_snapshot_active,
)
from .daytona_telemetry import (
    collect_cgroup_telemetry,
    finalize_lifecycle_telemetry,
    lifecycle_telemetry,
)
from .image_runtime_metadata import derive_daytona_recipe
from .provider_retry import provision_with_rate_limit_retry

_LONG_EXEC_SECONDS = 240


class TaskStartupError(RuntimeError):
    """A task-owned public startup command returned a known exit code."""

    def __init__(self, phase: str, return_code: int | None):
        self.phase = phase
        self.return_code = return_code
        super().__init__(f"Daytona task startup failed during {phase}")


def load_docker_binding(binding_path: Path, expected_image: str) -> DockerEnvironment:
    """Read the task-owned binding used by both Harbor and reset trials."""
    binding = msgspec.json.decode(binding_path.read_bytes(), type=HarborTaskBinding)
    requirement = binding.environment
    if not isinstance(requirement, DockerEnvironment):
        raise TypeError("Daytona adapter requires a Docker binding")
    if requirement.image != expected_image:
        raise ValueError("task binding image differs from requested immutable image")
    return requirement


async def materialize_docker_binding(requirement, inputs, execute, upload_dir) -> None:
    """Apply the exact public Harbor startup sequence to a fresh sandbox."""
    workdir = requirement.workdir
    result = await execute(
        "mkdir -p "
        + " ".join(
            shlex.quote(path)
            for path in (workdir, "/logs/agent", "/logs/verifier", "/logs/artifacts", "/tests")
        ),
        cwd="/",
    )
    if result.return_code != 0:
        raise TaskStartupError("mkdir", result.return_code)
    if inputs.is_dir():
        await upload_dir(inputs, workdir)
    for directory in requirement.additional_directories:
        result = await execute(f"mkdir -p {shlex.quote(directory)}", cwd="/")
        if result.return_code != 0:
            raise TaskStartupError("additional_directory", result.return_code)
    for command in requirement.setup_commands:
        result = await execute(command, cwd="/", timeout_sec=1800)
        if result.return_code != 0:
            raise TaskStartupError("setup_command", result.return_code)


def _dt():
    path = Path(os.environ["CAPABILITY_DAYTONA_TOOLS"]) / "dt.py"
    # The reset runner freezes this source file in its input inventory. Loading
    # it via importlib would create an interpreter-specific __pycache__ there.
    module = types.ModuleType("capability_daytona_dt")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def _run_portable(sandbox, command, cwd, env, timeout) -> dict:
    """Execute through POSIX sh; the maintained helper assumes optional bash."""

    from daytona import SessionExecuteRequest

    started = time.time()
    full = f"/bin/sh -c {shlex.quote(command)}"
    if timeout <= _LONG_EXEC_SECONDS:
        try:
            response = sandbox.process.exec(
                full, cwd=cwd, env=env or None, timeout=timeout + 15
            )
            return {
                "exit": response.exit_code,
                "stdout": response.result or "",
                "stderr": None,
                "seconds": round(time.time() - started, 2),
                "timed_out": False,
            }
        except Exception as error:  # noqa: BLE001 - normalize provider failure
            message = f"{type(error).__name__}: {str(error)[:600]}"
            return {
                "exit": None,
                "stdout": "",
                "stderr": message,
                "seconds": round(time.time() - started, 2),
                "timed_out": "timeout" in message.lower(),
            }
    session_id = "cap-harbor-" + uuid.uuid4().hex[:12]
    sandbox.process.create_session(session_id)
    try:
        prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
        for key, value in (env or {}).items():
            prefix += f"export {key}={shlex.quote(str(value))} && "
        wrapped = f"{prefix}timeout {int(timeout)} {full}"
        launched = sandbox.process.execute_session_command(
            session_id,
            SessionExecuteRequest(command=wrapped, run_async=True),
            timeout=60,
        )
        deadline = started + timeout + 120
        exit_code = None
        while time.time() < deadline:
            status = sandbox.process.get_session_command(session_id, launched.cmd_id)
            if status.exit_code is not None:
                exit_code = status.exit_code
                break
            time.sleep(3 if time.time() - started < 120 else 8)
        logs = sandbox.process.get_session_command_logs(session_id, launched.cmd_id)
        return {
            "exit": exit_code,
            "stdout": logs.stdout or "",
            "stderr": logs.stderr or "",
            "seconds": round(time.time() - started, 2),
            "timed_out": exit_code in {None, 124},
        }
    finally:
        with contextlib.suppress(Exception):
            sandbox.process.delete_session(session_id)


class DaytonaHarborEnvironment(BaseEnvironment):
    """Materialize an immutable Docker binding as a network-blocked Daytona snapshot."""

    def __init__(
        self, *args, snapshot_prefix="cap-harbor", resource_profile=None, **kwargs
    ):
        self.snapshot_prefix = snapshot_prefix
        self.resource_profile = resolve_profile(
            resource_profile, default=CANDIDATE_DEFAULT
        )
        self.daytona = None
        self.sandbox = None
        self._resource_telemetry = None
        self._resource_receipt_paths = ()
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "taskcompendium-daytona"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self) -> None:
        image = self.task_env_config.docker_image
        if not image:
            raise ValueError(
                "Daytona requires the TaskCompendium immutable Docker image"
            )
        image_digest(image)
        if not sandbox_provider.credentials_present():
            raise RuntimeError(f"sandbox credentials are required: {sandbox_provider.credentials_hint()}")

    def _start_sync(self):
        adapter_started = time.monotonic()
        dt = _dt()
        self.daytona = dt.client()
        image = self.task_env_config.docker_image
        definition = derive_daytona_recipe(image)
        recipe_sha256 = hashlib.sha256(definition.encode()).hexdigest()
        snapshot_cache_name = snapshot_name(
            self.snapshot_prefix, definition, self.resource_profile
        )

        def create_snapshot(name):
            with tempfile.NamedTemporaryFile(
                "w", suffix=".Dockerfile", delete=False
            ) as dockerfile:
                dockerfile.write(definition)
                dockerfile_path = dockerfile.name
            try:
                self.daytona.snapshot.create(
                    CreateSnapshotParams(
                        name=name,
                        image=Image.from_dockerfile(dockerfile_path),
                        resources=Resources(
                            cpu=self.resource_profile.cpu,
                            memory=self.resource_profile.memory_gb,
                            disk=self.resource_profile.disk_gb,
                        ),
                    ),
                    timeout=3600,
                )
                return wait_for_snapshot_active(
                    self.daytona.snapshot.get, name, definition
                )
            finally:
                Path(dockerfile_path).unlink(missing_ok=True)

        try:
            snapshot = self.daytona.snapshot.get(snapshot_cache_name)
        except Exception as error:
            if not snapshot_not_found(error):
                raise
            try:
                snapshot = create_snapshot(snapshot_cache_name)
            except Exception as create_error:
                if not snapshot_conflict(create_error):
                    raise
                snapshot = None
        snapshot = wait_for_snapshot_active(
            self.daytona.snapshot.get,
            snapshot_cache_name,
            definition,
            initial_snapshot=snapshot,
        )

        lifecycle_path = (
            Path(str(self.trial_paths.trial_dir)) / "daytona-snapshot-lifecycle.json"
        )

        resource_telemetry = None

        def write_lifecycle(events):
            payload = {
                "schema_version": "capability-daytona-snapshot-lifecycle-v1",
                "requested_image": image,
                "requested_recipe_sha256": recipe_sha256,
                "requested_resource_profile": self.resource_profile.receipt(),
                "resource_telemetry": resource_telemetry,
                "base_snapshot_name": snapshot_cache_name,
                "events": events,
            }
            lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = lifecycle_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.replace(lifecycle_path)

        def create_sandbox(name):
            parameters = CreateSandboxFromSnapshotParams(
                snapshot=name,
                labels={"envgen": "1", "envgen_purpose": "harbor-trial"},
                ephemeral=True,
                auto_stop_interval=0,
                ttl_minutes=180,
                network_block_all=True,
            )
            return provision_with_rate_limit_retry(
                lambda: self.daytona.create(parameters, timeout=600)
            )

        sandbox_started = time.monotonic()
        self.sandbox, provisioning_attempts, active_snapshot, snapshot_events = (
            provision_with_snapshot_recovery(
                base_name=snapshot_cache_name,
                requested_image=image,
                initial_snapshot=snapshot,
                create_sandbox=create_sandbox,
                reconstruct=create_snapshot,
                recovery_suffix=uuid.uuid4().hex[:8],
                record=write_lifecycle,
            )
        )
        sandbox_provision_seconds = round(time.monotonic() - sandbox_started, 3)
        startup_telemetry = collect_cgroup_telemetry(
            lambda command: _run_portable(self.sandbox, command, "/", None, timeout=30),
            self.resource_profile,
        )
        startup_telemetry["sandbox_provision_seconds"] = sandbox_provision_seconds
        startup_telemetry["adapter_start_seconds"] = round(
            time.monotonic() - adapter_started, 3
        )
        resource_telemetry = lifecycle_telemetry(startup_telemetry)
        write_lifecycle(snapshot_events)
        provider_record = {
            # The claimed adapter is the provider that actually ran, never a
            # constant: a silo sandbox recorded as Daytona is a false receipt.
            "adapter": sandbox_provider.adapter_id(),
            "sandbox_provider": sandbox_provider.provider(),
            "daytona_sdk_version": version("daytona"),
            "image": image,
            "snapshot_recipe_sha256": recipe_sha256,
            "requested_resource_profile": self.resource_profile.receipt(),
            "resource_telemetry": resource_telemetry,
            "network_block_all": True,
            "sandbox_id": self.sandbox.id,
            "snapshot": active_snapshot["name"],
            "snapshot_id": active_snapshot["id"],
            "snapshot_requested_image_sha256": active_snapshot[
                "requested_image_sha256"
            ],
            "snapshot_events": snapshot_events,
            "provisioning_attempts": provisioning_attempts,
        }
        record_path = Path(str(self.trial_paths.trial_dir)) / "daytona-environment.json"
        record_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = record_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(provider_record, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(record_path)
        self._resource_telemetry = resource_telemetry
        self._resource_receipt_paths = (lifecycle_path, record_path)

    async def start(self, force_build: bool) -> None:
        await asyncio.to_thread(self._start_sync)
        requirement = load_docker_binding(
            self.environment_dir.parent / "binding.json",
            self.task_env_config.docker_image,
        )
        await materialize_docker_binding(
            requirement, self.environment_dir / "inputs", self.exec, self.upload_dir
        )

    def _record_final_telemetry_sync(self, delete: bool) -> None:
        if self.sandbox is None or self._resource_telemetry is None:
            return
        final = collect_cgroup_telemetry(
            lambda command: _run_portable(self.sandbox, command, "/", None, timeout=30),
            self.resource_profile,
        )
        self._resource_telemetry = finalize_lifecycle_telemetry(
            self._resource_telemetry, final, delete=delete
        )
        for path in self._resource_receipt_paths:
            try:
                record = json.loads(path.read_text())
                record["resource_telemetry"] = self._resource_telemetry
                temporary = path.with_suffix(".tmp")
                temporary.write_text(
                    json.dumps(record, indent=2, sort_keys=True) + "\n"
                )
                temporary.replace(path)
            except Exception as error:  # noqa: BLE001 - telemetry must not prevent cleanup
                self._resource_telemetry["final_receipt_write_error_type"] = type(
                    error
                ).__name__

    async def stop(self, delete: bool) -> None:
        sandbox = self.sandbox
        if sandbox is None:
            return
        try:
            await asyncio.to_thread(self._record_final_telemetry_sync, delete)
        finally:
            try:
                if delete:
                    await asyncio.to_thread(sandbox.delete)
            finally:
                self.sandbox = None

    def _sandbox(self):
        if self.sandbox is None:
            raise RuntimeError("Daytona sandbox is not running")
        return self.sandbox

    async def exec(
        self, command, cwd=None, env=None, timeout_sec=None, user=None
    ) -> ExecResult:
        del user
        effective_cwd = cwd if cwd is not None else self.task_env_config.workdir
        result = await asyncio.to_thread(
            _run_portable,
            self._sandbox(),
            command,
            effective_cwd,
            self._merge_env(env),
            int(timeout_sec or 600),
        )
        code = result["exit"] if type(result["exit"]) is int else 124
        return ExecResult(
            stdout=result["stdout"], stderr=result["stderr"] or "", return_code=code
        )

    async def upload_file(self, source_path, target_path) -> None:
        source = Path(source_path)
        await self.exec(
            f"mkdir -p {shlex.quote(str(PurePosixPath(target_path).parent))}"
        )
        await asyncio.to_thread(
            self._sandbox().fs.upload_file, source.read_bytes(), target_path
        )

    async def upload_dir(self, source_dir, target_dir) -> None:
        result = await asyncio.to_thread(
            _dt().upload_path, self._sandbox(), Path(source_dir), target_dir
        )
        if result["exit"]:
            raise RuntimeError(f"Daytona directory upload failed: {result}")

    async def download_file(self, source_path, target_path) -> None:
        data = await asyncio.to_thread(self._sandbox().fs.download_file, source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data or b"")

    async def download_dir(self, source_dir, target_dir) -> None:
        await self.download_dir_with_exclusions(
            source_dir=source_dir, target_dir=target_dir, exclude=[]
        )
