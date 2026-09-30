"""Run a frozen five-cycle Daytona reset-conformance check.

This controller never executes task or candidate code locally.  It creates five
fresh, network-blocked, ephemeral sandboxes from one recipe- and
resource-bound snapshot.  A v2 plan first applies the hash-bound Harbor task
startup in each sandbox; a v1 plan checks the bare image.  The remote inspector
returns public file hashes, process command names/counts, and only its own
environment variable names.  The report is conformance evidence, never an
admission decision.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shlex
import stat
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from capability_pipeline.daytona_resources import (
    DaytonaResourceProfile,
    profile_from_receipt,
    snapshot_name,
)

_SCHEMA = "capability-reset-conformance-plan-v1"
_TASK_SCHEMA = "capability-reset-conformance-plan-v2"
_DIGEST = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
_HEX = re.compile(r"^[0-9a-f]{64}$")
_MAX_COMMAND = 4096


@dataclass(frozen=True)
class ResetPlan:
    image: str
    recipe: str
    recipe_sha256: str
    profile: DaytonaResourceProfile
    public_root: str
    public_files: dict[str, str]
    process_allowed: frozenset[str]
    process_max_count: int
    environment_allowed: frozenset[str]
    environment_required: frozenset[str]
    environment_forbidden: frozenset[str]
    readiness_command: str
    readiness_timeout_seconds: int
    mutation_command: str | None
    mutation_timeout_seconds: int | None
    mutation_files: dict[str, str] | None
    task_bundle: Path | None = None
    task_binding_sha256: str | None = None
    task_inputs_sha256: str | None = None
    task_toml_sha256: str | None = None


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _inputs_sha256(inputs: Path) -> str:
    """Bind input bytes and empty directories; reject links and special files."""
    if not inputs.is_dir() or inputs.is_symlink():
        raise ValueError("task inputs directory is missing or linked")
    entries = [[".", "directory", stat.S_IMODE(inputs.stat().st_mode)]]
    for path in sorted(inputs.rglob("*")):
        if path.is_symlink():
            raise ValueError("task inputs may not contain symlinks")
        relative = path.relative_to(inputs).as_posix()
        if path.is_dir():
            entries.append([relative, "directory", stat.S_IMODE(path.stat().st_mode)])
        elif path.is_file():
            entries.append([relative, "file", stat.S_IMODE(path.stat().st_mode),
                            _sha256_bytes(path.read_bytes())])
        else:
            raise ValueError("task inputs contain a special file")
    return _sha256_bytes(_canonical(entries).encode())


def _safe_relative(path: object) -> str:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("public file path must be a nonempty string")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or pure.as_posix() != path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("public file path must be a canonical relative path")
    return path


def _safe_public_root(path: object) -> str:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("reset plan public_root must be an absolute POSIX path")
    pure = PurePosixPath(path)
    system_roots = ("/", "/proc", "/sys", "/dev", "/run")
    if (
        not pure.is_absolute()
        or pure.as_posix() != path
        or any(part in {".", ".."} for part in pure.parts)
        or any(path == root or path.startswith(root + "/") for root in system_roots)
    ):
        raise ValueError("reset plan public_root is not a safe public directory")
    return path


def _hash_inventory(value: object, *, label: str, allow_empty: bool = False) -> dict[str, str]:
    if not isinstance(value, Mapping) or (not value and not allow_empty):
        raise ValueError(f"{label} must be {'a ' if allow_empty else 'a nonempty '}path-to-sha256 object")
    parsed: dict[str, str] = {}
    for path, digest in value.items():
        relative = _safe_relative(path)
        if (
            relative in parsed
            or not isinstance(digest, str)
            or not _HEX.fullmatch(digest)
        ):
            raise ValueError(f"{label} has an invalid path or SHA-256")
        parsed[relative] = digest
    return dict(sorted(parsed.items()))


def _name_set(value: object, *, label: str, required: bool = False) -> frozenset[str]:
    if not isinstance(value, list) or (required and not value):
        raise ValueError(f"{label} must be {'a nonempty ' if required else 'an '}array")
    if any(
        not isinstance(name, str) or not name or "=" in name or "\x00" in name
        for name in value
    ):
        raise ValueError(f"{label} contains an invalid environment or process name")
    parsed = frozenset(value)
    if len(parsed) != len(value):
        raise ValueError(f"{label} contains duplicate names")
    return parsed


def _command(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\x00" in value
        or len(value) > _MAX_COMMAND
    ):
        raise ValueError(f"{label} must be a bounded nonempty command")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_plan_payload(payload: object) -> ResetPlan:
    """Validate a plan completely before contacting a provider."""
    required_keys = {
        "schema_version",
        "image",
        "recipe",
        "recipe_sha256",
        "resource_profile",
        "public_root",
        "public_files",
        "process_policy",
        "environment_name_policy",
        "readiness",
        "cycles",
    }
    if not isinstance(payload, Mapping) or set(payload) not in {
        frozenset(required_keys),
        frozenset(required_keys | {"mutation"}),
        frozenset(required_keys | {"task_binding"}),
        frozenset(required_keys | {"mutation", "task_binding"}),
    }:
        raise ValueError("reset plan has an invalid top-level shape")
    if payload["schema_version"] not in {_SCHEMA, _TASK_SCHEMA}:
        raise ValueError("reset plan has an unsupported schema version")
    task_binding = payload.get("task_binding")
    if (payload["schema_version"] == _TASK_SCHEMA) != (task_binding is not None):
        raise ValueError("task-bound reset requires the v2 schema and task_binding")
    image = payload["image"]
    recipe = payload["recipe"]
    recipe_sha256 = payload["recipe_sha256"]
    public_root = payload["public_root"]
    if not isinstance(image, str) or not _DIGEST.fullmatch(image):
        raise ValueError(
            "reset plan image must be an immutable canonical digest reference"
        )
    if not isinstance(recipe, str) or not recipe or not isinstance(recipe_sha256, str):
        raise ValueError("reset plan recipe must be bound")
    if _sha256_bytes(recipe.encode()) != recipe_sha256:
        raise ValueError("reset plan recipe SHA-256 does not match recipe bytes")
    public_root = _safe_public_root(public_root)
    if _positive_int(payload["cycles"], label="cycles") != 5:
        raise ValueError("reset plan must contain exactly five cycles")
    profile = profile_from_receipt(payload["resource_profile"])
    public_files = _hash_inventory(
        payload["public_files"], label="public_files",
        allow_empty=payload["schema_version"] == _TASK_SCHEMA,
    )
    task_bundle = None
    task_binding_sha256 = task_inputs_sha256 = task_toml_sha256 = None
    if task_binding is not None:
        if not isinstance(task_binding, Mapping) or set(task_binding) != {
            "bundle_path", "binding_sha256", "inputs_sha256", "task_toml_sha256"
        }:
            raise ValueError("task_binding has an invalid shape")
        if not isinstance(task_binding["bundle_path"], str):
            raise ValueError("task bundle path must be an absolute string")
        task_bundle = Path(task_binding["bundle_path"])
        if not task_bundle.is_absolute() or any(
            path.is_symlink() for path in (task_bundle, *task_bundle.parents)
        ):
            raise ValueError("task bundle must be an absolute ordinary directory")
        task_binding_sha256 = task_binding["binding_sha256"]
        task_inputs_sha256 = task_binding["inputs_sha256"]
        task_toml_sha256 = task_binding["task_toml_sha256"]
        if not all(isinstance(value, str) and _HEX.fullmatch(value) for value in
                   (task_binding_sha256, task_inputs_sha256, task_toml_sha256)):
            raise ValueError("task binding fingerprints must be SHA-256")
        binding_path = task_bundle / "binding.json"
        task_toml_path = task_bundle / "task.toml"
        if (task_bundle / "environment").is_symlink():
            raise ValueError("task environment directory is linked")
        if binding_path.is_symlink() or not binding_path.is_file():
            raise ValueError("task binding file is missing or linked")
        if task_toml_path.is_symlink() or not task_toml_path.is_file():
            raise ValueError("task TOML is missing or linked")
        if _sha256_bytes(binding_path.read_bytes()) != task_binding_sha256:
            raise ValueError("task binding fingerprint mismatch")
        if _sha256_bytes(task_toml_path.read_bytes()) != task_toml_sha256:
            raise ValueError("task TOML fingerprint mismatch")
        if _inputs_sha256(task_bundle / "environment" / "inputs") != task_inputs_sha256:
            raise ValueError("task inputs fingerprint mismatch")
        try:
            binding_json = json.loads(binding_path.read_bytes())
            bound_image = binding_json["environment"]["image"]
            bound_kind = binding_json["environment"]["kind"]
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("task binding is invalid JSON or lacks an environment") from error
        if bound_kind != "docker" or bound_image != image:
            raise ValueError("task binding image differs from requested immutable image")
        task_environment = tomllib.loads(task_toml_path.read_text()).get("environment")
        if not isinstance(task_environment, Mapping):
            raise ValueError("task TOML lacks environment configuration")
        if task_environment.get("env"):
            raise ValueError("task-bound reset does not support injected task environment variables")
        if (task_environment.get("docker_image") != image or
                task_environment.get("workdir", "/app") !=
                binding_json["environment"].get("workdir", "/app") or
                task_environment.get("allow_internet") is not False):
            raise ValueError("task TOML differs from the bound Docker environment")

    process = payload["process_policy"]
    if not isinstance(process, Mapping) or set(process) != {
        "allowed_comm",
        "max_count",
    }:
        raise ValueError("process_policy must bind allowed_comm and max_count")
    process_allowed = _name_set(
        process["allowed_comm"], label="allowed_comm", required=True
    )
    process_max_count = _positive_int(process["max_count"], label="max_count")

    environment = payload["environment_name_policy"]
    if not isinstance(environment, Mapping) or set(environment) != {
        "allowed_names",
        "required_names",
        "forbidden_names",
    }:
        raise ValueError("environment_name_policy has an invalid shape")
    environment_allowed = _name_set(
        environment["allowed_names"], label="allowed_names", required=True
    )
    environment_required = _name_set(
        environment["required_names"], label="required_names"
    )
    environment_forbidden = _name_set(
        environment["forbidden_names"], label="forbidden_names"
    )
    if (
        not environment_required <= environment_allowed
        or environment_allowed & environment_forbidden
    ):
        raise ValueError("environment_name_policy is contradictory")

    readiness = payload["readiness"]
    if not isinstance(readiness, Mapping) or set(readiness) != {
        "command",
        "timeout_seconds",
    }:
        raise ValueError("readiness has an invalid shape")
    mutation = payload.get("mutation")
    mutation_command = mutation_timeout = mutation_files = None
    if mutation is not None:
        if not isinstance(mutation, Mapping) or set(mutation) != {
            "command",
            "timeout_seconds",
            "expected_files",
        }:
            raise ValueError("mutation has an invalid shape")
        mutation_command = _command(mutation["command"], label="mutation command")
        mutation_timeout = _positive_int(
            mutation["timeout_seconds"], label="mutation timeout_seconds"
        )
        mutation_files = _hash_inventory(
            mutation["expected_files"], label="mutation expected_files",
            allow_empty=payload["schema_version"] == _TASK_SCHEMA,
        )
        if mutation_files == public_files:
            raise ValueError("mutation expected_files must differ from public_files")
    return ResetPlan(
        image=image,
        recipe=recipe,
        recipe_sha256=recipe_sha256,
        profile=profile,
        public_root=public_root,
        public_files=public_files,
        process_allowed=process_allowed,
        process_max_count=process_max_count,
        environment_allowed=environment_allowed,
        environment_required=environment_required,
        environment_forbidden=environment_forbidden,
        readiness_command=_command(readiness["command"], label="readiness command"),
        readiness_timeout_seconds=_positive_int(
            readiness["timeout_seconds"], label="readiness timeout_seconds"
        ),
        mutation_command=mutation_command,
        mutation_timeout_seconds=mutation_timeout,
        mutation_files=mutation_files,
        task_bundle=task_bundle,
        task_binding_sha256=task_binding_sha256,
        task_inputs_sha256=task_inputs_sha256,
        task_toml_sha256=task_toml_sha256,
    )


def load_plan(path: Path) -> tuple[ResetPlan, str]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("reset plan is not JSON") from exc
    return validate_plan_payload(payload), _sha256_bytes(raw)


class ResetAdapter(Protocol):
    def ensure_snapshot(self, plan: ResetPlan) -> str: ...
    def create_sandbox(self, snapshot: str) -> tuple[Any, list[dict[str, Any]]]: ...
    def run(self, sandbox: Any, command: str, timeout: int) -> dict[str, Any]: ...
    def delete(self, sandbox: Any) -> dict[str, object]: ...


class DaytonaResetAdapter:
    """Thin production adapter; tests inject a local fake instead."""

    def __init__(self) -> None:
        from capability_pipeline.daytona_environment import _dt

        self.client = _dt().client()

    def ensure_snapshot(self, plan: ResetPlan) -> str:
        from daytona import CreateSnapshotParams, Image, Resources

        from capability_pipeline.daytona_snapshot import (
            snapshot_conflict,
            snapshot_not_found,
        )
        from capability_pipeline.image_runtime_metadata import derive_daytona_recipe

        if derive_daytona_recipe(plan.image) != plan.recipe:
            raise RuntimeError(
                "frozen reset plan recipe differs from maintained image recipe"
            )
        name = snapshot_name("cap-reset", plan.recipe, plan.profile)
        try:
            snapshot = self.client.snapshot.get(name)
        except Exception as error:
            if not snapshot_not_found(error):
                raise
        else:
            from capability_pipeline.daytona_snapshot import wait_for_snapshot_active
            wait_for_snapshot_active(self.client.snapshot.get, name, plan.recipe,
                                     initial_snapshot=snapshot)
            return name
        with tempfile.NamedTemporaryFile(
            "w", suffix=".Dockerfile", delete=False
        ) as dockerfile:
            dockerfile.write(plan.recipe)
            dockerfile_path = Path(dockerfile.name)
        try:
            try:
                self.client.snapshot.create(
                    CreateSnapshotParams(
                        name=name,
                        image=Image.from_dockerfile(str(dockerfile_path)),
                        resources=Resources(
                            cpu=plan.profile.cpu,
                            memory=plan.profile.memory_gb,
                            disk=plan.profile.disk_gb,
                        ),
                    ),
                    timeout=3600,
                )
            except Exception as error:
                if not snapshot_conflict(error):
                    raise
            from capability_pipeline.daytona_snapshot import wait_for_snapshot_active
            wait_for_snapshot_active(self.client.snapshot.get, name, plan.recipe)
        finally:
            dockerfile_path.unlink(missing_ok=True)
        return name

    def start_task(self, sandbox: Any, plan: ResetPlan) -> None:
        if plan.task_bundle is None:
            return
        from capability_pipeline.daytona_environment import (
            _dt,
            _run_portable,
            load_docker_binding,
            materialize_docker_binding,
        )
        binding_path = plan.task_bundle / "binding.json"
        task_toml_path = plan.task_bundle / "task.toml"
        inputs = plan.task_bundle / "environment" / "inputs"
        if any(path.is_symlink() for path in (
            plan.task_bundle, plan.task_bundle / "environment", inputs,
            binding_path, task_toml_path,
        )):
            raise ValueError("task bundle acquired a symlink before startup")
        if (_sha256_bytes(binding_path.read_bytes()) != plan.task_binding_sha256 or
                _sha256_bytes(task_toml_path.read_bytes()) != plan.task_toml_sha256 or
                _inputs_sha256(inputs) != plan.task_inputs_sha256):
            raise ValueError("task bundle changed before startup")
        requirement = load_docker_binding(binding_path, plan.image)

        async def execute(command, *, cwd, timeout_sec=600):
            result = await asyncio.to_thread(
                _run_portable, sandbox, command, cwd, None, timeout_sec
            )
            return type("Result", (), {
                "return_code": result["exit"], "stderr": result["stderr"],
                "stdout": result["stdout"]
            })()

        async def upload_dir(source, target):
            result = await asyncio.to_thread(_dt().upload_path, sandbox, source, target)
            if result["exit"]:
                raise RuntimeError("Daytona task input upload failed")

        asyncio.run(materialize_docker_binding(requirement, inputs, execute, upload_dir))

    def create_sandbox(self, snapshot: str) -> tuple[Any, list[dict[str, Any]]]:
        from daytona import CreateSandboxFromSnapshotParams

        from capability_pipeline.provider_retry import provision_with_rate_limit_retry

        return provision_with_rate_limit_retry(
            lambda: self.client.create(
                CreateSandboxFromSnapshotParams(
                    snapshot=snapshot,
                    labels={"envgen": "1", "envgen_purpose": "reset-conformance"},
                    ephemeral=True,
                    auto_stop_interval=0,
                    ttl_minutes=180,
                    network_block_all=True,
                ),
                timeout=600,
            )
        )

    def run(self, sandbox: Any, command: str, timeout: int) -> dict[str, Any]:
        from capability_pipeline.daytona_environment import _run_portable

        return _run_portable(sandbox, command, "/", None, timeout)

    def delete(self, sandbox: Any) -> dict[str, object]:
        from capability_pipeline.daytona_snapshot import wait_for_sandbox_deletion

        sandbox.delete()
        state, observations = wait_for_sandbox_deletion(self.client, sandbox.id)
        return {
            "verified_absent": state == "not_found",
            "observations": observations,
        }


def _probe_command(root: str) -> str:
    """Return remote-only inspection code without contents, values, or argv."""
    program = f"""import hashlib,json,os,pathlib
root=pathlib.Path({root!r})
if not root.is_dir(): raise SystemExit(20)
files={{}}; symlinks=[]
walk_errors=[]
for base, dirs, names in os.walk(root, followlinks=False, onerror=lambda _error: walk_errors.append(True)):
    base_path=pathlib.Path(base)
    for name in sorted(dirs + names):
        path=base_path/name
        relative=path.relative_to(root).as_posix()
        if path.is_symlink(): symlinks.append(relative)
    for name in sorted(names):
        path=base_path/name
        if path.is_file() and not path.is_symlink():
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1048576), b''): digest.update(chunk)
            files[path.relative_to(root).as_posix()]=digest.hexdigest()
if walk_errors: raise SystemExit(21)
counts={{}}
for item in pathlib.Path('/proc').glob('[0-9]*'):
    try:
        comm=(item/'comm').read_text().strip()
    except OSError:
        continue
    if comm: counts[comm]=counts.get(comm,0)+1
print(json.dumps({{'files':files,'symlinks':sorted(symlinks),'process_comm_counts':counts,'environment_names':sorted(os.environ)}},sort_keys=True,separators=(',',':')))
"""
    return "python3 -I -c " + shlex.quote(program)


def _safe_error(error: BaseException) -> dict[str, str]:
    return {"error_type": type(error).__name__}


def _run_exit(
    adapter: ResetAdapter, sandbox: Any, command: str, timeout: int
) -> dict[str, object]:
    started = time.monotonic()
    result = adapter.run(sandbox, command, timeout)
    return {
        "exit": result.get("exit"),
        "timed_out": result.get("timed_out") is True,
        "seconds": round(time.monotonic() - started, 3),
    }


def _inspect(
    adapter: ResetAdapter, sandbox: Any, plan: ResetPlan, expected_files: dict[str, str]
) -> dict[str, object]:
    result = adapter.run(sandbox, _probe_command(plan.public_root), 60)
    if result.get("exit") != 0 or result.get("timed_out") is True:
        return {
            "status": "probe_failed",
            "probe": {
                "exit": result.get("exit"),
                "timed_out": result.get("timed_out") is True,
            },
        }
    try:
        raw = json.loads(result.get("stdout", ""))
    except (TypeError, json.JSONDecodeError):
        return {"status": "probe_invalid"}
    if not isinstance(raw, Mapping) or set(raw) != {
        "files",
        "symlinks",
        "process_comm_counts",
        "environment_names",
    }:
        return {"status": "probe_invalid"}
    try:
        files = _hash_inventory(
            raw["files"], label="remote public_files",
            allow_empty=plan.task_bundle is not None,
        )
        if not isinstance(raw["symlinks"], list):
            raise TypeError("invalid symlink inventory")
        symlinks = [_safe_relative(value) for value in raw["symlinks"]]
        counts_raw = raw["process_comm_counts"]
        names = _name_set(raw["environment_names"], label="remote environment names")
        if not isinstance(counts_raw, Mapping) or any(
            type(count) is not int
            or count <= 0
            or not isinstance(comm, str)
            or not comm
            for comm, count in counts_raw.items()
        ):
            raise ValueError("invalid process counts")
        counts = dict(sorted(counts_raw.items()))
    except (TypeError, ValueError):
        return {"status": "probe_invalid"}
    unexpected_comm = sorted(set(counts) - plan.process_allowed)
    unexpected_names = sorted(names - plan.environment_allowed)
    required_missing = sorted(plan.environment_required - names)
    forbidden_present = sorted(names & plan.environment_forbidden)
    inventory_matches = files == expected_files and not symlinks
    process_matches = (
        not unexpected_comm and sum(counts.values()) <= plan.process_max_count
    )
    environment_matches = (
        not unexpected_names and not required_missing and not forbidden_present
    )
    return {
        "status": "passed"
        if inventory_matches and process_matches and environment_matches
        else "failed",
        "inventory_sha256": _sha256_bytes(_canonical(files).encode()),
        "file_count": len(files),
        "inventory_matches": inventory_matches,
        "symlink_paths": sorted(symlinks),
        "process": {
            "count": sum(counts.values()),
            "comm_counts": counts,
            "unexpected_comm": unexpected_comm,
            "matches": process_matches,
        },
        "environment": {
            "name_set_sha256": _sha256_bytes(_canonical(sorted(names)).encode()),
            "unexpected_names": unexpected_names,
            "required_missing_names": required_missing,
            "forbidden_present_names": forbidden_present,
            "matches": environment_matches,
        },
    }


def run_plan(
    plan: ResetPlan,
    plan_sha256: str,
    output: Path,
    *,
    adapter: ResetAdapter | None = None,
) -> dict[str, object]:
    """Execute five fresh cycles and write an append-only conformance report."""
    if output.exists():
        raise FileExistsError("reset conformance output already exists")
    output.mkdir(parents=True)
    adapter = adapter or DaytonaResetAdapter()
    report: dict[str, object] = {
        "schema_version": "capability-reset-conformance-report-v1",
        "plan_sha256": plan_sha256,
        "controller_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "requested_image": plan.image,
        "requested_recipe_sha256": plan.recipe_sha256,
        "requested_resource_profile": plan.profile.receipt(),
        "network_block_all_requested": True,
        "cycles_required": 5,
        "admission": "unassessed",
        "startup_scope": "task_bound" if plan.task_bundle else "image_only",
        "task_binding_sha256": plan.task_binding_sha256,
        "task_inputs_sha256": plan.task_inputs_sha256,
        "task_toml_sha256": plan.task_toml_sha256,
        "environment_name_scope": "remote inspector process only",
        "limitations": [
            "Provider network blocking is requested; network absence is unassessed.",
            "Private files outside public_root and credential values are unassessed.",
            "Image-local probes observe state; they do not prove absence against a hostile image.",
            "Task setup, readiness and optional mutation commands run remotely; this harness does not run a solver or grader.",
        ],
        "cycles": [],
    }
    try:
        snapshot = adapter.ensure_snapshot(plan)
    except Exception as error:  # noqa: BLE001 - provider boundary; no body is serialized
        report["snapshot"] = {"status": "failed", **_safe_error(error)}
        report["cycles"] = [
            {"cycle": number, "status": "not_started", "reason": "snapshot_unavailable"}
            for number in range(1, 6)
        ]
        report["conformance"] = "incomplete"
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        return report
    report["snapshot"] = {"status": "ready", "name": snapshot}
    seen_ids: set[str] = set()
    for number in range(1, 6):
        cycle: dict[str, object] = {
            "cycle": number,
            "status": "unknown",
            "snapshot": snapshot,
        }
        sandbox = None
        try:
            sandbox, provisioning_attempts = adapter.create_sandbox(snapshot)
            if not isinstance(provisioning_attempts, list) or any(
                not isinstance(attempt, Mapping) for attempt in provisioning_attempts
            ):
                raise RuntimeError("provider returned an invalid provisioning receipt")
            cycle["provisioning_attempts"] = provisioning_attempts
            sandbox_id = getattr(sandbox, "id", None)
            if (
                not isinstance(sandbox_id, str)
                or not sandbox_id
                or sandbox_id in seen_ids
            ):
                cycle.update(
                    status="failed",
                    sandbox_id=sandbox_id,
                    failure="sandbox_identity_invalid_or_reused",
                )
            else:
                seen_ids.add(sandbox_id)
                cycle["sandbox_id"] = sandbox_id
                if plan.task_bundle is not None:
                    starter = getattr(adapter, "start_task", None)
                    if starter is None:
                        raise RuntimeError("task-bound reset adapter cannot start a task")
                    starter(sandbox, plan)
                    cycle["task_startup"] = {"status": "completed"}
                readiness = _run_exit(
                    adapter,
                    sandbox,
                    plan.readiness_command,
                    plan.readiness_timeout_seconds,
                )
                cycle["readiness"] = readiness
                initial = _inspect(adapter, sandbox, plan, plan.public_files)
                cycle["initial_state"] = initial
                mutation: dict[str, object] | None = None
                if plan.mutation_command is not None:
                    mutation = _run_exit(
                        adapter,
                        sandbox,
                        plan.mutation_command,
                        plan.mutation_timeout_seconds or 1,
                    )
                    if mutation["exit"] == 0 and mutation["timed_out"] is False:
                        mutation["state"] = _inspect(
                            adapter, sandbox, plan, plan.mutation_files or {}
                        )
                        mutation["matches"] = (
                            mutation["state"].get("status") == "passed"
                        )
                    else:
                        mutation["matches"] = False
                    cycle["mutation"] = mutation
                cycle["status"] = (
                    "passed"
                    if readiness["exit"] == 0
                    and readiness["timed_out"] is False
                    and initial.get("status") == "passed"
                    and (mutation is None or mutation.get("matches") is True)
                    else "failed"
                )
        except Exception as error:  # noqa: BLE001 - provider boundary; no body is serialized
            cycle.update(status="failed", **_safe_error(error))
        finally:
            if sandbox is None:
                cycle["cleanup"] = {"attempted": False, "succeeded": False}
            else:
                try:
                    deletion = adapter.delete(sandbox)
                    if (
                        not isinstance(deletion, Mapping)
                        or deletion.get("verified_absent") is not True
                        or not isinstance(deletion.get("observations"), list)
                    ):
                        cycle["cleanup"] = {
                            "attempted": True,
                            "succeeded": False,
                            "failure": "deletion_unconfirmed",
                        }
                        cycle["status"] = "failed"
                    else:
                        cycle["cleanup"] = {
                            "attempted": True,
                            "succeeded": True,
                            "verified_absent": True,
                            "observations": deletion["observations"],
                        }
                except Exception as error:  # noqa: BLE001 - provider boundary; no body is serialized
                    cycle["cleanup"] = {
                        "attempted": True,
                        "succeeded": False,
                        **_safe_error(error),
                    }
                    cycle["status"] = "failed"
        report["cycles"].append(cycle)
    report["conformance"] = (
        "passed"
        if all(cycle["status"] == "passed" for cycle in report["cycles"])
        else "failed"
    )
    if plan.task_bundle is not None:
        try:
            stable = (
                _sha256_bytes((plan.task_bundle / "binding.json").read_bytes())
                == plan.task_binding_sha256
                and _sha256_bytes((plan.task_bundle / "task.toml").read_bytes())
                == plan.task_toml_sha256
                and _inputs_sha256(plan.task_bundle / "environment" / "inputs")
                == plan.task_inputs_sha256
            )
        except (OSError, ValueError):
            stable = False
        report["task_bundle_unchanged_at_finish"] = stable
        if not stable:
            report["conformance"] = "incomplete"
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        plan, plan_sha256 = load_plan(args.plan)
        report = run_plan(plan, plan_sha256, args.out)
    except (OSError, ValueError) as error:
        print(f"reset conformance refused: {type(error).__name__}", file=sys.stderr)
        return 2
    return 0 if report["conformance"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
