"""Trusted, credential-free-to-builder capture of reviewed Daytona snapshots.

This module does not approve a plan, publish an image, or return a usable OCI
reference.  A separate trusted controller must approve the exact plan hash and
review the captured rootfs before publication.

Every failure is classified so the controller can act on it instead of parking
the item:

* ``transient`` -- provider, transport or capacity trouble; retry with backoff.
* ``content``   -- a deterministic property of the reviewed task content (an
  environment that differs from the reviewed image config, a readiness command
  that fails, a recipe that cannot be re-materialised); the builder must repair it.
* ``harness``   -- our own pipeline broke (a missing dependency, credentials the
  job should have, an assumption about the provider that no longer holds); it
  must surface loudly and never loop silently.

Messages of the classified exceptions below are our own static text, so they may
be recorded; provider exception text is never recorded (only its type and HTTP
status), because it can carry URLs and tokens.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from scripts.capture_task_images import (
    CW_HOST,
    PROVIDER_BOUND_PATHS,
    REQUIRED_CAPTURE_EXCLUDES,
    _expected_env,
    _sanitize_legacy_receipt,
    _validate_archive_receipt,
    image_env,
)

SCHEMA = "capability-generic-image-capture-plan-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PRIVATE_ROOTS = ("/opt/evaluator", "/opt/verifier", "/private")
_CREDENTIAL_ROOTS = ("/run/secrets", "/root/.aws", "/root/.config/gcloud", "/root/.docker/config.json")
_SENSITIVE_ROOTS = _PRIVATE_ROOTS + _CREDENTIAL_ROOTS

TRANSIENT = "transient"
CONTENT = "content"
HARNESS = "harness"
FAILURE_CLASSES = (TRANSIENT, CONTENT, HARNESS)

# Receipt state written for a failure of each class (privacy and environment
# mismatches keep their own, more specific states).
FAILED_STATE = {
    TRANSIENT: "infrastructure_error",
    CONTENT: "capture_content_error",
    HARNESS: "capture_harness_error",
}


class PrivacyReviewRequired(ValueError):
    """A candidate image includes a reviewed private path or sensitive root."""


class CaptureError(RuntimeError):
    """A classified capture failure whose message is our own static text."""

    def __init__(self, message: str, *, step: str, failure_class: str = TRANSIENT,
                 detail: dict | None = None, error_type: str | None = None) -> None:
        if failure_class not in FAILURE_CLASSES:
            raise ValueError(f"unknown capture failure class {failure_class!r}")
        super().__init__(message)
        self.step = step
        self.failure_class = failure_class
        self.detail = dict(detail or {})
        self.error_type = error_type or type(self).__name__


class CaptureProvisionError(CaptureError):
    """A failure before any capture sandbox exists, so there is no receipt to write."""


class ImageConfigMismatch(CaptureError):
    """The sandbox environment differs from the reviewed image_config.Env."""

    def __init__(self, mismatches: list[dict], *, step: str = "environment") -> None:
        super().__init__("captured image environment differs from reviewed config", step=step,
                         failure_class=CONTENT, detail={"mismatches": mismatches})
        self.mismatches = mismatches


# Exception types that, raised from inside this process, mean our own code or
# environment is wrong rather than the provider or the task content.
_HARNESS_TYPES = (
    ImportError, NameError, AttributeError, TypeError, KeyError, IndexError, ValueError,
    AssertionError, NotImplementedError, ZeroDivisionError, RecursionError,
)


def classify_exception(error: BaseException) -> str:
    """Map any exception to transient / content / harness without reading its text."""
    if isinstance(error, CaptureError):
        return error.failure_class
    if isinstance(error, PrivacyReviewRequired):
        return CONTENT
    if isinstance(error, SystemExit):
        # dt.py / dtx.py / cw_presign.py exit when a credential the job should hold is absent.
        return HARNESS
    if isinstance(error, (OSError, TimeoutError)):
        return TRANSIENT
    if type(error).__module__ != "builtins":
        # Provider SDK, transport and object-store errors (SiloError, TransportError,
        # DaytonaError, botocore ClientError, JSONDecodeError on a provider payload).
        return TRANSIENT
    if isinstance(error, _HARNESS_TYPES):
        return HARNESS
    return TRANSIENT


def error_summary(error: BaseException) -> dict:
    """What may be recorded about any exception: never its provider text."""
    summary = {
        "error_type": getattr(error, "error_type", None) if isinstance(error, CaptureError) else type(error).__name__,
        "failure_class": classify_exception(error),
    }
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        summary["status_code"] = status
    if isinstance(error, ImportError) and error.name:
        summary["missing_module"] = error.name  # a module name, never provider text
    if isinstance(error, (CaptureError, PrivacyReviewRequired)):
        summary["error_message"] = str(error)
    if isinstance(error, CaptureError):
        summary["stage"] = error.step
        if error.detail:
            summary["detail"] = error.detail
    return summary


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source(workspace: Path, relative: Any, expected: Any) -> Path:
    if not isinstance(relative, str) or not relative or not isinstance(expected, str) or not _HEX.fullmatch(expected):
        raise ValueError("source binding is malformed")
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("source path is not workspace-relative")
    root = workspace.resolve(strict=True)
    target = workspace / path
    if any(node.is_symlink() for node in (target, *target.parents) if node == workspace or workspace in node.parents):
        raise ValueError("source path contains a symlink")
    if not target.resolve(strict=True).is_relative_to(root) or not target.is_file() or _hash(target) != expected:
        raise ValueError("source hash or workspace boundary differs")
    return target


def validate_plan(plan: Any, workspace: Path, capture_tools: Path) -> dict:
    """Validate explicit, controller-reviewed capture inputs without approving them."""
    if not isinstance(plan, dict) or plan.get("schema_version") != SCHEMA or plan.get("state") != "review_required":
        raise ValueError("capture plan must remain review_required")
    if not isinstance(plan.get("registry_host"), str) or re.fullmatch(r"[a-z0-9.-]+(?::[0-9]+)?", plan["registry_host"]) is None:
        raise ValueError("capture registry host is invalid")
    if workspace.is_symlink() or not workspace.is_dir():
        raise ValueError("capture workspace is unavailable")
    if capture_tools.is_symlink() or not capture_tools.is_dir():
        raise ValueError("capture tools are unavailable")
    sources = plan.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise ValueError("capture source inventory is empty")
    source_by_path = {}
    for row in sources:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "visibility"} or row["visibility"] not in {"public", "private"} or row["path"] in source_by_path:
            raise ValueError("capture source binding is malformed")
        _source(workspace, row["path"], row["sha256"])
        source_by_path[row["path"]] = row
    private_assets = plan.get("private_assets")
    if not isinstance(private_assets, list):
        raise TypeError("reviewed private asset inventory is absent")
    private_paths: set[str] = set()
    for asset in private_assets:
        if not isinstance(asset, dict) or set(asset) != {"workspace_path", "image_path", "sha256"}:
            raise ValueError("reviewed private asset binding is malformed")
        source = source_by_path.get(asset["workspace_path"])
        path = asset["image_path"]
        if (source is None or source["visibility"] != "private" or source["sha256"] != asset["sha256"]
                or not isinstance(path, str) or not path.startswith("/") or path == "/"
                or any(part in {"", ".", ".."} for part in Path(path).parts[1:]) or path in private_paths):
            raise ValueError("reviewed private asset binding is invalid")
        private_paths.add(path)
    implementation = plan.get("capture_implementation")
    tools = {
        "capture_rootfs.py": "capture_rootfs_sha256",
        "in_sandbox_capture.py": "in_sandbox_capture_sha256",
        "dtx.py": "dtx_sha256",
        "cw_presign.py": "cw_presign_sha256",
    }
    if not isinstance(implementation, dict) or set(implementation) != set(tools.values()) | {"dt_sha256"}:
        raise ValueError("capture implementation is unbound")
    for name, key in tools.items():
        path = capture_tools / name
        if path.is_symlink() or not path.is_file() or _hash(path) != implementation[key]:
            raise ValueError("capture implementation changed")
    # The existing dtx.sh prepends its parent before lazily importing `dt`.
    # Bind that exact resolution as an explicit part of the staged toolchain.
    parent_dt = capture_tools.parent / "dt.py"
    if parent_dt.is_symlink() or not parent_dt.is_file() or _hash(parent_dt) != implementation["dt_sha256"]:
        raise ValueError("dtx parent dt dependency changed")
    tree = ast.parse((capture_tools / "capture_rootfs.py").read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "EXCLUDES" for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("capture exclusion policy is ambiguous")
    excludes = ast.literal_eval(assignments[0].value)
    required = REQUIRED_CAPTURE_EXCLUDES - {"./var/lib/postgresql/data/*"}
    if not isinstance(excludes, list) or not required.issubset(set(excludes)):
        raise ValueError("capture exclusion policy lacks provider floor")
    roles = plan.get("images")
    if not isinstance(roles, list) or not roles or len(roles) > 2:
        raise ValueError("capture role inventory is invalid")
    names: set[str] = set()
    repositories: set[str] = set()
    for image in roles:
        if not isinstance(image, dict) or image.get("role") not in {"candidate", "private_verifier"} or image["role"] in names:
            raise ValueError("capture role is invalid or duplicated")
        names.add(image["role"])
        repository = image.get("repository")
        if (not isinstance(repository, str) or re.fullmatch(r"capability-env-gen/[a-z0-9][a-z0-9._-]*", repository) is None
                or repository in repositories):
            raise ValueError("capture repository is invalid or duplicated")
        repositories.add(repository)
        if image.get("architecture") != "amd64" or image.get("operating_system") != "linux":
            raise ValueError("capture OCI platform is unsupported")
        snapshot = image.get("source_snapshot")
        if not isinstance(snapshot, dict) or not all(isinstance(snapshot.get(k), str) and snapshot[k] for k in ("name", "id", "ref")):
            raise ValueError("capture source snapshot is incomplete")
        if not isinstance(image.get("authored_image_pointer"), str) or not image["authored_image_pointer"]:
            raise ValueError("capture authored image pointer is absent")
        recipe = image.get("source_recipe")
        if not isinstance(recipe, dict) or set(recipe) != {"path", "sha256"}:
            raise ValueError("capture source recipe is malformed")
        _source(workspace, recipe["path"], recipe["sha256"])
        if recipe["path"] not in source_by_path or source_by_path[recipe["path"]]["sha256"] != recipe["sha256"]:
            raise ValueError("capture recipe is absent from source inventory")
        inputs = image.get("input_files")
        if not isinstance(inputs, list) or not inputs or recipe["path"] not in inputs or len(inputs) != len(set(inputs)) or any(path not in source_by_path for path in inputs):
            raise ValueError("capture role input mapping is incomplete")
        if image["role"] == "candidate" and any(source_by_path[path]["visibility"] != "public" for path in inputs):
            raise ValueError("candidate image maps private source input")
        config = image.get("image_config")
        if not isinstance(config, dict) or not {"Env", "WorkingDir", "User", "Entrypoint", "Cmd"}.issubset(config):
            raise ValueError("capture OCI config provenance is incomplete")
        _expected_env(config["Env"])
        if not isinstance(config["WorkingDir"], str) or not isinstance(config["User"], str):
            raise TypeError("capture OCI config is malformed")
        for key in ("Entrypoint", "Cmd"):
            if config[key] is not None and (not isinstance(config[key], list) or any(not isinstance(x, str) for x in config[key])):
                raise ValueError("capture OCI command is malformed")
        lifecycle = image.get("capture_lifecycle")
        if not isinstance(lifecycle, dict) or set(lifecycle) != {"ready_command", "ready_attempts", "ready_timeout_seconds", "ready_interval_seconds", "quiesce_command", "quiesce_probe", "exclude_mounts", "exclude_absent_paths"}:
            raise ValueError("capture lifecycle is incomplete")
        if not isinstance(lifecycle["ready_command"], str) or not lifecycle["ready_command"] or type(lifecycle["ready_attempts"]) is not int or not 1 <= lifecycle["ready_attempts"] <= 120 or type(lifecycle["ready_timeout_seconds"]) is not int or not 1 <= lifecycle["ready_timeout_seconds"] <= 300 or type(lifecycle["ready_interval_seconds"]) is not int or not 0 <= lifecycle["ready_interval_seconds"] <= 30:
            raise ValueError("capture readiness policy is invalid")
        for key in ("quiesce_command", "quiesce_probe"):
            if lifecycle[key] is not None and (not isinstance(lifecycle[key], str) or not lifecycle[key]):
                raise ValueError("capture quiesce policy is invalid")
        if (lifecycle["quiesce_command"] is None) != (lifecycle["quiesce_probe"] is None):
            raise ValueError("capture quiesce command and probe must be paired")
        mounts = lifecycle["exclude_mounts"]
        if not isinstance(mounts, list) or len(mounts) != len(set(mounts)) or any(not isinstance(p, str) or not p.startswith("/") or p == "/" or ".." in Path(p).parts for p in mounts):
            raise ValueError("capture excluded mount policy is invalid")
        absent = lifecycle["exclude_absent_paths"]
        if not isinstance(absent, list) or len(absent) != len(set(absent)) or any(not isinstance(p, str) or not p.startswith("/") or p == "/" or ".." in Path(p).parts for p in absent):
            raise ValueError("capture excluded absent-path policy is invalid")
        if set(mounts) & set(absent):
            raise ValueError("excluded task path cannot be both mounted and absent")
        if "./var/lib/postgresql/data/*" in excludes and "/var/lib/postgresql/data" not in set(mounts) | set(absent):
            raise ValueError("capture tool excludes PostgreSQL data without mount-or-absence proof")
        hashes = image.get("required_ready_hashes")
        if not isinstance(hashes, dict) or any(not isinstance(p, str) or not p.startswith("/") or not isinstance(h, str) or not _HEX.fullmatch(h) for p, h in hashes.items()):
            raise ValueError("capture ready-state hashes are invalid")
        if not hashes:
            raise ValueError("image ready-state hashes are required")
        resources = image.get("resources")
        if (not isinstance(resources, dict) or set(resources) != {"cpu", "memory", "disk"}
                or any(type(resources[key]) is not int or resources[key] <= 0 for key in resources)):
            raise ValueError("capture image resources are invalid")
    return plan


def _exit_code(dtx: Any, sandbox: Any, command: str, timeout: int) -> int | None:
    """The command's exit code, or None when the provider never answered.

    ``dt.run_in_sandbox`` reports a transport failure or a provider-side exec
    timeout as ``exit: None``.  That is not evidence about the image, so no check
    may treat it as a failed check.
    """
    code = dtx.sh(sandbox, command, timeout=timeout).get("exit")
    return code if isinstance(code, int) else None


# Exit codes that mean "killed by a deadline" rather than "the check answered no".
_TIMEOUT_EXITS = frozenset({124, 137})


def _require(dtx: Any, sandbox: Any, command: str, timeout: int, *, step: str, message: str,
             on_failure: str, detail: dict | None = None) -> None:
    code = _exit_code(dtx, sandbox, command, timeout)
    if code == 0:
        return
    unanswered = code is None or code in _TIMEOUT_EXITS
    raise CaptureError(message, step=step, failure_class=TRANSIENT if unanswered else on_failure,
                       detail={**(detail or {}), "exit_code": code})


def _reviewed_excludes(capture_tools: Path) -> list[str]:
    """The reviewed capture_rootfs.py EXCLUDES literal (validate_plan pins its bytes)."""
    tree = ast.parse((capture_tools / "capture_rootfs.py").read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "EXCLUDES" for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("capture exclusion policy is ambiguous")
    excludes = ast.literal_eval(assignments[0].value)
    if not isinstance(excludes, list) or any(not isinstance(item, str) for item in excludes):
        raise ValueError("capture exclusion policy is malformed")
    return excludes


def snapshot_identity_matches(receipt: Any, image: dict) -> bool:
    """Whether a capture receipt's source snapshot is the reviewed one.

    Daytona snapshot ids are stable provider identities, so all of name, id and
    ref must match.  A silo snapshot id is a per-broker-lifetime handle: the
    same reviewed recipe materialised by another deployment, or after a broker
    restart, gets a fresh random id while its name and ref (both derived from
    the name) are unchanged.  For a receipt that the silo capture wrote, the
    binding is therefore name + ref + the exact reviewed recipe bytes, which
    the capture verified against the provider's own echo of the Dockerfile.
    """
    if not isinstance(receipt, dict) or not isinstance(receipt.get("source_snapshot"), dict):
        return False
    observed = receipt["source_snapshot"]
    expected = image["source_snapshot"]
    if all(observed.get(key) == expected[key] for key in ("name", "id", "ref")):
        return True
    return (
        receipt.get("sandbox_provider") == "silo"
        and all(observed.get(key) == expected[key] for key in ("name", "ref"))
        and receipt.get("source_recipe_verified") is True
        and receipt.get("source_recipe_sha256") == image["source_recipe"]["sha256"]
    )


# --------------------------------------------------------------------------- #
# Environment: the reviewed image_config.Env against a fresh sandbox's own env.
# --------------------------------------------------------------------------- #

_SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASS|CRED|AUTH|COOKIE|SESSION", re.IGNORECASE)


def _parse_env(text: str) -> dict[str, str]:
    """``env`` output as a mapping, parsed exactly as capture_rootfs.py parses it."""
    return dict(line.split("=", 1) for line in text.strip().splitlines() if "=" in line)


def _dedupe_path(value: str) -> list[str]:
    seen: set[str] = set()
    entries = []
    for entry in value.split(":"):
        if entry not in seen:
            seen.add(entry)
            entries.append(entry)
    return entries


def _shown(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if _SECRET_NAME.search(name):
        return f"<redacted: {len(value)} chars, sha256 {hashlib.sha256(value.encode()).hexdigest()[:12]}>"
    return value if len(value) <= 300 else value[:300] + "...<truncated>"


def env_comparison(observed: dict[str, str], expected: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Reviewed Env entries the sandbox does not reproduce, and equivalences applied.

    Each reviewed name is compared against the sandbox's RAW environment.
    ``image_env`` filters runtime-provided names such as HOME before deciding
    which names are "additional"; applying that filter to reviewed names made a
    reviewed ``HOME=/root`` unmatchable even when the sandbox had exactly that.

    The one equivalence accepted is PATH with repeated entries removed: command
    lookup takes the first match in order, so a later duplicate can never be
    reached and the two values resolve every command identically (the official
    python images carry ``/usr/local/bin`` twice).  Anything else must match
    byte for byte, because the publisher writes the reviewed Env into the image.
    """
    mismatches: list[dict] = []
    equivalences: list[str] = []
    for name, want in sorted(expected.items()):
        got = observed.get(name)
        if got == want:
            continue
        if name == "PATH" and got is not None and _dedupe_path(got) == _dedupe_path(want):
            equivalences.append("PATH: repeated entries ignored (lookup order unchanged)")
            continue
        row: dict[str, Any] = {"name": name, "reviewed": _shown(name, want), "observed": _shown(name, got)}
        if got is not None and any(f"{other}=" in got for other in expected if other != name):
            row["hint"] = ("the provider set this variable to the rest of a multi-assignment ENV line "
                           "(silo parses `ENV A=1 B=2` as A='1 B=2'); write one `ENV NAME=value` "
                           "instruction per variable")
        elif got is None:
            row["hint"] = "not set in a fresh sandbox from the reviewed snapshot"
        else:
            row["hint"] = "the reviewed image_config.Env value differs from the snapshot's own environment"
        mismatches.append(row)
    return mismatches, equivalences


# --------------------------------------------------------------------------- #
# Sandbox deletion that is verified, and survives a slow or flaky broker.
# --------------------------------------------------------------------------- #

# About three minutes in total.  A delete that timed out at the client has
# usually been accepted by the provider; absence is what we verify, by lookup.
CLEANUP_DELAYS = (0, 3, 5, 10, 15, 20, 30, 30, 30, 37)
REAP_DELAYS = (0, 5, 10, 15)


def _not_found(error: BaseException) -> bool:
    return getattr(error, "status_code", None) == 404 or "not found" in str(error).lower()


def delete_and_verify(client: Any, sandbox_id: str, *, sandbox: Any = None,
                      sleeper: Any = time.sleep, delays: tuple = CLEANUP_DELAYS) -> tuple[bool, list[dict]]:
    """Delete ``sandbox_id`` and return (absence verified by lookup, observations).

    A failed delete is retried on the next tick instead of ending the check; a
    not-found answer to the delete means the sandbox is already gone (a TTL reap,
    or an earlier delete that the client timed out on) and is confirmed by lookup.
    """
    observations: list[dict] = []
    delete_pending = True
    for delay in delays:
        if delay:
            sleeper(delay)
        if delete_pending:
            try:
                target = sandbox if sandbox is not None else client.get(sandbox_id)
                target.delete()
                delete_pending = False
            except Exception as error:  # noqa: BLE001 - recorded by type only
                if _not_found(error):
                    delete_pending = False
                    observations.append({"state": "not_found_on_delete"})
                else:
                    observations.append({"state": "delete_error", "error_type": type(error).__name__})
        try:
            client.get(sandbox_id)
            observations.append({"state": "present"})
        except Exception as error:  # noqa: BLE001
            if _not_found(error):
                observations.append({"state": "not_found"})
                return True, observations
            observations.append({"state": "error", "error_type": type(error).__name__})
    return False, observations


def _reap_labelled(client: Any, label: str, sleeper: Any) -> dict:
    """After an ambiguous create failure, delete any sandbox that carries our label."""
    try:
        sandboxes = list(client.list())
    except Exception as error:  # noqa: BLE001
        return {"listed": False, "error_type": type(error).__name__}
    matches = [box for box in sandboxes if (getattr(box, "labels", None) or {}).get("envgen_capture") == label]
    results = []
    for box in matches:
        absent, _ = delete_and_verify(client, box.id, sandbox=box, sleeper=sleeper, delays=REAP_DELAYS)
        results.append({"sandbox_id": box.id, "absence_verified": absent})
    return {"listed": True, "matched": len(matches), "results": results}


# --------------------------------------------------------------------------- #
# Provisioning: snapshot, sandbox, harness-side clients.
# --------------------------------------------------------------------------- #

SANDBOX_CREATE_ATTEMPTS = 3
SNAPSHOT_WAIT_SECONDS = 3600
SNAPSHOT_POLL_SECONDS = 15
_SNAPSHOT_PENDING = ("pending", "building", "creating", "queued", "pulling")


def _provision(step: str, action: Any) -> Any:
    """Run a pre-sandbox step; any failure becomes a classified provision error."""
    try:
        return action()
    except CaptureError:
        raise
    except (Exception, SystemExit) as error:  # noqa: BLE001 - SystemExit: dt.py/cw_presign exit on missing credentials
        status = getattr(error, "status_code", None)
        detail: dict[str, Any] = {"status_code": status} if isinstance(status, int) else {}
        if isinstance(error, ImportError) and error.name:
            detail["missing_module"] = error.name
        raise CaptureProvisionError(
            f"{step.replace('_', ' ')} failed", step=step, failure_class=classify_exception(error),
            error_type=type(error).__name__, detail=detail,
        ) from None


def _recipe_base(recipe_text: str) -> str | None:
    for line in recipe_text.splitlines():
        words = line.split()
        if words and words[0].upper() == "FROM":
            refs = [word for word in words[1:] if not word.startswith("--")]
            return refs[0] if refs else None
    return None


def _snapshot_state(snapshot: Any) -> str:
    state = getattr(snapshot, "state", "")
    return str(getattr(state, "value", state)).lower()


def _create_snapshot(client: Any, name: str, recipe_text: str, resources: dict) -> None:
    from .daytona_snapshot import snapshot_conflict

    try:
        client.snapshot.create(
            {"name": name, "image": recipe_text,
             "resources": {"cpu": resources["cpu"], "memory": resources["memory"], "disk": resources["disk"]}},
            timeout=SNAPSHOT_WAIT_SECONDS,
        )
    except Exception as error:  # noqa: BLE001 - classified below, text never recorded
        if snapshot_conflict(error):
            return  # a concurrent same-name create; the caller waits for it
        status = getattr(error, "status_code", None)
        base = _recipe_base(recipe_text) or ""
        if base.startswith("silo.local/"):
            raise CaptureProvisionError(
                "the reviewed recipe builds FROM a provider-local snapshot image that no registry "
                "serves, so the snapshot cannot be re-materialized", step="snapshot",
                failure_class=CONTENT, error_type=type(error).__name__,
                detail={"base_image": base[:200], "status_code": status},
            ) from None
        if status == 422:
            raise CaptureProvisionError(
                "the provider refused the reviewed recipe", step="snapshot", failure_class=CONTENT,
                error_type=type(error).__name__, detail={"status_code": status},
            ) from None
        raise CaptureProvisionError(
            "source snapshot could not be re-materialized", step="snapshot",
            failure_class=classify_exception(error), error_type=type(error).__name__,
            detail={"status_code": status} if isinstance(status, int) else {},
        ) from None


def _silo_snapshot(client: Any, expected: dict, recipe_text: str, resources: dict, *,
                   sleeper: Any = time.sleep, clock: Any = time.monotonic) -> tuple[Any, bool]:
    """Resolve the reviewed silo snapshot, re-materialising it from the recipe if needed.

    silo keeps snapshots in broker memory and per deployment, so a relaunched
    job can meet a broker that never saw the builder's snapshot (the broker in
    construct-003 restarted at 05:26Z and lost every earlier one).  Only a
    not-found answer permits creation, under the reviewed name, from the exact
    reviewed recipe bytes and the reviewed resource profile.  A snapshot still
    building is waited for; one whose build failed, and that carries exactly the
    reviewed recipe, is rebuilt once.
    """
    from .daytona_snapshot import snapshot_not_found

    name = expected["name"]
    recreated = rebuilt = False
    deadline = clock() + SNAPSHOT_WAIT_SECONDS
    while True:
        try:
            snapshot = client.snapshot.get(name)
        except Exception as error:  # noqa: BLE001 - only a snapshot 404 permits creation
            if not snapshot_not_found(error):
                status = getattr(error, "status_code", None)
                raise CaptureProvisionError(
                    "source snapshot lookup failed", step="snapshot", failure_class=classify_exception(error),
                    error_type=type(error).__name__, detail={"status_code": status} if isinstance(status, int) else {},
                ) from None
            if recreated:
                raise CaptureProvisionError("re-materialized source snapshot is missing", step="snapshot") from None
            _create_snapshot(client, name, recipe_text, resources)
            recreated = True
            continue
        state = _snapshot_state(snapshot)
        if "active" in state:
            return snapshot, recreated
        if any(word in state for word in _SNAPSHOT_PENDING):
            if clock() >= deadline:
                raise CaptureProvisionError("source snapshot did not become active", step="snapshot",
                                            detail={"state": state[:40]})
            sleeper(SNAPSHOT_POLL_SECONDS)
            continue
        ours = getattr(getattr(snapshot, "build_info", None), "dockerfile_content", None) == recipe_text
        if rebuilt or not ours:
            base = _recipe_base(recipe_text) or ""
            raise CaptureProvisionError(
                "source snapshot is not usable", step="snapshot",
                failure_class=CONTENT if (not ours or base.startswith("silo.local/")) else TRANSIENT,
                detail={"state": state[:40], "reviewed_recipe": ours, "base_image": base[:200]},
            )
        rebuilt = True
        try:
            client.snapshot.delete(name)
        except Exception as error:  # noqa: BLE001
            if not snapshot_not_found(error):
                raise CaptureProvisionError("failed source snapshot could not be removed", step="snapshot",
                                            error_type=type(error).__name__) from None
        _create_snapshot(client, name, recipe_text, resources)
        recreated = True


def _retry_delay(error: BaseException, attempt: int) -> float:
    headers = getattr(error, "headers", None) or {}
    try:
        advertised = float({str(k).lower(): v for k, v in dict(headers).items()}.get("retry-after"))
    except (TypeError, ValueError):
        advertised = None
    delay = advertised if advertised is not None and advertised >= 0 else 30.0 * attempt
    return max(5.0, min(120.0, delay))


def _create_silo_sandbox(client: Any, snapshot_name: str, label: str, *, sleeper: Any,
                         clock: Any) -> tuple[Any, float, list[dict]]:
    """Create the network-blocked capture sandbox, retrying capacity and transport failures.

    The broker already waits up to its placement window (480 s) before
    answering 429, so a few attempts cover minutes of capacity pressure.  A POST
    that timed out may still have created a sandbox; every sandbox carrying this
    capture's label is deleted before the next attempt so a retry cannot leak one.
    """
    params = {"snapshot": snapshot_name, "network_block_all": True, "ttl_minutes": 120,
              "labels": {"envgen": "1", "envgen_purpose": "capability-image-capture", "envgen_capture": label}}
    attempts: list[dict] = []
    started = clock()
    last: BaseException | None = None
    for attempt in range(1, SANDBOX_CREATE_ATTEMPTS + 1):
        begun = clock()
        try:
            sandbox = client.create(dict(params), timeout=600)
        except Exception as error:  # noqa: BLE001 - classified by type and status only
            last = error
            status = getattr(error, "status_code", None)
            name = type(error).__name__
            row: dict[str, Any] = {"attempt": attempt, "error_type": name, "seconds": round(clock() - begun, 1)}
            if isinstance(status, int):
                row["status_code"] = status
            transport = name == "TransportError" or isinstance(error, (OSError, TimeoutError))
            ambiguous = (transport and not getattr(error, "refused", False)) or status == 408
            if ambiguous:
                row["labelled_sandboxes_reaped"] = _reap_labelled(client, label, sleeper)
            if status in (400, 401, 403) and not transport:
                attempts.append(row)
                raise CaptureProvisionError("the provider refused the capture sandbox request",
                                            step="sandbox_create", failure_class=HARNESS, error_type=name,
                                            detail={"status_code": status, "attempts": attempts}) from None
            if attempt < SANDBOX_CREATE_ATTEMPTS:
                row["retry_after_seconds"] = _retry_delay(error, attempt)
                attempts.append(row)
                sleeper(row["retry_after_seconds"])
            else:
                attempts.append(row)
            continue
        attempts.append({"attempt": attempt, "state": "created", "seconds": round(clock() - begun, 1)})
        return sandbox, round(clock() - started, 2), attempts
    status = getattr(last, "status_code", None)
    raise CaptureProvisionError(
        "capture sandbox could not be created", step="sandbox_create",
        failure_class=classify_exception(last) if last is not None else TRANSIENT,
        error_type=type(last).__name__ if last is not None else None,
        detail={"status_code": status if isinstance(status, int) else None, "attempts": attempts},
    )


def _archive_failure(raw: Any) -> dict:
    """The archive process facts worth keeping when an archive is rejected."""
    capture = raw.get("capture") if isinstance(raw, dict) and isinstance(raw.get("capture"), dict) else raw
    if not isinstance(capture, dict):
        return {}
    keys = ("tar_exit", "gzip_exit", "parts", "compressed_bytes", "seconds", "error")
    facts = {key: capture.get(key) for key in keys if key in capture}
    tail = capture.get("tar_stderr_tail")
    if isinstance(tail, str) and tail:
        facts["tar_stderr_tail"] = tail[-2000:]
    return facts


# --------------------------------------------------------------------------- #
# One capture.
# --------------------------------------------------------------------------- #


def _record_failure(receipt: dict, error: BaseException, step: str) -> None:
    summary = error_summary(error)
    receipt["failure_class"] = summary["failure_class"]
    receipt["error_type"] = summary["error_type"]
    receipt["failed_step"] = error.step if isinstance(error, CaptureError) else step
    if "error_message" in summary:
        receipt["error_message"] = summary["error_message"]
    if isinstance(error, PrivacyReviewRequired):
        receipt["state"] = "privacy_review_required"
        receipt["failed_step"] = "sensitive_paths"
    elif isinstance(error, ImageConfigMismatch):
        receipt["state"] = "image_config_mismatch"
        receipt["environment_mismatch"] = error.mismatches
    else:
        receipt["state"] = FAILED_STATE[summary["failure_class"]]
        if isinstance(error, CaptureError) and error.detail:
            receipt["failure_detail"] = error.detail


def capture_role(
    plan_path: Path, workspace: Path, role: str, output: Path, *,
    approved_plan_sha256: str, dtx: Any, capture_tools: Path,
    runner: Any = subprocess.run, sleeper: Any = time.sleep,
    provider: str | None = None, client_factory: Any = None,
    s3_factory: Any = None, silo_capture: Any = None,
    prior_sandbox_ids: tuple[str, ...] | list[str] = (), clock: Any = time.monotonic,
) -> dict:
    """Capture one fresh sandbox; approval is an external exact-hash decision.

    ``provider`` defaults to ``CAPABILITY_SANDBOX_PROVIDER``.  On Daytona the
    source sandbox reaches only the object store and pushes its own archive
    (``capture_rootfs.py``).  On silo the sandbox has no network; the archive
    leaves through silo's file API and the harness uploads it
    (``silo_rootfs_capture``).  Either way the sandbox holds no credential.

    A failure before a sandbox exists raises ``CaptureProvisionError`` and
    writes no receipt; from sandbox creation on, a receipt is always written,
    with ``failure_class`` set whenever the capture did not complete.
    ``prior_sandbox_ids`` are capture sandboxes of earlier attempts whose
    deletion was never verified; they are deleted and verified first.
    """
    from . import sandbox_provider

    provider = provider or sandbox_provider.provider()
    if provider not in sandbox_provider.PROVIDERS:
        raise ValueError("capture sandbox provider is unknown")
    if output.exists():
        raise ValueError("capture receipt already exists")
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, workspace, capture_tools)
    if _hash(plan_path) != approved_plan_sha256:
        raise ValueError("capture plan differs from external approval")
    images = [image for image in plan["images"] if image["role"] == role]
    if len(images) != 1:
        raise ValueError("capture role is absent")
    image = images[0]
    silo = provider == sandbox_provider.SILO
    snapshot_expected = image["source_snapshot"]
    recipe = _source(workspace, image["source_recipe"]["path"], image["source_recipe"]["sha256"])
    recipe_text = recipe.read_text()
    expected_env = _expected_env(image["image_config"]["Env"])

    client = _provision("provider_client", client_factory or dtx.client)
    s3 = None
    if silo:
        if s3_factory is None:
            raise CaptureProvisionError("silo capture needs a harness-side object-store client",
                                        step="object_store_client", failure_class=HARNESS)
        # Built before any sandbox exists: a missing harness dependency must not burn one.
        s3 = _provision("object_store_client", s3_factory)
    recreated = False
    if silo:
        snapshot, recreated = _silo_snapshot(client, snapshot_expected, recipe_text, image["resources"],
                                             sleeper=sleeper, clock=clock)
    else:
        snapshot = _provision("snapshot", lambda: client.snapshot.get(snapshot_expected["name"]))
    observed = {key: getattr(snapshot, key) for key in ("name", "id", "ref", "cpu", "mem", "disk")}
    observed["state"] = str(snapshot.state)
    identity_keys = ("name", "ref") if silo else ("name", "id", "ref")
    if any(observed[key] != snapshot_expected[key] for key in identity_keys):
        raise CaptureProvisionError("source snapshot identity changed", step="snapshot", failure_class=CONTENT,
                                    detail={"differs": [key for key in identity_keys if observed[key] != snapshot_expected[key]]})
    if "ACTIVE" not in observed["state"].upper():
        raise CaptureProvisionError("source snapshot state is not active", step="snapshot",
                                    detail={"state": observed["state"][:40]})
    if getattr(getattr(snapshot, "build_info", None), "dockerfile_content", None) != recipe_text:
        raise CaptureProvisionError("source snapshot recipe changed", step="snapshot", failure_class=CONTENT)

    prior = []
    for sandbox_id in prior_sandbox_ids:
        absent, observations = delete_and_verify(client, sandbox_id, sleeper=sleeper, delays=REAP_DELAYS)
        prior.append({"sandbox_id": sandbox_id, "absence_verified": absent, "observations": observations})

    label = uuid.uuid4().hex
    if silo:
        sandbox, seconds, create_attempts = _create_silo_sandbox(client, snapshot_expected["name"], label,
                                                                 sleeper=sleeper, clock=clock)
    else:
        sandbox, seconds = _provision("sandbox_create", lambda: dtx.create(
            client, snapshot_expected["name"], purpose="capability-image-capture", ttl_min=120, domains=CW_HOST))
        create_attempts = [{"attempt": 1, "state": "created", "seconds": seconds}]
    receipt = {"schema_version": "capability-rootfs-capture-v1", "state": "infrastructure_error", "role": role, "plan_sha256": approved_plan_sha256, "sandbox_provider": provider, "source_snapshot": observed, "source_recipe_sha256": image["source_recipe"]["sha256"], "source_recipe_verified": True, "image_config_sha256": _json_hash(image["image_config"]), "input_files": {path: next(row["sha256"] for row in plan["source_files"] if row["path"] == path) for path in image["input_files"]}, "sandbox_id": sandbox.id, "sandbox_create_seconds": seconds, "sandbox_create_attempts": create_attempts, "capture_label": label, "failure_class": None, "capture": None, "cleanup": None}
    if prior:
        receipt["prior_sandbox_cleanup"] = prior
    if silo:
        receipt["source_snapshot_reviewed"] = dict(snapshot_expected)
        receipt["source_snapshot_rematerialized"] = recreated
        receipt["network_block_all"] = getattr(sandbox, "network_block_all", None)
    lifecycle = image["capture_lifecycle"]
    step = "readiness"
    try:
        exits: list[int | None] = []
        ready = False
        for attempt in range(1, lifecycle["ready_attempts"] + 1):
            code = _exit_code(dtx, sandbox, lifecycle["ready_command"], lifecycle["ready_timeout_seconds"])
            exits.append(code)
            if code == 0:
                ready = True
                break
            if attempt < lifecycle["ready_attempts"]:
                sleeper(lifecycle["ready_interval_seconds"])
        receipt["ready"] = {"attempts": attempt, "ready": ready, "exit_codes": exits[-10:]}
        if not ready:
            answered = [code for code in exits if code is not None and code not in _TIMEOUT_EXITS]
            raise CaptureError("capture sandbox did not become ready", step="readiness",
                               failure_class=CONTENT if answered else TRANSIENT,
                               detail={"exit_codes": exits[-10:]})
        step = "ready_hashes"
        for path, expected in image["required_ready_hashes"].items():
            result = dtx.sh(sandbox, "sha256sum -- " + shlex.quote(path), timeout=60)
            code = result.get("exit")
            if not isinstance(code, int):
                raise CaptureError("ready-state file hash could not be read", step=step)
            if code != 0 or (result.get("stdout") or "").split(maxsplit=1)[0:1] != [expected]:
                raise CaptureError("ready-state file hash differs", step=step, failure_class=CONTENT,
                                   detail={"path": path, "exit_code": code})
        step = "exclude_mounts"
        for path in lifecycle["exclude_mounts"]:
            command = f"test \"$(stat -c %d /)\" != \"$(stat -c %d {shlex.quote(path)})\" && findmnt -T {shlex.quote(path)} -n >/dev/null"
            _require(dtx, sandbox, command, 60, step=step, message="excluded state is not on a distinct mount",
                     on_failure=CONTENT, detail={"path": path})
        receipt["excluded_mounts_checked"] = lifecycle["exclude_mounts"]
        step = "exclude_absent_paths"
        for path in lifecycle["exclude_absent_paths"]:
            _require(dtx, sandbox, "test ! -e " + shlex.quote(path), 60, step=step,
                     message="excluded task path exists in capture source", on_failure=CONTENT, detail={"path": path})
        receipt["excluded_absent_paths_checked"] = lifecycle["exclude_absent_paths"]
        step = "provider_bound_mounts"
        if silo:
            from . import silo_rootfs_capture

            if receipt["network_block_all"] is not True:
                raise CaptureError("capture sandbox network isolation is unconfirmed", step=step, failure_class=HARNESS)
            # A distinct device is exactly what makes tar --one-file-system drop it.
            command = " && ".join(f'test "$(stat -c %d /)" != "$(stat -c %d {shlex.quote(p)})"' for p in silo_rootfs_capture.PROVIDER_BOUND_PATHS)
        else:
            command = "for p in " + " ".join(shlex.quote(p) for p in sorted(PROVIDER_BOUND_PATHS)) + '; do findmnt -T "$p" -n >/dev/null || exit 1; done'
        # A failure here means our model of the provider's mounts is wrong: our bug.
        _require(dtx, sandbox, command, 60, step=step, message="provider-bound mount check failed", on_failure=HARNESS)
        receipt["provider_bound_mounts_checked"] = True
        step = "sensitive_paths"
        forbidden = set(_CREDENTIAL_ROOTS)
        if role == "candidate":
            forbidden.update(_PRIVATE_ROOTS)
            forbidden.update(asset["image_path"] for asset in plan["private_assets"])
        checked = sorted(forbidden)
        code = _exit_code(dtx, sandbox, " && ".join("test ! -e " + shlex.quote(path) for path in checked), 120)
        if code is None or code in _TIMEOUT_EXITS:
            # An unanswered scan is not a privacy finding.
            raise CaptureError("sensitive-path scan did not complete", step=step, detail={"exit_code": code})
        if code != 0:
            raise PrivacyReviewRequired("image sensitive-path scan failed")
        receipt["sensitive_paths_checked"] = checked
        if role == "candidate":
            receipt["candidate_path_scan_passed"] = True
            receipt["candidate_private_paths_checked"] = checked
        step = "environment"
        # Checked before the archive: a mismatch is deterministic, and finding it
        # after streaming the whole rootfs wasted the sandbox and the upload.
        probe = dtx.sh(sandbox, "env", timeout=60)
        if not isinstance(probe.get("exit"), int):
            raise CaptureError("source environment probe did not complete", step=step)
        if probe["exit"] != 0:
            raise CaptureError("source environment probe failed", step=step, failure_class=HARNESS,
                               detail={"exit_code": probe["exit"]})
        mismatches, equivalences = env_comparison(_parse_env(probe.get("stdout") or ""), expected_env)
        if mismatches:
            raise ImageConfigMismatch(mismatches)
        receipt["environment_precheck"] = {"reviewed_values_matched": True, "equivalences": equivalences}
        step = "quiesce"
        if lifecycle["quiesce_command"] is not None:
            _require(dtx, sandbox, lifecycle["quiesce_command"], 300, step=step, message="capture quiesce failed", on_failure=CONTENT)
            _require(dtx, sandbox, lifecycle["quiesce_probe"], 60, step=step, message="capture quiesce probe failed", on_failure=CONTENT)
            receipt["quiesce_passed"] = True
        step = "archive_transport"
        # The transport staging key combines tag with second-resolution
        # time. Distinct tasks/attempts must not overwrite each other's
        # multipart object before the content-addressed copy completes.
        capture_tag = f"cap-{role}-{uuid.uuid4().hex}"
        receipt["capture_tag"] = capture_tag
        if silo:
            receipt["rootfs_transport"] = silo_rootfs_capture.transport_identity()
            try:
                raw = (silo_capture or silo_rootfs_capture.capture_rootfs)(
                    sandbox=sandbox, sh=dtx.sh, snapshot_name=snapshot_expected["name"], tag=capture_tag,
                    excludes=_reviewed_excludes(capture_tools) + list(silo_rootfs_capture.PROVIDER_EXCLUDES),
                    s3=s3, sleeper=sleeper,
                )
            except silo_rootfs_capture.SiloCaptureError as error:
                detail = {"kind": error.kind, **error.detail}
                if error.result:
                    detail["archive_failure"] = _archive_failure(error.result)
                raise CaptureError(str(error), step=step, failure_class=error.failure_class,
                                   detail=detail, error_type="SiloCaptureError") from None
            if raw.get("stream_helper"):
                receipt["rootfs_transport"]["stream_helper"] = raw["stream_helper"]
        else:
            with tempfile.TemporaryDirectory(prefix="task-image-capture-") as temporary:
                raw_path = Path(temporary) / "raw.json"
                completed = runner([sys.executable, str(capture_tools / "capture_rootfs.py"), "--snapshot", snapshot_expected["name"], "--sandbox", sandbox.id, "--tag", capture_tag, "--out", str(raw_path)], capture_output=True, text=True, timeout=4200, check=False)
                if completed.returncode != 0 or not raw_path.is_file():
                    raise CaptureError("rootfs capture failed", step=step, detail={"exit_code": completed.returncode})
                raw = json.loads(raw_path.read_text())
        step = "environment"
        raw_env = raw.pop("source_env", {})
        mismatches, equivalences = env_comparison(raw_env, expected_env)
        if mismatches:
            raise ImageConfigMismatch(mismatches)
        receipt["environment"] = {"reviewed_values_matched": True, "reviewed_names": sorted(expected_env), "additional_filtered_names": sorted(set(image_env(raw_env)) - set(expected_env)), "raw_source_environment_retained": False, "equivalences": equivalences}
        step = "archive_process"
        try:
            receipt["archive_process"] = _validate_archive_receipt(raw)
        except (RuntimeError, TypeError) as error:
            raise CaptureError("rootfs archive process failed", step=step,
                               detail={"check": str(error)[:200], "archive_failure": _archive_failure(raw)}) from None
        receipt["capture"] = _sanitize_legacy_receipt(raw)
        receipt["state"] = "captured_pending_privacy_and_publication"
    except Exception as error:  # noqa: BLE001 - preserve cleanup evidence, never log provider text.
        _record_failure(receipt, error, step)
    finally:
        absent, observations = delete_and_verify(client, sandbox.id, sandbox=sandbox, sleeper=sleeper)
        receipt["cleanup"] = {"delete_requested": True, "absence_verified": absent, "observations": observations}
        if not absent:
            receipt["state_before_cleanup"] = receipt["state"]
            receipt["state"] = "cleanup_unverified"
            if receipt.get("failure_class") is None:
                receipt["failure_class"] = TRANSIENT
                receipt["failed_step"] = "cleanup"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2)
            stream.write("\n")
    return receipt
