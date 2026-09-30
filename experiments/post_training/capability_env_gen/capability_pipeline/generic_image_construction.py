"""Freeze authored custom-image requests before independent GLM review."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .generic_image_capture import _source, validate_plan

# TaskCompendium also permits Docker Hub shorthand and tag+digest references.
# Requiring a registry hostname would misclassify pinned public base images as
# custom snapshots and force an unnecessary image build/publication.
_CANONICAL = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}\Z")
_DAYTONA_RECIPE_NAMESPACE = "envgen.daytona/"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("authored image input is missing or linked")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("authored image input must be an object")
    return value


def image_pointers(task: Path) -> list[dict[str, str]]:
    """Inventory candidate, verifier, and composite machine image pointers."""
    specification = _read(task / "specification.json")
    binding = _read(task / "binding.json")
    rows: list[dict[str, str]] = []

    def add(role: str, pointer: str, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, str) or not value:
            raise ValueError(f"invalid image pointer: {pointer}")
        rows.append({"role": role, "pointer": pointer, "image": value})

    requirements = specification.get("requirements", {})
    state = requirements.get("state", {}) if isinstance(requirements, dict) else {}
    if isinstance(state, dict):
        add("candidate", "specification.requirements.state.image", state.get("image"))
    environment = binding.get("environment", {})
    if isinstance(environment, dict):
        add("candidate", "binding.environment.image", environment.get("image"))
    steps = specification.get("steps", [])
    if not isinstance(steps, list):
        raise TypeError("task steps are malformed")
    for index, step in enumerate(steps):
        verifier = step.get("verifier", {}) if isinstance(step, dict) else {}
        runtime = verifier.get("runtime", {}) if isinstance(verifier, dict) else {}
        if isinstance(runtime, dict):
            add("private_verifier", f"specification.steps.{index}.verifier.runtime.image", runtime.get("image"))
        inner = verifier.get("verifier", {}) if isinstance(verifier, dict) else {}
        inner_runtime = inner.get("runtime", {}) if isinstance(inner, dict) else {}
        if isinstance(inner_runtime, dict):
            add("private_verifier", f"specification.steps.{index}.verifier.verifier.runtime.image", inner_runtime.get("image"))
    composite = task / "composite-verifier.json"
    if composite.is_file():
        config = _read(composite)
        for step_index, step in enumerate(config.get("steps", [])):
            if not isinstance(step, dict):
                raise TypeError("composite step is malformed")
            for check_index, check in enumerate(step.get("machine_checks", [])):
                if not isinstance(check, dict):
                    raise TypeError("composite machine check is malformed")
                add("private_verifier", f"composite.steps.{step_index}.machine_checks.{check_index}.image", check.get("image"))
    return rows


def custom_pointers(pointers: list[dict[str, str]], requested: set[str]) -> list[dict[str, str]]:
    """The pointers that need the reviewed capture/publication/migration path.

    The single classification shared by request_needed (entry) and
    generic_image_migration (exit): a pointer the builder requested, or in the
    builder's envgen.daytona/<snapshot>/dockerfile@sha256:<recipe> form, is
    custom even though it is digest-shaped.  That form fingerprints Dockerfile
    bytes, not an OCI manifest at a pullable registry, so its digest-shaped
    suffix must never bypass review/capture/publication -- nor migration.
    """
    return [
        row for row in pointers
        if row["image"].startswith(_DAYTONA_RECIPE_NAMESPACE)
        or row["image"] in requested
        or _CANONICAL.fullmatch(row["image"]) is None
    ]


def request_needed(workspace: Path, *, migrated_pointers: frozenset[str] | set[str] = frozenset()) -> dict:
    """Return exact custom pointers without changing the builder workspace.

    ``migrated_pointers`` are authored pointers that an applied image migration
    receipt already replaced with published digests.  The builder's
    image-capture-request.json still names them after migration (migration
    only rewrites the task documents), so they no longer bind a task pointer
    and must not be requested again.  The caller still proves the migrated
    documents match the receipt byte for byte.
    """
    task = workspace / "task"
    pointers = image_pointers(task)
    authored_request = task / "image-capture-request.json"
    requested: set[str] = set()
    if authored_request.exists() or authored_request.is_symlink():
        images = _read(authored_request).get("images")
        if (
            not isinstance(images, list)
            or not images
            or any(
                not isinstance(image, dict)
                or not isinstance(image.get("authored_image_pointer"), str)
                or not image["authored_image_pointer"]
                for image in images
            )
        ):
            raise ValueError("image capture request has no image roles")
        requested = {image["authored_image_pointer"] for image in images}
        current = {row["image"] for row in pointers}
        if migrated_pointers and requested <= set(migrated_pointers) and not requested & current:
            # Every requested pointer was replaced by an applied migration.
            requested = set()
        elif not requested.intersection(current):
            raise ValueError("image capture request does not bind task pointers")
    custom = custom_pointers(pointers, requested)
    return {"needed": bool(custom), "pointers": pointers, "custom_pointers": custom}


def freeze_request(workspace: Path, output: Path, capture_tools: Path) -> dict:
    """Copy authored source bytes and produce an immutable review plan."""
    task = workspace / "task"
    inventory = request_needed(workspace)
    custom = inventory["custom_pointers"]
    request_path = task / "image-capture-request.json"
    if not custom:
        return {"state": "not_required", "pointers": inventory["pointers"]}
    request = _read(request_path)
    if request.get("state") != "review_required":
        raise ValueError("builder image request must remain review_required")
    role_refs: dict[str, set[str]] = {}
    for row in custom:
        role_refs.setdefault(row["role"], set()).add(row["image"])
    if any(len(values) != 1 for values in role_refs.values()):
        raise ValueError("multiple custom images for one role need separate reviewed plans")
    images = request.get("images")
    if not isinstance(images, list) or {row.get("role") for row in images if isinstance(row, dict)} != set(role_refs):
        raise ValueError("image request roles differ from custom task pointers")
    for image in images:
        if image.get("authored_image_pointer") != next(iter(role_refs[image["role"]])):
            raise ValueError("custom task pointer differs from capture request")
    if "capture_implementation" in request:
        raise ValueError("builder cannot supply capture implementation identity")
    tool_names = {
        "capture_rootfs.py": "capture_rootfs_sha256",
        "in_sandbox_capture.py": "in_sandbox_capture_sha256",
        "dtx.py": "dtx_sha256",
        "cw_presign.py": "cw_presign_sha256",
    }
    implementation = {key: _sha(capture_tools / name) for name, key in tool_names.items()}
    implementation["dt_sha256"] = _sha(capture_tools.parent / "dt.py")
    plan = {**request, "capture_implementation": implementation}
    validate_plan(plan, workspace, capture_tools)
    attempt = output
    frozen = output / "input/workspace"
    if not output.exists():
        frozen.mkdir(parents=True)
        for row in request["source_files"]:
            source = _source(workspace, row["path"], row["sha256"])
            target = frozen / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (attempt / "input/request.json").write_bytes(request_path.read_bytes())
        (attempt / "input/plan.json").write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    plan_path = attempt / "input/plan.json"
    if _read(plan_path) != plan or _sha(attempt / "input/request.json") != _sha(request_path):
        raise ValueError("frozen image capture request changed")
    validate_plan(plan, frozen, capture_tools)
    return {
        "state": "review_required",
        "plan_path": str(plan_path),
        "plan_sha256": _sha(plan_path),
        "workspace": str(frozen),
        "request_sha256": _sha(request_path),
        "attempt": str(attempt),
        "pointers": custom,
    }


def prepare_construction_capture(item_root: Path, capture_tools: Path) -> dict:
    """Choose a source-keyed attempt and freeze its exact GLM review input."""
    workspace = item_root / "workspace"
    inventory = request_needed(workspace)
    if not inventory["needed"]:
        return {"state": "not_required", "pointers": inventory["pointers"]}
    request = _read(workspace / "task/image-capture-request.json")
    key = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    output = item_root / "diagnostics/image-capture" / f"attempt-{key}"
    return freeze_request(workspace, output, capture_tools)
