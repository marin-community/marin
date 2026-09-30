"""Cold-boot a reviewed OCI digest in two fresh isolated Daytona sandboxes."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .daytona_snapshot import (
    snapshot_not_found,
    validate_snapshot_recipe,
    wait_for_sandbox_deletion,
)
from .generic_image_capture import _CREDENTIAL_ROOTS, _PRIVATE_ROOTS, validate_plan
from .generic_image_publication import published_registry_host, validate_review

SCHEMA = "capability-generic-image-cold-pull-v1"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _recipe(reference: str, config: dict) -> str:
    if not isinstance(reference, str) or "@sha256:" not in reference:
        raise ValueError("cold pull requires a canonical digest reference")
    recipe = f"FROM {reference}\n"
    for key in ("Entrypoint", "Cmd"):
        if config[key] is not None:
            recipe += key.upper() + " " + json.dumps(config[key], separators=(",", ":")) + "\n"
    return recipe


def _checked_publication(plan: dict, plan_path: Path, publication_path: Path) -> tuple[dict, str]:
    publication = json.loads(publication_path.read_text())
    matches = [image for image in plan["images"] if image["role"] == publication.get("role")]
    if len(matches) != 1:
        raise ValueError("cold pull publication role is ambiguous")
    image = matches[0]
    transport = publication.get("publication", {})
    reference = transport.get("image")
    try:
        # A recorded object-store -> registry correction is honoured only when it
        # exactly matches what the publisher is allowed to correct.
        host = published_registry_host(plan, publication)
    except (TypeError, ValueError) as error:
        raise ValueError("cold pull publication is incomplete or differs from reviewed image") from error
    prefix = host + "/" + image["repository"] + "@"
    if (publication.get("schema_version") != "capability-task-image-publication-v1"
            or publication.get("state") != "published_pending_cold_pull"
            or publication.get("plan_sha256") != _sha(plan_path)
            or publication.get("rootfs_review", {}).get("state") != "passed"
            or publication["rootfs_review"].get("required_file_hashes") != image["required_ready_hashes"]
            or transport.get("state") != "integrity_verified"
            or not isinstance(reference, str) or not reference.startswith(prefix)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", reference.rpartition("@")[2]) is None
            or reference.rpartition("@")[2] != transport.get("manifest_digest")):
        raise ValueError("cold pull publication is incomplete or differs from reviewed image")
    return image, reference


def cold_pull(
    *, plan_path: Path, workspace: Path, capture_tools: Path,
    approval_path: Path, builder_session_ids: set[str], publication_path: Path,
    dtx: Any, create_snapshot: Callable[[Any, str, str, dict], None],
    sleeper: Callable[[float], None] = time.sleep,
) -> dict:
    """Verify portable launch/content; full task gates remain a separate step."""
    plan = json.loads(plan_path.read_text())
    validate_plan(plan, workspace, capture_tools)
    validate_review(approval_path, plan_path, builder_session_ids=builder_session_ids)
    image, reference = _checked_publication(plan, plan_path, publication_path)
    recipe = _recipe(reference, image["image_config"])
    name = "cap-cold-" + hashlib.sha256(recipe.encode()).hexdigest()[:24]
    client = dtx.client()
    receipt = {
        "schema_version": SCHEMA,
        "state": "pending",
        "role": image["role"],
        "image": reference,
        "plan_sha256": _sha(plan_path),
        "approval_sha256": _sha(approval_path),
        "publication_sha256": _sha(publication_path),
        "reconstruction_recipe": recipe,
        "sandboxes": [],
        "cleanup": [],
        "task_gates": "pending",
    }
    try:
        try:
            snapshot = client.snapshot.get(name)
            receipt["snapshot_created"] = False
        except Exception as error:  # Only a snapshot 404 permits creation.
            if not snapshot_not_found(error):
                raise
            create_snapshot(client, name, recipe, image["resources"])
            snapshot = client.snapshot.get(name)
            receipt["snapshot_created"] = True
        receipt["snapshot"] = {
            **validate_snapshot_recipe(snapshot, expected_name=name, expected_dockerfile=recipe),
            "id": str(snapshot.id), "ref": str(snapshot.ref),
        }
        state = getattr(snapshot.state, "value", snapshot.state)
        if str(state).lower() not in {"active", "snapshotstate.active"}:
            raise RuntimeError("cold image snapshot is not active")
        for index in range(2):
            sandbox, seconds = dtx.create(client, name, purpose="capability-generic-cold-pull", ttl_min=60, block_all=True)
            record = {"sandbox_id": sandbox.id, "create_seconds": seconds, "network_block_all": None, "ready": False, "required_file_hashes": {}}
            receipt["sandboxes"].append(record)
            try:
                observed = client.get(sandbox.id)
                record["network_block_all"] = getattr(observed, "network_block_all", None)
                if record["network_block_all"] is not True:
                    raise RuntimeError("cold sandbox network isolation is unconfirmed")
                lifecycle = image["capture_lifecycle"]
                for attempt in range(1, lifecycle["ready_attempts"] + 1):
                    ready = dtx.sh(sandbox, lifecycle["ready_command"], timeout=lifecycle["ready_timeout_seconds"])["exit"] == 0
                    if ready:
                        break
                    if attempt < lifecycle["ready_attempts"]:
                        sleeper(lifecycle["ready_interval_seconds"])
                record["ready_attempts"] = attempt
                record["ready"] = ready
                if not ready:
                    raise RuntimeError("cold sandbox did not become ready")
                for path, expected in image["required_ready_hashes"].items():
                    result = dtx.sh(sandbox, "sha256sum -- " + shlex.quote(path), timeout=60)
                    if result["exit"] != 0 or (result.get("stdout") or "").split(maxsplit=1)[0:1] != [expected]:
                        raise RuntimeError("cold image content differs from reviewed files")
                    record["required_file_hashes"][path] = expected
                forbidden = set(_CREDENTIAL_ROOTS)
                if image["role"] == "candidate":
                    forbidden.update(_PRIVATE_ROOTS)
                    forbidden.update(asset["image_path"] for asset in plan["private_assets"])
                checked = sorted(forbidden)
                command = " && ".join("test ! -e " + shlex.quote(path) for path in checked)
                if dtx.sh(sandbox, command, timeout=120)["exit"] != 0:
                    raise RuntimeError("cold image sensitive-path check failed")
                record["sensitive_paths_checked"] = checked
            finally:
                try:
                    sandbox.delete()
                except Exception as error:  # noqa: BLE001
                    record["delete_error_type"] = type(error).__name__
                state, observations = wait_for_sandbox_deletion(client, sandbox.id, sleeper=sleeper)
                receipt["cleanup"].append({"sandbox_id": sandbox.id, "state": state, "observations": observations})
                if state != "not_found":
                    raise RuntimeError("cold sandbox deletion is unconfirmed")
        ids = [row["sandbox_id"] for row in receipt["sandboxes"]]
        if len(set(ids)) != 2:
            raise RuntimeError("cold boots reused a sandbox")
        receipt["state"] = "passed_pending_task_gates"
    except Exception as error:  # noqa: BLE001 - preserve full completed evidence.
        receipt["error_type"] = type(error).__name__
    return receipt
