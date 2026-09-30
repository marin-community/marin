"""Validate reviewed inputs for task-image capture and publication.

This module is credential-free and performs no provider, object-store, or
registry operations.  A trusted publication worker consumes a validated
contract only after its review blockers have been resolved.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


class PublicationContractError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_publication_contract(document: Any, *, workspace: Path) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("schema_version") != "capability-image-publication-plan-v1":
        raise PublicationContractError("wrong publication contract schema")
    if document.get("state") not in {"review_required", "approved_for_capture"}:
        raise PublicationContractError("invalid publication contract state")
    blockers = document.get("review_blockers")
    if not isinstance(blockers, list) or any(not isinstance(x, str) or not x for x in blockers):
        raise PublicationContractError("review blockers must be explicit strings")
    if document["state"] == "approved_for_capture" and blockers:
        raise PublicationContractError("capture cannot be approved with unresolved blockers")

    sources = document.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise PublicationContractError("source files are required")
    for source in sources:
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise PublicationContractError("source file binding is malformed")
        path = Path(source["path"])
        if path.is_absolute() or ".." in path.parts:
            raise PublicationContractError("source path escapes workspace")
        target = workspace / path
        if not target.is_file() or _sha256(target) != source["sha256"]:
            raise PublicationContractError(f"source hash mismatch: {path}")

    roles = document.get("images")
    if not isinstance(roles, list) or {x.get("role") for x in roles if isinstance(x, dict)} != {"candidate", "private_verifier"}:
        raise PublicationContractError("candidate and private verifier images are both required")
    repositories: set[str] = set()
    for image in roles:
        if image.get("repository") in repositories or re.fullmatch(
            r"capability-env-gen/[a-z0-9][a-z0-9._-]*", image.get("repository", "")
        ) is None:
            raise PublicationContractError("invalid or reused publication repository")
        repositories.add(image["repository"])
        snapshot = image.get("source_snapshot")
        if not isinstance(snapshot, dict) or re.fullmatch(
            r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", snapshot.get("id", "")
        ) is None or not snapshot.get("name") or not snapshot.get("ref"):
            raise PublicationContractError("source snapshot identity is incomplete")
        recipe = image.get("source_recipe")
        if not isinstance(recipe, dict) or set(recipe) != {"path", "sha256"}:
            raise PublicationContractError("source recipe binding is malformed")
        recipe_path = Path(recipe["path"])
        target_recipe = recipe_path if recipe_path.is_absolute() else workspace / recipe_path
        if (
            not target_recipe.is_file()
            or re.fullmatch(r"[0-9a-f]{64}", recipe.get("sha256", "")) is None
            or _sha256(target_recipe) != recipe["sha256"]
        ):
            raise PublicationContractError("source recipe hash mismatch")
        config = image.get("image_config")
        if not isinstance(config, dict) or not {"Env", "WorkingDir", "User", "Entrypoint", "Cmd"}.issubset(config):
            raise PublicationContractError("image config provenance is incomplete")
        if not isinstance(config["Env"], list) or any(not isinstance(x, str) for x in config["Env"]):
            raise PublicationContractError("image Env must be an explicit string list")
        if not isinstance(config["WorkingDir"], str) or not isinstance(config["User"], str):
            raise PublicationContractError("image WorkingDir and User must be explicit")
        for key in ("Entrypoint", "Cmd"):
            if config[key] is not None and (
                not isinstance(config[key], list) or any(not isinstance(x, str) for x in config[key])
            ):
                raise PublicationContractError(f"invalid image config {key}")
        lifecycle = image.get("capture_lifecycle")
        required = {"ready_probe", "shutdown", "shutdown_probe", "reset_validation"}
        if not isinstance(lifecycle, dict) or set(lifecycle) != required or any(
            not isinstance(lifecycle[key], str) or not lifecycle[key] for key in required
        ):
            raise PublicationContractError("capture lifecycle is incomplete")
        boundary = image.get("privacy_boundary")
        if not isinstance(boundary, dict) or not isinstance(boundary.get("reject_paths"), list):
            raise PublicationContractError("privacy boundary is incomplete")
        if image["role"] == "candidate" and not boundary["reject_paths"]:
            raise PublicationContractError("candidate image requires private-path rejection")
        hashes = image.get("required_ready_hashes", {})
        if not isinstance(hashes, dict) or any(
            not isinstance(path, str)
            or not path.startswith("/")
            or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            for path, checksum in hashes.items()
        ):
            raise PublicationContractError("ready-state hashes are malformed")
        if image["role"] == "candidate" and not hashes:
            raise PublicationContractError("candidate ready-state hashes are required")
    return document
