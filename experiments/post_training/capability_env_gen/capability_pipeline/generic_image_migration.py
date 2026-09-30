"""Apply reviewed OCI digests to exact authored task image pointers."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from .generic_image_capture import validate_plan
from .generic_image_cold_pull import SCHEMA as COLD_SCHEMA
from .generic_image_cold_pull import _checked_publication, _recipe
from .generic_image_construction import custom_pointers, image_pointers
from .generic_image_publication import validate_review
from .image_review_contract import validate_retained_packet

SCHEMA = "capability-generic-image-migration-v1"
_DIGEST = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}\Z")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("migration evidence is missing or linked")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError("migration evidence must be an object")
    return value


def _set_pointer(document: dict, parts: list[str], old: str, new: str) -> None:
    current = document
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    if not isinstance(current, dict) or current.get(parts[-1]) != old:
        raise ValueError("authored pointer changed before migration")
    current[parts[-1]] = new


def _verified_cold(cold: dict, *, plan_path: Path, approval_path: Path,
                   publication_path: Path, image: dict, reference: str) -> None:
    if (cold.get("schema_version") != COLD_SCHEMA
            or cold.get("state") != "passed_pending_task_gates"
            or cold.get("role") != image["role"] or cold.get("image") != reference
            or cold.get("plan_sha256") != _sha(plan_path)
            or cold.get("approval_sha256") != _sha(approval_path)
            or cold.get("publication_sha256") != _sha(publication_path)
            or cold.get("reconstruction_recipe") != _recipe(reference, image["image_config"])):
        raise ValueError("cold-pull evidence differs from reviewed publication")
    sandboxes = cold.get("sandboxes")
    cleanup = cold.get("cleanup")
    if (not isinstance(sandboxes, list) or len(sandboxes) != 2
            or not isinstance(cleanup, list) or len(cleanup) != 2):
        raise ValueError("two cold sandboxes and cleanup receipts are required")
    ids = [row.get("sandbox_id") for row in sandboxes if isinstance(row, dict)]
    if (len(ids) != 2 or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != 2):
        raise ValueError("cold sandbox identities are incomplete or reused")
    for row in sandboxes:
        if (row.get("network_block_all") is not True or row.get("ready") is not True
                or row.get("required_file_hashes") != image["required_ready_hashes"]
                or not isinstance(row.get("sensitive_paths_checked"), list)
                or "/run/secrets" not in row["sensitive_paths_checked"]):
            raise ValueError("cold sandbox checks are incomplete")
    recipe = _recipe(reference, image["image_config"])
    expected_snapshot = "cap-cold-" + _sha_bytes(recipe.encode())[:24]
    snapshot = cold.get("snapshot")
    if (not isinstance(snapshot, dict) or snapshot.get("snapshot_name") != expected_snapshot
            or snapshot.get("dockerfile_sha256") != _sha_bytes(recipe.encode())
            or snapshot.get("evidence") != "provider-build-info-exact-match"
            or not isinstance(snapshot.get("id"), str) or not snapshot["id"]
            or not isinstance(snapshot.get("ref"), str) or not snapshot["ref"]):
        raise ValueError("cold snapshot identity differs from exact recipe")
    if any(not isinstance(row, dict) or row.get("sandbox_id") != ids[index]
           or row.get("state") != "not_found" or not isinstance(row.get("observations"), list)
           or not row["observations"] for index, row in enumerate(cleanup)):
        raise ValueError("cold sandbox cleanup is unconfirmed")


def migrate_image_pointers(
    *, task: Path, plan_path: Path, frozen_workspace: Path, capture_tools: Path,
    approval_path: Path, builder_session_ids: set[str],
    publication_paths: dict[str, Path], cold_pull_paths: dict[str, Path],
    output: Path,
) -> dict:
    """Change only reviewed image values, retaining exact pre-migration bytes."""
    if output.exists():
        receipt_path = output / "migration.json"
        if receipt_path.is_file() and _read(receipt_path).get("state") == "prepared":
            raise ValueError(
                "image migration was interrupted: compare task files with "
                "migration.json expected hashes and originals/ before recovery"
            )
        raise ValueError("image migration receipt already exists")
    plan = _read(plan_path)
    validate_plan(plan, frozen_workspace, capture_tools)
    approval = validate_review(approval_path, plan_path, builder_session_ids=builder_session_ids)
    manifest, _ = validate_retained_packet(
        approval_path.parent, plan_sha256=_sha(plan_path),
        snapshot_hash=approval["snapshot_hash"],
        manifest_sha256=approval["input_manifest_sha256"],
    )
    reviewed_files = manifest["files"]
    for name in (
        "specification.json",
        "binding.json",
        "composite-verifier.json",
        "judge-calibration.json",
    ):
        relative = "workspace/task/" + name
        expected = reviewed_files.get(relative)
        path = task / name
        if path.is_symlink():
            raise ValueError("current task document contains a link")
        if name not in {"composite-verifier.json", "judge-calibration.json"} and expected is None:
            raise ValueError("GLM review packet omitted required task document")
        if (expected is None) != (not path.exists()):
            raise ValueError("current task document presence differs from GLM review")
        if expected is not None and (path.is_symlink() or not path.is_file() or _sha(path) != expected):
            raise ValueError("current task document bytes differ from GLM review")
    roles = {image["role"] for image in plan["images"]}
    if set(publication_paths) != roles or set(cold_pull_paths) != roles:
        raise ValueError("every reviewed role needs publication and cold-pull evidence")
    pointers = image_pointers(task)
    expected = {image["role"]: image["authored_image_pointer"] for image in plan["images"]}
    # The same classification that sent the item here (request_needed): the
    # frozen plan's authored pointers are custom even when digest-shaped
    # (envgen.daytona recipe fingerprints, requested name@sha256 pointers).
    custom = custom_pointers(pointers, set(expected.values()))
    if (not custom or {row["role"] for row in custom} != roles
            or any(row["image"] != expected[row["role"]] for row in custom)):
        raise ValueError("current authored pointers differ from frozen reviewed plan")
    references = {}
    evidence = {}
    for image in plan["images"]:
        role = image["role"]
        publication_path = publication_paths[role]
        cold_path = cold_pull_paths[role]
        publication = _read(publication_path)
        verified_image, reference = _checked_publication(plan, plan_path, publication_path)
        if verified_image != image or _DIGEST.fullmatch(reference) is None:
            raise ValueError("publication does not match reviewed role")
        if publication.get("review_sha256") != _sha(approval_path):
            raise ValueError("publication review binding differs")
        _verified_cold(_read(cold_path), plan_path=plan_path, approval_path=approval_path,
                       publication_path=publication_path, image=image, reference=reference)
        references[role] = reference
        evidence[role] = {"publication_sha256": _sha(publication_path),
                          "cold_pull_sha256": _sha(cold_path), "image": reference}

    names = {
        "specification": "specification.json",
        "binding": "binding.json",
        "composite": "composite-verifier.json",
        "judge_calibration": "judge-calibration.json",
    }
    before = {key: (task / name).read_bytes() for key, name in names.items()
              if (task / name).is_file() and not (task / name).is_symlink()}
    if (task / names["composite"]).is_symlink():
        raise ValueError("composite verifier document is linked")
    if "specification" not in before or "binding" not in before:
        raise ValueError("task image documents are missing")
    documents = {key: json.loads(value) for key, value in before.items()}
    updated = copy.deepcopy(documents)
    if "composite" in updated and updated["composite"].get("specification_sha256") != _sha_bytes(before["specification"]):
        raise ValueError("composite specification binding differs before migration")
    if "judge_calibration" in updated and (
        updated["judge_calibration"].get("schema_version")
        != "taskcompendium-judge-calibration-v1"
        or updated["judge_calibration"].get("specification_sha256")
        != _sha_bytes(before["specification"])
    ):
        raise ValueError("judge calibration specification binding differs before migration")
    changes = []
    for row in custom:
        location, _, suffix = row["pointer"].partition(".")
        _set_pointer(updated[location], suffix.split("."), row["image"], references[row["role"]])
        changes.append({**row, "published_image": references[row["role"]]})
    after = {key: (json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
                   if value != documents[key] else before[key])
             for key, value in updated.items()}
    if "composite" in after and after["specification"] != before["specification"]:
        updated["composite"]["specification_sha256"] = _sha_bytes(after["specification"])
        after["composite"] = json.dumps(updated["composite"], sort_keys=True, indent=2).encode() + b"\n"
    if "judge_calibration" in after and after["specification"] != before["specification"]:
        updated["judge_calibration"]["specification_sha256"] = _sha_bytes(after["specification"])
        after["judge_calibration"] = (
            json.dumps(updated["judge_calibration"], sort_keys=True, indent=2).encode()
            + b"\n"
        )
    # The computed object is the full permitted semantic delta. The original
    # bytes are archived before any task file is replaced.
    output.mkdir(parents=True)
    originals = output / "originals"
    originals.mkdir()
    for key, value in before.items():
        (originals / names[key]).write_bytes(value)
    receipt = {"schema_version": SCHEMA, "state": "prepared", "plan_sha256": _sha(plan_path),
               "approval_sha256": _sha(approval_path), "roles": evidence, "changes": changes,
               "documents": {names[key]: {"before_sha256": _sha_bytes(value),
                                         "after_sha256": _sha_bytes(after[key])}
                             for key, value in before.items()}}
    receipt_path = output / "migration.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    for key, value in before.items():
        path = task / names[key]
        if path.is_symlink() or path.read_bytes() != value:
            raise ValueError("authored task changed during image migration")
    for key, value in before.items():
        path = task / names[key]
        if after[key] == value:
            continue
        replacement = path.with_name(path.name + ".image-migration-tmp")
        with replacement.open("xb") as stream:
            stream.write(after[key])
        replacement.chmod(path.stat().st_mode & 0o777)
        replacement.replace(path)
    receipt["state"] = "applied"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt
