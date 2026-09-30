"""Hash-bound, one-shot validation for private verifier image migrations.

This path is deliberately narrower than construction repair.  It restores a
complete failed controller output, proves that only the declared private
runtime image and its derived hashes changed, archives prior gate evidence,
and invokes the maintained synthesis validator exactly once.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .inference import atomic_json
from .synthesis import _handoff, load_accepted

_REGISTRY_DIGEST = re.compile(
    r"[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+(?::[a-z0-9._-]+)?@sha256:[0-9a-f]{64}"
)
_SPEC_POINTER = ("steps", 0, "verifier", "verifier", "runtime", "image")


class ImageMigrationError(ValueError):
    """The staged migration is not the reviewed image-only transformation."""


@dataclass(frozen=True)
class ImageMigrationPlan:
    bundle: Path
    restore_seed: Path
    item_name: str
    item: dict[str, Any]
    old_image: str
    new_image: str
    old_specification_sha256: str
    new_specification_sha256: str
    changed_paths: tuple[str, ...]
    repair_attempts: tuple[int, ...]
    protected_hashes: tuple[tuple[str, str], ...]
    migration_receipt_sha256: str
    bundle_manifest_sha256: str


@dataclass(frozen=True)
class PreparedImageMigration:
    plan: Any
    root: Path
    item_root: Path
    history_root: Path
    operation_path: Path


def _read_json(path: Path) -> Any:
    try:
        with path.open() as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ImageMigrationError(
            f"unreadable JSON artifact {path}: {error}"
        ) from error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ImageMigrationError(f"{label} is not a SHA-256 digest")
    return value


def _safe_relative(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise ImageMigrationError(f"{label} is not a relative path")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ImageMigrationError(f"{label} is not a safe relative path")
    return path


def _file_inventory(root: Path, *, omit: Path | None = None) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != omit
    }


def _differences(
    before: object, after: object, path: tuple[object, ...] = ()
) -> list[tuple[object, ...]]:
    if type(before) is not type(after):
        return [path]
    if isinstance(before, dict):
        if before.keys() != after.keys():
            return [path]
        differences = []
        for key in before:
            differences.extend(_differences(before[key], after[key], (*path, key)))
        return differences
    if isinstance(before, list):
        if len(before) != len(after):
            return [path]
        differences = []
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            differences.extend(_differences(left, right, (*path, index)))
        return differences
    return [] if before == after else [path]


def _assignment(tree: ast.AST, name: str) -> ast.Assign:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    ]
    if len(matches) != 1:
        raise ImageMigrationError(f"generator must assign {name} exactly once")
    return matches[0]


def _normalized_generator_ast(path: Path, expected_image: str) -> str:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ImageMigrationError(f"invalid task generator {path}: {error}") from error
    assignment = _assignment(tree, "VERIFIER_IMAGE")
    try:
        value = ast.literal_eval(assignment.value)
    except (ValueError, TypeError) as error:
        raise ImageMigrationError("VERIFIER_IMAGE must be a string literal") from error
    if value != expected_image:
        raise ImageMigrationError(
            "task generator VERIFIER_IMAGE does not match migration"
        )
    assignment.value = ast.Constant(value="<private-runtime-image>")
    return ast.dump(tree, include_attributes=False)


def _expected_changed_paths(item_name: str) -> set[str]:
    prefix = f"items/{item_name}"
    return {
        f"{prefix}/harbor/manifest.json",
        f"{prefix}/harbor/specification.json",
        f"{prefix}/workspace/build_task.py",
        f"{prefix}/workspace/task/harbor/manifest.json",
        f"{prefix}/workspace/task/harbor/specification.json",
        f"{prefix}/workspace/task/specification.json",
    }


def _validate_bundle_manifest(bundle: Path) -> tuple[dict[str, Any], str]:
    path = bundle / "manifest.json"
    manifest = _read_json(path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "capability-portable-runtime-bundle-v1"
        or manifest.get("state") != "prepared_pending_fresh_runtime_validation"
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ImageMigrationError(
            "portable-runtime bundle manifest has the wrong schema"
        )
    actual = _file_inventory(bundle, omit=path)
    if manifest["files"] != actual:
        raise ImageMigrationError("portable-runtime bundle inventory hash mismatch")
    for key, relative in (
        ("accepted_sha256", "accepted.json"),
        ("migration_receipt_sha256", "migration-receipt.json"),
        ("structural_validation_sha256", "structural-validation.json"),
    ):
        expected = _require_digest(manifest.get(key), f"bundle {key}")
        if _sha256(bundle / relative) != expected:
            raise ImageMigrationError(f"portable-runtime {relative} hash mismatch")
    return manifest, _sha256(path)


def _validate_history(
    restore_seed: Path,
    item_name: str,
    receipt: dict[str, Any],
    pull_files: dict[str, str],
) -> tuple[int, ...]:
    restore = receipt.get("restore_seed")
    controller = (
        restore.get("controller_history") if isinstance(restore, dict) else None
    )
    attempts = (
        controller.get("repair_attempts_retained")
        if isinstance(controller, dict)
        else None
    )
    if (
        not isinstance(attempts, list)
        or not attempts
        or any(type(value) is not int or value < 1 for value in attempts)
        or attempts != list(range(1, len(attempts) + 1))
    ):
        raise ImageMigrationError("migration lacks sequential consumed repair attempts")
    names = {f"attempt-{value}" for value in attempts}
    for directory in ("repairs", "repair-history"):
        parent = restore_seed / directory / item_name
        actual = (
            {path.name for path in parent.iterdir() if path.is_dir()}
            if parent.is_dir()
            else set()
        )
        if actual != names:
            raise ImageMigrationError(
                f"{directory} does not preserve every consumed repair attempt"
            )
        for name in names:
            prefix = f"{directory}/{item_name}/{name}/"
            if not any(path.startswith(prefix) for path in pull_files):
                raise ImageMigrationError(f"source manifest lacks {prefix}")
    status_path = restore_seed / "items" / item_name / "status.json"
    status = _read_json(status_path)
    rounds = status.get("repairs") if isinstance(status, dict) else None
    if (
        status.get("state") != "failed"
        or not isinstance(rounds, list)
        or [record.get("round") for record in rounds if isinstance(record, dict)]
        != attempts
    ):
        raise ImageMigrationError(
            "terminal failed status does not retain repair budget history"
        )
    return tuple(attempts)


def _validate_specs(
    bundle: Path,
    restore_seed: Path,
    item_name: str,
    receipt: dict[str, Any],
    expected_image: str,
) -> tuple[str, str, str]:
    image = receipt.get("runtime_image")
    if not isinstance(image, dict) or image.get("new") != expected_image:
        raise ImageMigrationError(
            "migration runtime image does not match the reviewed image"
        )
    old_image = image.get("old")
    if not isinstance(old_image, str) or old_image == expected_image:
        raise ImageMigrationError("migration old runtime image is invalid")
    if not _REGISTRY_DIGEST.fullmatch(expected_image):
        raise ImageMigrationError(
            "new runtime image is not a canonical registry digest"
        )

    item = restore_seed / "items" / item_name
    relative_specs = (
        "harbor/specification.json",
        "workspace/task/specification.json",
        "workspace/task/harbor/specification.json",
    )
    before_specs = []
    after_specs = []
    before_root = bundle / "migration-before" / "items" / item_name
    for relative in relative_specs:
        before_specs.append(_read_json(before_root / relative))
        after_specs.append(_read_json(item / relative))
    if any(value != before_specs[0] for value in before_specs[1:]):
        raise ImageMigrationError("pre-migration TaskSpec copies disagree")
    if any(value != after_specs[0] for value in after_specs[1:]):
        raise ImageMigrationError("migrated TaskSpec copies disagree")
    if _differences(before_specs[0], after_specs[0]) != [_SPEC_POINTER]:
        raise ImageMigrationError(
            "TaskSpec migration contains an extra semantic change"
        )
    current: object = after_specs[0]
    for component in _SPEC_POINTER:
        current = current[component]  # type: ignore[index]
    if current != expected_image:
        raise ImageMigrationError("TaskSpec ContainerRuntime.image was not migrated")

    specification = receipt.get("task_specification")
    if not isinstance(specification, dict):
        raise ImageMigrationError("migration lacks TaskSpec hash binding")
    old_sha = _require_digest(specification.get("old_sha256"), "old TaskSpec hash")
    new_sha = _require_digest(specification.get("new_sha256"), "new TaskSpec hash")
    for relative in relative_specs:
        if (
            _sha256(before_root / relative) != old_sha
            or _sha256(item / relative) != new_sha
        ):
            raise ImageMigrationError("TaskSpec raw hash binding mismatch")

    relative_manifests = (
        "harbor/manifest.json",
        "workspace/task/harbor/manifest.json",
    )
    manifest_differences = {
        ("specification_sha256",),
        ("verifier_runtimes", 0, "image"),
    }
    for relative in relative_manifests:
        before = _read_json(before_root / relative)
        after = _read_json(item / relative)
        if set(_differences(before, after)) != manifest_differences:
            raise ImageMigrationError("lowered manifest has an extra migration change")
        runtimes = after.get("verifier_runtimes") if isinstance(after, dict) else None
        if (
            after.get("specification_sha256") != new_sha
            or not isinstance(runtimes, list)
            or len(runtimes) != 1
            or runtimes[0].get("image") != expected_image
        ):
            raise ImageMigrationError(
                "lowered manifest does not bind migrated TaskSpec"
            )

    before_build = before_root / "workspace/build_task.py"
    after_build = item / "workspace/build_task.py"
    if _normalized_generator_ast(before_build, old_image) != _normalized_generator_ast(
        after_build, expected_image
    ):
        raise ImageMigrationError("task generator changed beyond VERIFIER_IMAGE")
    return old_image, old_sha, new_sha


def validate_image_migration_bundle(
    bundle: Path,
    *,
    expected_image: str,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> ImageMigrationPlan:
    """Validate a complete, immutable image-only migration bundle."""
    bundle = bundle.resolve()
    manifest, manifest_sha = _validate_bundle_manifest(bundle)
    if manifest_sha != _require_digest(
        expected_manifest_sha256, "reviewed bundle manifest hash"
    ):
        raise ImageMigrationError("bundle manifest differs from the reviewed migration")
    receipt_sha = _sha256(bundle / "migration-receipt.json")
    if receipt_sha != _require_digest(
        expected_receipt_sha256, "reviewed migration receipt hash"
    ):
        raise ImageMigrationError(
            "migration receipt differs from the reviewed migration"
        )
    receipt = _read_json(bundle / "migration-receipt.json")
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != "capability-portable-runtime-migration-v1"
        or receipt.get("execution", {}).get("runtime_validation")
        != "pending_fresh_full_gates"
        or receipt.get("execution", {}).get("acceptance_claimed") is not False
    ):
        raise ImageMigrationError(
            "image migration receipt has the wrong schema or state"
        )
    historical = receipt.get("historical_evidence")
    incident = (
        _safe_relative(historical.get("incident_copy_path"), "incident audit path")
        if isinstance(historical, dict)
        else None
    )
    incident_sha = (
        _require_digest(historical.get("incident_sha256"), "incident audit hash")
        if isinstance(historical, dict)
        else None
    )
    if (
        incident is None
        or incident_sha is None
        or _sha256(bundle / incident) != incident_sha
        or manifest.get("incident_sha256") != incident_sha
    ):
        raise ImageMigrationError("migration incident evidence hash mismatch")
    structural = _read_json(bundle / "structural-validation.json")
    if not isinstance(structural, dict) or structural.get("state") != "passed":
        raise ImageMigrationError("migration structural validation has not passed")

    restore_seed = bundle / "restore-seed"
    source = receipt.get("source")
    source_metadata = bundle / "source-metadata"
    if not isinstance(source, dict) or _require_digest(
        source.get("pull_manifest_sha256"), "source pull manifest hash"
    ) != _sha256(source_metadata / "pull-manifest.json"):
        raise ImageMigrationError(
            "source pull manifest does not match migration receipt"
        )
    if _require_digest(
        source.get("snapshot_capture_sha256"), "source snapshot capture hash"
    ) != _sha256(source_metadata / "snapshot-capture.json"):
        raise ImageMigrationError(
            "source snapshot capture does not match migration receipt"
        )
    item_name = receipt.get("item_name")
    if not isinstance(item_name, str) or not item_name:
        raise ImageMigrationError("migration item name is missing")
    item_root = restore_seed / "items" / item_name
    if not item_root.is_dir():
        raise ImageMigrationError("migration item is absent from complete restore seed")

    accepted = load_accepted(bundle / "accepted.json", 1)
    if len(accepted) != 1:
        raise ImageMigrationError("image migration requires exactly one accepted item")
    item = accepted[0]
    if _read_json(item_root / "contract/accepted.json") != item:
        raise ImageMigrationError("frozen accepted proposal differs from item contract")
    proposal_hash = item["proposal_hash"]
    if not item_name.endswith(proposal_hash[:12]):
        raise ImageMigrationError("migration item name does not bind proposal hash")

    pull = _read_json(source_metadata / "pull-manifest.json")
    pull_files = pull.get("files") if isinstance(pull, dict) else None
    if (
        pull.get("schema_version") != "capability-snapshot-pull-v1"
        or pull.get("complete_manifest") is not True
        or pull.get("complete_snapshot") is not True
        or pull.get("remote_final") is not True
        or not isinstance(pull_files, dict)
        or any(
            not isinstance(path, str)
            or not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
            for path, value in pull_files.items()
        )
    ):
        raise ImageMigrationError(
            "restore seed lacks a complete terminal pull manifest"
        )

    expected_changed = _expected_changed_paths(item_name)
    manifest_changed = manifest.get("changed_paths")
    if (
        not isinstance(manifest_changed, list)
        or set(manifest_changed) != expected_changed
    ):
        raise ImageMigrationError(
            "bundle changed-path declaration is not the six-file migration"
        )
    changed_files = receipt.get("changed_files")
    if not isinstance(changed_files, list):
        raise ImageMigrationError("migration receipt lacks changed-file hashes")
    changed_by_path = {}
    for record in changed_files:
        if not isinstance(record, dict):
            raise ImageMigrationError("migration changed-file record is malformed")
        path = _safe_relative(record.get("path"), "changed-file path")
        try:
            relative = str(path.relative_to("restore-seed"))
        except ValueError as error:
            raise ImageMigrationError(
                "changed-file path is outside restore-seed"
            ) from error
        if relative in changed_by_path:
            raise ImageMigrationError("migration changed-file path is duplicated")
        changed_by_path[relative] = record
    if set(changed_by_path) != expected_changed:
        raise ImageMigrationError(
            "migration receipt does not bind exactly six changed files"
        )

    before_declaration = receipt.get("before_files")
    declared_before = (
        before_declaration.get("files")
        if isinstance(before_declaration, dict)
        else None
    )
    if not isinstance(declared_before, dict) or declared_before != {
        path: record.get("old_sha256") for path, record in changed_by_path.items()
    }:
        raise ImageMigrationError("migration receipt does not bind exact before files")

    restore_declaration = receipt.get("restore_seed")
    retained_members = (
        restore_declaration.get("source_members_retained")
        if isinstance(restore_declaration, dict)
        else None
    )
    source_count = source.get("source_file_count")
    if (
        not isinstance(retained_members, list)
        or any(not isinstance(path, str) for path in retained_members)
        or len(retained_members) != len(set(retained_members))
        or type(source_count) is not int
        or source_count != len(retained_members)
    ):
        raise ImageMigrationError("migration receipt lacks complete source membership")
    current_restore = _file_inventory(restore_seed)
    if (
        set(current_restore) != set(retained_members)
        or set(current_restore) != set(pull_files)
        or {"pull-manifest.json", "snapshot-capture.json"} & set(current_restore)
    ):
        raise ImageMigrationError(
            "restore seed member set differs from terminal source"
        )

    prefix = f"items/{item_name}/"
    expected_item_members = {path for path in pull_files if path.startswith(prefix)}
    actual_item_members = {
        str(path.relative_to(restore_seed))
        for path in item_root.rglob("*")
        if path.is_file()
    }
    if actual_item_members != expected_item_members:
        raise ImageMigrationError(
            "migrated item member set differs from terminal source"
        )
    before_root = bundle / "migration-before"
    for relative, source_sha in pull_files.items():
        current = restore_seed / relative
        if not current.is_file():
            raise ImageMigrationError(f"restore seed is missing {relative}")
        if relative in expected_changed:
            record = changed_by_path[relative]
            old_sha = _require_digest(
                record.get("old_sha256"), f"old hash for {relative}"
            )
            new_sha = _require_digest(
                record.get("new_sha256"), f"new hash for {relative}"
            )
            before = before_root / relative
            if (
                source_sha != old_sha
                or not before.is_file()
                or _sha256(before) != old_sha
            ):
                raise ImageMigrationError(
                    f"pre-migration source hash mismatch for {relative}"
                )
            if _sha256(current) != new_sha:
                raise ImageMigrationError(f"migrated file hash mismatch for {relative}")
        elif _sha256(current) != source_sha:
            raise ImageMigrationError(
                f"undeclared terminal artifact change: {relative}"
            )

    reconstructed_source = dict(current_restore)
    for relative, old_sha in declared_before.items():
        reconstructed_source[relative] = old_sha
    source_payload = "".join(
        f"{path}\0{reconstructed_source[path]}\n"
        for path in sorted(reconstructed_source)
    )
    if hashlib.sha256(source_payload.encode()).hexdigest() != _require_digest(
        source.get("source_tree_sha256"), "terminal source tree hash"
    ):
        raise ImageMigrationError("reconstructed terminal source tree hash mismatch")

    attempts = _validate_history(restore_seed, item_name, receipt, pull_files)
    old_image, old_spec_sha, new_spec_sha = _validate_specs(
        bundle, restore_seed, item_name, receipt, expected_image
    )
    status = _read_json(item_root / "status.json")
    if status.get("proposal_hash") != proposal_hash:
        raise ImageMigrationError("terminal status proposal hash mismatch")
    sessions = status.get("sessions")
    if (
        not isinstance(sessions, list)
        or not sessions
        or any(
            not isinstance(session, dict) or session.get("status") != "complete"
            for session in sessions
        )
    ):
        raise ImageMigrationError(
            "migration cannot resume incomplete construction sessions"
        )
    session_names = [session.get("session") for session in sessions]
    if any(not isinstance(name, str) or not name for name in session_names) or len(
        session_names
    ) != len(set(session_names)):
        raise ImageMigrationError("terminal status has invalid construction sessions")
    declared_sessions = [
        session.get("session") for session in item["proposal"].get("builder_plan", [])
    ]
    if session_names != declared_sessions:
        raise ImageMigrationError(
            "terminal sessions differ from the accepted builder DAG"
        )
    for name in session_names:
        persisted = _read_json(item_root / "sessions" / name / "status.json")
        if persisted.get("session") != name or persisted.get("status") != "complete":
            raise ImageMigrationError(
                "migration cannot invoke a builder for an incomplete persisted session"
            )
        if _handoff(item_root / "workspace", name) is None:
            raise ImageMigrationError(
                "migration cannot invoke a builder for an invalid persisted handoff"
            )
    return ImageMigrationPlan(
        bundle=bundle,
        restore_seed=restore_seed,
        item_name=item_name,
        item=item,
        old_image=old_image,
        new_image=expected_image,
        old_specification_sha256=old_spec_sha,
        new_specification_sha256=new_spec_sha,
        changed_paths=tuple(sorted(expected_changed)),
        repair_attempts=attempts,
        protected_hashes=tuple(
            sorted(
                (str(path.relative_to(item_root)), _sha256(path))
                for path in (item_root / "workspace").rglob("*")
                if path.is_file()
                and path.parts[
                    len((item_root / "workspace").parts) : len(
                        (item_root / "workspace").parts
                    )
                    + 2
                ]
                != ("tools", "daytona")
            )
        ),
        migration_receipt_sha256=_sha256(bundle / "migration-receipt.json"),
        bundle_manifest_sha256=manifest_sha,
    )


def _tree_digest(root: Path) -> str:
    files = _file_inventory(root)
    payload = "".join(f"{path}\0{value}\n" for path, value in files.items())
    return hashlib.sha256(payload.encode()).hexdigest()


def prepare_image_migration_revalidation(
    plan: Any, output_root: Path
) -> PreparedImageMigration:
    """Restore history and archive prior gates without running validation."""
    output_root = output_root.resolve()
    if output_root.exists():
        raise ImageMigrationError("image-migration revalidation output must be new")
    shutil.copytree(plan.restore_seed, output_root, copy_function=shutil.copy2)
    item_root = output_root / "items" / plan.item_name
    history_root = (
        output_root / "image-migration-history" / plan.item_name / "revalidation-1"
    )
    history_root.mkdir(parents=True, exist_ok=False)

    for name in ("status.json",):
        source = item_root / name
        if source.is_file():
            shutil.copy2(source, history_root / name)
    controller_history = history_root / "controller"
    controller_history.mkdir()
    for name in ("run.json", "report.json", "tasks.json", "terminal.json"):
        source = output_root / name
        if source.is_file():
            shutil.copy2(source, controller_history / name)

    for name in (
        "harbor",
        "runtime-trials",
        "judge-calibration",
        "runtime-evidence.json",
        "solver-transcripts.json",
        "independent-adversary.json",
        "authored-oracle.json",
    ):
        source = item_root / name
        if source.exists():
            shutil.move(str(source), history_root / name)
    for name in ("quality", "attack-adjudication", "validated"):
        source = output_root / name / plan.item_name
        if source.exists():
            destination = history_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), destination)

    frozen_input = output_root / "image-migration-input"
    frozen_input.mkdir()
    for name in (
        "accepted.json",
        "manifest.json",
        "migration-receipt.json",
        "structural-validation.json",
    ):
        shutil.copy2(plan.bundle / name, frozen_input / name)
    shutil.copytree(plan.bundle / "migration-before", frozen_input / "migration-before")
    for directory in ("audits", "evidence", "source-metadata"):
        source = plan.bundle / directory
        if source.is_dir():
            shutil.copytree(source, frozen_input / directory)

    repair_digest = _tree_digest(output_root / "repairs" / plan.item_name)
    repair_history_digest = _tree_digest(
        output_root / "repair-history" / plan.item_name
    )
    operation = {
        "schema_version": "capability-image-migration-revalidation-v1",
        "state": "prepared",
        "item_name": plan.item_name,
        "old_specification_sha256": plan.old_specification_sha256,
        "new_specification_sha256": plan.new_specification_sha256,
        "changed_paths": list(plan.changed_paths),
        "consumed_repair_attempts": list(plan.repair_attempts),
        "repair_tree_sha256": repair_digest,
        "repair_history_tree_sha256": repair_history_digest,
        "protected_artifacts": dict(plan.protected_hashes),
        "migration_receipt_sha256": plan.migration_receipt_sha256,
        "bundle_manifest_sha256": plan.bundle_manifest_sha256,
        "prior_gate_archive": str(history_root),
        "started_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "result": None,
    }
    pointer_changes = getattr(plan, "pointer_changes", None)
    if pointer_changes is None:
        operation.update(old_image=plan.old_image, new_image=plan.new_image)
    else:
        operation["image_pointer_changes"] = [
            {
                "role": change.role,
                "old": change.old,
                "new": change.new,
                "publication_receipt_sha256": change.publication_receipt_sha256,
                "cold_pull_receipt_sha256": change.cold_pull_receipt_sha256,
            }
            for change in pointer_changes
        ]
    operation_path = output_root / "controller" / "image-migration-revalidation.json"
    atomic_json(operation_path, operation)
    prior_status_path = history_root / "status.json"
    prior_status = _read_json(prior_status_path)
    current_status = {
        "key": prior_status.get("key"),
        "proposal_hash": plan.item["proposal_hash"],
        "sessions": prior_status.get("sessions", []),
        "repairs": prior_status.get("repairs", []),
        "state": "pending_image_migration_revalidation",
        "issues": [
            "fresh maintained gates have not started; the prior terminal status is archived"
        ],
        "item_root": str(item_root),
        "image_migration_revalidation": {
            "operation_state": "prepared",
            "operation_artifact": str(operation_path.relative_to(output_root)),
            "prior_state": prior_status.get("state"),
            "prior_status_artifact": str(prior_status_path.relative_to(output_root)),
            "prior_status_sha256": _sha256(prior_status_path),
        },
    }
    atomic_json(item_root / "status.json", current_status)
    return PreparedImageMigration(
        plan=plan,
        root=output_root,
        item_root=item_root,
        history_root=history_root,
        operation_path=operation_path,
    )


def run_prepared_image_migration(
    prepared: PreparedImageMigration,
    attempt: Callable[[dict[str, Any], Path], dict[str, Any]],
) -> dict[str, Any]:
    """Run one maintained validation attempt and retain its exact outcome."""
    operation = _read_json(prepared.operation_path)
    if operation.get("state") != "prepared":
        raise ImageMigrationError("image migration is not in the prepared state")
    operation["state"] = "running"
    atomic_json(prepared.operation_path, operation)
    current_status_path = prepared.item_root / "status.json"
    current_status = _read_json(current_status_path)
    current_status.update(
        state="image_migration_revalidation_running",
        issues=["fresh maintained gates are running"],
    )
    migration_status = current_status.get("image_migration_revalidation")
    if not isinstance(migration_status, dict):
        raise ImageMigrationError("prepared migration status is missing")
    migration_status["operation_state"] = "running"
    atomic_json(current_status_path, current_status)
    try:
        result = attempt(prepared.plan.item, prepared.root)
        if not isinstance(result, dict) or not isinstance(result.get("state"), str):
            raise ImageMigrationError(
                "maintained validation returned an invalid result"
            )
        if (
            _tree_digest(prepared.root / "repairs" / prepared.plan.item_name)
            != operation["repair_tree_sha256"]
            or _tree_digest(prepared.root / "repair-history" / prepared.plan.item_name)
            != operation["repair_history_tree_sha256"]
        ):
            raise ImageMigrationError("maintained validation changed repair history")
        for relative, expected in prepared.plan.protected_hashes:
            path = prepared.item_root / relative
            if not path.is_file() or _sha256(path) != expected:
                raise ImageMigrationError(
                    f"maintained validation changed protected task artifact: {relative}"
                )
        result["image_migration_revalidation"] = {
            "migration_receipt_sha256": prepared.plan.migration_receipt_sha256,
            "bundle_manifest_sha256": prepared.plan.bundle_manifest_sha256,
            "prior_gate_archive": str(prepared.history_root),
            "consumed_repair_attempts": list(prepared.plan.repair_attempts),
            "semantic_repair_performed": False,
            "maintained_gate_attempts": 1,
        }
        atomic_json(prepared.item_root / "status.json", result)
        atomic_json(prepared.root / "tasks.json", [result])
        operation.update(
            state="completed",
            completed_at=datetime.now(UTC).isoformat(),
            result={
                "state": result["state"],
                "runtime_validated": result.get("runtime_validated") is True,
                "status_sha256": _sha256(prepared.item_root / "status.json"),
            },
        )
        atomic_json(prepared.operation_path, operation)
        atomic_json(
            prepared.root / "report.json",
            {
                "stage": "image_migration_revalidation",
                "state": (
                    "complete"
                    if result["state"] == "quality_accepted"
                    else "needs_continuation"
                ),
                "task_state": result["state"],
                "runtime_validated_tasks": int(result.get("runtime_validated") is True),
                "quality_accepted_tasks": int(result["state"] == "quality_accepted"),
                "semantic_repair_performed": False,
                "maintained_gate_attempts": 1,
            },
        )
        return result
    except Exception as error:
        operation.update(
            state="error",
            completed_at=datetime.now(UTC).isoformat(),
            result={"exception_type": type(error).__name__, "message": str(error)},
        )
        atomic_json(prepared.operation_path, operation)
        failed_status = _read_json(current_status_path)
        failed_status.update(
            state="image_migration_revalidation_error",
            issues=[f"maintained gates raised {type(error).__name__}"],
        )
        failed_migration = failed_status.get("image_migration_revalidation")
        if isinstance(failed_migration, dict):
            failed_migration["operation_state"] = "error"
        atomic_json(current_status_path, failed_status)
        raise
