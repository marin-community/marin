"""Produce and validate explicit multi-image TaskSpec pointer migrations."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .image_migration import ImageMigrationError, _differences, _read_json, _sha256
from .image_runtime_metadata import (
    ImageRuntimeMetadataError,
    derive_daytona_recipe,
)

_REGISTRY_DIGEST = re.compile(
    r"[a-z0-9.-]+(?::[0-9]+)?/[a-z0-9._/-]+"
    r"(?::[a-z0-9._-]+)?@sha256:[0-9a-f]{64}"
)
_CANDIDATE_SPEC_POINTER = ("requirements", "state", "image")
_VERIFIER_SPEC_POINTER = ("steps", 0, "verifier", "runtime", "image")
_BINDING_POINTER = ("environment", "image")
_CANDIDATE_MANIFEST_POINTER = ("binding", "environment", "image")
_VERIFIER_MANIFEST_POINTER = ("verifier_runtimes", 0, "image")
_SPEC_FILES = (
    "harbor/specification.json",
    "workspace/task/specification.json",
    "workspace/task/harbor/specification.json",
)
_BINDING_FILES = (
    "harbor/binding.json",
    "workspace/task/binding.json",
    "workspace/task/harbor/binding.json",
)
_MANIFEST_FILES = (
    "harbor/manifest.json",
    "workspace/task/harbor/manifest.json",
)
_GENERATOR = "workspace/task/build_task.py"
_OLD_VALIDATION = """    for image in (verifier_image, harbor_image):
        digest = image.removeprefix("sha256:")
        if not (image.startswith("sha256:") or "@sha256:" in image) \\
                or len(digest) != 64:
            raise SystemExit("image references must be immutable digests")
"""
_NEW_VALIDATION = """    for image in (verifier_image, harbor_image):
        repository, separator, digest = image.rpartition("@sha256:")
        if (not separator or "/" not in repository or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise SystemExit("image references must be canonical registry digests")
"""


@dataclass(frozen=True)
class ImagePointerChange:
    role: str
    old: str
    new: str
    publication_receipt_sha256: str
    cold_pull_receipt_sha256: str


@dataclass(frozen=True)
class PointerImageMigrationPlan:
    bundle: Path
    restore_seed: Path
    item_name: str
    item: dict[str, Any]
    pointer_changes: tuple[ImagePointerChange, ...]
    old_specification_sha256: str
    new_specification_sha256: str
    changed_paths: tuple[str, ...]
    repair_attempts: tuple[int, ...]
    protected_hashes: tuple[tuple[str, str], ...]
    migration_receipt_sha256: str
    bundle_manifest_sha256: str


def canonical_registry_digest(value: Any) -> str:
    if not isinstance(value, str) or _REGISTRY_DIGEST.fullmatch(value) is None:
        raise ImageMigrationError("image pointer is not a canonical registry digest")
    return value


def patch_generator_image_validation(source: str) -> str:
    """Apply the one reviewed generator change needed for registry digests."""
    if source.count(_OLD_VALIDATION) != 1 or _NEW_VALIDATION in source:
        raise ImageMigrationError(
            "generator does not contain the expected legacy image validation"
        )
    return source.replace(_OLD_VALIDATION, _NEW_VALIDATION)


def _value(document: Any, pointer: tuple[object, ...]) -> Any:
    current = document
    for component in pointer:
        try:
            current = current[component]
        except (KeyError, IndexError, TypeError) as error:
            raise ImageMigrationError(
                f"document lacks image pointer {pointer}"
            ) from error
    return current


def _set_value(document: Any, pointer: tuple[object, ...], value: str) -> None:
    current = document
    for component in pointer[:-1]:
        try:
            current = current[component]
        except (KeyError, IndexError, TypeError) as error:
            raise ImageMigrationError(
                f"document lacks image pointer {pointer}"
            ) from error
    current[pointer[-1]] = value


def _dump(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))


def _inventory(root: Path, *, omit: Path | None = None) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != omit
    }


def _tree_digest(files: dict[str, str]) -> str:
    payload = "".join(f"{path}\0{files[path]}\n" for path in sorted(files))
    return hashlib.sha256(payload.encode()).hexdigest()


def _safe_source_files(source: Path) -> dict[str, str]:
    pull = _read_json(source / "pull-manifest.json")
    files = pull.get("files") if isinstance(pull, dict) else None
    if (
        pull.get("schema_version") != "capability-snapshot-pull-v1"
        or pull.get("complete_manifest") is not True
        or pull.get("complete_snapshot") is not True
        or pull.get("remote_final") is not True
        or not isinstance(files, dict)
        or not files
    ):
        raise ImageMigrationError("source is not a complete terminal pull")
    for relative, digest in files.items():
        path = Path(relative)
        if (
            not isinstance(relative, str)
            or path.is_absolute()
            or ".." in path.parts
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ImageMigrationError("terminal pull inventory is malformed")
        candidate = source / path
        if not candidate.is_file() or _sha256(candidate) != digest:
            raise ImageMigrationError(f"terminal pull member mismatch: {relative}")
    return files


def _validate_publication_evidence(
    role: str,
    image: str,
    publication: dict[str, Any],
    cold_pull: dict[str, Any],
    *,
    publication_sha256: str,
) -> None:
    published = publication.get("publication")
    producer_role = "private_verifier" if role == "verifier" else role
    if (
        publication.get("schema_version") != "capability-task-image-publication-v1"
        or publication.get("state") != "published_pending_cold_pull"
        or publication.get("role") != producer_role
        or not isinstance(publication.get("plan_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", publication["plan_sha256"]) is None
        or not isinstance(published, dict)
        or published.get("schema_version") != "capability-oci-transport-v1"
        or published.get("state") != "integrity_verified"
        or published.get("image") != image
        or published.get("manifest_digest")
        != "sha256:" + image.rpartition("@sha256:")[2]
    ):
        raise ImageMigrationError(f"{role} publication receipt is not final and bound")
    cleanup = cold_pull.get("cleanup")
    sandboxes = cold_pull.get("sandboxes")
    try:
        expected_recipe = derive_daytona_recipe(image)
    except ImageRuntimeMetadataError as error:
        raise ImageMigrationError(
            f"{role} cold-pull image runtime metadata is invalid"
        ) from error
    expected_recipe_sha256 = hashlib.sha256(expected_recipe.encode()).hexdigest()
    snapshot = cold_pull.get("snapshot")
    if (
        cold_pull.get("schema_version") != "capability-image-cold-pull-v1"
        or cold_pull.get("state") != "passed"
        or cold_pull.get("role") != producer_role
        or cold_pull.get("image") != image
        or cold_pull.get("plan_sha256") != publication["plan_sha256"]
        or cold_pull.get("publication_sha256") != publication_sha256
        or not isinstance(snapshot, dict)
        or snapshot.get("snapshot_name")
        != f"cap-cold-{expected_recipe_sha256[:24]}"
        or snapshot.get("dockerfile_sha256") != expected_recipe_sha256
        or not isinstance(snapshot.get("id"), str)
        or not snapshot["id"]
        or not isinstance(snapshot.get("ref"), str)
        or not snapshot["ref"]
        or not isinstance(sandboxes, list)
        or len(sandboxes) < 2
        or not isinstance(cleanup, list)
        or len(cleanup) < 2
    ):
        raise ImageMigrationError(f"{role} cold-pull receipt is not a passed probe")
    expected_files = publication.get("rootfs_review", {}).get("required_file_hashes")
    identifiers = []
    for sandbox in sandboxes:
        if (
            not isinstance(sandbox, dict)
            or not isinstance(sandbox.get("sandbox_id"), str)
            or not sandbox["sandbox_id"]
            or sandbox.get("network_block_all") is not True
            or sandbox.get("required_file_hashes") != expected_files
            or sandbox.get("registry_reachability", {}).get("reachable") is not False
            or not isinstance(sandbox.get("database"), dict)
            or sandbox.get("database_and_workspace_pristine") is not True
        ):
            raise ImageMigrationError(f"{role} cold-pull sandbox evidence is incomplete")
        identifiers.append(sandbox["sandbox_id"])
    if len(set(identifiers)) != len(identifiers):
        raise ImageMigrationError(f"{role} cold-pull reused a sandbox")
    if sandboxes[0].get("database_and_workspace_mutated") is not True:
        raise ImageMigrationError(f"{role} cold-pull did not prove mutation isolation")
    if any(
        not isinstance(row, dict)
        or row.get("sandbox_id") not in identifiers
        or row.get("verified_absent") is not True
        for row in cleanup
    ):
        raise ImageMigrationError(f"{role} cold-pull cleanup is unverified")


def _changed_paths(item_name: str) -> tuple[str, ...]:
    prefix = f"items/{item_name}/"
    return tuple(
        sorted(
            prefix + relative
            for relative in (
                *_SPEC_FILES,
                *_BINDING_FILES,
                *_MANIFEST_FILES,
                _GENERATOR,
            )
        )
    )


def produce_pointer_migration_bundle(
    source: Path,
    output: Path,
    *,
    candidate_image: str,
    verifier_image: str,
    evidence: dict[str, tuple[Path, Path]],
) -> Path:
    """Create a reviewable v2 bundle without executing generated task code."""
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ImageMigrationError("pointer migration output must be new")
    images = {
        "candidate": canonical_registry_digest(candidate_image),
        "verifier": canonical_registry_digest(verifier_image),
    }
    if set(evidence) != set(images):
        raise ImageMigrationError("publication evidence must cover both image roles")
    source_files = _safe_source_files(source)
    item_names = sorted(
        {
            Path(relative).parts[1]
            for relative in source_files
            if len(Path(relative).parts) > 2 and Path(relative).parts[0] == "items"
        }
    )
    if len(item_names) != 1:
        raise ImageMigrationError("pointer migration needs exactly one terminal item")
    item_name = item_names[0]
    output.mkdir(parents=True)
    restore = output / "restore-seed"
    for relative in source_files:
        destination = restore / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    metadata = output / "source-metadata"
    metadata.mkdir()
    for name in ("pull-manifest.json", "snapshot-capture.json"):
        shutil.copy2(source / name, metadata / name)
    item = restore / "items" / item_name
    accepted = _read_json(item / "contract/accepted.json")
    (output / "accepted.json").write_text(json.dumps([accepted], indent=2) + "\n")

    before = output / "migration-before"
    changed = _changed_paths(item_name)
    for relative in changed:
        source_path = restore / relative
        target = before / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)

    specification = _read_json(item / _SPEC_FILES[0])
    old_candidate = _value(specification, _CANDIDATE_SPEC_POINTER)
    old_verifier = _value(specification, _VERIFIER_SPEC_POINTER)
    old_spec_sha = _sha256(item / _SPEC_FILES[0])
    for relative in _SPEC_FILES:
        document = _read_json(item / relative)
        if (
            _value(document, _CANDIDATE_SPEC_POINTER) != old_candidate
            or _value(document, _VERIFIER_SPEC_POINTER) != old_verifier
        ):
            raise ImageMigrationError("terminal TaskSpec image copies disagree")
        _set_value(document, _CANDIDATE_SPEC_POINTER, images["candidate"])
        _set_value(document, _VERIFIER_SPEC_POINTER, images["verifier"])
        _dump(item / relative, document)
    new_spec_sha = _sha256(item / _SPEC_FILES[0])
    if any(_sha256(item / relative) != new_spec_sha for relative in _SPEC_FILES):
        raise ImageMigrationError("migrated TaskSpec copies are not byte-identical")
    for relative in _BINDING_FILES:
        document = _read_json(item / relative)
        if _value(document, _BINDING_POINTER) != old_candidate:
            raise ImageMigrationError("terminal binding image copies disagree")
        _set_value(document, _BINDING_POINTER, images["candidate"])
        _dump(item / relative, document)
    for relative in _MANIFEST_FILES:
        document = _read_json(item / relative)
        if (
            _value(document, _CANDIDATE_MANIFEST_POINTER) != old_candidate
            or _value(document, _VERIFIER_MANIFEST_POINTER) != old_verifier
        ):
            raise ImageMigrationError("terminal manifest image copies disagree")
        _set_value(document, _CANDIDATE_MANIFEST_POINTER, images["candidate"])
        _set_value(document, _VERIFIER_MANIFEST_POINTER, images["verifier"])
        document["specification_sha256"] = new_spec_sha
        _dump(item / relative, document)
    generator = item / _GENERATOR
    generator.write_text(patch_generator_image_validation(generator.read_text()))

    evidence_root = output / "evidence"
    evidence_root.mkdir()
    evidence_records = {}
    for role, image in images.items():
        publication_path, cold_path = evidence[role]
        publication, cold = _read_json(publication_path), _read_json(cold_path)
        publication_sha = _sha256(publication_path)
        _validate_publication_evidence(
            role, image, publication, cold, publication_sha256=publication_sha
        )
        publication_target = evidence_root / f"{role}-publication.json"
        cold_target = evidence_root / f"{role}-cold-pull.json"
        shutil.copy2(publication_path, publication_target)
        shutil.copy2(cold_path, cold_target)
        publication_sha = _sha256(publication_target)
        if cold.get("publication_sha256") != publication_sha:
            raise ImageMigrationError(f"{role} cold pull does not bind publication")
        evidence_records[role] = {
            "publication_path": str(publication_target.relative_to(output)),
            "publication_sha256": publication_sha,
            "cold_pull_path": str(cold_target.relative_to(output)),
            "cold_pull_sha256": _sha256(cold_target),
        }

    changed_records = []
    for relative in changed:
        changed_records.append(
            {
                "path": relative,
                "old_sha256": _sha256(before / relative),
                "new_sha256": _sha256(restore / relative),
            }
        )
    receipt = {
        "schema_version": "capability-image-pointer-migration-v2",
        "state": "prepared_pending_fresh_runtime_validation",
        "item_name": item_name,
        "source": {
            "snapshot_id": _read_json(source / "pull-manifest.json").get(
                "remote_snapshot_id"
            ),
            "pull_manifest_sha256": _sha256(metadata / "pull-manifest.json"),
            "snapshot_capture_sha256": _sha256(metadata / "snapshot-capture.json"),
            "source_tree_sha256": _tree_digest(source_files),
            "source_file_count": len(source_files),
        },
        "pointer_changes": [
            {
                "role": "candidate",
                "old": old_candidate,
                "new": images["candidate"],
                **evidence_records["candidate"],
            },
            {
                "role": "verifier",
                "old": old_verifier,
                "new": images["verifier"],
                **evidence_records["verifier"],
            },
        ],
        "task_specification": {
            "old_sha256": old_spec_sha,
            "new_sha256": new_spec_sha,
        },
        "changed_files": changed_records,
        "repair_attempts_retained": [1, 2],
        "execution": {
            "runtime_validation": "pending_fresh_full_gates",
            "acceptance_claimed": False,
        },
    }
    (output / "migration-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    structural = {
        "schema_version": "capability-image-pointer-structural-v1",
        "state": "passed",
        "scope": "metadata-only pointer and derived-file consistency; no task execution",
        "changed_paths": list(changed),
    }
    (output / "structural-validation.json").write_text(
        json.dumps(structural, indent=2, sort_keys=True) + "\n"
    )
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": "capability-portable-runtime-bundle-v2",
        "state": "prepared_pending_fresh_runtime_validation",
        "migration_receipt_sha256": _sha256(output / "migration-receipt.json"),
        "structural_validation_sha256": _sha256(output / "structural-validation.json"),
        "changed_paths": list(changed),
        "files": _inventory(output, omit=manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output


def validate_pointer_migration_bundle(
    bundle: Path,
    *,
    expected_images: dict[str, str],
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> PointerImageMigrationPlan:
    """Fail closed unless a v2 bundle is the exact reviewed two-pointer change."""
    bundle = bundle.resolve()
    images = {
        role: canonical_registry_digest(value)
        for role, value in expected_images.items()
    }
    if set(images) != {"candidate", "verifier"}:
        raise ImageMigrationError("expected images must cover candidate and verifier")
    manifest_path = bundle / "manifest.json"
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != "capability-portable-runtime-bundle-v2"
        or manifest.get("state") != "prepared_pending_fresh_runtime_validation"
        or manifest.get("files") != _inventory(bundle, omit=manifest_path)
        or _sha256(manifest_path) != expected_manifest_sha256
    ):
        raise ImageMigrationError("v2 bundle manifest differs from reviewed bytes")
    receipt_path = bundle / "migration-receipt.json"
    receipt = _read_json(receipt_path)
    if (
        _sha256(receipt_path) != expected_receipt_sha256
        or receipt.get("schema_version") != "capability-image-pointer-migration-v2"
        or receipt.get("state") != "prepared_pending_fresh_runtime_validation"
        or receipt.get("execution")
        != {
            "runtime_validation": "pending_fresh_full_gates",
            "acceptance_claimed": False,
        }
    ):
        raise ImageMigrationError("v2 migration receipt differs from reviewed bytes")
    structural = _read_json(bundle / "structural-validation.json")
    if (
        structural.get("state") != "passed"
        or structural.get("scope")
        != "metadata-only pointer and derived-file consistency; no task execution"
    ):
        raise ImageMigrationError("v2 structural validation is incomplete")
    records = receipt.get("pointer_changes")
    if not isinstance(records, list) or [
        record.get("role") for record in records if isinstance(record, dict)
    ] != ["candidate", "verifier"]:
        raise ImageMigrationError("v2 pointer changes are incomplete")
    changes = []
    for record in records:
        role = record["role"]
        if record.get("new") != images[role] or record.get("old") == images[role]:
            raise ImageMigrationError(f"{role} pointer migration is invalid")
        publication_path = bundle / record["publication_path"]
        cold_path = bundle / record["cold_pull_path"]
        if _sha256(publication_path) != record.get("publication_sha256") or _sha256(
            cold_path
        ) != record.get("cold_pull_sha256"):
            raise ImageMigrationError(f"{role} image evidence hash mismatch")
        publication, cold = _read_json(publication_path), _read_json(cold_path)
        _validate_publication_evidence(
            role,
            images[role],
            publication,
            cold,
            publication_sha256=record["publication_sha256"],
        )
        if cold.get("publication_sha256") != record["publication_sha256"]:
            raise ImageMigrationError(f"{role} cold pull lost publication binding")
        changes.append(
            ImagePointerChange(
                role,
                record["old"],
                record["new"],
                record["publication_sha256"],
                record["cold_pull_sha256"],
            )
        )

    restore = bundle / "restore-seed"
    item_name = receipt.get("item_name")
    if not isinstance(item_name, str) or not (restore / "items" / item_name).is_dir():
        raise ImageMigrationError("v2 migration terminal item is missing")
    source = receipt.get("source")
    source_files = _read_json(bundle / "source-metadata/pull-manifest.json").get(
        "files"
    )
    if (
        not isinstance(source, dict)
        or not isinstance(source_files, dict)
        or source.get("source_file_count") != len(source_files)
        or source.get("source_tree_sha256") != _tree_digest(source_files)
        or source.get("pull_manifest_sha256")
        != _sha256(bundle / "source-metadata/pull-manifest.json")
        or source.get("snapshot_capture_sha256")
        != _sha256(bundle / "source-metadata/snapshot-capture.json")
    ):
        raise ImageMigrationError("v2 terminal source binding is incomplete")
    declared_changes = receipt.get("changed_files")
    if not isinstance(declared_changes, list):
        raise ImageMigrationError("v2 changed-file receipt is missing")
    by_path = {
        record.get("path"): record
        for record in declared_changes
        if isinstance(record, dict)
    }
    expected_paths = set(_changed_paths(item_name))
    if (
        set(by_path) != expected_paths
        or set(manifest.get("changed_paths", [])) != expected_paths
    ):
        raise ImageMigrationError("v2 changed-file coverage is not exact")
    before = bundle / "migration-before"
    for relative, old_digest in source_files.items():
        current = restore / relative
        if not current.is_file():
            raise ImageMigrationError(f"v2 restore seed is missing {relative}")
        if relative in expected_paths:
            record = by_path[relative]
            if (
                record.get("old_sha256") != old_digest
                or _sha256(before / relative) != old_digest
                or record.get("new_sha256") != _sha256(current)
            ):
                raise ImageMigrationError(f"v2 changed-file hash mismatch: {relative}")
        elif _sha256(current) != old_digest:
            raise ImageMigrationError(f"v2 undeclared source change: {relative}")

    item_root = restore / "items" / item_name
    candidate, verifier = changes
    for relative in _SPEC_FILES:
        old, new = (
            _read_json(before / f"items/{item_name}/{relative}"),
            _read_json(item_root / relative),
        )
        if set(_differences(old, new)) != {
            _CANDIDATE_SPEC_POINTER,
            _VERIFIER_SPEC_POINTER,
        } or (
            _value(new, _CANDIDATE_SPEC_POINTER) != candidate.new
            or _value(new, _VERIFIER_SPEC_POINTER) != verifier.new
        ):
            raise ImageMigrationError("v2 TaskSpec contains an extra semantic change")
    new_spec_sha = _sha256(item_root / _SPEC_FILES[0])
    for relative in _BINDING_FILES:
        old, new = (
            _read_json(before / f"items/{item_name}/{relative}"),
            _read_json(item_root / relative),
        )
        if (
            _differences(old, new) != [_BINDING_POINTER]
            or _value(new, _BINDING_POINTER) != candidate.new
        ):
            raise ImageMigrationError("v2 binding contains an extra semantic change")
    expected_manifest_diffs = {
        ("specification_sha256",),
        _CANDIDATE_MANIFEST_POINTER,
        _VERIFIER_MANIFEST_POINTER,
    }
    for relative in _MANIFEST_FILES:
        old, new = (
            _read_json(before / f"items/{item_name}/{relative}"),
            _read_json(item_root / relative),
        )
        if set(_differences(old, new)) != expected_manifest_diffs or (
            new.get("specification_sha256") != new_spec_sha
            or _value(new, _CANDIDATE_MANIFEST_POINTER) != candidate.new
            or _value(new, _VERIFIER_MANIFEST_POINTER) != verifier.new
        ):
            raise ImageMigrationError("v2 manifest contains an extra derived change")
    old_generator = (before / f"items/{item_name}/{_GENERATOR}").read_text()
    if (item_root / _GENERATOR).read_text() != patch_generator_image_validation(
        old_generator
    ):
        raise ImageMigrationError("generator changed beyond registry digest validation")
    attempts = receipt.get("repair_attempts_retained")
    if attempts != [1, 2]:
        raise ImageMigrationError("v2 migration does not retain both repair rounds")
    for directory in ("repairs", "repair-history"):
        parent = restore / directory / item_name
        if {path.name for path in parent.iterdir() if path.is_dir()} != {
            "attempt-1",
            "attempt-2",
        }:
            raise ImageMigrationError("v2 migration repair history is incomplete")
    accepted = _read_json(item_root / "contract/accepted.json")
    if _read_json(bundle / "accepted.json") != [accepted]:
        raise ImageMigrationError("v2 accepted contract was changed")
    specification = receipt.get("task_specification")
    old_spec_sha = _sha256(before / f"items/{item_name}/{_SPEC_FILES[0]}")
    if specification != {
        "old_sha256": old_spec_sha,
        "new_sha256": new_spec_sha,
    }:
        raise ImageMigrationError("v2 TaskSpec hash binding is wrong")
    status = _read_json(item_root / "status.json")
    if status.get("state") != "failed" or [
        record.get("round")
        for record in status.get("repairs", [])
        if isinstance(record, dict)
    ] != [1, 2]:
        raise ImageMigrationError("v2 terminal status lost repair budget history")
    item = accepted
    return PointerImageMigrationPlan(
        bundle=bundle,
        restore_seed=restore,
        item_name=item_name,
        item=item,
        pointer_changes=tuple(changes),
        old_specification_sha256=old_spec_sha,
        new_specification_sha256=new_spec_sha,
        changed_paths=tuple(sorted(expected_paths)),
        repair_attempts=(1, 2),
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
        migration_receipt_sha256=_sha256(receipt_path),
        bundle_manifest_sha256=_sha256(manifest_path),
    )
