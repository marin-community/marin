"""Pack and restore a manifest-bound portable-runtime bundle as opaque bytes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import NoReturn

SCHEMA = "capability-portable-runtime-transport-v1"
BUNDLE_SCHEMAS = {
    "capability-portable-runtime-bundle-v1",
    "capability-portable-runtime-bundle-v2",
}
EVALUATION_SCHEMA = "capability-runtime-evaluation-bundle-v1"
EVALUATION_TRANSPORT_SCHEMA = "capability-runtime-evaluation-transport-v1"
REGRADE_SCHEMA = "capability-fixed-submission-regrade-bundle-v1"
REGRADE_TRANSPORT_SCHEMA = "capability-fixed-submission-regrade-transport-v1"


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name or not path.parts or path.is_absolute()
        or ".." in path.parts or str(path) != name
        or "\\" in name or "\x00" in name
    ):
        fail("portable-runtime transport contains an unsafe member")
    return path


def bundle_members(source: Path) -> dict[str, Path]:
    manifest_path = source / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        declared = manifest["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("portable-runtime bundle manifest is invalid") from error
    if manifest.get("schema_version") not in BUNDLE_SCHEMAS or not isinstance(
        declared, dict
    ):
        fail("portable-runtime bundle manifest has the wrong schema")
    members: dict[str, Path] = {"manifest.json": manifest_path}
    for name, expected in declared.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            fail("portable-runtime bundle inventory is malformed")
        path = source / safe_name(name)
        if path.is_symlink() or not path.is_file():
            fail("portable-runtime bundle member is unavailable")
        if sha256(path.read_bytes()) != expected:
            fail("portable-runtime bundle member hash mismatch")
        members[name] = path
    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(declared):
        fail("portable-runtime bundle file set differs from its manifest")
    return members


def evaluation_members(source: Path) -> dict[str, Path]:
    """Validate the self-contained frozen evaluation input inventory."""
    if source.is_symlink() or not source.is_dir():
        fail("evaluation bundle source is unavailable")
    if any(path.is_symlink() for path in source.rglob("*")):
        fail("evaluation bundle contains a symlink")
    manifest_path = source / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        declared = manifest["files"]
        plan_sha256 = manifest["plan_sha256"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation bundle manifest is invalid") from error
    if manifest.get("schema_version") != EVALUATION_SCHEMA or not isinstance(
        declared, dict
    ):
        fail("evaluation bundle manifest has the wrong schema")
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        fail("evaluation bundle manifest lacks a plan fingerprint")
    members: dict[str, Path] = {"manifest.json": manifest_path}
    for name, expected in declared.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            fail("evaluation bundle inventory is malformed")
        path = source / safe_name(name)
        if (
            path.is_symlink()
            or not path.is_file()
            or sha256(path.read_bytes()) != expected
        ):
            fail("evaluation bundle member hash mismatch")
        members[name] = path
    declared_directories = manifest.get("directories")
    if declared_directories is not None:
        if not isinstance(declared_directories, list):
            fail("evaluation bundle directory inventory is malformed")
        directories: set[str] = set()
        for name in declared_directories:
            if not isinstance(name, str):
                fail("evaluation bundle directory inventory is malformed")
            path = source / safe_name(name)
            if name in directories or path.is_symlink() or not path.is_dir():
                fail("evaluation bundle directory inventory is malformed")
            directories.add(name)
            members[name] = path
        if directories.intersection(declared):
            fail("evaluation bundle inventory overlaps files and directories")
        actual_directories = {
            path.relative_to(source).as_posix()
            for path in source.rglob("*")
            if path.is_dir()
        }
        if actual_directories != directories:
            fail("evaluation bundle directory set differs from its manifest")
    actual = {
        p.relative_to(source).as_posix()
        for p in source.rglob("*")
        if p.is_file() and p != manifest_path
    }
    if actual != set(declared) or "plan.json" not in declared:
        fail("evaluation bundle file set differs from its manifest")
    plan_path = source / "plan.json"
    if sha256(plan_path.read_bytes()) != plan_sha256:
        fail("evaluation bundle plan fingerprint mismatch")
    try:
        plan = json.loads(plan_path.read_text())
        inputs = plan["inputs"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("evaluation bundle plan is invalid") from error
    if not isinstance(inputs, dict) or not inputs:
        fail("evaluation bundle plan lacks inputs")
    for record in inputs.values():
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            fail("evaluation bundle plan input is malformed")
        relative = safe_name(record["path"])
        resolved = source / relative
        if (
            resolved.is_symlink()
            or not resolved.exists()
            or not resolved.is_relative_to(source)
        ):
            fail("evaluation bundle plan input escapes the bundle")
    return members


def regrade_members(source: Path) -> dict[str, Path]:
    """Verify the exact external-plan wrapper; never accept extra files or links."""
    if (
        source.is_symlink()
        or not source.is_dir()
        or any(p.is_symlink() for p in source.rglob("*"))
    ):
        fail("regrade bundle has an unavailable or linked source")
    try:
        manifest = json.loads((source / "manifest.json").read_text())
        files = manifest["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("regrade bundle manifest is invalid") from error
    if manifest.get("schema_version") != REGRADE_SCHEMA or not isinstance(files, dict):
        fail("regrade bundle schema is invalid")
    if (
        not isinstance(manifest.get("plan_sha256"), str)
        or len(manifest["plan_sha256"]) != 64
    ):
        fail("regrade bundle plan fingerprint is invalid")
    required = {
        "plan.json",
        "input/manifest.json",
        "input/package/manifest.json",
        "input/bundle/binding.json",
        "input/bundle/specification.json",
        "input/bundle/renderings.json",
        "input/bundle/controls.json",
        "input/tools/dt.py",
    }
    if not required <= set(files):
        fail("regrade bundle lacks required inputs")
    members = {"manifest.json": source / "manifest.json"}
    for name, expected in files.items():
        if (
            not isinstance(name, str)
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            fail("regrade bundle inventory is malformed")
        path = source / safe_name(name)
        if not path.is_file() or sha256(path.read_bytes()) != expected:
            fail("regrade bundle member hash mismatch")
        members[name] = path
    actual = {
        p.relative_to(source).as_posix()
        for p in source.rglob("*")
        if p.is_file() and p != source / "manifest.json"
    }
    if actual != set(files) or files["plan.json"] != manifest["plan_sha256"]:
        fail("regrade bundle inventory or plan fingerprint differs")
    directories = manifest.get("directories")
    if directories is not None:
        if not isinstance(directories, list):
            fail("regrade bundle directory inventory is malformed")
        declared_directories: set[str] = set()
        for name in directories:
            if not isinstance(name, str):
                fail("regrade bundle directory inventory is malformed")
            path = source / safe_name(name)
            if name in declared_directories or not path.is_dir():
                fail("regrade bundle directory inventory is malformed")
            declared_directories.add(name)
            members[name] = path
        actual_directories = {
            p.relative_to(source).as_posix()
            for p in source.rglob("*") if p.is_dir()
        }
        if actual_directories != declared_directories:
            fail("regrade bundle directory set differs from its manifest")
    evaluation_members(source / "input")
    return members


def build_evaluation_transport(
    source: Path, archive: Path, transport_path: Path, *, plan_sha256: str
) -> dict:
    members = evaluation_members(source)
    manifest_sha256 = sha256((source / "manifest.json").read_bytes())
    if plan_sha256 != sha256((source / "plan.json").read_bytes()):
        fail("evaluation transport plan fingerprint mismatch")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as output:
        for name, path in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.mtime = 0
            info.mode = stat.S_IMODE(path.stat().st_mode) & 0o777
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.size = 0
                output.addfile(info)
            else:
                data = path.read_bytes()
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
    compressed = gzip.compress(payload.getvalue(), mtime=0)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(compressed)
    transport = {
        "schema_version": EVALUATION_TRANSPORT_SCHEMA,
        "archive_name": archive.name,
        "archive_sha256": sha256(compressed),
        "member_count": len(members),
        "bundle_manifest_sha256": manifest_sha256,
        "plan_sha256": plan_sha256,
    }
    transport_path.write_text(json.dumps(transport, indent=2, sort_keys=True) + "\n")
    return transport


def extract_evaluation_transport(
    archive: Path,
    transport_path: Path,
    destination: Path,
    *,
    expected_manifest_sha256: str,
    expected_plan_sha256: str,
) -> dict:
    try:
        transport = json.loads(transport_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("evaluation transport manifest is invalid") from error
    if (
        transport.get("schema_version") != EVALUATION_TRANSPORT_SCHEMA
        or transport.get("archive_name") != archive.name
        or transport.get("bundle_manifest_sha256") != expected_manifest_sha256
        or transport.get("plan_sha256") != expected_plan_sha256
        or not isinstance(transport.get("member_count"), int)
        or transport["member_count"] < 1
    ):
        fail("evaluation transport manifest does not match frozen input")
    data = archive.read_bytes()
    if sha256(data) != transport.get("archive_sha256"):
        fail("evaluation transport archive fingerprint mismatch")
    if destination.exists() or destination.is_symlink():
        fail("evaluation transport destination must be new")
    with tempfile.TemporaryDirectory(prefix="capability-evaluation-") as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir()
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as source_tar:
                seen: set[str] = set()
                for member in source_tar.getmembers():
                    name = member.name
                    relative = safe_name(name)
                    if (
                        name in seen
                        or not (member.isfile() or member.isdir())
                        or member.issym()
                        or member.islnk()
                        or member.isdev()
                        or member.isfifo()
                    ):
                        fail("evaluation transport contains an unsafe member")
                    seen.add(name)
                    output = root / relative
                    if member.isdir():
                        output.mkdir(parents=True, exist_ok=False)
                        os.chmod(output, member.mode & 0o777)
                    else:
                        payload = source_tar.extractfile(member)
                        if payload is None:
                            fail("evaluation transport member is unreadable")
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with output.open("wb") as handle:
                            shutil.copyfileobj(payload, handle)
                        os.chmod(output, member.mode & 0o777)
        except (OSError, tarfile.TarError) as error:
            raise ValueError("evaluation transport archive is unreadable") from error
        if len(seen) != transport["member_count"]:
            fail("evaluation transport member count mismatch")
        evaluation_members(root)
        if (
            sha256((root / "manifest.json").read_bytes()) != expected_manifest_sha256
            or sha256((root / "plan.json").read_bytes()) != expected_plan_sha256
        ):
            fail("restored evaluation bundle fingerprint mismatch")
        shutil.copytree(root, destination)
    return transport


def build_regrade_transport(
    source: Path, archive: Path, transport_path: Path, *, plan_sha256: str
) -> dict:
    members = regrade_members(source)
    if sha256((source / "plan.json").read_bytes()) != plan_sha256:
        fail("regrade transport plan fingerprint mismatch")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as output:
        for name, path in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.mtime = 0
            info.mode = stat.S_IMODE(path.stat().st_mode) & 0o777
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                output.addfile(info)
            else:
                data = path.read_bytes()
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
    compressed = gzip.compress(payload.getvalue(), mtime=0)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(compressed)
    transport = {
        "schema_version": REGRADE_TRANSPORT_SCHEMA,
        "archive_name": archive.name,
        "archive_sha256": sha256(compressed),
        "member_count": len(members),
        "bundle_manifest_sha256": sha256((source / "manifest.json").read_bytes()),
        "plan_sha256": plan_sha256,
    }
    transport_path.write_text(json.dumps(transport, indent=2, sort_keys=True) + "\n")
    return transport


def extract_regrade_transport(
    archive: Path,
    transport_path: Path,
    destination: Path,
    *,
    expected_manifest_sha256: str,
    expected_plan_sha256: str,
) -> dict:
    try:
        transport = json.loads(transport_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("regrade transport manifest is invalid") from error
    if (
        transport.get("schema_version") != REGRADE_TRANSPORT_SCHEMA
        or transport.get("archive_name") != archive.name
        or transport.get("bundle_manifest_sha256") != expected_manifest_sha256
        or transport.get("plan_sha256") != expected_plan_sha256
        or type(transport.get("member_count")) is not int
        or transport["member_count"] < 1
    ):
        fail("regrade transport manifest does not match frozen input")
    data = archive.read_bytes()
    if sha256(data) != transport.get("archive_sha256"):
        fail("regrade transport archive fingerprint mismatch")
    if destination.exists() or destination.is_symlink():
        fail("regrade transport destination must be new")
    with tempfile.TemporaryDirectory(prefix="capability-regrade-") as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir()
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as source_tar:
                seen: set[str] = set()
                for member in source_tar.getmembers():
                    relative = safe_name(member.name)
                    if (
                        member.name in seen
                        or not (member.isfile() or member.isdir())
                        or member.issym()
                        or member.islnk()
                        or member.isdev()
                        or member.isfifo()
                    ):
                        fail("regrade transport contains an unsafe member")
                    seen.add(member.name)
                    output = root / relative
                    if member.isdir():
                        output.mkdir(parents=True, exist_ok=True)
                        os.chmod(output, member.mode & 0o777)
                        continue
                    payload = source_tar.extractfile(member)
                    if payload is None:
                        fail("regrade transport member is unreadable")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("wb") as handle:
                        shutil.copyfileobj(payload, handle)
                    os.chmod(output, member.mode & 0o777)
        except (OSError, tarfile.TarError) as error:
            raise ValueError("regrade transport archive is unreadable") from error
        if len(seen) != transport["member_count"]:
            fail("regrade transport member count mismatch")
        if set(regrade_members(root)) != seen:
            fail("regrade transport member set differs from its manifest")
        if (
            sha256((root / "manifest.json").read_bytes()) != expected_manifest_sha256
            or sha256((root / "plan.json").read_bytes()) != expected_plan_sha256
        ):
            fail("restored regrade bundle fingerprint mismatch")
        shutil.copytree(root, destination)
    return transport


def build_transport(
    source: Path,
    archive: Path,
    transport_path: Path,
    *,
    reviewed_manifest_sha256: str,
    migration_receipt_sha256: str,
) -> dict:
    members = bundle_members(source)
    if sha256((source / "manifest.json").read_bytes()) != reviewed_manifest_sha256:
        fail("reviewed portable-runtime manifest fingerprint mismatch")
    if (
        sha256((source / "migration-receipt.json").read_bytes())
        != migration_receipt_sha256
    ):
        fail("reviewed migration receipt fingerprint mismatch")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as output:
        for name, path in sorted(members.items()):
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = stat.S_IMODE(path.stat().st_mode) & 0o777
            output.addfile(info, io.BytesIO(data))
    compressed = gzip.compress(payload.getvalue(), mtime=0)
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(compressed)
    transport = {
        "schema_version": SCHEMA,
        "archive_name": archive.name,
        "archive_sha256": sha256(compressed),
        "member_count": len(members),
        "reviewed_bundle_manifest_sha256": reviewed_manifest_sha256,
        "migration_receipt_sha256": migration_receipt_sha256,
    }
    transport_path.write_text(json.dumps(transport, indent=2, sort_keys=True) + "\n")
    return transport


def extract_transport(
    archive: Path,
    transport_path: Path,
    destination: Path,
    *,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str,
) -> dict:
    try:
        transport = json.loads(transport_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("portable-runtime transport manifest is invalid") from error
    if (
        transport.get("schema_version") != SCHEMA
        or transport.get("archive_name") != archive.name
        or transport.get("reviewed_bundle_manifest_sha256") != expected_manifest_sha256
        or transport.get("migration_receipt_sha256") != expected_receipt_sha256
        or not isinstance(transport.get("member_count"), int)
        or transport["member_count"] < 1
    ):
        fail("portable-runtime transport manifest does not match reviewed input")
    data = archive.read_bytes()
    if sha256(data) != transport.get("archive_sha256"):
        fail("portable-runtime transport archive fingerprint mismatch")
    if destination.exists() or destination.is_symlink():
        fail("portable-runtime transport destination must be new")
    with tempfile.TemporaryDirectory(
        prefix="capability-portable-runtime-"
    ) as temporary:
        root = Path(temporary) / "bundle"
        root.mkdir()
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as source:
                seen: set[str] = set()
                for member in source.getmembers():
                    name = member.name
                    relative = safe_name(name)
                    if (
                        name in seen
                        or not member.isfile()
                        or member.issym()
                        or member.islnk()
                        or member.isdev()
                        or member.isfifo()
                    ):
                        fail("portable-runtime transport contains an unsafe member")
                    seen.add(name)
                    payload = source.extractfile(member)
                    if payload is None:
                        fail("portable-runtime transport member is unreadable")
                    output = root / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("wb") as handle:
                        shutil.copyfileobj(payload, handle)
                    os.chmod(output, member.mode & 0o777)
        except (OSError, tarfile.TarError) as error:
            raise ValueError(
                "portable-runtime transport archive is unreadable"
            ) from error
        if len(seen) != transport["member_count"]:
            fail("portable-runtime transport member count mismatch")
        bundle_members(root)
        if sha256((root / "manifest.json").read_bytes()) != expected_manifest_sha256:
            fail("restored portable-runtime manifest fingerprint mismatch")
        if (
            sha256((root / "migration-receipt.json").read_bytes())
            != expected_receipt_sha256
        ):
            fail("restored migration receipt fingerprint mismatch")
        shutil.copytree(root, destination)
    return transport


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack")
    evaluation_pack = subparsers.add_parser("pack-evaluation")
    evaluation_restore = subparsers.add_parser("restore-evaluation")
    regrade_pack = subparsers.add_parser("pack-regrade")
    regrade_restore = subparsers.add_parser("restore-regrade")
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--archive", type=Path, required=True)
    pack.add_argument("--transport", type=Path, required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--archive", type=Path, required=True)
    restore.add_argument("--transport", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    evaluation_pack.add_argument("--source", type=Path, required=True)
    evaluation_pack.add_argument("--archive", type=Path, required=True)
    evaluation_pack.add_argument("--transport", type=Path, required=True)
    evaluation_restore.add_argument("--archive", type=Path, required=True)
    evaluation_restore.add_argument("--transport", type=Path, required=True)
    evaluation_restore.add_argument("--destination", type=Path, required=True)
    regrade_pack.add_argument("--source", type=Path, required=True)
    regrade_pack.add_argument("--archive", type=Path, required=True)
    regrade_pack.add_argument("--transport", type=Path, required=True)
    regrade_restore.add_argument("--archive", type=Path, required=True)
    regrade_restore.add_argument("--transport", type=Path, required=True)
    regrade_restore.add_argument("--destination", type=Path, required=True)
    for command in (pack, restore):
        command.add_argument("--expected-manifest-sha256", required=True)
        command.add_argument("--expected-receipt-sha256", required=True)
    for command in (evaluation_pack, evaluation_restore):
        command.add_argument("--expected-manifest-sha256", required=True)
        command.add_argument("--plan-sha256", required=True)
    for command in (regrade_pack, regrade_restore):
        command.add_argument("--expected-manifest-sha256", required=True)
        command.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.command == "pack-regrade":
        if (
            sha256((args.source / "manifest.json").read_bytes())
            != args.expected_manifest_sha256
        ):
            fail("regrade transport manifest fingerprint mismatch")
        transport = build_regrade_transport(
            args.source, args.archive, args.transport, plan_sha256=args.plan_sha256
        )
    elif args.command == "restore-regrade":
        transport = extract_regrade_transport(
            args.archive,
            args.transport,
            args.destination,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_plan_sha256=args.plan_sha256,
        )
    elif args.command == "pack-evaluation":
        if (
            sha256((args.source / "manifest.json").read_bytes())
            != args.expected_manifest_sha256
        ):
            fail("evaluation transport manifest fingerprint mismatch")
        transport = build_evaluation_transport(
            args.source, args.archive, args.transport, plan_sha256=args.plan_sha256
        )
    elif args.command == "restore-evaluation":
        transport = extract_evaluation_transport(
            args.archive,
            args.transport,
            args.destination,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_plan_sha256=args.plan_sha256,
        )
    elif args.command == "pack":
        transport = build_transport(
            args.source,
            args.archive,
            args.transport,
            reviewed_manifest_sha256=args.expected_manifest_sha256,
            migration_receipt_sha256=args.expected_receipt_sha256,
        )
    else:
        transport = extract_transport(
            args.archive,
            args.transport,
            args.destination,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_receipt_sha256=args.expected_receipt_sha256,
        )
    print(json.dumps(transport, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
