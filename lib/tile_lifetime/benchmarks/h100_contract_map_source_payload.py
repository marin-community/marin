# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare and restore an exact source capsule for the H100 evidence runner."""

import argparse
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
ALLOWLIST_RELATIVE_PATH = Path("lib/tile_lifetime/benchmarks/h100_contract_map_source_allowlist.json")
CAPSULE_FILENAME = "h100-evidence-source-capsule.zip"
MANIFEST_FILENAME = "h100-evidence-source-manifest.json"
LAUNCHER_FILENAME = "h100_contract_map_source_payload.py"
RUNNER_RELATIVE_PATH = Path("lib/tile_lifetime/benchmarks/h100_contract_map_backend_runner.py")
CUDA_BIN = Path("/usr/local/cuda-13.2/bin")
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CAPSULE_BYTES = 20 * 1024 * 1024
MAX_PAYLOAD_BYTES = 25 * 1024 * 1024
MAX_EXTRACTED_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_MEMBERS = 10_000
SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_PARTS = frozenset({".git", ".ssh", ".aws", ".gnupg", "__pycache__"})
FORBIDDEN_NAMES = frozenset(
    {
        ".env",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "application_default_credentials.json",
        "credentials",
        "credentials.json",
        "id_ed25519",
        "id_rsa",
    }
)
REQUIRED_EXACT_PATHS = (
    ".agents/projects/tile_lifetime_compiler/h100_contract_map_backend_evidence.md",
    "LICENSE",
    "lib/tile_lifetime/README.md",
    "lib/tile_lifetime/benchmarks/h100_contract_map_backend_runner.py",
    "lib/tile_lifetime/benchmarks/h100_contract_map_backend_training.py",
    "lib/tile_lifetime/benchmarks/h100_contract_map_source_allowlist.json",
    "lib/tile_lifetime/benchmarks/h100_contract_map_source_payload.py",
    "lib/tile_lifetime/pyproject.toml",
    "pyproject.toml",
    "uv.lock",
)


def _run_git(repository: Path, *arguments: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout.rstrip("\n") if text else completed.stdout


def _require_sha(value: Any, pattern: re.Pattern[str], name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} has invalid digest syntax: {value!r}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _validate_relative_path(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a normalized relative path: {value!r}")
    path = PurePosixPath(value)
    if not value or str(path) != value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{name} must be a normalized relative path: {value!r}")
    if FORBIDDEN_PARTS.intersection(path.parts) or path.name in FORBIDDEN_NAMES:
        raise ValueError(f"{name} names forbidden credential or repository state: {value!r}")
    return path


def _validate_symlink_target(path: PurePosixPath, target_value: Any) -> str:
    if not isinstance(target_value, str):
        raise ValueError(f"symlink target must be a string: {target_value!r}")
    target = PurePosixPath(target_value)
    normalized_target = posixpath.normpath(str(path.parent / target))
    if (
        not target_value
        or "\0" in target_value
        or target.is_absolute()
        or normalized_target == ".."
        or normalized_target.startswith("../")
    ):
        raise ValueError(f"source capsule symlink escapes the workspace: {path} -> {target_value}")
    return target_value


def _source_identity(source_root: Path, source_sha: str) -> tuple[str, str]:
    source_sha = _require_sha(source_sha, SHA1_PATTERN, "source SHA")
    if _run_git(source_root, "rev-parse", "--show-object-format") != "sha1":
        raise ValueError("source capsule supports only SHA-1 Git repositories")
    head = str(_run_git(source_root, "rev-parse", "HEAD"))
    if head != source_sha:
        raise ValueError(f"source checkout is {head}, expected {source_sha}")
    if _run_git(source_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("source checkout must be globally clean before capsule preparation")
    tree = str(_run_git(source_root, "rev-parse", f"{source_sha}^{{tree}}"))
    return source_sha, _require_sha(tree, SHA1_PATTERN, "source tree")


def _git_tree_records(repository: Path, source_sha: str) -> dict[str, tuple[str, str]]:
    raw = _run_git(repository, "ls-tree", "-r", "-z", source_sha, text=False)
    assert isinstance(raw, bytes)
    records: dict[str, tuple[str, str]] = {}
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", maxsplit=1)
        mode, object_type, object_id = metadata.split()
        source_path = os.fsdecode(raw_path)
        if object_type != b"blob":
            records[source_path] = (mode.decode(), "")
            continue
        records[source_path] = (mode.decode(), object_id.decode())
    return records


def _git_blob(repository: Path, object_id: str, source_path: str) -> bytes:
    if not object_id:
        raise ValueError(f"source capsule path is not a Git blob: {source_path}")
    contents = _run_git(repository, "cat-file", "blob", object_id, text=False)
    assert isinstance(contents, bytes)
    return contents


def _load_allowlist(
    source_root: Path,
    tree_records: dict[str, tuple[str, str]],
) -> tuple[str, ...]:
    allowlist_path = ALLOWLIST_RELATIVE_PATH.as_posix()
    if allowlist_path not in tree_records:
        raise ValueError("source commit omits the source capsule allowlist")
    raw = _git_blob(source_root, tree_records[allowlist_path][1], allowlist_path)
    try:
        allowlist = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("source capsule allowlist is not valid JSON") from error
    if not isinstance(allowlist, dict) or set(allowlist) != {"exact", "recursive", "schema_version"}:
        raise ValueError("source capsule allowlist must use the closed schema")
    if _canonical_json(allowlist) != raw:
        raise ValueError("source capsule allowlist must use canonical JSON encoding")
    if type(allowlist["schema_version"]) is not int or allowlist["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"source capsule allowlist schema_version must be {SCHEMA_VERSION}")
    exact = allowlist["exact"]
    recursive = allowlist["recursive"]
    if not isinstance(exact, list) or not exact or not all(isinstance(value, str) for value in exact):
        raise ValueError("source capsule exact allowlist must be a nonempty string list")
    if not isinstance(recursive, list) or len(recursive) != 1:
        raise ValueError("source capsule recursive allowlist must contain exactly one package rule")
    rule = recursive[0]
    if rule != {"path": "lib/tile_lifetime/src/tile_lifetime", "suffix": ".py"}:
        raise ValueError("source capsule recursive allowlist must cover exactly the tile_lifetime Python package")

    paths = {_validate_relative_path(value, "allowlist path").as_posix() for value in exact}
    package_paths = {
        path
        for path, (_mode, object_id) in tree_records.items()
        if object_id and path.startswith(f"{rule['path']}/") and path.endswith(rule["suffix"])
    }
    paths.update(package_paths)
    if ALLOWLIST_RELATIVE_PATH.as_posix() not in paths:
        raise ValueError("source capsule allowlist must include itself")
    if exact != sorted(set(exact)):
        raise ValueError("source capsule exact allowlist must be unique and source ordered")
    if tuple(exact) != REQUIRED_EXACT_PATHS:
        raise ValueError("source capsule exact allowlist drifted from the reviewed runtime/config path set")

    missing = sorted(path for path in paths if path not in tree_records or not tree_records[path][1])
    if missing:
        raise ValueError(f"source capsule allowlist names paths absent from the immutable source tree: {missing}")
    return tuple(sorted(paths))


def _capsule_records(
    source_root: Path,
    paths: tuple[str, ...],
    tree_records: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    records: list[dict[str, Any]] = []
    contents_by_path: dict[str, bytes] = {}
    total_size = 0
    for source_path in paths:
        path = _validate_relative_path(source_path, "capsule path")
        mode, object_id = tree_records[source_path]
        blob = _git_blob(source_root, object_id, source_path)
        if mode in {"100644", "100755"}:
            member_type = "file"
            contents = blob
        elif mode == "120000":
            member_type = "symlink"
            contents = blob
            try:
                target = contents.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"source capsule symlink target is not UTF-8: {source_path}") from error
            _validate_symlink_target(path, target)
        else:
            raise ValueError(f"source capsule does not support Git mode {mode} at {source_path}")
        if len(contents) > MAX_MEMBER_BYTES:
            raise ValueError(f"source capsule member {source_path} exceeds {MAX_MEMBER_BYTES} bytes")
        total_size += len(contents)
        if total_size > MAX_EXTRACTED_BYTES:
            raise ValueError(f"source capsule expands beyond {MAX_EXTRACTED_BYTES} bytes")
        contents_by_path[source_path] = contents
        records.append(
            {
                "mode": mode,
                "path": source_path,
                "sha256": hashlib.sha256(contents).hexdigest(),
                "size": len(contents),
                "type": member_type,
            }
        )
    if len(records) > MAX_MEMBERS:
        raise ValueError(f"source capsule exceeds {MAX_MEMBERS} members")
    return records, contents_by_path


def _write_capsule(path: Path, contents_by_path: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, contents in sorted(contents_by_path.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, contents)


def prepare_source_payload(source_root: Path, source_sha: str, output_directory: Path) -> dict[str, Any]:
    """Create an immutable allowlisted source capsule from a clean exact checkout."""
    source_root = source_root.resolve()
    output_directory = output_directory.resolve()
    source_sha, source_tree = _source_identity(source_root, source_sha)
    tree_records = _git_tree_records(source_root, source_sha)
    allowlisted_paths = _load_allowlist(source_root, tree_records)
    records, contents_by_path = _capsule_records(source_root, allowlisted_paths, tree_records)
    if output_directory.exists():
        raise ValueError(f"payload output already exists: {output_directory}")
    if output_directory.is_relative_to(source_root):
        raise ValueError("payload output must be outside the exact source checkout")
    output_directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{output_directory.name}.", dir=output_directory.parent) as temp_name:
        payload = Path(temp_name) / "payload"
        payload.mkdir()
        capsule_path = payload / CAPSULE_FILENAME
        _write_capsule(capsule_path, contents_by_path)
        if capsule_path.stat().st_size > MAX_CAPSULE_BYTES:
            raise ValueError(f"source capsule exceeds {MAX_CAPSULE_BYTES} bytes")
        manifest = {
            "archive": {"filename": CAPSULE_FILENAME, "sha256": _sha256(capsule_path)},
            "members": records,
            "schema_version": SCHEMA_VERSION,
            "source": {"commit": source_sha, "tree": source_tree},
        }
        manifest_path = payload / MANIFEST_FILENAME
        manifest_path.write_bytes(_canonical_json(manifest))
        launcher_relative_path = "lib/tile_lifetime/benchmarks/h100_contract_map_source_payload.py"
        (payload / LAUNCHER_FILENAME).write_bytes(contents_by_path[launcher_relative_path])
        payload_size = sum(path.stat().st_size for path in payload.iterdir())
        if payload_size > MAX_PAYLOAD_BYTES:
            raise ValueError(f"source capsule transport exceeds {MAX_PAYLOAD_BYTES} bytes before Iris compression")
        if _source_identity(source_root, source_sha) != (source_sha, source_tree):
            raise ValueError("source identity changed while preparing the capsule")
        payload.replace(output_directory)

    result = dict(manifest)
    result["launcher_sha256"] = _sha256(output_directory / LAUNCHER_FILENAME)
    result["manifest_sha256"] = _sha256(output_directory / MANIFEST_FILENAME)
    result["payload_directory"] = str(output_directory)
    return result


def _validate_manifest(
    raw: bytes,
    *,
    expected_manifest_sha256: str,
    expected_source_sha: str,
    expected_source_tree: str,
) -> dict[str, Any]:
    expected_manifest_sha256 = _require_sha(expected_manifest_sha256, SHA256_PATTERN, "manifest SHA-256")
    expected_source_sha = _require_sha(expected_source_sha, SHA1_PATTERN, "source SHA")
    expected_source_tree = _require_sha(expected_source_tree, SHA1_PATTERN, "source tree")
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("source manifest SHA-256 does not match the trusted launch identity")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("source manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != {"archive", "members", "schema_version", "source"}:
        raise ValueError("source manifest must use the closed schema")
    if _canonical_json(manifest) != raw:
        raise ValueError("source manifest must use canonical JSON encoding")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"source manifest schema_version must be {SCHEMA_VERSION}")
    if manifest["source"] != {"commit": expected_source_sha, "tree": expected_source_tree}:
        raise ValueError("source manifest commit or tree differs from the trusted launch identity")
    archive = manifest["archive"]
    if not isinstance(archive, dict) or set(archive) != {"filename", "sha256"}:
        raise ValueError("source manifest archive record must use the closed schema")
    if archive["filename"] != CAPSULE_FILENAME:
        raise ValueError(f"source manifest archive filename must be {CAPSULE_FILENAME}")
    _require_sha(archive["sha256"], SHA256_PATTERN, "capsule SHA-256")
    members = manifest["members"]
    if not isinstance(members, list) or not members or len(members) > MAX_MEMBERS:
        raise ValueError("source manifest members must be a bounded nonempty list")
    paths: set[str] = set()
    total_size = 0
    for record in members:
        if not isinstance(record, dict) or set(record) != {"mode", "path", "sha256", "size", "type"}:
            raise ValueError("source capsule member must use the closed schema")
        path = _validate_relative_path(record["path"], "capsule path")
        if record["path"] in paths:
            raise ValueError(f"source capsule repeats path: {record['path']}")
        paths.add(record["path"])
        if record["type"] == "file":
            if record["mode"] not in {"100644", "100755"}:
                raise ValueError(f"source capsule file has invalid mode: {record}")
        elif record["type"] == "symlink":
            if record["mode"] != "120000":
                raise ValueError(f"source capsule symlink has invalid mode: {record}")
        else:
            raise ValueError(f"source capsule member has invalid type: {record}")
        if type(record["size"]) is not int or not 0 <= record["size"] <= MAX_MEMBER_BYTES:
            raise ValueError(f"source capsule member has invalid size: {record}")
        _require_sha(record["sha256"], SHA256_PATTERN, "capsule member SHA-256")
        total_size += record["size"]
        if total_size > MAX_EXTRACTED_BYTES:
            raise ValueError(f"source capsule expands beyond {MAX_EXTRACTED_BYTES} bytes")
        if record["type"] == "symlink" and not path.parts:
            raise ValueError("source capsule symlink path must be nonempty")
    if [record["path"] for record in members] != sorted(paths):
        raise ValueError("source capsule members must be in canonical path order")
    return manifest


def load_and_verify_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_source_sha: str,
    expected_source_tree: str,
) -> dict[str, Any]:
    """Load one trusted canonical source manifest without importing project code."""
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("source manifest must be a regular file")
    raw = manifest_path.read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError(f"source manifest exceeds {MAX_MANIFEST_BYTES} bytes")
    return _validate_manifest(
        raw,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_sha=expected_source_sha,
        expected_source_tree=expected_source_tree,
    )


def _safe_parent(root: Path, path: PurePosixPath) -> None:
    parent = root
    for part in path.parts[:-1]:
        parent /= part
        if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
            raise ValueError(f"source capsule parent is not a real directory: {path}")
        parent.mkdir(exist_ok=True)


def _verify_extracted_source(source_root: Path, manifest: dict[str, Any]) -> None:
    records = {record["path"]: record for record in manifest["members"]}
    actual_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_paths != set(records):
        raise ValueError(
            f"source capsule file set differs from manifest: missing={sorted(set(records) - actual_paths)}, "
            f"extra={sorted(actual_paths - set(records))}"
        )
    expected_directories = {
        parent.as_posix()
        for source_path in records
        for parent in PurePosixPath(source_path).parents
        if parent.as_posix() != "."
    }
    actual_directories = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if actual_directories != expected_directories:
        raise ValueError("source capsule directory set differs from manifest paths")
    for source_path, record in records.items():
        path = source_root.joinpath(*PurePosixPath(source_path).parts)
        if record["type"] == "symlink":
            if not path.is_symlink():
                raise ValueError(f"source capsule member is not the required symlink: {source_path}")
            contents = os.readlink(path).encode()
        else:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"source capsule member is not a regular file: {source_path}")
            expected_mode = 0o755 if record["mode"] == "100755" else 0o644
            if stat.S_IMODE(path.stat().st_mode) != expected_mode:
                raise ValueError(f"source capsule member mode differs from manifest: {source_path}")
            contents = path.read_bytes()
        if len(contents) != record["size"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
            raise ValueError(f"source capsule member content differs from manifest: {source_path}")


def restore_and_verify_source_payload(
    workspace: Path,
    *,
    expected_manifest_sha256: str,
    expected_source_sha: str,
    expected_source_tree: str,
) -> tuple[Path, dict[str, Any]]:
    """Extract and verify an exact allowlisted source capsule."""
    workspace = workspace.resolve()
    manifest_path = workspace / MANIFEST_FILENAME
    capsule_path = workspace / CAPSULE_FILENAME
    manifest = load_and_verify_manifest(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_source_sha=expected_source_sha,
        expected_source_tree=expected_source_tree,
    )
    if capsule_path.is_symlink() or not capsule_path.is_file():
        raise ValueError("source capsule must be a regular file")
    if capsule_path.stat().st_size > MAX_CAPSULE_BYTES:
        raise ValueError(f"source capsule exceeds {MAX_CAPSULE_BYTES} bytes")
    if _sha256(capsule_path) != manifest["archive"]["sha256"]:
        raise ValueError("source capsule SHA-256 does not match the source manifest")
    source_root = workspace / "source"
    if source_root.exists() or source_root.is_symlink():
        raise ValueError("source capsule extraction root must not already exist")
    temporary = Path(tempfile.mkdtemp(prefix=".source-capsule-", dir=workspace))
    try:
        records = {record["path"]: record for record in manifest["members"]}
        with zipfile.ZipFile(capsule_path) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(names) != len(set(names)) or set(names) != set(records):
                raise ValueError("source capsule members differ from the closed manifest")
            for member in members:
                record = records[member.filename]
                transport_mode = member.external_attr >> 16
                if member.is_dir() or stat.S_IFMT(transport_mode) != stat.S_IFREG:
                    raise ValueError(f"source capsule transport member is not regular: {member.filename}")
                if member.file_size != record["size"]:
                    raise ValueError(f"source capsule member size differs from manifest: {member.filename}")
                contents = archive.read(member)
                if hashlib.sha256(contents).hexdigest() != record["sha256"]:
                    raise ValueError(f"source capsule member SHA-256 differs from manifest: {member.filename}")
                relative = _validate_relative_path(member.filename, "capsule path")
                destination = temporary.joinpath(*relative.parts)
                _safe_parent(temporary, relative)
                if destination.exists() or destination.is_symlink():
                    raise ValueError(f"source capsule member repeats extraction path: {member.filename}")
                if record["type"] == "symlink":
                    try:
                        target = contents.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise ValueError(f"source capsule symlink target is not UTF-8: {member.filename}") from error
                    destination.symlink_to(_validate_symlink_target(relative, target))
                else:
                    destination.write_bytes(contents)
                    destination.chmod(0o755 if record["mode"] == "100755" else 0o644)
        _verify_extracted_source(temporary, manifest)
        temporary.replace(source_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return source_root, manifest


def runner_arguments(
    source_root: Path,
    source_sha: str,
    source_tree: str,
    manifest_path: Path,
    manifest_sha256: str,
    artifact_directory: Path,
    require_jax_version: str,
) -> tuple[str, ...]:
    if not require_jax_version.strip():
        raise ValueError("required JAX version must be nonempty")
    return (
        sys.executable,
        str(source_root / RUNNER_RELATIVE_PATH),
        "--execute",
        "--source-root",
        str(source_root),
        "--source-sha",
        source_sha,
        "--source-tree",
        source_tree,
        "--source-capsule-manifest",
        str(manifest_path),
        "--source-capsule-manifest-sha256",
        manifest_sha256,
        "--artifact-directory",
        str(artifact_directory),
        "--require-jax-version",
        require_jax_version,
        "--git",
        "/usr/bin/git",
        "--nvidia-smi",
        "/usr/bin/nvidia-smi",
        "--nvcc",
        str(CUDA_BIN / "nvcc"),
        "--ptxas",
        str(CUDA_BIN / "ptxas"),
        "--cuobjdump",
        str(CUDA_BIN / "cuobjdump"),
        "--ncu",
        "/usr/local/bin/ncu",
        "--nsys",
        "/usr/local/bin/nsys",
    )


def capsule_python_path(source_root: Path, inherited: str | None) -> str:
    """Return the closed capsule import roots followed by any image paths."""
    roots = (str(source_root), str(source_root / "lib/tile_lifetime/src"))
    return os.pathsep.join((*roots, *((inherited,) if inherited else ())))


def verify_launcher_identity(path: Path, expected_sha256: str) -> None:
    """Reject execution when the transported stdlib launcher changed."""
    expected_sha256 = _require_sha(expected_sha256, SHA256_PATTERN, "launcher SHA-256")
    if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha256:
        raise ValueError("source capsule launcher differs from the trusted launch identity")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-root", type=Path, required=True)
    prepare.add_argument("--source-sha", required=True)
    prepare.add_argument("--output-directory", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--launcher-sha256", required=True)
    run.add_argument("--manifest-sha256", required=True)
    run.add_argument("--source-sha", required=True)
    run.add_argument("--source-tree", required=True)
    run.add_argument("--artifact-directory", type=Path, required=True)
    run.add_argument("--require-jax-version", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    if args.command == "prepare":
        print(
            json.dumps(prepare_source_payload(args.source_root, args.source_sha, args.output_directory), sort_keys=True)
        )
        return
    verify_launcher_identity(Path(__file__).resolve(), args.launcher_sha256)
    workspace = args.workspace.resolve()
    source_root, _manifest = restore_and_verify_source_payload(
        workspace,
        expected_manifest_sha256=args.manifest_sha256,
        expected_source_sha=args.source_sha,
        expected_source_tree=args.source_tree,
    )
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPATH"] = capsule_python_path(source_root, os.environ.get("PYTHONPATH"))
    arguments = runner_arguments(
        source_root,
        args.source_sha,
        args.source_tree,
        workspace / MANIFEST_FILENAME,
        args.manifest_sha256,
        args.artifact_directory.resolve(),
        args.require_jax_version,
    )
    os.execv(arguments[0], arguments)


if __name__ == "__main__":
    main()
