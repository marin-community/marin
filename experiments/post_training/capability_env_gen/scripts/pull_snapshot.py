"""Read one immutable result manifest and verify its selected content objects."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


class SnapshotError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_members(manifest: dict) -> dict[str, str]:
    files = manifest.get("files")
    if not isinstance(files, dict) or not isinstance(manifest.get("snapshot_id"), str):
        raise SnapshotError("invalid snapshot manifest")
    reserved = {"pull-manifest.json", "snapshot-capture.json"}
    for name, digest in files.items():
        if not isinstance(name, str) or not name:
            raise SnapshotError("invalid member name")
        path = PurePosixPath(name)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or "\\" in name
            or "\x00" in name
            or path.parts[0] in reserved
        ):
            raise SnapshotError(f"unsafe member: {name!r}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise SnapshotError(f"invalid digest for {name}")
    # A file cannot also be another member's parent directory.
    for name in files:
        if any(str(parent) in files for parent in PurePosixPath(name).parents):
            raise SnapshotError(f"file/directory collision: {name}")
    return files


def safe_local(root: Path, relative: str) -> Path:
    current = root
    for component in PurePosixPath(relative).parts:
        current = current / component
        if current.is_symlink():
            raise SnapshotError(f"local symlink in destination: {relative}")
    return current


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise SnapshotError("local metadata temporary is a symlink")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def pull_snapshot(fs, remote: str, source: str, destination: Path, *,
                  prefixes: tuple[str, ...] = (), workers: int = 32) -> dict:
    """Resume the captured snapshot, never silently advance to the latest one."""
    if workers < 1 or workers > 128:
        raise SnapshotError("workers must be between 1 and 128")
    if destination.is_symlink():
        raise SnapshotError("destination must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    capture_path = safe_local(destination, "snapshot-capture.json")
    receipt_path = safe_local(destination, "pull-manifest.json")
    if capture_path.exists():
        capture = json.loads(capture_path.read_text())
        if capture.get("source") != source or capture.get("prefixes") != list(prefixes):
            raise SnapshotError("destination belongs to a different source or selection")
        raw = capture["remote_manifest_json"].encode()
        if sha256(raw) != capture.get("remote_manifest_sha256"):
            raise SnapshotError("captured manifest checksum mismatch")
    else:
        if any(destination.iterdir()):
            raise SnapshotError("new snapshot requires an empty destination")
        raw = fs.cat(remote.rstrip("/") + "/_manifests/latest.json")
        manifest = json.loads(raw)
        validate_members(manifest)
        capture = {
            "schema_version": "capability-snapshot-capture-v1",
            "source": source,
            "prefixes": list(prefixes),
            "remote_manifest_json": raw.decode(),
            "remote_manifest_sha256": sha256(raw),
        }
        atomic_json(capture_path, capture)
    manifest = json.loads(raw)
    files = validate_members(manifest)
    selected = {
        name: digest for name, digest in files.items()
        if not prefixes or any(name == p or name.startswith(p.rstrip("/") + "/") for p in prefixes)
    }
    if prefixes and not selected:
        raise SnapshotError("selection matches no snapshot files")
    # A failed resumed transfer must not leave an apparent current success receipt.
    receipt_path.unlink(missing_ok=True)

    def fetch(entry: tuple[str, str]) -> tuple[str, str]:
        name, expected = entry
        local = safe_local(destination, name)
        if local.is_file() and sha256(local.read_bytes()) == expected:
            return name, expected
        data = fs.cat(remote.rstrip("/") + "/_objects/" + expected)
        if sha256(data) != expected:
            raise SnapshotError(f"object checksum mismatch: {name}")
        local.parent.mkdir(parents=True, exist_ok=True)
        # No unverified bytes are published. Independent names never share a temp path.
        temporary = local.with_name(local.name + ".snapshot-part")
        if temporary.is_symlink() or temporary.relative_to(destination).as_posix() in files:
            raise SnapshotError(f"unsafe temporary path: {name}")
        temporary.write_bytes(data)
        temporary.replace(local)
        return name, expected

    with ThreadPoolExecutor(max_workers=workers) as pool:
        verified = dict(pool.map(fetch, selected.items()))
    receipt = {
        "schema_version": "capability-snapshot-pull-v1",
        "source": source,
        "remote_snapshot_id": manifest["snapshot_id"],
        "remote_manifest_sha256": sha256(raw),
        "remote_created_utc": manifest.get("created_utc"),
        "remote_final": manifest.get("final"),
        "pulled_utc": datetime.now(UTC).isoformat(),
        "prefixes": list(prefixes),
        "complete_manifest": len(verified) == len(files),
        "complete_snapshot": len(verified) == len(files) and not manifest.get("omitted"),
        "omitted": manifest.get("omitted", {}),
        "files": verified,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    # Local metadata tests do not require the optional cloud stack.
    import fsspec
    from rigging.filesystem.s3_compat import configure_coreweave_s3

    configure_coreweave_s3()
    fs, remote = fsspec.core.url_to_fs(args.source.rstrip("/"))
    try:
        receipt = pull_snapshot(fs, remote, args.source.rstrip("/"), args.destination,
                                prefixes=tuple(args.prefix), workers=args.workers)
    except Exception as error:  # noqa: BLE001 -- redact backend transport messages at CLI boundary
        print(json.dumps({"ok": False, "error_type": type(error).__name__,
                          "error": str(error) if isinstance(error, SnapshotError) else "snapshot transfer failed"}))
        return 1
    print(json.dumps({"ok": True, "snapshot_id": receipt["remote_snapshot_id"],
                      "files": len(receipt["files"]), "complete_manifest": receipt["complete_manifest"],
                      "complete_snapshot": receipt["complete_snapshot"], "omitted": len(receipt["omitted"]),
                      "manifest": str(args.destination / "pull-manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
