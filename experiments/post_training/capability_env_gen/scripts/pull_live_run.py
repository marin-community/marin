#!/usr/bin/env python3
"""Copy verified-by-hash live-run artifacts from CoreWeave S3 for local review.

The running pilot predates the content-addressed snapshot uploader, so this tool
does not treat its old manifest as an atomic restore point.  It reads completed
item trees directly, records the exact bytes it copied, and can additionally
collect every artifact for a named failed inference identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import fsspec
from rigging.filesystem.s3_compat import configure_coreweave_s3

KEEP = {"request.json", "response.json", "result.json", "rejected-result.json", "status.json", "cache-rejection.json"}
COMPACT_RECORDS = {"response.json", "result.json", "rejected-result.json", "status.json", "cache-rejection.json"}


def read_json(fs, path: str) -> dict | None:
    try:
        value = json.loads(fs.cat(path))
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def safe(relative: str) -> bool:
    path = PurePosixPath(relative)
    return not path.is_absolute() and ".." not in path.parts


def copy(fs, remote: str, local: Path) -> str:
    data = fs.cat(remote)
    local.parent.mkdir(parents=True, exist_ok=True)
    temporary = local.with_name(local.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, local)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--failed-identity", required=True)
    parser.add_argument("--completed", action="store_true", help="also copy all completed item trees")
    parser.add_argument("--all-item-records", action="store_true", help="copy status/result/response JSON for every item, excluding SSE logs")
    args = parser.parse_args()

    configure_coreweave_s3()
    fs, prefix = fsspec.core.url_to_fs(args.source.rstrip("/"))
    prefix = prefix.rstrip("/")
    files = sorted(fs.find(prefix))
    groups: dict[str, list[str]] = {}
    for remote in files:
        relative = remote.removeprefix(prefix + "/")
        parts = PurePosixPath(relative).parts
        if len(parts) >= 4 and parts[0] == "items":
            groups.setdefault(str(PurePosixPath(*parts[:-1])), []).append(remote)

    selected: set[str] = set()
    failures: list[dict[str, str]] = []
    for name in ("input_pilot.json", "run.json", "plans.json", "proposals.json", "reviews-round-0.json", "reviews-round-1.json", "accepted.json", "rejected.json", "report.json", "worker.log"):
        remote = f"{prefix}/{name}"
        if fs.exists(remote):
            selected.add(remote)

    def inspect(entry):
        group, members = entry
        request = read_json(fs, f"{prefix}/{group}/request.json")
        status = read_json(fs, f"{prefix}/{group}/status.json")
        complete = f"{prefix}/{group}/result.json" in members and (status or {}).get("state") == "complete"
        return group, members, request, status, complete

    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = [pool.submit(inspect, item) for item in groups.items()]
        for future in as_completed(futures):
            group, members, request, status, completed = future.result()
            matched = (request or {}).get("identity") == args.failed_identity
            if args.all_item_records:
                selected.update(member for member in members if PurePosixPath(member).name in COMPACT_RECORDS)
            elif args.completed and completed:
                selected.update(member for member in members if PurePosixPath(member).name in KEEP)
            if matched:
                selected.update(member for member in members if PurePosixPath(member).name in KEEP or PurePosixPath(member).name.startswith("events."))
            if matched:
                failures.append({"group": group, "state": str((status or {}).get("state")), "error": str((status or {}).get("error", ""))})

    copied: dict[str, str] = {}
    def copy_one(remote: str) -> tuple[str, str]:
        relative = remote.removeprefix(prefix + "/")
        if not safe(relative):
            raise RuntimeError(f"unsafe remote relative path: {relative}")
        return relative, copy(fs, remote, args.destination / relative)
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(copy_one, remote) for remote in sorted(selected)]
        for future in as_completed(futures):
            relative, checksum = future.result()
            copied[relative] = checksum
    manifest = {
        "pulled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": args.source,
        "completed_item_artifacts": len(copied),
        "failed_identity": args.failed_identity,
        "failed_matches": failures,
        "sha256": copied,
    }
    output = args.destination / "local-pull-manifest.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"copied": len(copied), "failure_matches": len(failures), "manifest": str(output)}))
    return 0 if failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
