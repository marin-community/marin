# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Acquire and verify task archives from a pinned public Harbor Hub release."""

import argparse
import hashlib
import json
import logging
import re
import shutil
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from tempfile import TemporaryFile
from typing import Any, BinaryIO
from urllib.parse import quote

import requests
from rigging.filesystem.storage_path import StoragePath

LOGGER = logging.getLogger(__name__)
REGISTRY_HOST = "ofhuhcpkvzjlejydnvyd.supabase.co"
REGISTRY_URL = f"https://{REGISTRY_HOST}"
# Harbor's public client key. Public releases are readable without a user session.
REGISTRY_KEY = "sb_publishable_Z-vuQbpvpG-PStjbh4yE0Q_e-d3MTIH"
PAGE_SIZE = 1000
REQUEST_TIMEOUT = (10, 60)
CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _TaskArchive:
    content_hash: str
    archive_path: str


def _registry_rows(session: requests.Session, table: str, params: dict[str, str | int]) -> list[dict[str, Any]]:
    with session.get(f"{REGISTRY_URL}/rest/v1/{table}", params=params, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        return response.json()


def _release_archives(session: requests.Session, release: str) -> dict[str, _TaskArchive]:
    dataset, digest = release.rsplit("@", 1)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
        raise ValueError("Select a release by content digest, not a tag")
    org, name = dataset.split("/")
    versions = _registry_rows(
        session,
        "dataset_version",
        {
            "select": "id,content_hash,package:package_id!inner(org:org_id!inner(name))",
            "package.type": "eq.dataset",
            "package.name": f"eq.{name}",
            "package.org.name": f"eq.{org}",
            "content_hash": f"eq.{digest.removeprefix('sha256:')}",
        },
    )
    if len(versions) != 1 or f"sha256:{versions[0]['content_hash']}" != digest:
        raise ValueError(f"Hub did not resolve the pinned release: {release}")
    version = f"eq.{versions[0]['id']}"
    if _registry_rows(session, "dataset_version_file", {"select": "id", "dataset_version_id": version, "limit": 1}):
        raise ValueError("Dataset-level scoring files require an explicit adapter")

    refs: dict[str, _TaskArchive] = {}
    offset = 0
    while True:
        page = _registry_rows(
            session,
            "dataset_version_task",
            {
                "select": (
                    "task_version:task_version_id(content_hash,archive_path,"
                    "package:package_id(name,org:org_id(name)))"
                ),
                "dataset_version_id": version,
                "order": "task_version_id",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
        )
        for row in page:
            task = row["task_version"]
            package = task["package"]
            task_digest = task["content_hash"]
            ref = f"{package['org']['name']}/{package['name']}@sha256:{task_digest}"
            if ref in refs:
                raise ValueError(f"Duplicate task identity in release: {ref}")
            refs[ref] = _TaskArchive(task_digest, task["archive_path"])
        if len(page) < PAGE_SIZE:
            return refs
        offset += PAGE_SIZE


def _task_content_hash(stream: BinaryIO) -> str:
    # Hub tarballs contain the package files selected by its packager. The digest
    # covers sorted relative paths and file digests, independent of tar metadata.
    files: dict[str, str] = {}
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Invalid package path: {member.name}")
            if member.isdir():
                continue
            name = path.as_posix()
            if not member.isfile() or name in files or not path.parts:
                raise ValueError(f"Invalid package member: {member.name}")
            file = archive.extractfile(member)
            assert file is not None
            with file:
                files[name] = hashlib.file_digest(file, "sha256").hexdigest()
    digest = hashlib.sha256()
    for name, file_hash in sorted(files.items()):
        digest.update(f"{name}\0{file_hash}\n".encode())
    return digest.hexdigest()


def _download_task(session: requests.Session, ref: str, task: _TaskArchive, output: StoragePath) -> None:
    identity = hashlib.sha256(ref.encode()).hexdigest()
    destination = output / f"{identity}.tar.gz"
    if destination.exists():
        return
    url = f"{REGISTRY_URL}/storage/v1/object/authenticated/packages/{quote(task.archive_path, safe='/')}"
    with TemporaryFile() as archive:
        with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            for chunk in response.iter_content(CHUNK_BYTES):
                archive.write(chunk)
        archive.seek(0)
        if _task_content_hash(archive) != task.content_hash:
            raise ValueError(f"Downloaded task differs from the pinned package: {ref}")
        archive.seek(0)
        pending = output / f"{identity}.partial"
        with pending.open("wb", auto_mkdir=True) as target:
            shutil.copyfileobj(archive, target)
        pending.rename(destination)
    LOGGER.info("Acquired %s", ref)


def acquire_release(release: str, output: StoragePath, selected: Sequence[str]) -> None:
    """Cache original task archives after verifying their native content hashes."""
    with requests.Session() as session:
        session.headers["apikey"] = REGISTRY_KEY
        refs = _release_archives(session, release)
        unknown = set(selected) - refs.keys()
        if unknown:
            raise ValueError(f"Tasks absent from pinned release: {sorted(unknown)}")
        selected_refs = sorted(set(selected) if selected else refs)
        for ref in selected_refs:
            _download_task(session, ref, refs[ref], output / "tasks")

    (output / "release.json").write_text(
        json.dumps(
            {
                "release": release,
                "population_count": len(refs),
                "selected": selected_refs,
                "task_refs": sorted(refs),
            },
            indent=2,
        )
        + "\n",
        auto_mkdir=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--task", action="append", default=[], help="Immutable org/task@sha256:... reference")
    selection.add_argument("--all", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    acquire_release(args.release, StoragePath(args.output), args.task)


if __name__ == "__main__":
    main()
