# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Stage a pinned HuggingFace checkpoint into a durable object-store prefix, verified.

One invocation stages one repository at one immutable commit. Files stream one at a
time through a small local scratch directory, so a 134 GB checkpoint never needs more
than one shard of local disk. Every file is hashed locally before upload; LFS files are
checked against the sha256 HuggingFace records for the blob, and every upload is checked
for size on the remote. A per-file sidecar under ``.staged/`` records each verified
upload so a restarted job skips what is already done.

The destination is wrapped in :func:`levanter.model_cache.cache_to_prefix`, so the
completion marker ``.cache_metadata.json`` is written last and concurrent jobs for the
same destination serialize under a distributed lock. A destination with the marker is
a cache hit and is not touched.

Example (run from a marin checkout so the workspace ships with the job)::

    uv run iris --cluster=marin job run --no-wait --cpu 4 --memory 16GB --disk 40GB \
      --enable-extra-resources --zone us-central2-b --timeout 43200 \
      --job-name rsi-stage-S -- python -m experiments.weight_merging.rsi.stage_parents \
      --alias S --repo open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02 \
      --revision 50fe2f488aee7730bbccc721024b35229075602e \
      --prefix gs://marin-us-central2/scratch/karan/rsi-merge/parents
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import socket
import sys
import tempfile
import time
from dataclasses import asdict, dataclass

import fsspec
from huggingface_hub import hf_hub_download, model_info
from levanter.model_cache import DEFAULT_COMPLETE_MARKER, CacheMetadata, cache_to_prefix

logger = logging.getLogger("stage_parents")

_COMMIT = re.compile(r"[0-9a-f]{40}")
_STAGED_DIR = ".staged"
_MANIFEST = "staging_manifest.json"
_CHUNK = 16 * 1024 * 1024


@dataclass(frozen=True)
class ExpectedFile:
    name: str
    size: int | None
    lfs_sha256: str | None


@dataclass(frozen=True)
class StagedFile:
    name: str
    bytes: int
    sha256: str
    lfs: bool
    seconds: float


class StagingError(RuntimeError):
    pass


def _sha256_file(path: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def expected_files(repo: str, revision: str, complete_marker: str) -> list[ExpectedFile]:
    """List the repository at *revision* with sizes and LFS hashes; verify the commit."""
    info = model_info(repo, revision=revision, files_metadata=True)
    if info.sha is None or info.sha.lower() != revision:
        raise StagingError(f"{repo}@{revision} resolved to sha {info.sha!r}; refusing to stage a moving ref")
    files: list[ExpectedFile] = []
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if name == complete_marker or name.startswith(f"{_STAGED_DIR}/") or name == _MANIFEST:
            # Merge outputs built from a cached parent carry levanter's completion marker
            # along into the HF repo. It is not part of the model; only cache_to_prefix may
            # write it here, so it is left out rather than copied.
            logger.warning("skipping reserved path %r present in %s@%s", name, repo, revision)
            continue
        lfs = getattr(sibling, "lfs", None)
        lfs_sha = None
        size = sibling.size
        if lfs is not None:
            lfs_sha = lfs.sha256 if hasattr(lfs, "sha256") else lfs.get("sha256")
            size = size if size is not None else (lfs.size if hasattr(lfs, "size") else lfs.get("size"))
        files.append(ExpectedFile(name=name, size=size, lfs_sha256=lfs_sha))
    if not files:
        raise StagingError(f"{repo}@{revision} lists no files")
    return files


def destination(prefix: str, alias: str, repo: str, revision: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", repo.strip("/"))
    return f"{prefix.rstrip('/')}/{alias}--{readable}@{revision[:12]}"


def _sidecar(dest: str, name: str) -> str:
    return f"{dest}/{_STAGED_DIR}/{name}.json"


def _already_staged(fs: fsspec.AbstractFileSystem, dest: str, expected: ExpectedFile) -> StagedFile | None:
    sidecar = _sidecar(dest, expected.name)
    if not fs.exists(sidecar):
        return None
    with fs.open(sidecar, "r") as handle:
        record = StagedFile(**json.load(handle))
    remote = f"{dest}/{expected.name}"
    if not fs.exists(remote) or fs.info(remote)["size"] != record.bytes:
        logger.warning("sidecar for %s does not match remote object; re-staging", expected.name)
        return None
    if expected.lfs_sha256 and record.sha256 != expected.lfs_sha256:
        logger.warning("sidecar hash for %s does not match HF LFS hash; re-staging", expected.name)
        return None
    return record


def _stage_one(
    fs: fsspec.AbstractFileSystem, dest: str, repo: str, revision: str, expected: ExpectedFile, scratch: str
) -> StagedFile:
    started = time.monotonic()
    local_path = hf_hub_download(repo, expected.name, revision=revision, local_dir=scratch)
    try:
        sha, size = _sha256_file(local_path)
        if expected.size is not None and size != expected.size:
            raise StagingError(f"{expected.name}: downloaded {size} bytes, HF lists {expected.size}")
        if expected.lfs_sha256 is not None and sha != expected.lfs_sha256:
            raise StagingError(f"{expected.name}: sha256 {sha} != HF LFS sha256 {expected.lfs_sha256}")
        remote = f"{dest}/{expected.name}"
        fs.makedirs(remote.rsplit("/", 1)[0], exist_ok=True)
        fs.put_file(local_path, remote)
        remote_size = fs.info(remote)["size"]
        if remote_size != size:
            raise StagingError(f"{expected.name}: remote has {remote_size} bytes after upload, expected {size}")
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)
    record = StagedFile(
        name=expected.name,
        bytes=size,
        sha256=sha,
        lfs=expected.lfs_sha256 is not None,
        seconds=round(time.monotonic() - started, 1),
    )
    sidecar = _sidecar(dest, expected.name)
    fs.makedirs(sidecar.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(sidecar, "w") as handle:
        json.dump(asdict(record), handle, sort_keys=True)
    return record


def stage(alias: str, repo: str, revision: str, prefix: str, complete_marker: str = DEFAULT_COMPLETE_MARKER) -> str:
    revision = revision.lower()
    if not _COMMIT.fullmatch(revision):
        raise StagingError(f"--revision must be a 40-hex commit, got {revision!r}")
    dest = destination(prefix, alias, repo, revision)
    files = expected_files(repo, revision, complete_marker)
    total = sum(f.size or 0 for f in files)
    logger.info("%s: %s@%s -> %s (%d files, %.1f GB)", alias, repo, revision, dest, len(files), total / 1e9)

    def populate(fs: fsspec.AbstractFileSystem, cache_path: str) -> None:
        started = time.time()
        staged: list[StagedFile] = []
        done_bytes = 0
        with tempfile.TemporaryDirectory(prefix=f"stage_{alias}_") as scratch:
            for index, expected in enumerate(files, 1):
                record = _already_staged(fs, cache_path, expected)
                if record is None:
                    record = _stage_one(fs, cache_path, repo, revision, expected, scratch)
                    logger.info(
                        "[%d/%d] %s %.2f GB in %.0fs (%.0f MB/s)",
                        index,
                        len(files),
                        expected.name,
                        record.bytes / 1e9,
                        record.seconds,
                        record.bytes / 1e6 / max(record.seconds, 0.1),
                    )
                else:
                    logger.info("[%d/%d] %s already staged, skipping", index, len(files), expected.name)
                staged.append(record)
                done_bytes += record.bytes
        manifest = {
            "alias": alias,
            "repo": repo,
            "revision": revision,
            "destination": cache_path,
            "files": [asdict(r) for r in staged],
            "file_count": len(staged),
            "total_bytes": done_bytes,
            "lfs_files_verified_against_hf_sha256": sum(1 for r in staged if r.lfs),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
        }
        with fs.open(f"{cache_path}/{_MANIFEST}", "w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        logger.info(
            "%s: staged %d files, %.1f GB, manifest at %s/%s",
            alias,
            len(staged),
            done_bytes / 1e9,
            cache_path,
            _MANIFEST,
        )

    return cache_to_prefix(
        dest, populate, metadata=CacheMetadata(source_revision=revision), complete_marker=complete_marker
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--alias", required=True, help="Short name used in the destination directory, e.g. S")
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. org/model")
    parser.add_argument("--revision", required=True, help="40-hex immutable commit")
    parser.add_argument("--prefix", required=True, help="Destination prefix, e.g. gs://bucket/path/parents")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    path = stage(args.alias, args.repo, args.revision, args.prefix)
    print(f"STAGED {args.alias} {args.repo}@{args.revision} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
