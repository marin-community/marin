# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the regional v0 copy against its published hashes and list materialized nemotron_sft_v3 chat outputs."""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import fsspec
from marin.datakit.sft_sources import all_sft_sources
from rigging.filesystem.storage_path import StoragePath

V0_COPY = "s3://marin-us-east-02a/marin/weight-merging/campaign3-20261011/dare-linear-02/model"


def sha256(path: str) -> tuple[str, str, int]:
    digest = hashlib.sha256()
    size = 0
    with fsspec.open(path, "rb") as stream:
        while chunk := stream.read(64 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return path.rsplit("/", 1)[1], digest.hexdigest(), size


def main() -> None:
    expected = json.loads(sys.argv[1])
    fs, _ = fsspec.core.url_to_fs(V0_COPY)
    names = sorted("s3://" + p for p in fs.ls(V0_COPY, detail=False))
    print("V0_COPY_FILES " + json.dumps([n.rsplit("/", 1)[1] for n in names]), flush=True)
    with ThreadPoolExecutor(8) as pool:
        hashes = {name: (digest, size) for name, digest, size in pool.map(sha256, names)}
    report = {
        name: {"sha256": digest, "bytes": size, "matches_hf": expected.get(name) == digest}
        for name, (digest, size) in hashes.items()
    }
    print("V0_COPY_HASHES " + json.dumps(report), flush=True)
    for name, source in sorted(all_sft_sources().items()):
        if not name.startswith("nemotron_sft_v3/"):
            continue
        path = source.chat_normalized.output_path
        files = []
        if StoragePath(path).exists():
            files = [
                (p, s)
                for p, s in (
                    (f["name"], f["size"]) for f in fsspec.core.url_to_fs(path)[0].find(path, detail=True).values()
                )
            ]
        print(
            "V3_SOURCE "
            + json.dumps(
                {
                    "name": name,
                    "path": path,
                    "files": len(files),
                    "bytes": sum(s for _, s in files),
                    "sample": [p for p, _ in files[:3]],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
