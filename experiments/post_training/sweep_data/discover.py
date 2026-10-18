# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the regional v0 copy against its published hashes and list materialized nemotron_sft_v3 chat outputs."""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from marin.datakit.sft_sources import all_sft_sources
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.sweep_data.prepare_agentic import STUDENT_REGIONAL_COPY


def name_digest_size(path: StoragePath) -> tuple[str, str, int]:
    """Stream one file; return its basename, sha256 hex digest and size in bytes."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return path.name, digest.hexdigest(), size


def main() -> None:
    expected = json.loads(sys.argv[1])
    paths = sorted(StoragePath(STUDENT_REGIONAL_COPY).ls(), key=str)
    print("V0_COPY_FILES " + json.dumps([path.name for path in paths]), flush=True)
    with ThreadPoolExecutor(8) as pool:
        hashes = {name: (digest, size) for name, digest, size in pool.map(name_digest_size, paths)}
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
            fs, root = url_to_fs(path)
            files = [(f["name"], f["size"]) for f in fs.find(root, detail=True).values()]
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
