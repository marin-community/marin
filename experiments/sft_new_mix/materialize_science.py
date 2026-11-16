# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the sealed science export's exact token IDs as candidate build inputs."""

import argparse
import gzip
import hashlib
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from levanter.store.cache import write_levanter_cache

SCIENCE_REVISION = "1583049c1df02b418e07d14047eb37db9df46eba"


def materialize_file(job: tuple[str, str, str]) -> list[dict]:
    path, expected_sha256, output = job
    with open(path, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError("Science input file digest mismatch")
    groups = defaultdict(list)
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            source = row["id"].split(":", 1)[0]
            groups[source].append(np.asarray(row["input_ids"], dtype=np.int32))
    receipts = []
    for source, arrays in groups.items():
        shard_name = Path(path).parent.name + "-" + Path(path).name.removesuffix(".jsonl.gz")
        destination = str(Path(output) / source.rsplit("/", 1)[-1] / shard_name)
        result = write_levanter_cache(
            ({"input_ids": ids} for ids in arrays),
            destination,
            metadata={"science_revision": SCIENCE_REVISION, "accepted_file_sha256": expected_sha256},
            batch_size=32,
            max_pending_batches=2,
        )
        if result["count"] != len(arrays):
            raise ValueError("Science cache count mismatch")
        receipts.append(
            {
                "source": source,
                "path": destination,
                "rows": len(arrays),
                "tokens": sum(map(len, arrays)),
                "accepted_file": path,
                "accepted_file_sha256": expected_sha256,
                "source_revision": SCIENCE_REVISION,
            }
        )
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_bytes = (args.dataset / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    files = [
        row
        for row in manifest["files"]
        if Path(row["path"]).name.startswith("accepted-") and row["path"].endswith(".jsonl.gz")
    ]
    jobs = [(str(args.dataset / row["path"]), row["sha256"], str(args.output)) for row in files]
    receipts = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in pool.map(materialize_file, jobs):
            receipts.extend(result)
            print(
                json.dumps(
                    {
                        "files_completed": len({row["accepted_file"] for row in receipts}),
                        "rows": sum(row["rows"] for row in receipts),
                    }
                ),
                flush=True,
            )
    if sum(row["rows"] for row in receipts) != 99_453 or sum(row["tokens"] for row in receipts) != 1_667_075_429:
        raise ValueError("Science seal population does not match publication receipt")
    receipt = {
        "science_revision": SCIENCE_REVISION,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "shards": receipts,
    }
    (args.output / "manifest.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
