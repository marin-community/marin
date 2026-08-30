# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print per-bin row counts and grades for one built pool version.

Runs where the experiment bucket is readable (an Iris CPU job on the training
cluster).
"""

import json
from collections import Counter

import click
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath, prefix_join

POOL_ROOT = "s3://marin-us-east-02a/marin/documents/curriculum-rl-pool"


@click.command(help=__doc__)
@click.option("--version", required=True)
def main(version: str) -> None:
    for filename in ("train.parquet", "validation.parquet"):
        path = prefix_join(POOL_ROOT, version, filename)
        with StoragePath(path).open("rb") as handle:
            table = pq.read_table(handle, columns=["extra_info"])
        counts: Counter[tuple[int, str]] = Counter()
        for info in table.column("extra_info").to_pylist():
            counts[(int(info["grade"]), str(info["data_source"]))] += 1
        print(f"== {filename}")
        for (grade, source), n in sorted(counts.items()):
            print(json.dumps({"grade": grade, "bin": source, "rows": n}))


if __name__ == "__main__":
    main()
