# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Search an agent-phase SWE-smith container for the clean code the bug patch removed (issue #10212).

For each package: take the source lines the 'Bug Patch' commit removed that occur nowhere in the bugged tree (from the
swesmith/<repo> mirror), build environment/ with no network, run the history commands an agent would try
(validate_swesmith.LEAK_COMMANDS), and report how many clean lines their output contains and which files anywhere in
the image contain one. A leaky upstream package prints clean lines; a fixed package should print none.

Usage:
    python -m experiments.post_training.sft_shootout_pool.leak_probe <tasks.parquet> <label> <mirrors> <id> ...
"""

import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from experiments.post_training.sft_shootout_pool.build_swesmith import MIRROR_NAME
from experiments.post_training.sft_shootout_pool.swesmith_packages import instance_id, read_package
from experiments.post_training.sft_shootout_pool.swesmith_restore import clean_only_lines, mirror
from experiments.post_training.sft_shootout_pool.validate_swesmith import docker, leak_probe


def main() -> None:
    parquet, label, mirrors, wanted = sys.argv[1], sys.argv[2], Path(sys.argv[3]), set(sys.argv[4:])
    for row in pq.read_table(parquet).to_pylist():
        if row["path"] not in wanted:
            continue
        instance = instance_id(read_package(row["task_binary"]))
        lines = clean_only_lines(mirror(MIRROR_NAME.match(instance)[1], mirrors), instance)
        with tempfile.TemporaryDirectory() as temporary:
            with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:*") as archive:
                archive.extractall(temporary, filter="data")
            image = f"pool-leak-probe:{row['path']}-{label}"
            docker("build", "-q", "-t", image, f"{temporary}/environment")
            try:
                result = leak_probe(image, lines)
            finally:
                docker("rmi", "-f", image, check=False)
        print("LEAK_PROBE " + json.dumps({"task": row["path"], "package": label} | result), flush=True)


if __name__ == "__main__":
    main()
