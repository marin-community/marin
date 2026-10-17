# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task #215: per-task repo and fix lines from DCAgent's SWE-smith packages (inputs to audit.py).

Usage: package_index.py <task_repo.json> <fix_lines.json>

Reads DCAgent/swesmith-sandboxes-with_tests-oracle_verified_120s@78b0d879 tasks.parquet (9,439 Harbor packages):
- task_repo.json: {task id: {repo: 'owner/name', instance_id}} from tests/config.json's instance_id.
- fix_lines.json: {task id: [line, ...]}: solution/solve.sh applies the bug patch in reverse, so the patch's '-' lines
  are the clean code. Kept: stripped '-' lines in non-test files, 15+ characters, equal to no '+' line.
"""

import io
import json
import re
import sys
import tarfile

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

PACKAGES = ("DCAgent/swesmith-sandboxes-with_tests-oracle_verified_120s", "78b0d8791f9036b90ea303561254c9d617024258")
TEST_FILE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.(py|go)$")
MIN_LINE = 15


def read_package(binary: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(binary), mode="r:*") as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def fix_lines(solve: str) -> list[str]:
    patch = re.search(r"<< '__SOLUTION__'\n(.*?)\n__SOLUTION__", solve, re.S)
    if not patch:
        return []
    minus, plus, current = set(), set(), None
    for line in patch[1].splitlines():
        if line.startswith("diff --git "):
            current = line.split(" b/", 1)[-1]
            continue
        if current is None or TEST_FILE.search(current) or line.startswith(("---", "+++")):
            continue
        if line.startswith("-"):
            minus.add(line[1:].strip())
        elif line.startswith("+"):
            plus.add(line[1:].strip())
    return sorted(line for line in minus - plus if len(line) >= MIN_LINE)


def main():
    path = hf_hub_download(PACKAGES[0], "tasks.parquet", repo_type="dataset", revision=PACKAGES[1])
    repos, fixes = {}, {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=200, columns=["path", "task_binary"]):
        for task, binary in zip(batch.column("path").to_pylist(), batch.column("task_binary").to_pylist(), strict=True):
            package = read_package(binary)
            instance = json.loads(package["tests/config.json"])["instance_id"]
            owner, name = re.match(r"^(.+?)__(.+?)\.[0-9a-f]{8}\b", instance).groups()
            repos[task] = {"repo": f"{owner}/{name}", "instance_id": instance}
            fixes[task] = fix_lines(package["solution/solve.sh"].decode(errors="replace"))
    with open(sys.argv[1], "w") as stream:
        json.dump(repos, stream)
    with open(sys.argv[2], "w") as stream:
        json.dump(fixes, stream)
    print(f"tasks {len(repos)}, with fix lines {sum(bool(v) for v in fixes.values())}", flush=True)


if __name__ == "__main__":
    main()
