# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Look for copies of a SWE-smith repo's own source outside /testbed in its base image (task #206, pool v3).

Clean copies of a repo can ship elsewhere in the image (setuptools vendors typeguard; conda's package cache holds
boltons). For each repo: take up to SAMPLES source files from the swesmith/<repo> mirror (not tests), keep each
path's last two components (package directory and file name, e.g. typeguard/_checkers.py), list every regular file
in the repo's base image outside /testbed, /proc, /sys and /dev, and report the files whose path ends with one of
those suffixes. Any hit is a candidate copy of clean code to inspect. mini-swe-agent's tool environment is installed
at agent setup and is not in the base image; #207's probe covers it.

Usage:
    python -m experiments.post_training.sft_shootout_pool.image_check <swesmith-packages.parquet> <mirror dir> <repo> ...
"""

import json
import random
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from experiments.post_training.sft_shootout_pool.build_swesmith import MIRROR_NAME
from experiments.post_training.sft_shootout_pool.sample import upstream_repo
from experiments.post_training.sft_shootout_pool.swesmith_packages import base_image, instance_id, read_package
from experiments.post_training.sft_shootout_pool.swesmith_restore import git, mirror

SAMPLES = 20
SOURCE_SUFFIXES = (".py", ".go", ".pyx", ".c")
LIST_FILES = (
    "find / \\( -path /proc -o -path /sys -o -path /dev -o -path /testbed \\) -prune -o -type f -print 2>/dev/null"
)


def source_suffixes(repository: Path) -> list[str]:
    files = git(repository, "ls-tree", "-r", "--name-only", "HEAD").decode().splitlines()
    candidates = sorted(
        path
        for path in files
        if path.endswith(SOURCE_SUFFIXES)
        and path.count("/") >= 1
        and "test" not in path.lower()
        and not path.rsplit("/", 1)[-1].startswith("__")
    )
    picked = random.Random(f"image-check:{repository.name}").sample(candidates, min(SAMPLES, len(candidates)))
    return sorted({"/".join(path.split("/")[-2:]) for path in picked})


def main() -> None:
    parquet, mirrors, wanted = sys.argv[1], Path(sys.argv[2]), set(sys.argv[3:])
    seen = {}
    for row in pq.read_table(parquet).to_pylist():
        package = read_package(row["task_binary"])
        repo = upstream_repo(json.loads(package["tests/config.json"])["repo"])
        if repo in wanted and repo not in seen:
            seen[repo] = (base_image(package), MIRROR_NAME.match(instance_id(package))[1])
    for repo, (image, name) in sorted(seen.items()):
        suffixes = source_suffixes(mirror(name, mirrors))
        listing = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", image, "bash", "-c", LIST_FILES],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        hits = sorted(path for path in listing if any(path.endswith("/" + suffix) for suffix in suffixes))
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, check=False)
        print(
            "IMAGE_CHECK "
            + json.dumps(
                {"repo": repo, "image": image, "files_listed": len(listing), "suffixes": suffixes, "hits": hits}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
