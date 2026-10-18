# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Look for copies of a SWE-smith repo's own source outside /testbed in its base image (pools v3-v4).

Clean copies of a repo can ship elsewhere in the image (setuptools vendors typeguard; conda's package cache holds
boltons). See PROBE_METHOD for the two passes. Any hit is a candidate to inspect by hand: console-script stubs in
bin/, license directories and unrelated files that share a name are not copies. mini-swe-agent's tool environment is
installed at agent setup and is not in the base image; the generation pilot's tool-environment probe covers it.

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
from experiments.post_training.sft_shootout_pool.repo_names import IMPORT_NAMES
from experiments.post_training.sft_shootout_pool.sample import upstream_repo
from experiments.post_training.sft_shootout_pool.swesmith_packages import base_image, instance_id, read_package
from experiments.post_training.sft_shootout_pool.swesmith_restore import git, mirror

SAMPLES = 20
SOURCE_SUFFIXES = (".py", ".go", ".pyx", ".c")
PROBE_METHOD = (
    "For each repo's base image (the FROM of a pool package's environment/Dockerfile), list every regular file outside "
    "/testbed, /proc, /sys and /dev with no network. Suffix pass: take 20 seeded source files (.py/.go/.pyx/.c, not "
    "tests, not dunder files) from the swesmith/<repo> mirror and flag listed files whose path ends with a sampled "
    "file's last two components. Name pass: flag listed files under a directory named after the repo's import name "
    "(repo_names.IMPORT_NAMES), module files named <import>.py, and Go module cache entries github.com/<owner>/<name>@. "
    "Hits were then inspected by hand. Code: experiments/post_training/sft_shootout_pool/image_check.py."
)
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
        owner, name_only = repo.lower().split("/")
        names = IMPORT_NAMES.get(repo, [])
        name_hits = sorted(
            path
            for path in listing
            if any(f"/{name}/" in path or path.endswith(f"/{name}.py") for name in names)
            or f"/github.com/{owner}/{name_only}@" in path.lower()
        )
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True, check=False)
        print(
            "IMAGE_CHECK "
            + json.dumps(
                {
                    "repo": repo,
                    "image": image,
                    "files_listed": len(listing),
                    "suffixes": suffixes,
                    "hits": hits,
                    "import_names": names,
                    "name_hits": name_hits[:50],
                    "name_hit_count": len(name_hits),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
