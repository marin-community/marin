# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop this node's page-cache pages of every file under a directory, without root.

``posix_fadvise(POSIX_FADV_DONTNEED)`` evicts a file's clean cached pages, on NFS too, so a following read goes to
the file server. The server's own cache is untouched. Prints the node's ``Cached`` memory before and after.

    python -m experiments.amd.june_67b_a2b.vultr.evict_page_cache /work/.../checkpoints/step-2
"""

import os
import socket
import sys


def cached_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1]) / 2**20
    raise RuntimeError("no Cached line in /proc/meminfo")


def evict(root: str) -> tuple[int, int]:
    files = total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            fd = os.open(os.path.join(directory, name), os.O_RDONLY)
            try:
                total += os.fstat(fd).st_size
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            files += 1
    return files, total


def main() -> None:
    before = cached_gib()
    files, total = evict(sys.argv[1])
    print(
        f"{socket.gethostname()}: evicted {files} files ({total / 2**30:.1f} GiB); "
        f"page cache {before:.1f} -> {cached_gib():.1f} GiB"
    )


if __name__ == "__main__":
    main()
