# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Subprocess entry point for Zephyr shard execution.

Invoked as ``python -m zephyr.subprocess_worker <task_file> <result_file>``.
Provides full memory isolation — all allocations are reclaimed on exit.
"""

import sys

from zephyr.execution import _subprocess_execute_shard


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m zephyr.subprocess_worker <task_file> <result_file>", file=sys.stderr)
        sys.exit(1)
    _subprocess_execute_shard(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
