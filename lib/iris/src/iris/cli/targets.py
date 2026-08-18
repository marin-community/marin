# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource target parsing shared by bulk workload commands."""

import csv
import sys


def collect_resource_ids(targets: tuple[str, ...], read_stdin: bool) -> list[str]:
    """Combine positional IDs with the first ID column from stdin CSV rows."""
    consume_stdin = read_stdin or "-" in targets
    collected = [target for target in targets if target != "-"]
    if not consume_stdin:
        return collected

    for row in csv.reader(sys.stdin):
        if row and row[0].strip().startswith("/"):
            collected.append(row[0].strip())
    return collected
