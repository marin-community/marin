# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource target parsing shared by bulk workload commands."""

import csv
import sys

import click


def workload_action_options(command):
    """Apply the options shared by Task and Attempt lifecycle actions."""
    options = [
        click.option("--stdin", is_flag=True, help="Read additional target IDs from stdin CSV rows."),
        click.option("--reason", default="", help="Record an operator reason on the target Attempts."),
        click.option("--dry-run", is_flag=True, help="Print the targets without changing them."),
    ]
    for option in reversed(options):
        command = option(command)
    return command


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
