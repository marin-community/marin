# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The status file of an executor step output.

Each step writes ``output_path/.executor_status``: one token, SUCCESS, FAILED, DEP_FAILED, or
RUNNING. Readers that do not run steps, such as dashboards, use this module. The lock that guards
a running step lives in ``marin.execution.step_status``.
"""

import json

from fsspec import AbstractFileSystem

from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import prefix_join

STATUS_RUNNING = "RUNNING"
STATUS_FAILED = "FAILED"
STATUS_SUCCESS = "SUCCESS"
STATUS_DEP_FAILED = "DEP_FAILED"


def get_status_path(output_path: str) -> str:
    """Return the path of the status file associated with `output_path`."""
    return prefix_join(output_path, ".executor_status")


def read_step_status(output_path: str) -> str | None:
    """Return the status of the step that writes `output_path`, or None when it has none."""
    path = get_status_path(output_path)
    return read_status_file(url_to_fs(path, use_listings_cache=False)[0], path)


def read_status_file(fs: AbstractFileSystem, path: str) -> str | None:
    """Return the status token stored at `path`, or None when the file is missing or empty.

    A status file holds a single token. Older executors wrote a JSON-lines event log; for
    those files the status is the last event's ``status`` field.
    """
    if not fs.exists(path):
        return None

    # Status changes while peers wait; a ranged S3 read pins the ETag from
    # its metadata lookup and fails if another worker updates the object.
    content = fs.cat_file(path).decode("utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]

    if not lines:
        return None

    if len(lines) == 1 and not lines[0].startswith("{"):
        return lines[0]

    return _parse_event_log_status(lines) or lines[-1]


def _parse_event_log_status(lines: list[str]) -> str | None:
    """Return the latest status from a JSON-lines event log."""
    last_status: str | None = None
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            status = event.get("status")
            if isinstance(status, str):
                last_status = status
    return last_status
