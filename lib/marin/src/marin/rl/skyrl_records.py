# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for native launch files and FineStore attempt records."""

import yaml
from finestore.rl.run_records import (
    RESOLVED_LAUNCH_FILENAME,
    TERMINAL_RECEIPT_FILENAME,
    AttemptRecord,
    LaunchReceipt,
    read_attempt,
    read_attempt_metrics,
    read_attempts,
)
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join


def read_skyrl_launch_config(run_output: str) -> dict:
    """Read the native resolved-launch envelope's compiled configuration."""
    fs, key = filesystem_for(prefix_join(run_output, RESOLVED_LAUNCH_FILENAME))
    return yaml.safe_load(fs.cat(key))["config"]


def read_stored_skyrl_receipt(run_output: str) -> AttemptRecord | None:
    """Read the current native attempt, including its receipt and startup facts."""
    terminal = None
    fs, key = filesystem_for(prefix_join(run_output, TERMINAL_RECEIPT_FILENAME))
    if fs.exists(key):
        terminal = LaunchReceipt.model_validate_json(fs.cat(key))
        config = terminal.config
    else:
        try:
            config = read_skyrl_launch_config(run_output)
        except FileNotFoundError:
            return None
    attempt_id = config["run"]["attempt_id"]
    record = read_attempt(config["artifacts"]["export_root"], attempt_id)
    if record is not None:
        return record
    if terminal is None:
        return None
    return AttemptRecord(run_id=config["run"]["id"], attempt_id=attempt_id, launch=terminal)


def read_stored_skyrl_attempts(run_output: str) -> tuple[str | None, dict[str, AttemptRecord]]:
    """Read the current ID and native training attempts without querying Iris."""
    current = read_stored_skyrl_receipt(run_output)
    if current is None:
        try:
            config = read_skyrl_launch_config(run_output)
        except FileNotFoundError:
            return None, {}
    else:
        config = current.native.config
    attempts = {
        attempt_id: record
        for attempt_id, record in read_attempts(config["artifacts"]["export_root"]).items()
        if record.launch is not None
        and record.run_id == config["run"]["id"]
        and record.native.config["run"]["mode"] != "checkpoint_export"
    }
    if current is not None:
        attempts[current.attempt_id] = current
    return config["run"]["attempt_id"], attempts


def read_skyrl_metrics(run_output: str, attempt_id: str | None = None) -> dict[int, dict]:
    """Read one native attempt's metrics and bounded checkpoint ancestors."""
    config = read_skyrl_launch_config(run_output)
    return read_attempt_metrics(config["artifacts"]["export_root"], attempt_id or config["run"]["attempt_id"])
