# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from rigging.filesystem.step_status import STATUS_FAILED, STATUS_SUCCESS, get_status_path, read_step_status


def test_step_without_a_status_file_has_no_status(tmp_path: Path):
    assert read_step_status(str(tmp_path)) is None


def test_status_is_the_stored_token(tmp_path: Path):
    Path(get_status_path(str(tmp_path))).write_text(STATUS_SUCCESS)

    assert read_step_status(str(tmp_path)) == STATUS_SUCCESS


def test_status_of_an_event_log_is_its_last_event(tmp_path: Path):
    events = [{"status": "RUNNING"}, {"status": STATUS_FAILED, "message": "boom"}]
    Path(get_status_path(str(tmp_path))).write_text("\n".join(json.dumps(event) for event in events))

    assert read_step_status(str(tmp_path)) == STATUS_FAILED
