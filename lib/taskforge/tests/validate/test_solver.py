# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from taskforge.validate.trials import write_evidence


def test_an_attempt_file_appears_only_once_its_whole_payload_is_written(tmp_path, monkeypatch):
    path = tmp_path / "solver" / "0" / "attempt-0.json"

    def interrupted(self, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_evidence(path, b'{"outcome": "graded"}')

    assert list(path.parent.iterdir()) == []
