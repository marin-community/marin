# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def local_pipeline_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prefix = str(tmp_path / "marin")
    monkeypatch.setenv("MARIN_PREFIX", prefix)
    monkeypatch.setenv("MARIN_TEMP_PREFIX", prefix)
