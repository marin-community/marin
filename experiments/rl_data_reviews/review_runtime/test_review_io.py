# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral checks for native verifier source evidence."""

import importlib
import json
import sys
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from experiments.rl_data_reviews.review_runtime.review_io import capture_native_sources, native_calls


@pytest.fixture
def isolated_verifyit_imports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in list(sys.modules):
        if name == "verifyit" or name.startswith("verifyit."):
            monkeypatch.delitem(sys.modules, name)
    yield
    for name in list(sys.modules):
        if name == "verifyit" or name.startswith("verifyit."):
            sys.modules.pop(name)


@pytest.mark.usefixtures("isolated_verifyit_imports")
def test_verifyit_call_captures_source_and_installed_revision(tmp_path, monkeypatch):
    package = tmp_path / "verifyit"
    package.mkdir()
    (package / "__init__.py").write_text("from .checks import check\n")
    (package / "checks.py").write_text("def check(value):\n    return value == 7\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    distribution = SimpleNamespace(
        version="0.8.1",
        read_text=lambda name: json.dumps(
            {
                "url": "https://github.com/marin-community/verifyit",
                "vcs_info": {"vcs": "git", "commit_id": "d3edc5d240d53edbd0c0e4a53c0e112629550f7e"},
            }
        ),
    )
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: distribution)
    with native_calls() as called:
        verifier = importlib.import_module("verifyit.checks")
        assert verifier.check(7)
    capture_native_sources(tmp_path / "review", called)

    index = json.loads((tmp_path / "review/native-code-index.json").read_text())
    check = next(entry for entry in index if entry["module"] == "verifyit.checks")
    assert check["called_in_attempt"] is True
    assert (tmp_path / "review" / check["path"]).read_bytes() == (package / "checks.py").read_bytes()
    assert check["package"] == {
        "version": "0.8.1",
        "source_url": "https://github.com/marin-community/verifyit",
        "source_commit": "d3edc5d240d53edbd0c0e4a53c0e112629550f7e",
    }
