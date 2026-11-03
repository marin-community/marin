# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from taskforge.llm.test_support import unique_run_name


def test_unique_run_names_differ_within_one_second(monkeypatch):
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw3")
    names = {unique_run_name("live") for _ in range(50)}
    assert len(names) == 50
    assert all(name.startswith("live-") and "-gw3-" in name for name in names)
