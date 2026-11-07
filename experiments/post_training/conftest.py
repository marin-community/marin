# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load Snowball fixture producers only when requested; sibling packages collect independently."""

import pytest


@pytest.fixture
def selection(tmp_path, monkeypatch):
    import experiments.post_training.snowball_rl_test_fixtures as support  # noqa: PLC0415

    return support.selection(tmp_path, monkeypatch)


@pytest.fixture
def source_provenance(monkeypatch):
    import experiments.post_training.snowball_rl_test_fixtures as support  # noqa: PLC0415

    return support.source_provenance(monkeypatch)


@pytest.fixture
def synthetic_current_controls(monkeypatch):
    import experiments.post_training.snowball_rl_test_fixtures as support  # noqa: PLC0415

    return support.synthetic_current_controls(monkeypatch)


@pytest.fixture
def native_json():
    import experiments.post_training.snowball_rl_test_fixtures as support  # noqa: PLC0415

    return support.native_json()


@pytest.fixture
def recorded_runs(selection, tmp_path, monkeypatch, request):
    import experiments.post_training.snowball_rl_test_fixtures as support  # noqa: PLC0415

    return support.recorded_runs(selection, tmp_path, monkeypatch, request)
