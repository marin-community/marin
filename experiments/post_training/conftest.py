# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def publication_transport(monkeypatch):
    from experiments.post_training.inventory.publication_test_support import (  # noqa: PLC0415
        publication_transport as make_transport,
    )

    return make_transport(monkeypatch)


@pytest.fixture
def recorded_runs(selection, tmp_path, monkeypatch, publication_transport, request):
    from experiments.post_training.inventory.publication_test_support import (  # noqa: PLC0415
        recorded_runs as make_runs,
    )

    return make_runs(selection, tmp_path, monkeypatch, publication_transport, request)
