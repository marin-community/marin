# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
from finestore.testing.simulated_s3 import simulated_s3


@pytest.fixture
def simulated_bucket():
    """A zero-latency simulated S3 bucket set routed for every ``s3://`` path."""
    with simulated_s3(latency=0.0) as fs:
        yield fs


@pytest.fixture(params=["local", "simulated-s3"])
def archive_root(request, tmp_path) -> Iterator[str]:
    """An empty archive root on local disk or on the simulated S3 backend."""
    if request.param == "local":
        yield str(tmp_path / "run")
        return
    with simulated_s3(latency=0.0):
        yield "s3://finestore-tests/run"
