# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-backed fixtures for Zephyr adapter tests."""

import time
from pathlib import Path

import pytest
from fray.iris_backend import FrayIrisClient
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.config import load_config, make_local_config
from iris.cluster.lifecycle import connect_cluster
from iris.cluster.types import Entrypoint, ResourceSpec

ZEPHYR_ROOT = Path(__file__).resolve().parents[2]
IRIS_CONFIG = Path(__file__).resolve().parents[3] / "iris" / "config" / "ci-test.yaml"


def _parent_holder_entrypoint() -> None:
    time.sleep(3600)


@pytest.fixture(scope="module")
def iris_cluster():
    config = make_local_config(load_config(IRIS_CONFIG))
    with connect_cluster(config) as url:
        yield url


@pytest.fixture(scope="module")
def iris_integration_client(iris_cluster):
    iris_client = IrisClient.remote(iris_cluster, workspace=ZEPHYR_ROOT)
    client = FrayIrisClient.from_iris_client(iris_client)
    parent_job = iris_client.submit(
        entrypoint=Entrypoint.from_callable(_parent_holder_entrypoint),
        name="test",
        resources=ResourceSpec(cpu=1, memory="512m"),
    )
    try:
        with iris_ctx_scope(IrisContext(job_id=parent_job.job_id, client=iris_client)):
            yield client
    finally:
        iris_client.terminate(parent_job.job_id)
        client.shutdown(wait=True)
