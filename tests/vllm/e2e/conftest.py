# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for vLLM integration tests running through Marin's Iris cluster."""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import cloudpickle
import pytest
from iris.cli.connect import open_iris_client
from iris.client import IrisClient

from . import june_67b_a2b

MARIN_ROOT = Path(__file__).resolve().parents[3]
MARIN_GPU_CLUSTER = "cw-us-east-02a"
VLLM_ATTENTION_BACKENDS = ("FLASH_ATTN", "TRITON_ATTN")

# Iris serializes the direct test callable by value; register its shared test helper too.
cloudpickle.register_pickle_by_value(june_67b_a2b)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--vllm-attention-backend",
        choices=VLLM_ATTENTION_BACKENDS,
        default="FLASH_ATTN",
        help="Attention backend for the June 67B vLLM e2e.",
    )


@pytest.fixture
def vllm_attention_backend(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--vllm-attention-backend"))


@pytest.fixture
def marin_gpu_client() -> Iterator[IrisClient]:
    # This test drives a live CoreWeave GPU node; it can only run where the cluster's
    # kube credentials are present. In CI (and any workstation without them) opening the
    # client raises ConfigException, so skip rather than error the whole integration run.
    # Import kubernetes lazily: it ships with iris[controller] and is absent from the
    # unit-test env, which still collects (imports) this module before deselecting it.
    from kubernetes.config.config_exception import ConfigException  # noqa: PLC0415

    with contextlib.ExitStack() as stack:
        try:
            client = stack.enter_context(open_iris_client(cluster_name=MARIN_GPU_CLUSTER, workspace=MARIN_ROOT))
        except ConfigException as exc:
            pytest.skip(f"CoreWeave cluster {MARIN_GPU_CLUSTER!r} unavailable (no kube-config): {exc}")
        yield client
