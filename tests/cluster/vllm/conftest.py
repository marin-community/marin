# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM backend options and checks against caller-owned endpoints."""

import pytest
from marin.testing.inference.endpoint import VllmEndpoint
from marin.testing.inference.endpoint_checks import TestSpec as InferenceTestSpec
from marin.testing.inference.endpoint_checks import read_test_spec

VLLM_ATTENTION_BACKENDS = ("FLASH_ATTN", "TRITON_ATTN")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--vllm-attention-backend",
        choices=VLLM_ATTENTION_BACKENDS,
        default="FLASH_ATTN",
        help="Attention backend for the Snowball vLLM e2e.",
    )
    parser.addoption("--endpoint-url", help="Dedicated vLLM endpoint URL")
    parser.addoption("--correctness-spec", help="JSON test specification path")
    parser.addoption("--correctness-model", help="Override the HTTP adapter's default model")
    parser.addoption("--runtime-source", help="Optional runtime provenance")
    parser.addoption("--correctness-output", help="Save a typed report at this directory or storage URI")


@pytest.fixture
def vllm_attention_backend(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--vllm-attention-backend"))


@pytest.fixture
def correctness_endpoint(request: pytest.FixtureRequest) -> VllmEndpoint:
    url = request.config.getoption("--endpoint-url")
    if not url:
        raise pytest.UsageError("Supply --endpoint-url for the endpoint correctness check")
    endpoint = VllmEndpoint(url, model=request.config.getoption("--correctness-model"))
    runtime_source = request.config.getoption("--runtime-source")
    if runtime_source:
        endpoint.metadata["runtime_source"] = runtime_source
    return endpoint


@pytest.fixture
def correctness_spec(request: pytest.FixtureRequest) -> InferenceTestSpec:
    path = request.config.getoption("--correctness-spec")
    if not path:
        raise pytest.UsageError("Supply --correctness-spec for the endpoint correctness check")
    return read_test_spec(path)
