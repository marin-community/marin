# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM backend options and checks against caller-owned endpoints."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from marin.testing.inference.goldens import ModelFixture, read_fixture

VLLM_ATTENTION_BACKENDS = ("FLASH_ATTN", "TRITON_ATTN")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--vllm-attention-backend",
        choices=VLLM_ATTENTION_BACKENDS,
        default="FLASH_ATTN",
        help="Attention backend for the Snowball vLLM e2e.",
    )
    parser.addoption("--endpoint-url", help="Dedicated vLLM endpoint URL")
    parser.addoption("--correctness-fixture", help="JSON model fixture path")
    parser.addoption("--correctness-model", help="Served model name")
    parser.addoption("--runtime-source", help="Expected vLLM source commit")
    parser.addoption("--correctness-output", type=Path, help="Retained JSON report path")


@pytest.fixture
def vllm_attention_backend(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--vllm-attention-backend"))


@dataclass(frozen=True)
class CorrectnessEndpoint:
    url: str
    model: str
    fixture: ModelFixture
    runtime_source: str
    output: Path


@pytest.fixture
def correctness_endpoint(request: pytest.FixtureRequest, tmp_path: Path) -> CorrectnessEndpoint:
    options = ("--endpoint-url", "--correctness-fixture", "--correctness-model", "--runtime-source")
    missing = [option for option in options if not request.config.getoption(option)]
    if missing:
        raise pytest.UsageError(f"Supply {' '.join(missing)} for the endpoint correctness check")
    return CorrectnessEndpoint(
        url=request.config.getoption("--endpoint-url"),
        model=request.config.getoption("--correctness-model"),
        fixture=read_fixture(request.config.getoption("--correctness-fixture")),
        runtime_source=request.config.getoption("--runtime-source"),
        output=request.config.getoption("--correctness-output") or tmp_path / "correctness.json",
    )
