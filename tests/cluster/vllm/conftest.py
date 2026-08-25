# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific options for the Snowball e2es.

The ``marin_gpu_client`` fixture these tests use lives in the shared
``tests/cluster/conftest.py``; this module only adds the attention-backend option.
"""

import pytest

VLLM_ATTENTION_BACKENDS = ("FLASH_ATTN", "TRITON_ATTN")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--vllm-attention-backend",
        choices=VLLM_ATTENTION_BACKENDS,
        default="FLASH_ATTN",
        help="Attention backend for the Snowball vLLM e2e.",
    )


@pytest.fixture
def vllm_attention_backend(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--vllm-attention-backend"))
