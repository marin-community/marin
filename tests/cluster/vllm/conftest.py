# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific options for the June 67B e2es.

The ``marin_gpu_client`` fixture these tests use lives in the shared
``tests/cluster/conftest.py``; this module only adds the attention-backend option
and registers the shared checkpoint helper for by-value pickling.
"""

import cloudpickle
import pytest

from tests.cluster.vllm import backend_parity, june_67b_a2b

VLLM_ATTENTION_BACKENDS = ("FLASH_ATTN", "TRITON_ATTN")

# Iris serializes the direct test callable by value; register its shared test helpers too.
cloudpickle.register_pickle_by_value(june_67b_a2b)
cloudpickle.register_pickle_by_value(backend_parity)


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
