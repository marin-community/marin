# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM-specific options for the Snowball e2es.

The ``marin_gpu_client`` fixture these tests use lives in the shared
``tests/cluster/conftest.py``; this module only adds the attention-backend option
and registers remote-job helpers for by-value pickling.
"""

import cloudpickle
import pytest

from tests.cluster.vllm import (
    backend_parity,
    snowball,
    snowball_checkpoint,
    snowball_export,
    snowball_exported_levanter,
    snowball_levanter,
    snowball_vllm,
    snowball_vllm_production,
    snowball_vllm_production_oracle,
)

VLLM_ATTENTION_BACKENDS = ("FLASH_ATTN", "TRITON_ATTN")

# Iris serializes the direct test callable by value; register its shared test helpers too.
cloudpickle.register_pickle_by_value(snowball)
cloudpickle.register_pickle_by_value(snowball_checkpoint)
cloudpickle.register_pickle_by_value(backend_parity)
cloudpickle.register_pickle_by_value(snowball_levanter)
cloudpickle.register_pickle_by_value(snowball_export)
cloudpickle.register_pickle_by_value(snowball_exported_levanter)
cloudpickle.register_pickle_by_value(snowball_vllm)
cloudpickle.register_pickle_by_value(snowball_vllm_production)
cloudpickle.register_pickle_by_value(snowball_vllm_production_oracle)


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
