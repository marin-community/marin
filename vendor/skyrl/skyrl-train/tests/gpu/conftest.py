import os

import pytest
import ray
from tests.gpu.utils import ray_init_for_tests


@pytest.fixture(autouse=True)
def clear_attention_backend_environment():
    # Megatron Core sets these when a test builds a model in-process, and a Ray cluster started
    # later copies them to workers whose backend may differ, so each test starts without them.
    for name in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        os.environ.pop(name, None)


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    ray_init_for_tests()
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
