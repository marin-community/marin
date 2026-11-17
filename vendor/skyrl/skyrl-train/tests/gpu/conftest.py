import os

import pytest
import ray
from tests.gpu.utils import ray_init_for_tests


ATTENTION_BACKEND_ENV_VARS = ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")


@pytest.fixture(autouse=True)
def isolate_attention_backend_environment():
    # Megatron Core writes its attention backend into these variables when it builds a model in
    # the pytest process, and a Ray cluster started later hands them to every worker. A worker that
    # selects another backend then fails Core's check, so each test starts and ends without them.
    for name in ATTENTION_BACKEND_ENV_VARS:
        os.environ.pop(name, None)
    yield
    for name in ATTENTION_BACKEND_ENV_VARS:
        os.environ.pop(name, None)


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    ray_init_for_tests()
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
