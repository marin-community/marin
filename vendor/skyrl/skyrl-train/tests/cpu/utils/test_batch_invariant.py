import sys
import types

import pytest

from skyrl_train.batch_invariant import enable_trainer_batch_invariance
from skyrl_train.env_vars import VLLM_BATCH_INVARIANT_ENV
from skyrl_train.inference_engines.ray_wrapped_inference_engine import _build_inference_engine_runtime_env
from skyrl_train.utils.utils import prepare_runtime_environment
from tests.cpu.util import example_dummy_config


def test_batch_invariant_reaches_ray_and_nested_vllm_workers(monkeypatch):
    cfg = example_dummy_config()
    cfg.trainer.numerics.mode = "batch_invariant"

    ray_environment = prepare_runtime_environment(cfg)
    monkeypatch.setenv(VLLM_BATCH_INVARIANT_ENV, ray_environment[VLLM_BATCH_INVARIANT_ENV])

    assert ray_environment[VLLM_BATCH_INVARIANT_ENV] == "1"
    assert _build_inference_engine_runtime_env()["env_vars"][VLLM_BATCH_INVARIANT_ENV] == "1"


def test_trainer_requires_registered_kernels_after_vllm_activation(monkeypatch):
    batch_invariant_module = types.ModuleType("vllm.model_executor.determinism.batch_invariant")
    batch_invariant_module.init_batch_invariance = lambda: None
    monkeypatch.setitem(sys.modules, batch_invariant_module.__name__, batch_invariant_module)
    monkeypatch.setenv(VLLM_BATCH_INVARIANT_ENV, "1")

    with pytest.raises(RuntimeError, match="without exposing registered CUDA overrides"):
        enable_trainer_batch_invariance(True)
