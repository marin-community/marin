from skyrl_train.batch_invariant import BATCH_INVARIANT_NCCL_ENV
from skyrl_train.env_vars import VLLM_BATCH_INVARIANT_ENV
from skyrl_train.inference_engines.ray_wrapped_inference_engine import _build_inference_engine_runtime_env
from skyrl_train.utils.utils import prepare_runtime_environment
from tests.cpu.util import example_dummy_config


def test_batch_invariant_reaches_ray_and_nested_vllm_workers(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: True)
    cfg = example_dummy_config()
    cfg.trainer.numerics.mode = "batch_invariant"

    ray_environment = prepare_runtime_environment(cfg)
    monkeypatch.setenv(VLLM_BATCH_INVARIANT_ENV, ray_environment[VLLM_BATCH_INVARIANT_ENV])

    assert ray_environment[VLLM_BATCH_INVARIANT_ENV] == "1"
    assert BATCH_INVARIANT_NCCL_ENV.items() <= ray_environment.items()
    assert _build_inference_engine_runtime_env()["env_vars"][VLLM_BATCH_INVARIANT_ENV] == "1"
