import pytest
from marinskyrl.environment_contract import VLLM_BATCH_INVARIANT_ENV, EnvVarManager, EnvVarScope
from marinskyrl.runtime_options import NumericsMode
from omegaconf import OmegaConf


@pytest.mark.parametrize("numerics_mode", list(NumericsMode))
def test_only_batch_invariant_numerics_sets_vllm_batch_invariant_in_ray_and_inference_workers(numerics_mode):
    manager = EnvVarManager.from_config(
        OmegaConf.create({"trainer": {"numerics": {"mode": numerics_mode.value}}}), environ={}
    )

    expected = "1" if numerics_mode is NumericsMode.BATCH_INVARIANT else None
    for scope in (EnvVarScope.RAY_WORKER, EnvVarScope.INFERENCE_WORKER):
        assert manager.environment_for(scope).get(VLLM_BATCH_INVARIANT_ENV) == expected
