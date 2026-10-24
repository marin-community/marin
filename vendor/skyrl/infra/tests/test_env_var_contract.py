from pathlib import Path

import pytest
import yaml
from marinskyrl.environment_contract import (
    ENV_VAR_SPECS,
    EXECUTION_UID_ENV,
    EnvVarScope,
    EnvVarSource,
    EnvVarManager,
)
from omegaconf import OmegaConf
from skyrl_train.config.numerics import NumericsMode

EXECUTION_UID = "iris:01JABCDEF0123456789"
WORKER_ENVIRONMENT = {
    "SKYRL_ENABLE_NUMA_AFFINITY": "1",
    "RAY_USE_UVLOOP": "0",
    "UV_USE_IO_URING": "0",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
}


def test_managed_names_have_one_owner_and_config_control():
    names = [spec.name for spec in ENV_VAR_SPECS]
    assert len(names) == len(set(names))

    config = yaml.safe_load(Path("skyrl-train/skyrl_train/config/ppo_base_config.yaml").read_text())
    for spec in ENV_VAR_SPECS:
        if spec.source not in {EnvVarSource.CONFIG, EnvVarSource.DERIVED}:
            continue
        value = config
        for component in spec.owner.split("."):
            assert component in value, f"{spec.name} has no config owner at {spec.owner}"
            value = value[component]


@pytest.mark.parametrize(
    ("numerics_mode", "worker_environment"),
    [
        (NumericsMode.NATIVE, WORKER_ENVIRONMENT),
        (NumericsMode.BATCH_INVARIANT, {**WORKER_ENVIRONMENT, "VLLM_BATCH_INVARIANT": "1"}),
    ],
)
def test_process_boundary_settings_project_only_to_their_scopes(numerics_mode, worker_environment):
    config = OmegaConf.create(
        {
            "trainer": {
                "debug_mode": "off",
                "collective_phase_diagnostics": False,
                "placement": {"enable_numa_affinity": True},
                "numerics": {"mode": numerics_mode.value},
            },
            "generator": {"fuse_weights": True},
        }
    )

    manager = EnvVarManager.from_config(config, environ={EXECUTION_UID_ENV: EXECUTION_UID})

    # The head's attempt uid would override each node's own once it reaches a worker's runtime_env.
    assert manager.environment_for(EnvVarScope.DRIVER) == {EXECUTION_UID_ENV: EXECUTION_UID}
    assert manager.environment_for(EnvVarScope.TASK_RUNTIME) == {EXECUTION_UID_ENV: EXECUTION_UID}
    for scope in (EnvVarScope.RAY_WORKER, EnvVarScope.INFERENCE_WORKER):
        environ = {}
        assert manager.apply_to_process(scope, environ=environ) == worker_environment
        assert environ == worker_environment
