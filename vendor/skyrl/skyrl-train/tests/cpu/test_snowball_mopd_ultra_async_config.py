"""Contract checks for the asynchronous Snowball MOPD smoke against its synchronous baseline."""

from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest


from cloud.iris.rl_config_translation import compose_skyrl_config, load_rl_recipe, parse_rl_config
from cloud.iris.role_plan import derive_role_plan
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.utils.harbor_errors import ErrorTreatment, classify_exception_type
import skyrl_train.objective.losses  # noqa: F401  (registers policy losses for validate_cfg)
from skyrl_train.utils import validate_cfg

CONFIGS = Path(__file__).parents[3] / "cloud" / "iris" / "configs"
ASYNC_CONFIG = CONFIGS / "snowball_mopd_ultra_async_smoke.yaml"
SYNC_CONFIG = CONFIGS / "snowball_mopd_ultra_smoke.yaml"
STUDENT = "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92"


@pytest.mark.parametrize(
    "recipe",
    [
        "snowball_mopd_ultra_32k",
        "snowball_ultra_rlvr1_split64",
        "snowball_ultra_rlvr1_colocated64",
        "snowball_ultra_rlvr2_split64",
        "snowball_ultra_rlvr2_colocated64",
    ],
)
def test_snowball_ultra_recipes_mask_infrastructure_failures(recipe):
    parsed = parse_rl_config(str(CONFIGS / (recipe + ".yaml")), model_override=STUDENT)
    cfg = compose_skyrl_config(
        parsed,
        {"job_name": "mopd-32k-test", "experiments_dir": "/tmp/exp", "num_nodes": 9},
        SimpleNamespace(gpus_per_node=8),
    ).config

    if recipe == "snowball_mopd_ultra_32k":
        assert cfg.trainer.policy.megatron_config.context_parallel_size == 1
        assert cfg.environment.task_sessions.nemotron_ultra.grading == "skip"
    settings = HarborTaskSettings.from_config(cfg.terminal_bench_config)
    assert settings.agent_timeout == cfg.terminal_bench_config.harbor.override_timeout_sec
    for name in ("VerifierTimeoutError", "ModelServerError"):
        assert classify_exception_type(name, settings.error_handling) is ErrorTreatment.MASK


def test_async_smoke_runs_the_in_process_async_trainer_and_passes_trainer_validation():
    parsed = parse_rl_config(str(ASYNC_CONFIG), model_override=STUDENT)
    compiled = compose_skyrl_config(
        parsed,
        {"job_name": "mopd-async-smoke-test", "experiments_dir": "/tmp/exp", "num_nodes": 8},
        SimpleNamespace(gpus_per_node=8),
    )

    assert compiled.entrypoint == "skyrl_train.entrypoints.main_base"
    validate_cfg(compiled.config)


def test_async_smoke_matches_the_sync_baseline_geometry_and_prompt_count():
    async_config = OmegaConf.to_container(load_rl_recipe(str(ASYNC_CONFIG)), resolve=True)
    sync_config = OmegaConf.to_container(load_rl_recipe(str(SYNC_CONFIG)), resolve=True)

    assert derive_role_plan(async_config) == derive_role_plan(sync_config)
    assert async_config["teachers"] == sync_config["teachers"]
    assert async_config["context_budget"] == sync_config["context_budget"]
    # One optimizer update per step in both schedules, so equal steps mean equal prompts.
    for key in ("train_batch_size", "policy_mini_batch_size", "max_steps"):
        assert async_config["trainer"][key] == sync_config["trainer"][key]
