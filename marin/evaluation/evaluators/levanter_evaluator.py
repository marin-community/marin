import dataclasses
import logging
from dataclasses import dataclass, replace

import levanter.eval_harness as eval_harness
import levanter.infra.cli_helpers
import ray
from levanter.infra.ray_tpu import run_on_pod_resumable

from marin.training.training import (
    _add_default_env_variables,
    _add_run_env_variables,
    _check_for_wandb_key,
    _update_config_to_use_out_path,
)
from marin.utilities.dataclass_utils import shallow_asdict
from marin.utilities.gcs_utils import get_vm_region

logger = logging.getLogger(__name__)


@dataclass
class LmEvalOnPodConfig(eval_harness.EvalHarnessMainConfig):
    output_path: str | None = None
    """Path to save the evaluation output to. If None, a default path will be used."""

    tpu_type: str | None = None
    """Example: "v4-128". If None, the evaluation will run locally."""

    env: dict = dataclasses.field(default_factory=dict)
    """Environment variables to set for the evaluation run."""

    bypass_path_checks: bool = False
    """If True, don't check that paths are set and are in the same region as the VM."""

    node_count: int = 1
    """Number of TPU slices for evals."""


@ray.remote(num_cpus=0.1, runtime_env={"pip": ["levanter>=1.2.dev1074"]})
def run_levanter_lm_eval(config: LmEvalOnPodConfig):
    """
    Run the Levanter's LM-Eval-Harness on a Ray cluster.

    Similar to the structure of run_levanter_train_lm but for running evals.
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()

    if config.output_path is not None:
        logger.info(f"Using output path: {config.output_path}")
        config = _update_config_to_use_out_path(config)

    env = _add_default_env_variables(config.env, default_launch_config.get("env"))
    _check_for_wandb_key(env)
    env = _add_run_env_variables(env)
    config = replace(config, env=env)

    if not config.bypass_path_checks and config.tpu_type is not None:
        ray.get(ray.remote(_doublecheck_paths_eval).options(num_cpus=0.1).remote(config, must_save_checkpoints=True))

    eval_config = _upcast_eval_config(config)

    @ray.remote
    def eval_lm_task():
        eval_harness.run_eval_harness_main(eval_config)

    if config.tpu_type is not None:
        return run_on_pod_resumable(eval_lm_task, config.tpu_type, max_retries_failure=10)
    else:
        return ray.get(eval_lm_task.remote())


def _upcast_eval_config(config: LmEvalOnPodConfig) -> eval_harness.EvalHarnessMainConfig:
    """
    upcast LmEvalOnPodConfig to EvalHarnessMainConfig by stripping the TPU type and env
    """
    dict_config = shallow_asdict(config)
    fields_to_remote = set(dict_config.keys()) - set(eval_harness.EvalHarnessMainConfig.__dataclass_fields__.keys())
    for field in fields_to_remote:
        del dict_config[field]
    eval_config = eval_harness.EvalHarnessMainConfig(**dict_config)

    return eval_config


def _doublecheck_paths_eval(config: LmEvalOnPodConfig, must_save_checkpoints):
    """
    Double-check paths specifically for eval configs.
    """
    local_ok = config.bypass_path_checks or config.tpu_type is None
    try:
        region = get_vm_region()
        logger.info(f"VM region: {region}")
    except ValueError as e:
        if local_ok:
            logger.warning("Could not determine the region of the VM. This is fine if you're running locally.")
            return
        raise ValueError("Could not determine the region of the VM. This is required for path checks.") from e

    # _check_path_in_region("data.cache_dir", config.data.cache_dir, none_ok=True, region=region, local_ok=local_ok)
    # # now check all subcaches if applicable
    # if isinstance(config.data, LMMixtureDatasetConfig):
    #     for key, subcache in config.data.configs.items():
    #         _check_path_in_region(
    #             f"data.configs[{key}].cache_dir", subcache.cache_dir, none_ok=True, region=region, local_ok=local_ok
    #         )
    # _check_path_in_region(
    #     "trainer.checkpointer.base_path",
    #     config.trainer.checkpointer.base_path,
    #     none_ok=not must_save_checkpoints,
    #     region=region,
    #     local_ok=local_ok,
    # )

    # if config.hf_save_path is not None:
    #     _check_path_in_region(
    #         "hf_save_path", config.hf_save_path, none_ok=not must_save_checkpoints, region=region, local_ok=local_ok
    #     )
    # else:
    #     logger.warning("hf_save_path is not set. This is fine if you don't want HF checkpoints.")

    return config
