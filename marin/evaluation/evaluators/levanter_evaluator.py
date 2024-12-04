import os
import subprocess
import time
from abc import ABC
from typing import ClassVar

import ray
import requests

import levanter.eval_harness as eval_harness

from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import Dependency, Evaluator, ModelConfig
from marin.evaluation.utils import kill_process_on_port
from marin.utils import remove_tpu_lockfile_on_exit

@dataclass
class LmEvalOnPodConfig(eval_harness.LmEvalHarnessConfig):
    output_path: str | None = None
    """Path to save the evaluation output to. If None, a default path will be used."""

    tpu_type: str | None = None
    """Example: "v4-128". If None, the evaluation will run locally."""

    env: dict = dataclasses.field(default_factory=dict)
    """Environment variables to set for the evaluation run."""

    bypass_path_checks: bool = False
    """If True, don't check that paths are set and are in the same region as the VM."""

    node_count: int = 1
    """Number of TPU slices for training."""


@ray.remote(num_cpus=0.1, runtime_env={"pip": ["levanter>=1.2.dev1074"]})
def run_levanter_lm_eval(config: LmEvalOnPodConfig):
    """
    Run the Levanter LM Eval Harness on a Ray cluster.

    Similar to run_levanter_train_lm but for SFT training.
    """
    default_launch_config = levanter.infra.cli_helpers.load_config()

    if config.output_path is not None:
        logger.info(f"Using output path: {config.output_path}")
        config = _update_config_to_use_out_path(config)

    env = _add_default_env_variables(config.env, default_launch_config.get("env"))
    _check_for_wandb_key(env)
    env = _add_run_env_variables(env)
    config = replace(config, env=env)

    config = _suppress_ray_config(config)
    config = _enforce_run_id(config)
    logger.info(f"Using run ID: {config.trainer.id}")

    runtime_env = RuntimeEnv(env_vars=config.env, pip=["levanter>=1.2.dev1074"])

    if not config.bypass_path_checks and config.tpu_type is not None:
        ray.get(
            ray.remote(_doublecheck_paths_sft)
            .options(runtime_env=runtime_env, num_cpus=0.1)
            .remote(config, must_save_checkpoints=True)
        )

    eval_config = _upcast_eval_config(config)

    @ray.remote(runtime_env=runtime_env)
    def eval_lm_task():
        eval_harness.run_eval_harness_main(eval_config)

    if config.tpu_type is not None:
        return run_on_pod_resumable(eval_lm_task, config.tpu_type, max_retries_failure=10)
    else:
        return ray.get(eval_lm_task.remote())


def _upcast_eval_config(config: LmEvalOnPodConfig) -> eval_harness.LmEvalHarnessConfig:
    """
    upcast LmEvalOnPodConfig to LmEvalHarnessConfig by stripping the TPU type and env
    """
    dict_config = shallow_asdict(config)
    fields_to_remote = set(dict_config.keys()) - set(eval_harness.LmEvalHarnessConfig.__dataclass_fields__.keys())
    for field in fields_to_remote:
        del dict_config[field]
    eval_config = eval_harness.LmEvalHarnessConfig(**dict_config)

    return eval_config

def _enforce_run_id(config: EvalLmOnPodConfig) -> EvalLmOnPodConfig:
    """
    Levanter will auto-generate a run ID if it's not set. We want to enforce that it's set, so that it resumes
    properly after preemption.

    Look for:
        * config.trainer.id
        * config.env.RUN_ID
        * environment variable RUN_ID
        * default to a random UID
    """
    run_id = config.trainer.id

    if run_id is None:
        run_id = config.env.get("RUN_ID", os.environ.get("RUN_ID"))

    if not run_id:
        run_id = levanter.infra.cli_helpers.default_run_id()
        logger.warning(f"Run ID not set. Using default: {run_id}")

    return replace(config, trainer=replace(config.trainer, id=run_id))


def _suppress_ray_config(config: EvalLmOnPodConfig) -> EvalLmOnPodConfig:
    """
    Levanter wants to auto-start the Ray cluster, but we're already in a Ray cluster. Disable that.
    """
    # my other kingdom for lenses
    if config.trainer.ray.auto_start_cluster:
        logger.info("Ray cluster is set to auto-start, but that's not what we want for Marin. Disabling.")
        config = replace(
            config,
            trainer=replace(
                config.trainer,
                ray=replace(config.trainer.ray, auto_start_cluster=False, start_workers=False),
            ),
        )
    elif config.trainer.ray.start_workers:
        logger.info("Ray cluster is set to start workers, but that's not what we want for Marin. Disabling.")
        config = replace(
            config,
            trainer=replace(config.trainer, ray=replace(config.trainer.ray, start_workers=False)),
        )
    return config

