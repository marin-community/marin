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


