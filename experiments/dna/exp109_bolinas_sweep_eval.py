# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off VEP eval of a transfer sweep checkpoint on v5p-8."""

import dataclasses
import logging
import os

import jmp
import levanter.eval_harness as eval_harness
from fray.v2 import Entrypoint, JobRequest, TpuConfig, create_environment, current_client
from levanter.distributed import RayConfig
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig

from experiments.dna.defaults import dna_effective_seq_len
from experiments.dna.exp109_bolinas_scaling_sweep import (
    REFERENCE_HPARAMS,
    TOKENIZER,
    TRANSFER_HIDDEN_SIZE,
    _build_model_config,
)
from experiments.dna.smoke_tests.eval_traitgym import EvalTraitGymConfig, _build_eval_env, _run_eval_on_tpu
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255, convert_to_levanter_task_config
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main

logger = logging.getLogger(__name__)

CHECKPOINT = "mirror://checkpoints/dna-bolinas-transfer-v0.12-learning_rate-2-fbe714/checkpoints/step-4767"
RESOURCES = ResourceConfig.with_tpu("v5p-8")


def run_eval(config: EvalTraitGymConfig):
    env = _build_eval_env(config)
    model_config = _build_model_config(TRANSFER_HIDDEN_SIZE, REFERENCE_HPARAMS.initializer_range)

    eval_config = eval_harness.EvalHarnessMainConfig(
        eval_harness=eval_harness.LmEvalHarnessConfig(
            task_spec=convert_to_levanter_task_config([TRAITGYM_MENDELIAN_V2_255]),
            include_path="experiments/evals/custom_tasks",
            log_samples=False,
            max_packed_segments=1,
        ),
        tokenizer=config.tokenizer,
        checkpoint_path=config.checkpoint_path,
        checkpoint_is_hf=False,
        trainer=TrainerConfig(
            tracker=NoopConfig(),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            per_device_eval_parallelism=4096,
            ray=RayConfig(auto_start_cluster=False),
        ),
        model=dataclasses.replace(model_config, max_seq_len=dna_effective_seq_len(256, config.tokenizer)),
    )

    logger.info("Evaluating checkpoint: %s", config.checkpoint_path)

    client = current_client()
    extras = ["eval"]
    if isinstance(config.resources.device, TpuConfig):
        extras.append("tpu")

    job_request = JobRequest(
        name="eval_vep",
        entrypoint=Entrypoint.from_callable(
            _run_eval_on_tpu,
            args=[eval_config, env.get("WANDB_API_KEY"), env.get("HF_TOKEN")],
        ),
        resources=config.resources,
        environment=create_environment(env_vars=env, extras=extras),
    )
    job = client.submit(job_request)
    job.wait(raise_on_failure=True)
    logger.info("Evaluation complete.")


eval_step = ExecutorStep(
    name="eval/dna-bolinas-transfer-v0.12-lr2-vep-v0.1",
    description="Evaluate transfer sweep checkpoint on TraitGym Mendelian VEP.",
    fn=run_eval,
    config=EvalTraitGymConfig(
        checkpoint_path=CHECKPOINT,
        tokenizer=TOKENIZER,
        resources=RESOURCES,
        wandb_api_key=os.environ.get("WANDB_API_KEY"),
        hf_token=os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[eval_step])
