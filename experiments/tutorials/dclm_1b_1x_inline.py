# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How To: Replicating DCLM 1B/1x in Marin (inline-protocol style).

Every training decision is built here, in plain Levanter config, rather than
hidden behind a higher-level wrapper: the model, the data mixture, the optimizer,
the token budget, precision, parallelism, evals, and checkpoint cadence are all
visible in this file. The library provides the *features* it composes —
``run_levanter_train_lm`` (run Levanter on a pod), ``lm_mixture_data_config``
(assemble a weighted mixture), the eval-harness conversion — not a config builder.

A golden test (``tests/experiment/test_train_lm_golden.py``) pins this to the
``default_train`` recipe's resolved config so the readable inline code and the
executed config cannot drift.
"""

import os
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig, truncate_wandb_run_name
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import add_validation_sets_to_mixture, lm_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.dclm import DCLM_MIXTURE_WEIGHTS, dclm_components_llama3
from experiments.recipes import core_tasks, marin_validation

SEQ_LEN = 2048
BATCH_SIZE = 256
NUM_TRAIN_TOKENS = 28.8e9  # 1B-1x DCLM competition scale (Chinchilla-optimal for 1.4B)
NUM_TRAIN_STEPS = int(NUM_TRAIN_TOKENS) // (BATCH_SIZE * SEQ_LEN)
RESOURCES = ResourceConfig.with_tpu("v4-128")

# 1.4B-parameter Llama, DCLM 1B-1x competition scale.
llama_1_4b_dclm = LlamaConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=24,
)


def build() -> ExecutorStep:
    """The DCLM 1B/1x training step, with every decision built inline."""
    data = add_validation_sets_to_mixture(
        lm_mixture_data_config(components=dict(dclm_components_llama3), weights=dict(DCLM_MIXTURE_WEIGHTS)),
        dict(marin_validation(llama3_tokenizer)),
    )

    evals = core_tasks(every=10000)
    harness = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(list(evals.tasks)))

    inner_config = TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                name=None,
                tags=["HOWTOS", "DCLM_1B_1X"],
                group=None,
                # Mirror the run's metrics next to its output so they survive the run.
                replicate_path=this_output_path(),
            ),
            # Compute in bf16, keep master params / optimizer state in f32.
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            per_device_parallelism=-1,
            num_train_steps=NUM_TRAIN_STEPS,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                # Rolling resumption checkpoint on a 10-minute wall-clock heartbeat.
                save_interval=timedelta(minutes=10),
                keep=[],
            ),
            model_averaging=None,
            # Single-host data parallelism (model=1 means no tensor sharding); the token
            # axes map onto the data/replica mesh axes Levanter's MoE path expects.
            mesh=MeshConfig(
                axes={"replica": 1, "data": -1, "model": 1},
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            allow_partial_checkpoint=False,
            per_device_eval_parallelism=-1,
            max_eval_batches=None,
            # DCLM's batch size need not divide evenly across the pod.
            allow_nondivisible_batch_size=True,
            quantization=None,
            initialize_from=None,
            use_explicit_mesh_axes=False,
        ),
        initialize_from_checkpoint_path=None,
        initialize_from_hf=False,
        pad_tokenizer_to_match_model=False,
        z_loss_weight=1e-4,
        train_seq_len=SEQ_LEN,
        model=llama_1_4b_dclm,
        optimizer=AdamConfig(learning_rate=3e-3, weight_decay=0.033, warmup=5000, min_lr_ratio=0.1),
        hf_save_steps=None,
        hf_generation_eos_token_ids=None,
        data_seed=None,
        eval_harness_steps=evals.every,
        eval_harness=harness,
        adapter=NoAdaptorConfig(),
    )

    return ExecutorStep(
        name=os.path.join("checkpoints", truncate_wandb_run_name("dclm_1b_1x_how_to")),
        description=f"Train dclm_1b_1x_how_to for {NUM_TRAIN_STEPS} steps * {BATCH_SIZE} batch * {SEQ_LEN} seq.",
        fn=run_levanter_train_lm,
        resources=RESOURCES,
        config=TrainLmOnPodConfig(
            train_config=inner_config,
            resources=RESOURCES,
            output_path=this_output_path(),
            env_vars=None,
        ),
    )


if __name__ == "__main__":
    executor_main(
        steps=[build()],
        description="DCLM 1B/1X baseline, written as an inline protocol.",
    )
