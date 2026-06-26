# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How To: Replicating DCLM 1B/1x in Marin (lazy-artifact style).

Every training decision is built here, in plain Levanter config, rather than hidden
behind a higher-level wrapper: the model, the data mixture, the optimizer, the token
budget, precision, parallelism, evals, and checkpoint cadence are all visible in this
file. The run is a lazy :class:`~marin.execution.lazy.Checkpoint` — no executor, no
import-time step graph, no content-addressing — that lowers and runs through the
``StepRunner``. The library provides the *features* it composes
(``run_levanter_train_lm``, ``mixture``, the eval-harness conversion), not a config
builder.

A golden test (``tests/experiment/test_train_lm_golden.py``) pins this to the
``default_train`` recipe's resolved decisions so the readable inline code and the
executed config cannot drift.
"""

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
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin.execution.lazy import Checkpoint, Recipe, RunContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.evals.uncheatable_lazy import uncheatable_validation
from experiments.llama import llama3_tokenizer
from experiments.paloma_lazy import paloma_validation
from experiments.pretraining_datasets.dclm_lazy import DCLM_MIXTURE_WEIGHTS, dclm_datasets
from experiments.recipes import core_tasks

SEQ_LEN = 2048
BATCH_SIZE = 256
NUM_TRAIN_TOKENS = 28.8e9  # 1B-1x DCLM competition scale (Chinchilla-optimal for 1.4B)
NUM_TRAIN_STEPS = int(NUM_TRAIN_TOKENS) // (BATCH_SIZE * SEQ_LEN)

# The TPU each training job is dispatched onto. A run-arg, not part of the
# checkpoint's identity: re-running on different hardware is the same artifact.
TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-128")

# 1.4B-parameter Llama, DCLM 1B-1x competition scale.
llama_1_4b_dclm = LlamaConfig(
    max_seq_len=SEQ_LEN,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=24,
)


def _train(pod_config: TrainLmOnPodConfig) -> None:
    """Dispatch the inline TrainLmConfig as its own Fray training job.

    The launcher step runs inline; ``run_levanter_train_lm`` runs Levanter on the TPU,
    baking the checkpoint paths under the artifact's output and imputing the run id.
    """
    remote(run_levanter_train_lm, resources=pod_config.resources)(pod_config)


def build(*, version: str = "v1") -> Checkpoint:
    """The DCLM 1B/1x training run as a lazy checkpoint, every decision built inline."""
    train = dclm_datasets(tokenizer=llama3_tokenizer)
    validation = [*paloma_validation(tokenizer=llama3_tokenizer), *uncheatable_validation(tokenizer=llama3_tokenizer)]
    weighted = {train[name]: DCLM_MIXTURE_WEIGHTS[name] for name in train}

    evals = core_tasks(every=10000)
    harness = LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(list(evals.tasks)))

    def build_config(ctx: RunContext) -> TrainLmOnPodConfig:
        inner_config = TrainLmConfig(
            data=mixture(ctx, weighted, validation=validation),
            trainer=TrainerConfig(
                id="dclm_1b_1x_how_to",
                tracker=WandbConfig(
                    project="marin",
                    name=None,
                    tags=["HOWTOS", "DCLM_1B_1X"],
                    group=None,
                    # Mirror the run's metrics next to its output so they survive the run.
                    replicate_path=ctx.out,
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
                # Single-host data parallelism (model=1 means no tensor sharding); the
                # token axes map onto the data/replica mesh axes the marin path expects.
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
        return TrainLmOnPodConfig(
            train_config=inner_config,
            resources=ctx.run_arg("train_resources"),
            output_path=ctx.out,
            env_vars=None,
        )

    return Checkpoint(
        name="checkpoints/dclm_1b_1x_how_to",
        version=version,
        recipe=Recipe(
            fn=_train,
            build_config=build_config,
            deps=(*train.values(), *validation),
            run_args={"train_resources": TRAIN_RESOURCES},
        ),
    )


if __name__ == "__main__":
    # Lower the checkpoint to a StepSpec graph and run it: the data tokenizes (cached),
    # then one TPU training job runs. No executor, no import-time graph.
    StepRunner().run([lower(build())])
