# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Template: grug-native trial run.

Runs a small grug-native pretraining trial (~130M, 2000 steps) via Marin's
executor stack. This is intended as a lightweight end-to-end validation run.
"""

import os
from dataclasses import dataclass
from datetime import timedelta

import jmp
from experiments.defaults import default_validation_sets
from experiments.tootsie.exp1295_32b import nemotron_mix
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.data.text import LmDataConfig
from levanter.grug.model import GrugModelConfig
from levanter.grug_native.config import GrugEvalConfig, GrugNativeRunConfig, GrugTrainerConfig
from levanter.grug_native.runtime import as_grug_runtime
from levanter.grug_native.train import run_grug_native
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture


@dataclass(frozen=True)
class GrugNativeTrialConfig:
    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    steps_per_eval: int
    max_eval_batches: int | None
    seed: int
    z_loss_weight: float
    ema_beta: float | None
    enable_validation: bool = True


GRUG_130M_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=1792,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=4096,
    head_dim=None,
)

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


def _resolve_run_id(default_run_id: str) -> str:
    """Resolve run id with env overrides similar to ferry flows."""
    run_id = os.environ.get("GRUG_NATIVE_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def run_grug_native_trial(config: GrugNativeTrialConfig) -> None:
    eval_interval = max(1, config.steps_per_eval) if config.enable_validation else None

    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        steps_per_eval=max(1, config.steps_per_eval),
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            name=config.run_id,
            tags=["grug-native", "template"],
            group="grug-native-trial",
            replicate_path=config.output_path,
        ),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=True,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )

    run_config = GrugNativeRunConfig(
        model=config.model,
        data=config.data,
        optimizer=AdamConfig(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_schedule="cosine",
            decay=0.2,
            min_lr_ratio=0.1,
            warmup=1000,
        ),
        trainer=GrugTrainerConfig(
            trainer=as_grug_runtime(trainer),
            log_every=1,
            z_loss_weight=config.z_loss_weight,
            ema_beta=config.ema_beta,
        ),
        eval=GrugEvalConfig(
            steps_per_eval=eval_interval,
            max_eval_batches=config.max_eval_batches,
            eval_current=config.enable_validation,
            eval_ema=config.enable_validation and bool(config.ema_beta is not None),
        ),
    )
    run_grug_native(run_config)


RESOLVED_RUN_ID = _resolve_run_id("grug-native-trial")


grug_native_trial = ExecutorStep(
    name="templates/grug-native-trial",
    fn=run_grug_native_trial,
    config=GrugNativeTrialConfig(
        model=versioned(GRUG_130M_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=versioned(RESOLVED_RUN_ID),
        steps=versioned(2_000),
        batch_size=versioned(512),
        learning_rate=versioned(3e-3),
        weight_decay=versioned(0.1),
        steps_per_eval=versioned(200),
        max_eval_batches=versioned(8),
        seed=versioned(0),
        z_loss_weight=versioned(1e-4),
        ema_beta=versioned(None),
        enable_validation=versioned(True),
    ),
    resources=ResourceConfig.with_tpu("v5p-8"),
)


if __name__ == "__main__":
    executor_main(
        steps=[grug_native_trial],
        description="Template grug-native 130M trial run (~2000 steps) on Nemotron mix.",
    )
