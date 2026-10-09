# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transformer vs Parallax head-to-head (arXiv 2605.29157, Tables 2 and 5) at Qwen3-0.6B on the v16384 data.

The paper's 0.6B Muon comparison with its recipe: Qwen3 backbone (28 layers, d 1024, SwiGLU 3072, 16/8 heads of
128, RoPE theta 1e6, tied embeddings, QK RMSNorm), Parallax in every layer with RMSNorm and RoPE on the probe; Muon
lr 5e-3 (wd 0.1, momentum 0.95, 5 quintic Newton-Schulz steps) with the non-matrix parameters on Adam (beta 0.8 /
0.95, eps 1e-7) at 0.3x (embeddings) and 0.015x (norms) of the Muon lr; global grad clip 1.0; WSD with no warmup
and a linear decay to 0 over the last 20%; 960 x 4096-token batches (3.93M tokens). The paper trains 20k steps
(78.6B tokens); ``--steps`` sets the budget. Differences: our tokenizer (hero-bpe-v16384) and data, and bf16
linears instead of fp8.

Plan: ``.agents/projects/parallax-replication.md``. Launch (from the repo root, ``WANDB_API_KEY`` set):

    uv run python -m experiments.parallax_replication --arm parallax --steps 5000 --submit --target-cluster cw-rno2a
"""

import os
import shlex
import sys
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.adaptor import NoAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.eval_harness import LmEvalHarnessConfig
from levanter.layers.rotary import DefaultRotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.muon import MuonConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, convert_to_levanter_task_config
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.experiment.train import MARIN_PRECISION, _marin_mesh
from marin.training.training import (
    LevanterCheckpoint,
    TrainLmOnPodConfig,
    resolve_training_env,
    run_levanter_train_lm,
)

from experiments.datasets.paloma import _PALOMA_DETOK_RAW, paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.fast_track.launch import (
    PALOMA_DETOK_VERSION,
    V16384_CACHE_DIR,
    V16384_TOKENIZER,
    _flat_cache_data_config,
)

VERSION = "2026.10.09"
SEQ_LEN = 4096
BATCH_SIZE = 960  # 3.93M tokens
PER_DEVICE_BATCH = 8  # gradient-accumulation microbatch per H100
ARMS = ("transformer", "parallax")
WANDB_PROJECT = "marin_moe"
# Levanter's in-training harness needs its pinned lm-eval fork (levanter's ``lm_eval`` extra), which the default
# GPU training environment does not install.
LM_EVAL_PACKAGE = (
    "lm-eval@git+https://github.com/stanford-crfm/lm-evaluation-harness.git@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab"
)

# The paper's evals (all 0-shot, lm-eval-harness). WikiText perplexity comes from Paloma's wikitext_103 set.
EVAL_TASKS = (
    EvalTaskConfig("lambada_openai", 0, task_alias="lambada_openai_0shot"),
    EvalTaskConfig("boolq", 0, task_alias="boolq_0shot"),
    EvalTaskConfig("hellaswag", 0, task_alias="hellaswag_0shot"),
    EvalTaskConfig("piqa", 0, task_alias="piqa_0shot"),
    EvalTaskConfig("arc_easy", 0, task_alias="arc_easy_0shot"),
    EvalTaskConfig("arc_challenge", 0, task_alias="arc_challenge_0shot"),
    EvalTaskConfig("winogrande", 0, task_alias="winogrande_0shot"),
    EvalTaskConfig("openbookqa", 0, task_alias="openbookqa_0shot"),
    EvalTaskConfig("sciq", 0, task_alias="sciq_0shot"),
)


def qwen3_0p6b(parallax: bool) -> Qwen3Config:
    """The paper's Table 5 backbone."""
    return Qwen3Config(
        max_seq_len=SEQ_LEN,
        hidden_dim=1024,
        intermediate_dim=3072,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        tie_word_embeddings=True,
        rope=DefaultRotaryEmbeddingsConfig(theta=1e6),
        parallax=parallax,
        parallax_rope=True,
    )


def paper_muon() -> MuonConfig:
    """The paper's Table 2b Muon + WSD recipe."""
    return MuonConfig(
        learning_rate=5e-3,
        weight_decay=0.1,
        momentum=0.95,
        backend_steps=5,
        coefficient_type="quintic",
        beta1=0.8,
        beta2=0.95,
        epsilon=1e-7,
        embedding_lr_mult=0.3,
        norm_lr_mult=0.015,
        max_grad_norm=1.0,
        max_grad_norm_scope="global",
        warmup=0.0,
        decay=0.2,
        lr_schedule="linear",
        min_lr_ratio=0.0,
    )


def _train_job(pod_config: TrainLmOnPodConfig) -> None:
    """``marin.experiment.train._train_job`` for GPU, plus the lm-eval package the harness imports."""
    env_vars = resolve_training_env(pod_config.env_vars, pod_config.resources)
    remote(run_levanter_train_lm, resources=pod_config.resources, env_vars=env_vars, pip_packages=[LM_EVAL_PACKAGE])(
        pod_config
    )


def build(arm: str, steps: int) -> ArtifactStep[LevanterCheckpoint]:
    run_id = f"plx-{arm}-{steps}"
    validation = [
        *uncheatable_datasets(tokenizer=V16384_TOKENIZER, tag=V16384_TOKENIZER).values(),
        *paloma_datasets(
            tokenizer=V16384_TOKENIZER, tag=V16384_TOKENIZER, raw_prefix=_PALOMA_DETOK_RAW, version=PALOMA_DETOK_VERSION
        ).values(),
    ]

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        inner = TrainLmConfig(
            data=_flat_cache_data_config(
                ctx=ctx, validation=validation, tokenizer=V16384_TOKENIZER, train_cache_dir=V16384_CACHE_DIR
            ),
            trainer=TrainerConfig(
                id=run_id,
                tracker=WandbConfig(
                    project=WANDB_PROJECT,
                    name=run_id,
                    tags=["parallax-replication", arm],
                    replicate_path=ctx.output_path,
                ),
                mp=jmp.get_policy(MARIN_PRECISION),
                train_batch_size=BATCH_SIZE,
                per_device_parallelism=PER_DEVICE_BATCH,
                per_device_eval_parallelism=PER_DEVICE_BATCH,
                num_train_steps=steps,
                steps_per_eval=1000,
                checkpointer=CheckpointerConfig(save_interval=timedelta(minutes=20), keep=[]),
                mesh=_marin_mesh(1),
            ),
            model=qwen3_0p6b(parallax=arm == "parallax"),
            optimizer=paper_muon(),
            z_loss_weight=None,  # the paper trains without a z-loss
            train_seq_len=SEQ_LEN,
            eval_harness=LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(list(EVAL_TASKS))),
            eval_harness_steps=steps,
            adapter=NoAdaptorConfig(),
        )
        return TrainLmOnPodConfig(
            train_config=inner,
            resources=ctx.runtime_arg("train_resources"),
            output_path=ctx.output_path,
            env_vars=None,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"parallax_replication/{run_id}", VERSION),
        version=VERSION,
        artifact_type=LevanterCheckpoint,
        run=_train_job,
        build_config=build_config,
        deps=tuple(validation),
        runtime_args={
            "train_resources": ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="600g", disk="900g", replicas=1)
        },
    )


@click.command()
@click.option("--arm", type=click.Choice(ARMS), required=True)
@click.option("--steps", type=int, required=True, help="Training steps (the paper: 20000).")
@click.option("--submit", is_flag=True, help="Re-exec this script as an Iris coordinator job.")
@click.option("--target-cluster", default="cw-rno2a", show_default=True)
@click.option("--priority", default="interactive", show_default=True)
def main(arm: str, steps: int, submit: bool, target_cluster: str, priority: str) -> None:
    if not submit:
        StepRunner().run([build(arm, steps).lower()])
        return
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        raise click.ClickException("WANDB_API_KEY must be set to submit")
    args = [a for a in sys.argv[1:] if a not in ("--submit",)]
    cmd = [
        "uv", "run", "iris", "--cluster", "marin", "job", "run", "--no-wait", "--enable-extra-resources",
        "--target-cluster", target_cluster, "--priority", priority,
        "--job-name", f"plx-{arm}-{steps}-coord",
        "-e", "WANDB_API_KEY", wandb_key, "-e", "WANDB_PROJECT", WANDB_PROJECT,
        "--", "python", "-m", "experiments.parallax_replication", *args,
    ]  # fmt: skip
    click.echo("submitting: " + " ".join(shlex.quote("$WANDB_API_KEY" if c == wandb_key else c) for c in cmd), err=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
