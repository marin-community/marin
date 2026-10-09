# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Transformer vs Parallax head-to-head (arXiv 2605.29157, Tables 2 and 5) at Qwen3-0.6B on the v16384 data.

The paper's 0.6B Muon comparison with its recipe: Qwen3 backbone (28 layers, d 1024, SwiGLU 3072, 16/8 heads of
128, RoPE theta 1e6, tied embeddings, QK RMSNorm), Parallax in every layer with RMSNorm and RoPE on the probe; Muon
lr 5e-3 (wd 0.1, Nesterov momentum 0.95, 5 quintic Newton-Schulz steps, sqrt(max(1, out/in)) scaling) with the
non-matrix parameters on Adam (beta 0.8 / 0.95, eps 1e-7) at 0.3x (embeddings) and 0.015x (norms) of the Muon lr;
global grad clip 1.0; WSD with no warmup and a linear decay to 0 over the last 20%; 960 x 4096-token batches (3.93M
tokens); no z-loss. The paper trains 20k steps (78.6B tokens); ``--steps`` sets the budget. Differences: our
tokenizer (hero-bpe-v16384, so a 16k instead of 152k vocab) and data, and bf16 linears instead of fp8. The
downstream evals (LAMBADA and the 0-shot suite) run on the final checkpoint after training; WikiText perplexity
comes from the Paloma wikitext_103 validation loss logged during training.

Launch (from the repo root, ``WANDB_API_KEY`` set):

    uv run python -m experiments.grug.parallax.launch --arm parallax --steps 5000 --submit
"""

import os
import shlex
import sys
from dataclasses import dataclass
from datetime import timedelta

import click
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.grugmuon import GrugMuonConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint, resolve_checkpointer_output_path

from experiments.datasets.paloma import _PALOMA_DETOK_RAW, paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.fast_track.launch import (
    PALOMA_DETOK_VERSION,
    V16384_CACHE_DIR,
    V16384_TOKENIZER,
    V16384_VOCAB,
    _flat_cache_data_config,
)
from experiments.grug.parallax.model import GrugModelConfig
from experiments.grug.parallax.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug

VERSION = "2026.10.09"
ARMS = ("transformer", "parallax")
SEQ_LEN = 4096
BATCH_SIZE = 960  # 3.93M tokens
GRAD_ACCUM_MICROBATCHES = 15  # 8 sequences per H100 per microbatch
PAPER_LR = 5e-3
PROFILE_START = 20
PROFILE_STEPS = 5
WANDB_PROJECT = "marin_moe"
MP_POLICY = "params=float32,compute=bfloat16,output=bfloat16"
TRAIN_RESOURCES = ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="600g", disk="900g", replicas=1)


def qwen3_0p6b(arm: str) -> GrugModelConfig:
    """The paper's Table 5 backbone on our vocab."""
    return GrugModelConfig(
        vocab_size=V16384_VOCAB,
        hidden_dim=1024,
        intermediate_dim=3072,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=SEQ_LEN,
        parallax=arm == "parallax",
    )


def paper_muon(learning_rate: float = PAPER_LR) -> GrugMuonConfig:
    """The paper's Table 2b Muon + WSD recipe (the embedding/norm Adam lrs scale with ``learning_rate``)."""
    return GrugMuonConfig(
        learning_rate=learning_rate,
        weight_decay=0.1,
        momentum=0.95,
        nesterov=True,
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


@dataclass(frozen=True)
class ParallaxLaunchConfig:
    """Fingerprinted run config; ``run_grug_config`` adds the sharding-bearing trainer pieces at run time."""

    model: GrugModelConfig
    optimizer: GrugMuonConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    grad_accum_microbatches: int
    profile: bool = False  # capture an xprof trace of steps PROFILE_START..+PROFILE_STEPS


def run_grug_config(launch: ParallaxLaunchConfig) -> GrugRunConfig:
    trainer = TrainerConfig(
        id=launch.run_id,
        seed=0,
        train_batch_size=launch.batch_size,
        num_train_steps=launch.steps,
        profiler=ProfilerConfig(enabled=launch.profile, start_step=PROFILE_START, num_steps=PROFILE_STEPS),
        mp=jmp.get_policy(MP_POLICY),
        tracker=WandbConfig(
            project=WANDB_PROJECT,
            name=launch.run_id,
            tags=["parallax-replication", "grug", "parallax" if launch.model.parallax else "transformer"],
            replicate_path=launch.output_path,
        ),
        use_explicit_mesh_axes=True,
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=resolve_checkpointer_output_path(
            CheckpointerConfig(save_interval=timedelta(minutes=20), keep=None), launch.output_path
        ),
    )
    return GrugRunConfig(
        model=launch.model,
        data=launch.data,
        resources=launch.resources,
        output_path=launch.output_path,
        optimizer=launch.optimizer,
        trainer=GrugTrainerConfig(
            trainer=trainer,
            z_loss_weight=0.0,  # the paper trains without a z-loss
            grad_accum_microbatches=launch.grad_accum_microbatches,
        ),
        eval=GrugEvalConfig(eval_batch_size=128, steps_per_eval=1000, max_eval_batches=8, eval_ema=False),
    )


def _run(launch: ParallaxLaunchConfig) -> None:
    run_grug(run_grug_config(launch))


def run_name(arm: str, steps: int, batch_size: int, learning_rate: float, microbatches: int) -> str:
    """``plx-grug-<arm>-<steps>``, plus ``-b<batch>-lr<lr>-mb<n>`` when the run departs from the paper's batch."""
    name = f"plx-grug-{arm}-{steps}"
    if (batch_size, learning_rate, microbatches) != (BATCH_SIZE, PAPER_LR, GRAD_ACCUM_MICROBATCHES):
        name += f"-b{batch_size}-lr{learning_rate:g}-mb{microbatches}"
    return name


def build(
    arm: str,
    steps: int,
    *,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = PAPER_LR,
    microbatches: int = GRAD_ACCUM_MICROBATCHES,
    profile: bool = False,
) -> ArtifactStep[LevanterCheckpoint]:
    run_id = run_name(arm, steps, batch_size, learning_rate, microbatches) + ("-prof" if profile else "")
    validation = [
        *uncheatable_datasets(tokenizer=V16384_TOKENIZER, tag=V16384_TOKENIZER).values(),
        *paloma_datasets(
            tokenizer=V16384_TOKENIZER, tag=V16384_TOKENIZER, raw_prefix=_PALOMA_DETOK_RAW, version=PALOMA_DETOK_VERSION
        ).values(),
    ]

    def build_config(ctx: StepContext) -> ParallaxLaunchConfig:
        return ParallaxLaunchConfig(
            model=qwen3_0p6b(arm),
            optimizer=paper_muon(learning_rate),
            data=_flat_cache_data_config(
                ctx=ctx, validation=validation, tokenizer=V16384_TOKENIZER, train_cache_dir=V16384_CACHE_DIR
            ),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            grad_accum_microbatches=microbatches,
            profile=profile,
        )

    return ArtifactStep(
        name=user_namespaced_name(f"parallax_replication/{run_id}", VERSION),
        version=VERSION,
        artifact_type=LevanterCheckpoint,
        run=_run,
        build_config=build_config,
        deps=tuple(validation),
        runtime_args={"train_resources": TRAIN_RESOURCES},
    )


@click.command()
@click.option("--arm", type=click.Choice(ARMS), required=True)
@click.option("--steps", type=int, required=True, help="Training steps (the paper: 20000).")
@click.option("--submit", is_flag=True, help="Re-exec this launcher as an Iris coordinator job.")
@click.option("--target-cluster", default="cw-rno2a", show_default=True)
@click.option("--priority", default="interactive", show_default=True)
@click.option("--batch-size", type=int, default=BATCH_SIZE, show_default=True, help="Sequences per step.")
@click.option("--lr", "learning_rate", type=float, default=PAPER_LR, show_default=True, help="Peak Muon lr.")
@click.option("--microbatches", type=int, default=GRAD_ACCUM_MICROBATCHES, show_default=True)
@click.option("--profile", is_flag=True, help="Capture an xprof trace (run id gets a -prof suffix).")
def main(
    arm: str,
    steps: int,
    submit: bool,
    target_cluster: str,
    priority: str,
    batch_size: int,
    learning_rate: float,
    microbatches: int,
    profile: bool,
) -> None:
    if not submit:
        StepRunner().run(
            [
                build(
                    arm,
                    steps,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    microbatches=microbatches,
                    profile=profile,
                ).lower()
            ]
        )
        return
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        raise click.ClickException("WANDB_API_KEY must be set to submit")
    args = [a for a in sys.argv[1:] if a != "--submit"]
    job_run_id = run_name(arm, steps, batch_size, learning_rate, microbatches) + ("-prof" if profile else "")
    cmd = [
        "uv", "run", "iris", "--cluster", "marin", "job", "run", "--no-wait", "--enable-extra-resources",
        "--target-cluster", target_cluster, "--priority", priority,
        "--job-name", f"{job_run_id}-coord",
        "-e", "WANDB_API_KEY", wandb_key, "-e", "WANDB_PROJECT", WANDB_PROJECT,
        "--", "python", "-m", "experiments.grug.parallax.launch", *args,
    ]  # fmt: skip
    click.echo("submitting: " + " ".join(shlex.quote("$WANDB_API_KEY" if c == wandb_key else c) for c in cmd), err=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
