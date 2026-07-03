# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenizer bake-off launcher: one grug-moe proxy run for a single tokenizer arm.

A thin variant of :mod:`experiments.grug.moe.launch_cw_scale` that makes the tokenizer the
independent variable. It differs from the scale launcher in exactly the three ways the
bake-off needs (everything else — the SCALE_* shape/mesh/batch/step knobs — is inherited):

1. **Tokenizer is chosen by ``BAKEOFF_ARM``** (default ``marin-128k``), resolved through
   the arm registry in :mod:`experiments.tokenize.bakeoff_tokenizers`. That one choice sets
   both the data tokenization *and* the model ``vocab_size`` (the scale launcher hardcodes
   llama3 and 128256 independently).
2. **A held-out validation set is attached** — the Uncheatable-Eval subsets, tokenized with
   the arm's tokenizer, so every arm is scored on the same raw bytes.
3. **BPB eval is on** (``GrugEvalConfig(compute_bpb=True)``); the scale launcher passes
   ``eval=None``.

Run one arm at one compute point (the isoFLOP ladder is several of these — vary SCALE_STEPS):

    uv run iris --cluster=cw-rno2a job run --cpu 2 --memory 3GB --extra cpu \
      --job-name grug-bakeoff-marin-c0 \
      -e BAKEOFF_ARM marin-128k \
      -e SCALE_GPU_REPLICAS 1 -e SCALE_EXPERT_AXIS 4 -e SCALE_HIDDEN_DIM 1024 \
      -e SCALE_NUM_LAYERS 16 -e SCALE_NUM_EXPERTS 32 -e SCALE_TOP_K 4 \
      -e SCALE_BATCH 128 -e SCALE_SEQ_LEN 1024 -e SCALE_STEPS 2000 \
      -e SCALE_TRACKER wandb -e RUN_ID bakeoff-marin-c0 \
      -- python -m experiments.grug.moe.launch_tokenizer_bakeoff

Use ``SCALE_TRACKER=wandb`` for a durable, queryable ``eval/bpb`` history; the default
``json_logger`` only writes to the run log (scrape with
``experiments.tokenize.collect_metrics``).
"""

import dataclasses
import datetime
import os

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, run_grug_moe_trial
from experiments.grug.moe.launch_cw_scale import (
    _SLIMPAJAMA_SHUFFLE,
    GPUS_PER_NODE,
    OUTPUT_SUBDIR,
    SCALE_OPTIMIZER,
    SCALE_TRAINER_DEFAULTS,
    build_scale_model,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.tokenize.bakeoff_tokenizers import arm_by_name

# SlimPajama-6B tokenization OOMs at the default worker resources (matches launch.py).
_SLIMPAJAMA_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="64g")

# These runs live under their own subtree so the ladder stays grouped and separable from the
# throughput scale runs that share OUTPUT_SUBDIR's sibling.
BAKEOFF_SUBDIR = f"{OUTPUT_SUBDIR}/tokenizer-bakeoff"

# Held-out BPB eval every this many steps; cheap enough to leave on for short proxy runs.
_STEPS_PER_EVAL = 500


def slimpajama_6b_for(tokenizer: str) -> ArtifactStep:
    """SlimPajama-6B tokenized with ``tokenizer`` — the bake-off's shared training corpus.

    Same source and cache name as the scale launcher's llama3 handle; the tokenizer is part
    of the cache identity, so each arm gets its own cache and no arm reads another's tokens.
    """
    return tokenized(
        "slimpajama-6b",
        source="DKYoon/SlimPajama-6B",
        tokenizer=tokenizer,
        resources=_SLIMPAJAMA_TOKENIZE_RESOURCES,
        version="2026.06.28",
    )


def build_bakeoff_checkpoint(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """One tokenizer arm's proxy run as a lazy :class:`LevanterCheckpoint` from BAKEOFF_ARM + SCALE_* env."""
    arm = arm_by_name(os.environ.get("BAKEOFF_ARM", "marin-128k"))
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")

    replicas = env_int("SCALE_GPU_REPLICAS", 1)
    expert_axis = env_int("SCALE_EXPERT_AXIS", 4)
    replica_axis = env_int("SCALE_REPLICA_AXIS", 1)
    batch_size = env_int("SCALE_BATCH", 128)
    steps = env_int("SCALE_STEPS", 2000)
    processes_per_task = env_int("SCALE_PROCESSES_PER_TASK", 1)

    # The model shape comes from SCALE_* exactly as in the scale launcher; only the vocab is
    # the arm's, so the output head and embedding table size track the tokenizer under test.
    model = dataclasses.replace(build_scale_model(), vocab_size=arm.vocab_size)
    if model.num_experts % expert_axis != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by SCALE_EXPERT_AXIS={expert_axis}")

    data_axis = (replicas * GPUS_PER_NODE) // (replica_axis * expert_axis)
    batch_shards = replica_axis * data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"SCALE_BATCH={batch_size} must be divisible by batch shards={batch_shards}")

    resources = ResourceConfig.with_gpu("H100", count=GPUS_PER_NODE, cpu=32, ram="256g", disk="256g", replicas=replicas)

    use_wandb = os.environ.get("SCALE_TRACKER", "json_logger").lower() == "wandb"
    json_logger_name = os.environ.get("SCALE_JSON_LOGGER", "grug_moe_scale.metrics")
    wandb_project = os.environ.get("WANDB_PROJECT", "marin_moe")

    grug_trainer = GrugTrainerConfig(
        expert_axis_size=expert_axis,
        replica_axis_size=replica_axis,
        **SCALE_TRAINER_DEFAULTS,
    )
    mp = os.environ.get("SCALE_MP", "params=float32,compute=bfloat16,output=bfloat16")

    train = slimpajama_6b_for(arm.ref)
    validation = list(uncheatable_datasets(tokenizer=arm.ref).values())
    name = f"grug-bakeoff-{arm.name}-d{model.hidden_dim}-L{model.num_layers}"

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        if use_wandb:
            tracker = WandbConfig(
                project=wandb_project,
                tags=["grug", "moe", "cw", "h100", "tokenizer-bakeoff", arm.name],
                group="tokenizer-flop-bakeoff",
                name=None,
                replicate_path=ctx.output_path,
            )
        else:
            tracker = JsonLoggerConfig(logger_name=json_logger_name)
        return GrugMoeLaunchConfig(
            model=model,
            data=mixture(ctx, {train: 1.0}, validation=validation, shuffle=_SLIMPAJAMA_SHUFFLE),
            output_path=ctx.output_path,
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            steps=steps,
            batch_size=batch_size,
            seed=0,
            mp=mp,
            tracker=tracker,
            optimizer=SCALE_OPTIMIZER,
            grug_trainer=grug_trainer,
            processes_per_task=processes_per_task,
            eval=GrugEvalConfig(
                compute_bpb=True,
                eval_batch_size=batch_size,
                steps_per_eval=env_int("SCALE_STEPS_PER_EVAL", _STEPS_PER_EVAL),
                max_eval_batches=16,
                eval_current=True,
                eval_ema=False,
            ),
            profiler=ProfilerConfig(enabled=False),
            checkpointer=CheckpointerConfig(
                base_path=f"/tmp/grug-bakeoff-ckpt/{run_id}",
                append_run_id_to_base_path=False,
                save_interval=None,
                keep=None,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(f"{BAKEOFF_SUBDIR}/{name}-{run_id}", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(train, *validation),
        runtime_args={"train_resources": resources},
    )


if __name__ == "__main__":
    StepRunner().run([build_bakeoff_checkpoint().lower()])
