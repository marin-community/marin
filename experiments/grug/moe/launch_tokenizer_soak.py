# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""24h tokenizer soak launcher: one 10B-total / ~500M-active grug-moe run per tokenizer arm.

The lock-down counterpart to :mod:`experiments.grug.moe.launch_tokenizer_bakeoff`. Where the
bake-off is a short isoFLOP proxy (hidden 1024) on SlimPajama alone, each soak run is a single
long run at a representative model size on a representative multi-domain mixture, so the
tokenizer ranking is confirmed at scale on the kind of data grug-moe actually trains on.

Three differences from the bake-off launcher:

1. **Representative mixture** (:data:`SOAK_SOURCES`) — English web + Python code + several
   Wikipedia languages + math, each tokenized with the arm's own tokenizer and combined by
   weight. (The production grug-moe datakit mixture is only available pre-tokenized under one
   tokenizer and in another region, so it cannot be re-tokenized per arm; this raw-text mix
   stands in for it, matching its web-heavy multi-domain composition.)
2. **Target-shaped model** — the SCALE_* env sets a ~10B-total / ~500M-active MoE (hidden 2560,
   8 layers, 128 experts, top-4 — the target run's width/expert/GQA structure, downscaled in
   depth), and this launcher overrides ``sliding_window`` to the target's 2,048 (the scale
   builder otherwise ties it to seq_len).
3. **wandb by default** — a 24h run needs a durable, queryable ``eval/bpb`` history.

Everything else (mesh math, optimizer, BPB eval, n-gram toggle) is inherited from the bake-off /
scale launchers. Launch one arm (see ``experiments/tokenize/EXPERIMENT_LOG.md`` EXP-011 for the
full 8-arm matrix and the driver script):

    uv run iris --cluster=cw-rno2a job run --no-wait --cpu 2 --memory 3GB --extra cpu \
      --job-name soak-superbpe-64k \
      -e BAKEOFF_ARM soak-superbpe-64k \
      -e SCALE_GPU_REPLICAS 8 -e SCALE_EXPERT_AXIS 8 -e SCALE_HIDDEN_DIM 2560 \
      -e SCALE_NUM_LAYERS 8 -e SCALE_NUM_EXPERTS 128 -e SCALE_TOP_K 4 \
      -e SCALE_BATCH 512 -e SCALE_SEQ_LEN 4096 -e SCALE_STEPS 40000 \
      -e SCALE_TRACKER wandb -e SCALE_RAM 512g -e RUN_ID soak-64k \
      -- python -m experiments.grug.moe.launch_tokenizer_soak
"""

import dataclasses
import datetime
import math
import os

from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.grug.ngram_embed import NgramEmbedConfig
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, run_grug_moe_trial
from experiments.grug.moe.launch_cw_scale import (
    GPUS_PER_NODE,
    OUTPUT_SUBDIR,
    SCALE_OPTIMIZER,
    SCALE_TRAINER_DEFAULTS,
    build_scale_model,
)
from experiments.grug.moe.launch_tokenizer_bakeoff import bakeoff_validation
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.tokenize.bakeoff_tokenizers import arm_by_name

# Soak runs live under their own subtree so they stay grouped and separable from the proxy ladder.
SOAK_SUBDIR = f"{OUTPUT_SUBDIR}/tokenizer-soak"

# The target run's sliding window (sw2k); the scale builder otherwise sets sliding_window=seq_len.
SOAK_SLIDING_WINDOW = 2048

# Held-out BPB eval every this many steps -> ~24-40 curve points over a 24h run.
_STEPS_PER_EVAL = 1000

# Tokenization workers need headroom (SlimPajama shards OOM at the default) but not a GPU.
_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="128g")


@dataclasses.dataclass(frozen=True)
class SoakSource:
    """One raw HF component of the soak mixture: what to tokenize, its config, and its weight."""

    key: str  # short stable id -> cache/component name
    source: str  # HF dataset id
    weight: float  # mixture weight (all sources' weights are normalized by the sampler)
    text_key: str = "text"
    hf_name: str | None = None  # HF dataset config/subset (finemath-3plus, 20231101.de, ...)
    sample_count: int | None = None  # cap docs so per-arm tokenization stays fast


# Representative multi-domain mixture (weights sum to 1.0): web-heavy, with Python code, three
# Wikipedia languages spanning Latin/Cyrillic/CJK scripts, and math. The larger sources are
# doc-capped so all 8 arms' tokenizations finish promptly; the mix repeats over a 24h run with an
# identical schedule across arms, so relative BPB ordering is unaffected by the repetition.
SOAK_SOURCES: tuple[SoakSource, ...] = (
    SoakSource("web", "DKYoon/SlimPajama-6B", 0.50, text_key="text"),
    SoakSource("code", "codeparrot/codeparrot-clean-valid", 0.20, text_key="content"),
    SoakSource("ml-de", "wikimedia/wikipedia", 0.0667, hf_name="20231101.de", sample_count=1_500_000),
    SoakSource("ml-ru", "wikimedia/wikipedia", 0.0667, hf_name="20231101.ru", sample_count=1_500_000),
    SoakSource("ml-zh", "wikimedia/wikipedia", 0.0666, hf_name="20231101.zh", sample_count=1_500_000),
    SoakSource(
        "math", "HuggingFaceTB/finemath", 0.10, text_key="text", hf_name="finemath-3plus", sample_count=3_000_000
    ),
)


def soak_train_datasets(arm_name: str, tokenizer: str) -> dict[ArtifactStep, float]:
    """Each :data:`SOAK_SOURCES` component tokenized with ``tokenizer``, mapped to its weight.

    Cache names are suffixed with the arm so each arm builds its own tokenization rather than
    adopting another tokenizer's cache (the artifact store adopts by name@version).
    """
    return {
        tokenized(
            f"soak-train/{src.key}-{arm_name}",
            source=src.source,
            tokenizer=tokenizer,
            hf_name=src.hf_name,
            text_key=src.text_key,
            sample_count=src.sample_count,
            resources=_TOKENIZE_RESOURCES,
            version="2026.07.04",
        ): src.weight
        for src in SOAK_SOURCES
    }


def build_soak_checkpoint(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """One tokenizer arm's 24h soak run as a lazy :class:`LevanterCheckpoint` from BAKEOFF_ARM + SCALE_* env."""
    arm = arm_by_name(os.environ.get("BAKEOFF_ARM", "soak-superbpe-64k"))
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")

    replicas = env_int("SCALE_GPU_REPLICAS", 8)
    expert_axis = env_int("SCALE_EXPERT_AXIS", 8)
    replica_axis = env_int("SCALE_REPLICA_AXIS", 1)
    batch_size = env_int("SCALE_BATCH", 512)
    steps = env_int("SCALE_STEPS", 40_000)
    processes_per_task = env_int("SCALE_PROCESSES_PER_TASK", 1)

    # The arm sets the model vocab (and data tokenization); the SCALE_* env sets the 10B/500M
    # shape; we force the target's sliding window (the scale builder ties it to seq_len).
    model = dataclasses.replace(build_scale_model(), vocab_size=arm.vocab_size, sliding_window=SOAK_SLIDING_WINDOW)

    # BAKEOFF_NGRAM toggles the hashed multi-gram input embedding (Over-Encoding / LongCat); the
    # 64k+ngram arm carries the proxy study's winning config (4M buckets, rank-128, orders 2-4,
    # ratio 0.25, mean-combine). It adds input-side capacity at zero extra serving FLOPs.
    if os.environ.get("BAKEOFF_NGRAM"):
        orders = tuple(int(o) for o in os.environ.get("BAKEOFF_NGRAM_ORDERS", "2,3,4").split(","))
        rank = env_int("BAKEOFF_NGRAM_RANK", 128)
        width = rank if rank > 0 else model.hidden_dim
        ratio = float(os.environ.get("BAKEOFF_NGRAM_RATIO", "0.25"))
        model = dataclasses.replace(
            model,
            ngram=NgramEmbedConfig(
                orders=orders,
                num_hashes=env_int("BAKEOFF_NGRAM_HASHES", 2),
                hash_buckets=env_int("BAKEOFF_NGRAM_BUCKETS", 4_000_037),
                rank=rank if rank > 0 else None,
                combine=os.environ.get("BAKEOFF_NGRAM_COMBINE", "mean"),
                init_std_scale=ratio * model.initializer_std * math.sqrt(width),
            ),
        )

    if model.num_experts % expert_axis != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by SCALE_EXPERT_AXIS={expert_axis}")
    data_axis = (replicas * GPUS_PER_NODE) // (replica_axis * expert_axis)
    batch_shards = replica_axis * data_axis * expert_axis
    if batch_size % batch_shards != 0:
        raise ValueError(f"SCALE_BATCH={batch_size} must be divisible by batch shards={batch_shards}")

    ram = os.environ.get("SCALE_RAM", "512g")
    resources = ResourceConfig.with_gpu("H100", count=GPUS_PER_NODE, cpu=32, ram=ram, disk="512g", replicas=replicas)

    use_wandb = os.environ.get("SCALE_TRACKER", "wandb").lower() == "wandb"
    wandb_project = os.environ.get("WANDB_PROJECT", "marin_moe")
    json_logger_name = os.environ.get("SCALE_JSON_LOGGER", "grug_moe_soak.metrics")

    grug_trainer = GrugTrainerConfig(
        expert_axis_size=expert_axis,
        replica_axis_size=replica_axis,
        **SCALE_TRAINER_DEFAULTS,
    )
    mp = os.environ.get("SCALE_MP", "params=float32,compute=bfloat16,output=bfloat16")

    train = soak_train_datasets(arm.name, arm.ref)
    validation = bakeoff_validation(arm.name, arm.ref)
    variant = "-ngram" if os.environ.get("BAKEOFF_NGRAM") else ""
    name = f"grug-soak-{arm.name}{variant}-d{model.hidden_dim}-L{model.num_layers}"

    def build_config(ctx: StepContext) -> GrugMoeLaunchConfig:
        if use_wandb:
            tracker = WandbConfig(
                project=wandb_project,
                tags=["grug", "moe", "cw", "h100", "tokenizer-soak", arm.name],
                group="tokenizer-soak",
                name=None,
                replicate_path=ctx.output_path,
            )
        else:
            tracker = JsonLoggerConfig(logger_name=json_logger_name)
        return GrugMoeLaunchConfig(
            model=model,
            data=mixture(ctx, train, validation=validation),
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
                base_path=f"/tmp/grug-soak-ckpt/{run_id}",
                append_run_id_to_base_path=False,
                save_interval=None,
                keep=None,
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(f"{SOAK_SUBDIR}/{name}-{run_id}", version),
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_grug_moe_trial,
        build_config=build_config,
        deps=(*train, *validation),
        runtime_args={"train_resources": resources},
    )


if __name__ == "__main__":
    StepRunner().run([build_soak_checkpoint().lower()])
