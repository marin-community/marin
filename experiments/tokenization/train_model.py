# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""24h tokenizer soak launcher: one 10B-total / ~500M-active grug-moe run per tokenizer arm.

The lock-down counterpart to :mod:`experiments.tokenization.proxy_ladder`. Where the
bake-off is a short isoFLOP proxy (hidden 1024) on SlimPajama alone, each soak run is a single
long run at a representative model size on a representative multi-domain mixture, so the
tokenizer ranking is confirmed at scale on the kind of data grug-moe actually trains on.

Four differences from the bake-off launcher:

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
4. **Extended held-out eval** (:func:`soak_validation`) — the bake-off's English/code subsets
   plus held-out multilingual (German/Russian/Chinese) and math validation
   (:mod:`experiments.datasets.multilingual_math_eval`), so ``macro_bpb`` scores every domain
   :data:`SOAK_SOURCES` trains on, not just the 70% that the bake-off ladder's English/code eval
   covers.

Everything else (mesh math, optimizer, BPB eval, n-gram toggle) is inherited from the bake-off /
scale launchers. Launch one arm (see ``experiments/tokenization/EXPERIMENT_LOG.md`` EXP-011 for the
full 8-arm matrix and the driver script):

    uv run iris --cluster=cw-rno2a job run --no-wait --cpu 2 --memory 3GB --extra cpu \
      --job-name soak-superbpe-64k \
      -e BAKEOFF_ARM soak-superbpe-64k \
      -e SCALE_GPU_REPLICAS 8 -e SCALE_EXPERT_AXIS 8 -e SCALE_HIDDEN_DIM 2560 \
      -e SCALE_NUM_LAYERS 8 -e SCALE_NUM_EXPERTS 128 -e SCALE_TOP_K 4 \
      -e SCALE_BATCH 512 -e SCALE_SEQ_LEN 4096 -e SCALE_STEPS 40000 \
      -e SCALE_TRACKER wandb -e SCALE_RAM 512g -e RUN_ID soak-64k \
      -- python -m experiments.tokenization.train_model
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
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import hf_download, mixture, tokenized
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import LevanterCheckpoint

from experiments.datasets.multilingual_math_eval import multilingual_math_validation
from experiments.grug.moe.launch import GrugMoeLaunchConfig, env_int, run_grug_moe_trial
from experiments.grug.moe.launch_cw_scale import (
    GPUS_PER_NODE,
    OUTPUT_SUBDIR,
    SCALE_OPTIMIZER,
    SCALE_TRAINER_DEFAULTS,
    build_scale_model,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.tokenization.arms import arm_by_name
from experiments.tokenization.proxy_ladder import bakeoff_validation

# Soak runs live under their own subtree so they stay grouped and separable from the proxy ladder.
SOAK_SUBDIR = f"{OUTPUT_SUBDIR}/tokenizer-soak"

# The target run's sliding window (sw2k); the scale builder otherwise sets sliding_window=seq_len.
SOAK_SLIDING_WINDOW = 2048

# Held-out BPB eval every this many steps -> ~24-40 curve points over a 24h run.
_STEPS_PER_EVAL = 1000

# Tokenization workers need headroom (SlimPajama shards OOM at the default) but not a GPU.
_TOKENIZE_RESOURCES = ResourceConfig(ram="64g", disk="128g")

# Cache-once version for the shared HF downloads (bump to re-fetch from the Hub).
_HF_DL_VERSION = "2026.07.04"

# Each HF source is mirrored to S3 ONCE as a shared, revision-pinned raw-download artifact; every
# arm then tokenizes from that S3 copy (``raw=``/``glob=`` in soak_train_datasets) instead of each
# arm re-streaming the dataset from the Hub. Streaming per arm rate-limits the Hub (HTTP 429) under
# 8 concurrent arms; a single shared download does not. ``urls_glob`` fetches only the parquet/json
# we actually tokenize (e.g. just three Wikipedia languages of the ~300 in the repo).
_SLIMPAJAMA_RAW = hf_download(
    "raw/hf/slimpajama-6b",
    hf_id="DKYoon/SlimPajama-6B",
    revision="b5f90f419b7489cdba26fdbc8c022fcb5562f968",
    urls_glob=["data/train-*.parquet"],
    version=_HF_DL_VERSION,
)
_CODEPARROT_RAW = hf_download(
    "raw/hf/codeparrot-clean-valid",
    hf_id="codeparrot/codeparrot-clean-valid",
    revision="4db92d2ec0c1b4c41eeb439cfae16854511d9dcd",
    urls_glob=["*.json.gz"],
    version=_HF_DL_VERSION,
)
_WIKIPEDIA_RAW = hf_download(
    "raw/hf/wikipedia-20231101-deruzh",
    hf_id="wikimedia/wikipedia",
    revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    urls_glob=["20231101.de/*.parquet", "20231101.ru/*.parquet", "20231101.zh/*.parquet"],
    version=_HF_DL_VERSION,
)
_FINEMATH_RAW = hf_download(
    "raw/hf/finemath-3plus",
    hf_id="HuggingFaceTB/finemath",
    revision="e92b25a616738fe95dc186b64dfb19f9c8525594",
    urls_glob=["finemath-3plus/*.parquet"],
    version=_HF_DL_VERSION,
)


@dataclasses.dataclass(frozen=True)
class SoakSource:
    """One component of the soak mixture: a shared S3 raw download, a glob into it, and its weight."""

    key: str  # short stable id -> per-arm cache/component name
    raw: ArtifactStep[Artifact]  # shared HF->S3 download (built once, adopted by every arm)
    glob: str  # parquet/json glob within the download dir
    weight: float  # mixture weight (all sources' weights are normalized by the sampler)
    text_key: str = "text"


# Representative multi-domain mixture (weights sum to 1.0): web-heavy, with Python code, three
# Wikipedia languages spanning Latin/Cyrillic/CJK scripts, and math. The three Wikipedia components
# share one download (three globs into it). The mix repeats over a 24h run with an identical
# schedule across arms, so relative BPB ordering is unaffected by the repetition.
SOAK_SOURCES: tuple[SoakSource, ...] = (
    SoakSource("web", _SLIMPAJAMA_RAW, "data/train-*.parquet", 0.50),
    SoakSource("code", _CODEPARROT_RAW, "*.json.gz", 0.20, text_key="content"),
    SoakSource("ml-de", _WIKIPEDIA_RAW, "20231101.de/*.parquet", 0.0667),
    SoakSource("ml-ru", _WIKIPEDIA_RAW, "20231101.ru/*.parquet", 0.0667),
    SoakSource("ml-zh", _WIKIPEDIA_RAW, "20231101.zh/*.parquet", 0.0666),
    SoakSource("math", _FINEMATH_RAW, "finemath-3plus/*.parquet", 0.10),
)


def soak_train_datasets(arm_name: str, tokenizer: str) -> dict[ArtifactStep, float]:
    """Each :data:`SOAK_SOURCES` component tokenized with ``tokenizer`` from its shared S3 raw copy.

    Cache names are suffixed with the arm so each arm builds its own tokenization rather than
    adopting another tokenizer's cache (the artifact store adopts by name@version); the raw
    downloads, keyed only by source, are shared across all arms.
    """
    return {
        tokenized(
            f"soak-train/{src.key}-{arm_name}",
            tokenizer=tokenizer,
            raw=src.raw,
            glob=src.glob,
            text_key=src.text_key,
            resources=_TOKENIZE_RESOURCES,
            version="2026.07.04",
        ): src.weight
        for src in SOAK_SOURCES
    }


def soak_validation(arm_name: str, tokenizer: str) -> list[ArtifactStep]:
    """Held-out validation for the soak: the bake-off's English/code subsets plus multilingual + math.

    The isoFLOP bake-off ladder (:func:`experiments.tokenization.proxy_ladder.bakeoff_validation`)
    trains on SlimPajama alone, so its held-out set stays English/code-only for comparability with
    prior ladder runs. The soak additionally trains on :data:`SOAK_SOURCES`' German/Russian/Chinese
    Wikipedia and FineMath (30% of its mixture), so its ``macro_bpb`` needs matching held-out domains
    or those components are never scored.
    """
    return [*bakeoff_validation(arm_name, tokenizer), *multilingual_math_validation(arm_name, tokenizer).values()]


@dataclasses.dataclass(frozen=True)
class NgramSpec:
    """The hashed multi-gram input embedding (Over-Encoding / LongCat) config for a soak arm.

    Adds input-side capacity at zero extra serving FLOPs; ``ratio`` scales each n-gram term's
    init to the base embedding norm. The default is the proxy study's winning 64k config.
    """

    orders: tuple[int, ...] = (2, 3, 4)
    num_hashes: int = 2
    hash_buckets: int = 4_000_037
    rank: int = 128
    combine: str = "mean"
    ratio: float = 0.25


@dataclasses.dataclass(frozen=True)
class SoakParams:
    """Mesh, batch, and schedule knobs for one soak run (the ~10B/500M SCALE_* shape is inherited).

    ``ngram`` attaches the hashed n-gram embedding; ``None`` (the default) is the plain model.
    """

    replicas: int = 8
    expert_axis: int = 8
    replica_axis: int = 1
    batch_size: int = 512
    steps: int = 40_000
    processes_per_task: int = 1
    ram: str = "512g"
    steps_per_eval: int = _STEPS_PER_EVAL
    mp: str = "params=float32,compute=bfloat16,output=bfloat16"
    ngram: NgramSpec | None = None


def soak_checkpoint(
    arm,
    params: SoakParams = SoakParams(),
    *,
    run_id: str,
    use_wandb: bool = True,
    wandb_project: str = "marin_moe",
    json_logger_name: str = "grug_moe_soak.metrics",
    version: str = "dev",
) -> ArtifactStep[LevanterCheckpoint]:
    """One tokenizer arm's 24h soak run as a lazy :class:`LevanterCheckpoint`.

    ``arm`` sets both the data tokenization and the model ``vocab_size``; ``params`` sets the mesh,
    batch, schedule, and optional n-gram embedding. The target's sliding window is forced (the
    scale builder otherwise ties it to seq_len).
    """
    # The arm sets the model vocab; the scale builder sets the 10B/500M shape; force the target sw.
    model = dataclasses.replace(build_scale_model(), vocab_size=arm.vocab_size, sliding_window=SOAK_SLIDING_WINDOW)
    if params.ngram is not None:
        n = params.ngram
        width = n.rank if n.rank > 0 else model.hidden_dim
        model = dataclasses.replace(
            model,
            ngram=NgramEmbedConfig(
                orders=n.orders,
                num_hashes=n.num_hashes,
                hash_buckets=n.hash_buckets,
                rank=n.rank if n.rank > 0 else None,
                combine=n.combine,
                init_std_scale=n.ratio * model.initializer_std * math.sqrt(width),
            ),
        )

    if model.num_experts % params.expert_axis != 0:
        raise ValueError(f"num_experts={model.num_experts} must be divisible by expert_axis={params.expert_axis}")
    data_axis = (params.replicas * GPUS_PER_NODE) // (params.replica_axis * params.expert_axis)
    batch_shards = params.replica_axis * data_axis * params.expert_axis
    if params.batch_size % batch_shards != 0:
        raise ValueError(f"batch_size={params.batch_size} must be divisible by batch shards={batch_shards}")

    resources = ResourceConfig.with_gpu(
        "H100", count=GPUS_PER_NODE, cpu=32, ram=params.ram, disk="512g", replicas=params.replicas
    )
    grug_trainer = GrugTrainerConfig(
        expert_axis_size=params.expert_axis,
        replica_axis_size=params.replica_axis,
        **SCALE_TRAINER_DEFAULTS,
    )

    train = soak_train_datasets(arm.name, arm.ref)
    validation = soak_validation(arm.name, arm.ref)
    variant = "-ngram" if params.ngram is not None else ""
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
            steps=params.steps,
            batch_size=params.batch_size,
            seed=0,
            mp=params.mp,
            tracker=tracker,
            optimizer=SCALE_OPTIMIZER,
            grug_trainer=grug_trainer,
            processes_per_task=params.processes_per_task,
            eval=GrugEvalConfig(
                compute_bpb=True,
                eval_batch_size=params.batch_size,
                steps_per_eval=params.steps_per_eval,
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


def _ngram_from_env() -> NgramSpec | None:
    """The :class:`NgramSpec` from ``BAKEOFF_NGRAM*`` env, or ``None`` when n-gram is off."""
    if not os.environ.get("BAKEOFF_NGRAM"):
        return None
    return NgramSpec(
        orders=tuple(int(o) for o in os.environ.get("BAKEOFF_NGRAM_ORDERS", "2,3,4").split(",")),
        num_hashes=env_int("BAKEOFF_NGRAM_HASHES", 2),
        hash_buckets=env_int("BAKEOFF_NGRAM_BUCKETS", 4_000_037),
        rank=env_int("BAKEOFF_NGRAM_RANK", 128),
        combine=os.environ.get("BAKEOFF_NGRAM_COMBINE", "mean"),
        ratio=float(os.environ.get("BAKEOFF_NGRAM_RATIO", "0.25")),
    )


def build_soak_checkpoint(*, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """The soak checkpoint for the in-cluster ``python -m`` entrypoint, from ``BAKEOFF_ARM`` + SCALE_* env."""
    arm = arm_by_name(os.environ.get("BAKEOFF_ARM", "soak-superbpe-64k"))
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    params = SoakParams(
        replicas=env_int("SCALE_GPU_REPLICAS", 8),
        expert_axis=env_int("SCALE_EXPERT_AXIS", 8),
        replica_axis=env_int("SCALE_REPLICA_AXIS", 1),
        batch_size=env_int("SCALE_BATCH", 512),
        steps=env_int("SCALE_STEPS", 40_000),
        processes_per_task=env_int("SCALE_PROCESSES_PER_TASK", 1),
        ram=os.environ.get("SCALE_RAM", "512g"),
        steps_per_eval=env_int("SCALE_STEPS_PER_EVAL", _STEPS_PER_EVAL),
        mp=os.environ.get("SCALE_MP", "params=float32,compute=bfloat16,output=bfloat16"),
        ngram=_ngram_from_env(),
    )
    return soak_checkpoint(
        arm,
        params,
        run_id=run_id,
        use_wandb=os.environ.get("SCALE_TRACKER", "wandb").lower() == "wandb",
        wandb_project=os.environ.get("WANDB_PROJECT", "marin_moe"),
        json_logger_name=os.environ.get("SCALE_JSON_LOGGER", "grug_moe_soak.metrics"),
        version=version,
    )


if __name__ == "__main__":
    StepRunner().run([build_soak_checkpoint().lower()])
