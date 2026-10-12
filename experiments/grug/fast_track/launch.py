# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""H100 dense-vs-MoE scaling ladder for the 16k-vocab BPE tokenizer study.

Rungs d512 / d768 / d1024 / d1280 map the model, data, and optimizer onto Hopper nodes and train on
the in-region 16k BPE flat cache. Each variant (dense / MoE) has a baseline recipe; a run either
data-matches or compute-matches it (``--match``, default data), or sets ``--batch-size`` /
``--num-steps`` explicitly. See README.md for the results table and launch commands.
"""

import dataclasses
import hashlib
import importlib.metadata
import math
import os
import shlex
import sys
import types
import typing
from collections.abc import Mapping
from datetime import timedelta
from enum import StrEnum
from typing import Any

import click
import fsspec
import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.progress_watchdog import ProgressWatchdogConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import DEFAULT_JAX_CONFIG, TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name
from marin.training.training import temporary_checkpoint_base_path
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import prefix_join

from experiments.datasets.paloma import _PALOMA_DETOK_RAW, paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.grug.checkpointing import RESTORE_BARRIER_TIMEOUT
from experiments.grug.fast_track.heuristic import MoeHeuristic
from experiments.grug.fast_track.model import (
    AttnResLayerBackward,
    GrugModelConfig,
    LocalMixer,
)
from experiments.grug.fast_track.train import (
    TRAIN_LOADER_BUFFER_SIZE,
    GrugEvalConfig,
    GrugRunConfig,
    GrugTrainerConfig,
    LayerSkipRule,
    MasterParamMode,
    RaggedTransport,
    WatchMode,
    _compute_match_flops,
    dropless_moe_implementation,
    run_grug,
)
from experiments.grug.moe.launch_datakit_moe_mix import _val_component

# Run defaults.
H100_LADDER_SIZES = ("d512", "d768", "d1024", "d1280", "d1536")
# Hero-shaped validation rung on one GB200 rack as the hero uses it: 16 nodes x 4 GPUs, expert parallel over all 64.
GB200_SIZES = ("gb200-d6144",)
LADDER_SIZES = H100_LADDER_SIZES + GB200_SIZES
# GB200 runs load every CUDA module at startup: a run that loads the shared XLA autotune cache autotunes nothing at
# compile time, so kernels (e.g. XLA's CUB sort) would first load lazily inside the first step while NCCL kernels spin,
# and every rank hangs in the driver's module load (lc1-gb200-tp1..tp4). It must be set before the CUDA driver
# initializes, so it goes in the job environment, not the training process.
GB200_CUDA_MODULE_LOADING = "CUDA_MODULE_LOADING=EAGER"
DEFAULT_WANDB_PROJECT = "marin_moe"
# Default tokens-per-active-param budgets defining each variant's baseline: dense compute-optimal, MoE 3x.
DENSE_TPP = 20
MOE_TPP = 60

# Fixed baseline active-param counts per (size, dense) for the recorded dense/MoE baseline runs.
# These pin the reference budget so a candidate architecture's own param count cannot move the
# baseline it is compared against (data-match holds these tokens; compute-match holds 6*N*tokens).
# They are NOT recomputed from the candidate model on purpose. Regenerate deliberately only when the
# baseline recipe itself changes, via `_active_params(_h100_ladder_model(_h100_ladder_rung(size), dense=dense))`.
_BASELINE_ACTIVE_PARAMS: dict[tuple[str, bool], int] = {
    ("d512", False): 20_840_448,
    ("d512", True): 18_087_936,
    ("d768", False): 60_555_264,
    ("d768", True): 53_477_376,
    ("d1024", False): 162_004_992,
    ("d1024", True): 144_703_488,
    ("d1280", False): 291_307_520,
    ("d1280", True): 261_488_640,
}

# Fixed baseline training FLOPs/example of the recorded dense/MoE baselines, from the trainer's
# `_compute_match_flops` (which prices the lm_head and attention matmuls that `_active_params` omits). These
# pin the compute-match budget so it holds the baseline's *true* FLOPs, not an active-param proxy, and
# so a candidate architecture cannot move the budget it is compared against. NOT recomputed from the
# candidate. Regenerate deliberately only when the baseline recipe changes, via
# `_compute_match_flops(model_config=_h100_ladder_model(_h100_ladder_rung(size), dense=dense))`.
_BASELINE_FLOPS_PER_EXAMPLE: dict[tuple[str, bool], int] = {
    ("d512", False): 1_133_066_059_776,
    ("d512", True): 1_065_420_324_864,
    ("d768", False): 2_575_067_774_976,
    ("d768", True): 2_401_121_599_488,
    ("d1024", False): 5_929_672_114_176,
    ("d1024", True): 5_504_470_351_872,
    ("d1280", False): 9_975_229_317_120,
    ("d1280", True): 9_242_400_522_240,
}


class Recipe(StrEnum):
    """Architecture of the ladder model."""

    BASELINE = "baseline"
    """Sliding-window/global GQA softmax attention with half-RoPE and QK-norm."""
    KMA = "kma"
    """KDA local layers, MLA (KV latent) + Inkling global layers, Block AttnRes; see ``_kma_model``."""


class MatchMode(StrEnum):
    """How to budget a run against its variant's baseline (dense@DENSE_TPP / MoE@MOE_TPP).

    DATA holds the baseline's total tokens; COMPUTE holds its total FLOPs. They coincide unless the
    run's active-param count differs from the baseline's. Both use the baseline batch unless
    ``--batch-size`` overrides it, rescaling the step count to hold the matched quantity.
    """

    DATA = "data"
    COMPUTE = "compute"


# Data. In-region 16k BPE-ladder cache (train split), document-shuffled so sequential reads interleave domains.
V16384_CACHE_DIR = "s3://marin-us-east-02a/marin/datakit/hero_tok/v16384_shuf/train"
V16384_TOKENIZER = "hero-bpe-v16384"
V16384_VOCAB = 16384
# Paloma caches rebuilt from the in-region detokenized raw (dodges the failed 2026.06.28 stubs).
PALOMA_DETOK_VERSION = "2026.09.17"
# In-process read cache for the tensorstore data loader. 1 GB is ample for the flat cache.
TENSORSTORE_CACHE_BYTES = 1_000_000_000


# Model geometry shared across rungs.
SEQ_LEN = 4096
SLIDING_WINDOW = 2048
GLOBAL_EVERY = 4
_EP_CAPACITY_FACTOR = 1.05  # receiver and sender EP capacity, kept paired.

# Fault tolerance / watchdog.
WATCH_INTERVAL = 10
RESUME_SAVE_INTERVAL = timedelta(minutes=20)
STEP_TIMEOUT = timedelta(minutes=15)
PROCESS_STALL_TIMEOUT = timedelta(hours=1)
# Budget for the stall diagnostic (thread stacks and a py-spy native dump) before the watchdog exits.
STALL_DIAGNOSTIC_TIMEOUT = timedelta(seconds=90)
STARTUP_TIMEOUT = timedelta(seconds=2 * RESTORE_BARRIER_TIMEOUT)
MAX_RETRIES_FAILURE = 3
Z_LOSS_WEIGHT = 1e-4
MAX_TASK_FAILURES = 3
PROFILE_NUM_STEPS = 5


class ThroughputResult(Artifact):
    """Metrics artifact for a fast_track ladder run."""


@dataclasses.dataclass(frozen=True)
class SmallShape:
    hidden_dim: int
    num_layers: int
    num_heads: int
    local_kv_heads: int
    global_kv_heads: int


@dataclasses.dataclass(frozen=True)
class H100LadderRung:
    shape: SmallShape
    gpus_per_task: int
    baseline_batch: int  # baseline global batch for this rung (larger rungs run bigger batches)
    # KMA memory/speed trade-offs (same math): where HBM allows, keep forward intermediates for backward
    # (the AttnRes layer residuals, the KDA state pass's per-chunk states); where it does not,
    # rematerialize the attention branch inside each AttnRes layer's backward.
    attn_res_layer_backward: AttnResLayerBackward = AttnResLayerBackward.RECOMPUTE
    kda_save_chunk_states: bool = False
    attn_res_remat_attention: bool = False
    device: str = "H100"
    nodes: int = 1
    # Sequential local-expert rounds of the ragged EP transport (``moe_ragged_chunks``): fewer rounds, fewer
    # collectives, at the cost of the transport buffers' peak memory.
    moe_ragged_chunks: int = 2
    # Hero-scale memory: bf16 device params with a pinned-host fp32 master and offloaded optimizer state.
    hero_memory: bool = False

    @property
    def global_device_count(self) -> int:
        return self.gpus_per_task * self.nodes


def _h100_ladder_rung(size: str) -> H100LadderRung:
    if size == "d512":
        return H100LadderRung(
            SmallShape(512, 6, 4, 1, 1),
            gpus_per_task=8,
            baseline_batch=128,
            attn_res_layer_backward=AttnResLayerBackward.SAVE,  # KMA d512: 34.5 GiB peak vs 11.7 recomputing
            kda_save_chunk_states=True,
        )
    if size == "d768":
        return H100LadderRung(SmallShape(768, 8, 6, 1, 1), gpus_per_task=8, baseline_batch=128)
    if size == "d1024":
        return H100LadderRung(SmallShape(1024, 12, 8, 2, 1), gpus_per_task=8, baseline_batch=256)
    if size == "d1280":
        return H100LadderRung(
            SmallShape(1280, 14, 10, 2, 1),
            gpus_per_task=8,
            baseline_batch=256,
            attn_res_remat_attention=True,  # KMA d1280 OOMs without it
        )
    if size == "d1536":
        # 24 layers, 12 heads x 128, GQA 3:1; no recorded baseline budget (pass --num-steps).
        return H100LadderRung(SmallShape(1536, 24, 12, 4, 4), gpus_per_task=8, baseline_batch=256)
    if size == "gb200-d6144":
        return H100LadderRung(
            SmallShape(6144, 48, 48, 1, 1),
            gpus_per_task=4,
            nodes=16,
            baseline_batch=1024,
            device="GB200",
            hero_memory=True,
            # One round: 17.54 s/step vs 17.71 s with two at the hero gate (lc1-gb200-rc1 vs prof201), and it fits.
            moe_ragged_chunks=1,
        )
    raise ValueError(f"size must be one of {list(LADDER_SIZES)}, got {size!r}")


def _h100_ladder_model(rung: H100LadderRung, dense: bool = False) -> GrugModelConfig:
    """Build the ladder model at the rung's width: MoE (384 experts, top-8, pooled-wave EP), or a
    dense baseline (``dense``) of one DenseMLP(hidden, 3*hidden) SwiGLU per block."""
    shape = rung.shape
    hidden = shape.hidden_dim
    return GrugModelConfig(
        vocab_size=V16384_VOCAB,
        dense_mlp=dense,
        hidden_dim=hidden,
        intermediate_dim=3 * hidden if dense else hidden // 2,
        shared_expert_intermediate_dim=hidden // 2,
        num_shared_experts=2,
        num_experts=384,
        num_experts_per_token=8,
        num_layers=shape.num_layers + shape.num_layers % 2,
        num_heads=shape.num_heads,
        num_kv_heads=max(shape.local_kv_heads, shape.global_kv_heads),
        local_kv_heads=shape.local_kv_heads,
        global_kv_heads=shape.global_kv_heads,
        head_dim=128,
        max_seq_len=SEQ_LEN,
        sliding_window=SLIDING_WINDOW,
        global_every=GLOBAL_EVERY,
        capacity_factor=_EP_CAPACITY_FACTOR,
        initializer_std=0.5 / math.sqrt(hidden),
        qk_mult=1.3,
        sconv=True,
        pooled_transport_capacity_factor=_EP_CAPACITY_FACTOR,
        latent_dim=None if dense else hidden // 2,  # dense carries no LatentMoE
        attn_res_layer_backward=rung.attn_res_layer_backward,
        kda_save_chunk_states=rung.kda_save_chunk_states,
        attn_res_remat_attention=rung.attn_res_remat_attention,
        moe_ragged_chunks=rung.moe_ragged_chunks,
    )


def _kma_model(model: GrugModelConfig) -> GrugModelConfig:
    """The KMA recipe on a ladder model: KDA local layers (K3 layer, dt range 0.02-0.5), MLA global
    layers with a 512-dim KV latent, full-rank Q and no QK-norm (qk_mult 1), the Inkling relative-position
    bias in place of RoPE, Block AttnRes with 8 blocks, and SConv only on the attention/MLP branch
    outputs (no K SConv)."""
    return dataclasses.replace(
        model,
        local_mixer=LocalMixer.KDA,
        mla=True,
        inkling_relpos=True,
        qk_norm=False,
        qk_mult=1.0,
        attn_res=True,
        attn_res_num_blocks=8,
        sconv_sites=("attn", "mlp"),
    )


def _keep_policy(entry: str) -> tuple[int, int | None]:
    """``EVERY[:UNTIL]`` -> ``(every, until)``."""
    every, _, until = entry.partition(":")
    return int(every), int(until) if until else None


def _active_params(cfg: GrugModelConfig) -> int:
    """Active (non-embedding) params per token, summed over layers: attention plus either the dense
    MLP or the router + top-k routed experts + LatentMoE projections + shared experts."""
    d = cfg.hidden_dim
    attn = 2 * d * cfg.num_heads * cfg.head_dim + 2 * d * cfg.num_kv_heads * cfg.head_dim
    if cfg.dense_mlp:
        return cfg.num_layers * (attn + 3 * d * cfg.intermediate_dim)
    routed = cfg.num_experts_per_token * (2 * cfg.expert_in_dim + cfg.expert_out_dim) * cfg.intermediate_dim
    latent_down = 0 if cfg.latent_dim is None else d * cfg.latent_dim
    latent_proj = latent_down + (d * cfg.expert_out_dim if cfg.has_latent_up else 0)
    shared = cfg.num_shared_experts * 3 * d * cfg.shared_expert_intermediate_dim
    router = d * (cfg.num_experts + cfg.num_null_experts)
    return cfg.num_layers * (attn + router + routed + latent_proj + shared)


def _flat_cache_data_config(
    *,
    ctx: StepContext,
    validation,
    tokenizer: str,
    train_cache_dir: str,
) -> LmDataConfig:
    """Straight-through single-source training on one pre-built flat cache (tokenizer-ablation runs).

    Mirrors the Harrier config's validation wiring, but trains on a single pre-tokenized cache at
    constant weight -- no mixture phases, no simulated epoching. ``validation`` sets are folded in as
    zero-weight components (built as executor deps, tokenized with the same ``tokenizer``).
    """
    components = {
        "train": DatasetComponent(
            source=None,
            cache_dir=train_cache_dir,
            format=TextLmDatasetFormat(),
            tags=["train"],
            flat_cache=True,
        )
    }
    if ctx.is_fingerprint:
        val_components = {item.name: _val_component(ctx.artifact_path(item)) for item in validation}
    else:
        val_components = {item.name: ctx.resolved(item).as_component() for item in validation}
    collisions = components.keys() & val_components.keys()
    if collisions:
        raise ValueError(f"validation components collide with the training component: {sorted(collisions)}")
    train_weights = {"train": 1.0, **{name: 0.0 for name in val_components}}
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components={**components, **val_components},
        train_weights=train_weights,
        auto_build_caches=False,
    )


def build_h100_ladder_run(
    *,
    run_id: str,
    size: str,
    match: MatchMode = MatchMode.DATA,
    num_steps: int | None = None,
    batch_size: int | None = None,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    version: str | None = None,
    tokenizer: str = V16384_TOKENIZER,
    train_cache_dir: str = V16384_CACHE_DIR,
    vocab_size: int = V16384_VOCAB,
    no_eval: bool = False,
    dense: bool = False,
    save_checkpoints: bool = False,
    recipe: Recipe = Recipe.BASELINE,
    attn_res_remat_attention: bool = False,
    seed: int = 0,
    profile_start_step: int | None = None,
    dump_hlo: bool = False,
    token_dump: bool = False,
    seed_xla_autotune_cache: bool = False,
    token_dump_skip_layers: tuple[int, ...] = (),
    layer_skip: LayerSkipRule | None = None,
    extra_mlp_start_frac: float = 0.5,
    resume_from: str | None = None,
    xla_memory_report: bool = False,
    master_param_mode: MasterParamMode | None = None,
    pgle_runs: int = 0,
    pipeline_depth: int = 0,
    loader_buffer_batches: int = TRAIN_LOADER_BUFFER_SIZE,
    ema_beta: float | None = None,
    ema_last_steps: int | None = None,
    max_retries_failure: int = MAX_RETRIES_FAILURE,
    model_settings: Mapping[str, str] | None = None,
    optimizer_settings: Mapping[str, str] | None = None,
    z_loss_weight: float = Z_LOSS_WEIGHT,
    ragged_transport: RaggedTransport = RaggedTransport.DEVICE,
    collective_overlap_limit: int | None = None,
    single_process: bool = False,
    grad_accum_microbatches: int = 1,
    checkpoint_keep: tuple[tuple[int, int | None], ...] = (),
    stop_after_steps: int | None = None,
) -> ArtifactStep[ThroughputResult]:
    """Build one H100 scaling-ladder rung.

    Budget resolution (see ``MatchMode``): ``batch_size`` defaults to the rung's baseline batch;
    ``num_steps`` overrides the step count directly, else it is derived to data- or compute-match the
    variant's baseline at that batch. ``recipe`` picks the architecture; ``attn_res_remat_attention``
    forces the KMA attention-branch remat on rungs that do not default to it. Evaluation runs at the
    end only. Permanent checkpoints
    default to the final step, with one rolling checkpoint every ``RESUME_SAVE_INTERVAL`` on region-local
    temporary storage.
    ``collective_overlap_limit`` overrides XLA's concurrent-collective limit (None: the mode default).
    ``ragged_transport`` picks the XLA kernel when ``model_settings`` select ``moe_implementation=ragged_all_to_all``;
    ``single_process`` runs one JAX process owning every GPU of the task, which its peer-writing kernels need.
    """
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    if not wandb_project.strip():
        raise ValueError("wandb_project must not be empty")

    rung = _h100_ladder_rung(size)
    model = dataclasses.replace(_h100_ladder_model(rung, dense=dense), vocab_size=vocab_size)
    if recipe is Recipe.KMA:
        model = _kma_model(model)
    if attn_res_remat_attention:
        if not model.attn_res:
            raise ValueError("attn_res_remat_attention requires the kma recipe")
        model = dataclasses.replace(model, attn_res_remat_attention=True)
    model = _apply_settings(model, model_settings or {})
    if master_param_mode is None:
        master_param_mode = MasterParamMode.FP32_PINNED_HOST if rung.hero_memory else MasterParamMode.DEVICE
    mp_policy = (
        "params=bfloat16,compute=bfloat16,output=bfloat16"
        if master_param_mode == MasterParamMode.FP32_PINNED_HOST
        else "params=float32,compute=bfloat16,output=bfloat16"
    )
    expert_axis_size = 1 if dense else rung.global_device_count
    if rung.global_device_count % (expert_axis_size * (1)) != 0:
        raise ValueError(
            f"expert_axis ({expert_axis_size}) * replica_axis ({(1)}) must divide "
            f"global_device_count ({rung.global_device_count})"
        )

    # Baseline: the recorded dense/MoE baseline for this size (DENSE_TPP/MOE_TPP at the rung's baseline
    # batch). baseline_active is a FIXED reference (not the candidate's), so an architecture change moves
    # the candidate's FLOPs/token but never the budget it is compared against.
    if (size, dense) not in _BASELINE_ACTIVE_PARAMS and num_steps is None:
        raise ValueError(f"No baseline budget recorded for (size={size!r}, dense={dense}); pass --num-steps")

    batch_size = batch_size if batch_size is not None else rung.baseline_batch
    if batch_size <= 0 or batch_size % rung.global_device_count != 0:
        raise ValueError(f"batch_size must be positive and divisible by {rung.global_device_count}, got {batch_size}")
    if grad_accum_microbatches < 1 or batch_size % (grad_accum_microbatches * rung.global_device_count) != 0:
        raise ValueError(
            f"batch_size {batch_size} must split into {grad_accum_microbatches} microbatches divisible by "
            f"{rung.global_device_count} devices"
        )
    if num_steps is None:
        baseline_active = _BASELINE_ACTIVE_PARAMS[(size, dense)]
        baseline_tpp = DENSE_TPP if dense else MOE_TPP
        baseline_steps = max(1, round(baseline_tpp * baseline_active / (rung.baseline_batch * SEQ_LEN)))
        baseline_tokens = rung.baseline_batch * baseline_steps * SEQ_LEN
        # Baseline's true total training FLOPs (fixed): flops/example * examples, at the baseline batch.
        baseline_flops = _BASELINE_FLOPS_PER_EXAMPLE[(size, dense)] * baseline_steps * rung.baseline_batch
        if match is MatchMode.DATA:
            # DATA holds the baseline's token budget.
            num_steps = max(1, round(baseline_tokens / (batch_size * SEQ_LEN)))
        else:
            # COMPUTE holds the baseline's *true* training FLOPs, derived from the candidate's own
            # flops/example (the trainer's `_compute_match_flops`, which counts the lm_head and attention that
            # `_active_params` omits) -- so an architecture change never buys or loses compute.
            candidate_flops_per_example = _compute_match_flops(model_config=model)
            num_steps = max(1, round(baseline_flops / (candidate_flops_per_example * batch_size)))
    elif num_steps <= 0:
        raise ValueError(f"--num-steps must be positive, got {num_steps}")

    # Only the end-of-run eval: the periodic one lands on the last step, where the forced final pass skips it.
    # no_eval disables it entirely below (the forced final callback would otherwise still run a full eval, so
    # pushing the interval past the end is not enough).
    steps_per_eval = num_steps
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=num_steps,
        batch_size=batch_size,
        hidden_dim=model.hidden_dim,
        seq_len=SEQ_LEN,
    )
    optimizer = _apply_settings(optimizer, optimizer_settings or {})
    grug_trainer = GrugTrainerConfig(
        data_seed=None,
        log_every=1,
        z_loss_weight=z_loss_weight,
        watch_mode=WatchMode.INLINE,
        save_checkpoints=save_checkpoints,
        expert_axis_size=expert_axis_size,
        replica_axis_size=(1),
        offload_opt_state=rung.hero_memory,
        master_param_mode=master_param_mode,
        ema_beta=ema_beta,
        ema_last_steps=ema_last_steps,
        grad_accum_microbatches=grad_accum_microbatches,
    )
    # GB200 nodes as the hero requests them (moe_hero_ep.hero_recipe); H100 nodes as the ladder always has.
    cpu, ram, disk = (120, "890g", "1t") if rung.device == "GB200" else (32, "600g", "900g")
    train_resources = ResourceConfig.with_gpu(
        rung.device,
        # The seed is one process on one GPU (GB200 runs one process per GPU), standing in for all of them.
        count=1 if seed_xla_autotune_cache else rung.gpus_per_task,
        cpu=cpu,
        ram=ram,
        disk=disk,
        replicas=1 if seed_xla_autotune_cache else rung.nodes,
    )
    autotune_cache_path = _xla_autotune_cache_path(model, rung) if rung.device == "GB200" else None
    if autotune_cache_path is not None and not seed_xla_autotune_cache and not _path_exists(autotune_cache_path):
        # Without a cache every rank autotunes and exchanges results, and that multi-process compile hangs ~7 times
        # in 8 on GB200. A one-GPU seed builds it in ~2 minutes.
        raise ValueError(
            f"No XLA autotune cache for this GB200 program ({autotune_cache_path}). Seed it first: rerun this "
            "command with --seed-xla-autotune-cache."
        )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)
    eval_tag = tokenizer.rsplit("/", 1)[-1]
    validation = [
        *uncheatable_datasets(tokenizer=tokenizer, tag=eval_tag).values(),
        *paloma_datasets(
            tokenizer=tokenizer, tag=eval_tag, raw_prefix=_PALOMA_DETOK_RAW, version=PALOMA_DETOK_VERSION
        ).values(),
    ]

    def build_config(ctx: StepContext) -> GrugRunConfig:
        permanent_checkpoint_path = prefix_join(ctx.output_path, "checkpoints")
        temporary_checkpoint_path = temporary_checkpoint_base_path(ctx.output_path)
        trainer = TrainerConfig(
            id=run_id,
            seed=seed,
            train_batch_size=batch_size,
            num_train_steps=num_steps,
            jax_config=(
                {**DEFAULT_JAX_CONFIG, "jax_enable_pgle": True, "jax_pgle_profiling_runs": pgle_runs}
                if pgle_runs
                else dict(DEFAULT_JAX_CONFIG)
            ),
            profiler=ProfilerConfig(
                enabled=profile_start_step is not None,
                start_step=profile_start_step if profile_start_step is not None else 0,
                num_steps=PROFILE_NUM_STEPS,
            ),
            mp=jmp.get_policy(mp_policy),
            tracker=WandbConfig(
                entity="marin-community",
                project=wandb_project,
                tags=["h100", "fasttrack", "vocab-16k"],
                group="fasttrack-scaling-ladder",
                name=run_id,
                replicate_path=ctx.output_path,
            ),
            watch=WatchConfig(interval=WATCH_INTERVAL),
            progress_watchdog=ProgressWatchdogConfig(
                step_timeout=STEP_TIMEOUT,
                process_timeout=PROCESS_STALL_TIMEOUT,
                startup_timeout=STARTUP_TIMEOUT,
                diagnostic_timeout=STALL_DIAGNOSTIC_TIMEOUT,
            ),
            use_explicit_mesh_axes=True,
            require_accelerator=True,
            allow_nondivisible_batch_size=False,
            load_checkpoint_path=[
                *([resume_from] if resume_from else []),
                permanent_checkpoint_path,
                temporary_checkpoint_path,
            ],
            checkpointer=CheckpointerConfig(
                base_path=permanent_checkpoint_path,
                temporary_base_path=temporary_checkpoint_path,
                save_interval=RESUME_SAVE_INTERVAL,
                keep=[dict(every=every, until=until) for every, until in checkpoint_keep] or None,
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
        )
        data = _flat_cache_data_config(
            ctx=ctx, validation=validation, tokenizer=tokenizer, train_cache_dir=train_cache_dir
        )
        return GrugRunConfig(
            model=model,
            data=data,
            resources=ctx.runtime_arg("train_resources"),
            tensorstore_cache_bytes=TENSORSTORE_CACHE_BYTES,
            optimizer=optimizer,
            trainer=dataclasses.replace(
                grug_trainer,
                trainer=trainer,
                hlo_dump_path=prefix_join(ctx.output_path, "train_step.hlo.txt") if dump_hlo else None,
                token_dump_path=(prefix_join(ctx.output_path, "token_dump.npz") if token_dump else None),
                token_dump_skip_layers=token_dump_skip_layers,
                xla_autotune_cache_path=autotune_cache_path,
                preload_library_kernels=rung.device == "GB200",
                xla_autotune_seed_topology=(f"1x{rung.global_device_count}x1" if seed_xla_autotune_cache else None),
                layer_skip=layer_skip,
                extra_mlp_start_frac=extra_mlp_start_frac,
                completion_marker_path=prefix_join(ctx.output_path, "final_eval_done.json"),
                pipeline_depth=pipeline_depth,
                loader_buffer_batches=loader_buffer_batches,
                xla_memory_report_path=prefix_join(ctx.output_path, "xla_memory") if xla_memory_report else None,
            ),
            eval=(
                None
                if no_eval
                else GrugEvalConfig(
                    steps_per_eval=steps_per_eval,
                    eval_batch_size=rung.global_device_count,
                    compute_bpb=True,
                    dropless_eval=True,
                    dropless_eval_moe_implementation=dropless_moe_implementation(model),
                )
            ),
            stop_after_steps=num_steps if stop_after_steps is None else stop_after_steps,
            processes_per_task=1 if single_process or seed_xla_autotune_cache else rung.gpus_per_task,
            ragged_transport=ragged_transport,
            collective_overlap_limit=collective_overlap_limit,
            max_retries_failure=max_retries_failure,
            max_task_failures=MAX_TASK_FAILURES,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=ThroughputResult,
        run=run_grug,
        build_config=build_config,
        deps=(*validation,),
        runtime_args={"train_resources": train_resources},
    )


_WANDB_PROJECT = "marin_moe"


def _submit_to_cluster(run_id: str, target_cluster: str | None, priority: str, job_env: tuple[str, ...]) -> None:
    """Re-exec this launcher as an Iris H100 job: wrap the same launcher args in ``iris job run ... --
    python -m ...launch <args> --run``. Replaces the old ``irun`` shell wrapper. Never returns."""
    launch_args = [a for a in sys.argv[1:] if a != "--submit"]
    if "--run" not in launch_args:
        launch_args.append("--run")
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        raise click.ClickException("WANDB_API_KEY must be set in the environment to submit a cluster run.")
    placement_args = ["--target-cluster", target_cluster] if target_cluster else ["--reserve", "H100"]
    cmd = [
        "uv",
        "run",
        "iris",
        "--cluster",
        "marin",
        "job",
        "run",
        "--no-wait",
        "--enable-extra-resources",
        *placement_args,
        "--priority",
        priority,
        "--job-name",
        f"{run_id}-coord",
        "-e",
        "WANDB_API_KEY",
        wandb_key,
        "-e",
        "WANDB_PROJECT",
        _WANDB_PROJECT,
        *_job_env_args(job_env),
        "--",
        "python",
        "-m",
        "experiments.grug.fast_track.launch",
        *launch_args,
    ]
    printable = " ".join(shlex.quote("$WANDB_API_KEY" if c == wandb_key else c) for c in cmd)
    click.echo(f"submitting: {printable}", err=True)
    os.execvp(cmd[0], cmd)


def _job_env_args(job_env: tuple[str, ...]) -> list[str]:
    """``iris job run -e KEY VALUE`` pairs for ``--job-env KEY=VALUE`` (the dispatcher forwards ``XLA_*`` and friends
    from the coordinator to the train child)."""
    args = []
    for item in job_env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise click.BadParameter(f"--job-env needs KEY=VALUE, got {item!r}")
        args += ["-e", key, value]
    return args


XLA_AUTOTUNE_CACHE_TTL_DAYS = 30


def _path_exists(path: str) -> bool:
    return fsspec.core.url_to_fs(path)[0].exists(path)


def _xla_autotune_cache_path(model: GrugModelConfig, rung: H100LadderRung) -> str:
    """Region-temp path of the shared XLA autotune results for this program: keyed by the full model config, the
    rung's mesh, the job's XLA flags and jaxlib, so only runs that compile the same fusions share them. A new key
    (any config change, including a changed default) needs a seed (``--seed-xla-autotune-cache``, ~2 min, one GPU)."""
    identity = repr((model, rung, os.environ.get("XLA_FLAGS", ""), importlib.metadata.version("jaxlib")))
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    return prefix_join(marin_temp_bucket(XLA_AUTOTUNE_CACHE_TTL_DAYS, "xla-autotune-results"), f"{key}.textproto")


def _layer_skip_rule(
    layers: str | None, rate: float, positions: str | None, token_ids_file: str | None
) -> LayerSkipRule | None:
    """The ``--layer-skip-*`` options as a ``LayerSkipRule`` (None without ``--layer-skip-layers``)."""
    if layers is None:
        if positions is not None or token_ids_file is not None:
            raise click.UsageError("--layer-skip-positions / --layer-skip-token-ids-file need --layer-skip-layers")
        return None
    span = None
    if positions is not None:
        lo, _, hi = positions.partition(":")
        span = (int(lo), int(hi) if hi else -1)
    ids: tuple[int, ...] = ()
    if token_ids_file is not None:
        with open(token_ids_file) as f:
            ids = tuple(sorted({int(t) for t in f.read().split()}))
    return LayerSkipRule(layers=tuple(int(x) for x in layers.split(",")), rate=rate, positions=span, token_ids=ids)


@click.command()
@click.option("--run-id", required=True, help="Run identifier for artifact and W&B names.")
@click.option(
    "--size", required=True, type=click.Choice(LADDER_SIZES), help="Ladder rung (H100 widths or the GB200 rack)."
)
@click.option(
    "--match",
    type=click.Choice([m.value for m in MatchMode]),
    default=MatchMode.DATA.value,
    show_default=True,
    help="Budget vs the variant's baseline: data-match tokens or compute-match FLOPs (ignored if --num-steps set).",
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Global sequence batch (default: the rung's baseline batch).",
)
@click.option(
    "--num-steps",
    type=click.IntRange(min=1),
    default=None,
    help="Override the step budget directly (else derived from --match).",
)
@click.option("--no-eval", is_flag=True, help="Disable in-run eval (clean MFU probes).")
@click.option("--dense", is_flag=True, help="Dense baseline: 3x hidden SwiGLU per block, no MoE.")
@click.option(
    "--save-checkpoints",
    is_flag=True,
    default=False,
    help="Save a permanent final checkpoint to S3 (off by default; also enables recovery).",
)
@click.option(
    "--recipe",
    type=click.Choice([r.value for r in Recipe]),
    default=Recipe.BASELINE.value,
    show_default=True,
    help="Architecture: the baseline, or KMA (KDA local + MLA/Inkling global layers + Block AttnRes).",
)
@click.option(
    "--attn-res-remat-attention",
    is_flag=True,
    help="KMA: recompute the attention branch in each AttnRes layer's backward (lower peak memory, one "
    "extra attention forward per layer; always on at d1280).",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Trainer seed (model init and data key); vary it to measure run-to-run noise.",
)
@click.option(
    "--profile-start-step",
    type=int,
    default=None,
    help="Capture a JAX profile of a few steps from this step (uploaded to the run's xprof directory).",
)
@click.option("--dump-hlo", is_flag=True, help="Write the compiled train-step HLO to <output>/train_step.hlo.txt.")
@click.option(
    "--extra-mlp-start-frac", type=float, default=0.5, show_default=True, help="extra_mlp_layers turn on here."
)
@click.option("--layer-skip-layers", default=None, help="Comma-separated layers some tokens skip (LayerSkipRule).")
@click.option("--layer-skip-rate", type=float, default=1.0, show_default=True, help="Skip probability of a match.")
@click.option("--layer-skip-positions", default=None, help="LO:HI document positions that skip (HI empty: open).")
@click.option(
    "--layer-skip-token-ids-file", default=None, help="Local file of whitespace-separated token ids that skip."
)
@click.option(
    "--token-dump",
    is_flag=True,
    help="After the last step, write each held-out token's next-token loss (and null slots per layer, with "
    "moe_null_layers) to <output>/token_dump.npz.",
)
@click.option(
    "--seed-xla-autotune-cache",
    is_flag=True,
    help="GB200: seed the shared XLA autotune cache from one node under a mock topology of the full rack, then stop.",
)
@click.option(
    "--token-dump-skip-layers",
    default=None,
    help="Comma-separated layers: the token dump also records each token's loss with every token skipping them.",
)
@click.option(
    "--resume-from",
    default=None,
    help="Load this checkpoint directory (another run's <output>/checkpoints) first.",
)
@click.option(
    "--master-param-mode",
    type=click.Choice([m.value for m in MasterParamMode]),
    default=None,
    help="Where the fp32 weights live (default: pinned host on hero-memory rungs, device otherwise).",
)
@click.option(
    "--xla-memory-report",
    is_flag=True,
    help="Dump XLA's train-step buffer assignment and memory-usage report to <output>/xla_memory/ (also on OOM).",
)
@click.option(
    "--pgle-runs",
    type=int,
    default=0,
    help="Profile-guided latency estimation: profile this many steps, then recompile with measured collective "
    "latencies for the latency-hiding scheduler (0: off).",
)
@click.option(
    "--pipeline-depth",
    type=click.IntRange(0, 1),
    default=0,
    help="Train steps dispatched before the previous step's host work (1: overlap, crashes near step 515; "
    "0: finish each step first).",
)
@click.option(
    "--loader-buffer-batches",
    type=int,
    default=TRAIN_LOADER_BUFFER_SIZE,
    help="Batches the train loader keeps on the devices ahead of the step.",
)
@click.option("--ema-beta", type=float, default=None, help="Weight-EMA decay; evals after the EMA start score the EMA.")
@click.option("--ema-last-steps", type=int, default=None, help="Run the weight EMA over only the last N steps.")
@click.option(
    "--max-retries",
    type=click.IntRange(min=0),
    default=MAX_RETRIES_FAILURE,
    show_default=True,
    help="Training-job retries after a failure (0 for debugging runs).",
)
@click.option(
    "--z-loss-weight",
    type=click.FloatRange(min=0.0),
    default=Z_LOSS_WEIGHT,
    show_default=True,
    help="Weight of the final-logit logsumexp z-loss (0 disables it).",
)
@click.option(
    "--ragged-transport",
    type=click.Choice([t.value for t in RaggedTransport]),
    default=RaggedTransport.DEVICE.value,
    show_default=True,
    help="XLA kernel for --model-set moe_implementation=ragged_all_to_all. device/one_shot need --single-process.",
)
@click.option(
    "--collective-overlap-limit",
    type=int,
    default=None,
    help="XLA's concurrent-collective limit (default: 1 for ragged MoE or inline watch, else 4).",
)
@click.option(
    "--single-process",
    is_flag=True,
    help="Run one JAX process owning every GPU (default: one process per GPU), so XLA can write peer buffers.",
)
@click.option(
    "--checkpoint-keep",
    default="",
    help="Comma-separated EVERY[:UNTIL] permanent-checkpoint policies (levanter keep), e.g. 350:350,1400 saves at "
    "350, 1400 and 2800 (with --save-checkpoints).",
)
@click.option(
    "--stop-after-steps",
    type=int,
    default=None,
    help="Stop at this step without changing the schedule's length (e.g. a probe-only job resumed at that step).",
)
@click.option(
    "--grad-accum",
    "grad_accum_microbatches",
    default=1,
    show_default=True,
    help="Microbatches per train batch (gradient accumulation): the same update with less activation memory.",
)
@click.option(
    "--model-set",
    multiple=True,
    help="Override a GrugModelConfig field, 'name=value' (repeatable; parsed as the field's declared type).",
)
@click.option(
    "--opt-set",
    multiple=True,
    help="Override a GrugMoeMuonHConfig field, 'name=value' (repeatable; parsed as the field's declared type).",
)
@click.option(
    "--priority",
    type=click.Choice(["production", "interactive", "batch"]),
    default="interactive",
    show_default=True,
    help="Iris scheduling priority for --submit.",
)
@click.option(
    "--job-env",
    multiple=True,
    help="Environment variable for the submitted job, KEY=VALUE (repeatable; e.g. XLA_PYTHON_CLIENT_MEM_FRACTION=0.9).",
)
@click.option(
    "--submit",
    is_flag=True,
    help="Submit as an Iris H100 job (wraps this launcher in `iris job run`); without it the "
    "launcher builds/prints the plan locally.",
)
@click.option(
    "--target-cluster",
    default=None,
    help="Pin the submitted job to this Iris cluster. Omit to let Iris select an H100 cluster.",
)
@build_options
def main(
    run_id: str,
    size: str,
    match: str,
    batch_size: int | None,
    num_steps: int | None,
    no_eval: bool,
    dense: bool,
    save_checkpoints: bool,
    recipe: str,
    attn_res_remat_attention: bool,
    seed: int,
    profile_start_step: int | None,
    dump_hlo: bool,
    token_dump: bool,
    seed_xla_autotune_cache: bool,
    token_dump_skip_layers: str | None,
    extra_mlp_start_frac: float,
    layer_skip_layers: str | None,
    layer_skip_rate: float,
    layer_skip_positions: str | None,
    layer_skip_token_ids_file: str | None,
    resume_from: str | None,
    xla_memory_report: bool,
    master_param_mode: str | None,
    pgle_runs: int,
    pipeline_depth: int,
    loader_buffer_batches: int,
    ema_beta: float | None,
    ema_last_steps: int | None,
    max_retries: int,
    z_loss_weight: float,
    ragged_transport: str,
    collective_overlap_limit: int | None,
    single_process: bool,
    grad_accum_microbatches: int,
    checkpoint_keep: str,
    stop_after_steps: int | None,
    model_set: tuple[str, ...],
    opt_set: tuple[str, ...],
    priority: str,
    job_env: tuple[str, ...],
    submit: bool,
    target_cluster: str | None,
) -> ArtifactStep[ThroughputResult]:
    if submit:
        if size in GB200_SIZES:
            job_env = (*job_env, GB200_CUDA_MODULE_LOADING)
        _submit_to_cluster(run_id, target_cluster, priority, job_env)  # re-execs iris; never returns
    return build_h100_ladder_run(
        run_id=run_id,
        size=size,
        match=MatchMode(match),
        batch_size=batch_size,
        num_steps=num_steps,
        no_eval=no_eval,
        dense=dense,
        save_checkpoints=save_checkpoints,
        recipe=Recipe(recipe),
        attn_res_remat_attention=attn_res_remat_attention,
        seed=seed,
        profile_start_step=profile_start_step,
        dump_hlo=dump_hlo,
        token_dump=token_dump,
        seed_xla_autotune_cache=seed_xla_autotune_cache,
        token_dump_skip_layers=(
            tuple(int(x) for x in token_dump_skip_layers.split(",")) if token_dump_skip_layers else ()
        ),
        extra_mlp_start_frac=extra_mlp_start_frac,
        layer_skip=_layer_skip_rule(layer_skip_layers, layer_skip_rate, layer_skip_positions, layer_skip_token_ids_file),
        resume_from=resume_from,
        xla_memory_report=xla_memory_report,
        master_param_mode=MasterParamMode(master_param_mode) if master_param_mode else None,
        pgle_runs=pgle_runs,
        pipeline_depth=pipeline_depth,
        loader_buffer_batches=loader_buffer_batches,
        ema_beta=ema_beta,
        ema_last_steps=ema_last_steps,
        max_retries_failure=max_retries,
        model_settings=_parse_settings(model_set),
        optimizer_settings=_parse_settings(opt_set),
        z_loss_weight=z_loss_weight,
        ragged_transport=RaggedTransport(ragged_transport),
        collective_overlap_limit=collective_overlap_limit,
        single_process=single_process,
        grad_accum_microbatches=grad_accum_microbatches,
        checkpoint_keep=tuple(_keep_policy(entry) for entry in checkpoint_keep.split(",") if entry),
        stop_after_steps=stop_after_steps,
    )


def _parse_settings(items: tuple[str, ...]) -> dict[str, str]:
    settings = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep:
            raise click.BadParameter(f"expected 'name=value', got {item!r}")
        settings[name.strip()] = value.strip()
    return settings


def _typed_setting(config: Any, name: str, text: str) -> Any:
    """Parse ``text`` as the declared type of ``config.<name>`` (``X | None`` parses as ``X``; ``none`` gives
    None; tuples are comma-separated)."""
    if name not in {f.name for f in dataclasses.fields(config)}:
        raise ValueError(f"{type(config).__name__} has no field {name!r}")
    annotation = typing.get_type_hints(type(config))[name]
    # "none" means None only for optional fields; a string field may use "none" as a value (muon_pre_norm).
    if text.lower() == "none" and type(None) in typing.get_args(annotation):
        return None
    return _parse_as(annotation, text, name)


def _parse_as(annotation: Any, text: str, name: str) -> Any:
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if typing.get_origin(annotation) in (types.UnionType, typing.Union):
        if type(None) in typing.get_args(annotation) and text == "None":
            return None
        # Members in declaration order; the first that parses wins.
        errors = []
        for member in args:
            try:
                return _parse_as(member, text, name)
            except ValueError as e:
                errors.append(str(e))
        raise ValueError(f"{name}: {text!r} matches no member of {annotation}: {errors}")
    if typing.get_origin(annotation) is tuple:
        item = args[0] if args else float
        return tuple(_parse_as(item, part, name) for part in text.split(",") if part)
    if annotation is bool:
        if text.lower() not in ("true", "false"):
            raise ValueError(f"{name} expects true/false, got {text!r}")
        return text.lower() == "true"
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation(text)
    if annotation in (int, float, str):
        return annotation(text)
    if typing.get_origin(annotation) is typing.Literal:
        if text not in args:
            raise ValueError(f"{name}: {text!r} is not one of {args}")
        return text
    raise ValueError(f"{name}: unsupported field type {annotation}")


def _apply_settings(config: Any, settings: Mapping[str, str]) -> Any:
    return dataclasses.replace(config, **{k: _typed_setting(config, k, v) for k, v in settings.items()})


if __name__ == "__main__":
    main()
