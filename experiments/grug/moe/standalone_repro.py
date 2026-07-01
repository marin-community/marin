# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone reproduction of the GPU MoE (May d=2560 Grug recipe) — issue #6759.

Runs the production grug MoE training step on synthetic data with **no Marin
infrastructure**: no iris, no fray dispatch, no GCS/R2, no W&B. It is meant to be
handed to external engineers to reproduce GPU MoE throughput/memory behaviour for
debugging, so everything outside the model + train step is stubbed out.

What it does:
  * builds the May d=2560 model shape (configurable) using only fields that exist
    on `main`'s `GrugModelConfig`
  * feeds a deterministic in-memory token stream (`SyntheticGrugDataset`, ported
    from the `codex/deepep-*` branch) so there is zero data I/O
  * builds the compact `(replica_dcn, data, expert, model)` mesh directly and runs
    the in-process training loop (`_run_grug_local`), bypassing Fray entirely

Default mesh: 2 nodes x 8xH100 = 16 GPUs, expert parallelism inside each NVLink
node (`--expert-axis 8`) and replication across nodes (`--replica-axis` defaults
to `jax.process_count()`). Override any axis/shape/length from the CLI.

Single host / single process (e.g. one 8xH100 box, smoke test):

    uv run python -m experiments.grug.moe.standalone_repro \
        --expert-axis 8 --batch 8 --steps 5

Two nodes of 8xH100 (run once per node, same coordinator):

    # node 0
    uv run python -m experiments.grug.moe.standalone_repro \
        --coordinator NODE0_IP:1234 --num-processes 2 --process-id 0
    # node 1
    uv run python -m experiments.grug.moe.standalone_repro \
        --coordinator NODE0_IP:1234 --num-processes 2 --process-id 1

When `--coordinator`/`--num-processes` are omitted, Levanter's `DistributedConfig`
auto-detects common multi-host launchers (SLURM, TPU, etc.); a plain single-process
launch stays single-host.
"""

import argparse
import logging
import os
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import wandb
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data import AsyncDataset
from levanter.data.text import LmDataConfig
from levanter.data.text.datasets import DirectDatasetComponent
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.optim import AdamConfig
from levanter.tracker.tracker import NoopConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.grug.moe.model import GrugAttentionImplementation, GrugModelConfig, MoeImplementation, RematMode
from experiments.grug.moe.train import GrugRunConfig, GrugTrainerConfig, _run_grug_local

logger = logging.getLogger(__name__)

# llama3 tokenizer vocab; the synthetic stream is tokenizer-free but the model
# embedding/lm-head still size to this.
VOCAB_SIZE = 128256

# May Recipe d=2560 shape (issue #6044). head_dim is fixed at 128; num_heads and
# num_kv_heads (~4:1 GQA) and the expert FFN width derive from hidden_dim exactly
# as `launch_cw_scale.py` does, so a different --hidden-dim stays self-consistent.
HEAD_DIM = 128
DEFAULT_HIDDEN_DIM = 2560
DEFAULT_NUM_LAYERS = 26
DEFAULT_NUM_EXPERTS = 256
DEFAULT_TOP_K = 4
DEFAULT_SEQ_LEN = 4096
DEFAULT_SLIDING_WINDOW = 2048

# Default mesh: 2 nodes x 8xH100. Expert parallelism spans the 8 GPUs inside a
# node; nodes replicate (replica_axis defaults to jax.process_count()).
DEFAULT_EXPERT_AXIS = 8

# Per-device batch matches the May run's regime: 256 global on 256 H100s == 1 per
# device at seq 4096, i.e. 16 global on the default 2x8 mesh.
DEFAULT_BATCH = 16
DEFAULT_STEPS = 30

# Throughput repro, not convergence. Modest schedule-stable Adam; muonh (the May
# optimizer) lives only on the deepep branches, so we stay on main's AdamConfig.
DEFAULT_LR = 6e-4
DEFAULT_WEIGHT_DECAY = 0.1

DEFAULT_MP = "params=float32,compute=bfloat16,output=bfloat16"
# The May profiling run; both backends exist on main. For a single-device smoke
# test use --expert-axis 1 --moe-impl scatter (no expert-axis collectives).
DEFAULT_MOE_IMPL: MoeImplementation = "ring"
DEFAULT_ATTN_IMPL: GrugAttentionImplementation = "gpu_fa4_cute"
DEFAULT_REMAT: RematMode = "save_moe"

# W&B project for --wandb runs (and for reattaching to log the profiler artifact).
WANDB_PROJECT = "marin_moe"

# Synthetic stream stride; coprime-ish with vocab so rows differ.
_SYNTHETIC_STRIDE = 9973


@dataclass(frozen=True)
class SyntheticGrugDataset(AsyncDataset[GrugLmExample]):
    """Deterministic in-memory token stream for distributed-systems probes.

    Tokens are a pure function of the example index (`positions + index*stride mod
    vocab`), so there is no data I/O and every host generates identical batches.
    Optional periodic EOS tokens plus segment ids exercise the cross-document
    attention mask exactly as real packed data would.
    """

    seq_len: int
    vocab_size: int
    num_examples: int
    eos_id: int | None = None
    eos_interval: int = 0
    block_cross_document_attention: bool = True

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.vocab_size <= 1:
            raise ValueError(f"vocab_size must be greater than 1, got {self.vocab_size}")
        if self.num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {self.num_examples}")
        if self.eos_interval < 0:
            raise ValueError(f"eos_interval must be non-negative, got {self.eos_interval}")
        if self.eos_interval > 0 and self.eos_id is None:
            raise ValueError("eos_id must be set when eos_interval is positive")

        # Runtime caches are intentionally not dataclass fields so frozen replace stays cheap.
        object.__setattr__(self, "_positions", np.arange(self.seq_len, dtype=np.int64))
        loss_weight = (np.arange(self.seq_len) < (self.seq_len - 1)).astype(np.float32)
        object.__setattr__(self, "_loss_weight", loss_weight)
        object.__setattr__(self, "_attn_mask", GrugAttentionMask.causal())

    async def async_len(self) -> int:
        return self.num_examples

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[GrugLmExample]:
        if not indices:
            return []
        tokens = self._tokens_for_indices(indices)
        return [self._example_from_tokens(row) for row in tokens]

    def _tokens_for_indices(self, indices: Sequence[int]) -> np.ndarray:
        positions = cast(np.ndarray, self.__dict__["_positions"])
        offsets = np.asarray(indices, dtype=np.int64)[:, None] * _SYNTHETIC_STRIDE
        tokens = (positions[None, :] + offsets) % self.vocab_size
        if self.eos_interval > 0:
            assert self.eos_id is not None  # guaranteed by __post_init__
            tokens[:, self.eos_interval - 1 :: self.eos_interval] = self.eos_id
        return tokens.astype(np.int32, copy=False)

    def _example_from_tokens(self, tokens: np.ndarray) -> GrugLmExample:
        loss_weight = cast(np.ndarray, self.__dict__["_loss_weight"])
        attn_mask = cast(GrugAttentionMask, self.__dict__["_attn_mask"])
        token_array = jnp.asarray(tokens, dtype=jnp.int32)
        loss_weight_array = jnp.asarray(loss_weight)
        if self.eos_interval > 0 and self.block_cross_document_attention:
            assert self.eos_id is not None
            eos_mask = np.roll(tokens, 1) == self.eos_id
            eos_mask[0] = False
            segment_ids = jnp.asarray(np.cumsum(eos_mask, dtype=np.int32))
            attn_mask = attn_mask.with_segment_ids(segment_ids)
        return GrugLmExample(tokens=token_array, loss_weight=loss_weight_array, attn_mask=attn_mask)


def synthetic_grug_data(
    *,
    seq_len: int,
    vocab_size: int,
    num_examples: int,
    eos_id: int | None = None,
    eos_interval: int = 0,
    block_cross_document_attention: bool = True,
) -> LmDataConfig:
    """Wrap `SyntheticGrugDataset` as an `LmDataConfig` with a passthrough tokenizer."""
    dataset = SyntheticGrugDataset(
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_examples=num_examples,
        eos_id=eos_id,
        eos_interval=eos_interval,
        block_cross_document_attention=block_cross_document_attention,
    )
    return LmDataConfig(
        tokenizer="passthrough",
        vocab_size=vocab_size,
        cache_dir=None,
        auto_build_caches=False,
        shuffle=False,
        block_cross_document_attention=block_cross_document_attention,
        components={"synthetic": DirectDatasetComponent(datasets={"train": dataset, "validation": dataset})},
        train_weights={"synthetic": 1.0},
    )


def build_model(args: argparse.Namespace) -> GrugModelConfig:
    """May d=2560 shape using only `main`'s `GrugModelConfig` fields.

    num_heads / num_kv_heads (~4:1 GQA) and the expert FFN width derive from
    hidden_dim, matching `launch_cw_scale.py`, so --hidden-dim stays consistent.
    """
    hidden_dim = args.hidden_dim
    if hidden_dim % HEAD_DIM != 0:
        raise ValueError(f"--hidden-dim {hidden_dim} must be a multiple of head_dim {HEAD_DIM}")
    num_heads = hidden_dim // HEAD_DIM
    num_kv_heads = max(1, num_heads // 4)
    while num_heads % num_kv_heads != 0:
        num_kv_heads -= 1
    intermediate_dim = hidden_dim // 2
    return GrugModelConfig(
        vocab_size=args.vocab_size,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=intermediate_dim,
        num_experts=args.num_experts,
        num_experts_per_token=args.top_k,
        num_layers=args.num_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=HEAD_DIM,
        max_seq_len=args.seq_len,
        sliding_window=args.sliding_window,
        attention_implementation=args.attn_impl,
        moe_implementation=args.moe_impl,
        remat_mode=args.remat,
    )


def build_run_config(args: argparse.Namespace) -> GrugRunConfig:
    model = build_model(args)
    data = synthetic_grug_data(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_examples=args.num_examples,
        eos_id=args.vocab_size - 1 if args.eos_interval > 0 else None,
        eos_interval=args.eos_interval,
    )

    distributed = DistributedConfig(
        coordinator_address=args.coordinator,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    tracker = WandbConfig(project=WANDB_PROJECT, tags=["grug-moe-standalone"]) if args.wandb else NoopConfig()
    # Profile rank 0 only, and skip the perfetto trace: writing it is slow and, for a
    # large multi-process trace, corrupts/hangs (BadGzipFile). The xplane .pb that xprof
    # reads is still written. Every rank still adds the callback so the stop barrier syncs.
    profiler = ProfilerConfig(
        enabled=args.profiler_steps > 0,
        start_step=args.profiler_start,
        num_steps=args.profiler_steps,
        create_perfetto_trace=False,
        profile_process_index=0,
    )
    # Local-only checkpointer with no periodic saves: _run_grug_local always builds
    # one, but we keep everything off GCS/R2 for an external-engineer repro.
    checkpointer = CheckpointerConfig(base_path=str(Path(args.output_dir) / "checkpoints"), save_interval=None)

    trainer = TrainerConfig(
        id=args.run_id,
        seed=args.seed,
        train_batch_size=args.batch,
        num_train_steps=args.steps,
        mp=jmp.get_policy(args.mp),
        tracker=tracker,
        profiler=profiler,
        checkpointer=checkpointer,
        distributed=distributed,
        log_dir=Path(args.output_dir) / "logs",
        use_explicit_mesh_axes=True,
        require_accelerator=args.require_accelerator,
    )
    grug_trainer = GrugTrainerConfig(
        trainer=trainer,
        log_every=args.log_every,
        z_loss_weight=args.z_loss_weight,
        expert_axis_size=args.expert_axis,
        replica_axis_size=args.replica_axis,
    )
    # resources is only consumed by the Fray dispatch path (run_grug); _run_grug_local
    # ignores it, but GrugRunConfig requires it, so describe the default node shape.
    resources = ResourceConfig.with_gpu("h100", count=8)
    return GrugRunConfig(
        model=model,
        data=data,
        resources=resources,
        optimizer=AdamConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_schedule="cosine",
            warmup=10,
            min_lr_ratio=0.1,
        ),
        trainer=grug_trainer,
        eval=None,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone GPU MoE (May d=2560) reproduction on synthetic data (#6759).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mesh = p.add_argument_group("mesh (default: 2 nodes x 8xH100)")
    mesh.add_argument("--expert-axis", type=int, default=DEFAULT_EXPERT_AXIS, help="expert-parallel axis (within node)")
    mesh.add_argument(
        "--replica-axis",
        type=int,
        default=None,
        help="cross-slice replica axis; defaults to jax.process_count()",
    )

    dist = p.add_argument_group("multi-host (omit for single-process auto-detect)")
    dist.add_argument("--coordinator", type=str, default=None, help="coordinator address host:port")
    dist.add_argument("--num-processes", type=int, default=None, help="total JAX processes")
    dist.add_argument("--process-id", type=int, default=None, help="this process's id in [0, num-processes)")

    shape = p.add_argument_group("model shape (May d=2560 defaults)")
    shape.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    shape.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    shape.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    shape.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="experts per token")
    shape.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    shape.add_argument("--sliding-window", type=int, default=DEFAULT_SLIDING_WINDOW)
    shape.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    shape.add_argument("--moe-impl", default=DEFAULT_MOE_IMPL, help="ring|ragged_all_to_all|deepep|scatter|sonic")
    shape.add_argument("--attn-impl", default=DEFAULT_ATTN_IMPL, help="gpu_fa4_cute|gpu_fa4_thd|reference|tpu_splash")
    shape.add_argument("--remat", default=DEFAULT_REMAT, help="recompute_all|save_moe")

    run = p.add_argument_group("run")
    run.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="global train batch size")
    run.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="number of train steps")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--mp", default=DEFAULT_MP, help="jmp mixed-precision policy")
    run.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    run.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    run.add_argument("--z-loss-weight", type=float, default=1e-4)
    run.add_argument("--log-every", type=int, default=1)
    run.add_argument("--run-id", default="grug-moe-standalone", help="trainer run id")
    run.add_argument("--output-dir", default="/tmp/grug-moe-standalone", help="local dir for logs/checkpoints")
    run.add_argument("--require-accelerator", action="store_true", help="fail if no GPU/TPU is present")
    run.add_argument("--wandb", action="store_true", help="log to W&B (default: NoopConfig, no tracking)")

    data = p.add_argument_group("synthetic data")
    data.add_argument("--num-examples", type=int, default=1 << 20, help="distinct synthetic examples")
    data.add_argument(
        "--eos-interval",
        type=int,
        default=0,
        help="insert EOS every N tokens (0 disables); enables cross-document segment masking",
    )

    prof = p.add_argument_group("profiler (off by default)")
    prof.add_argument("--profiler-steps", type=int, default=0, help="steps to profile; 0 disables")
    prof.add_argument("--profiler-start", type=int, default=8, help="step to start profiling at")

    return p.parse_args(argv)


def _archive_profiler_dir(profile_dir: Path, *, run_id: str) -> Path:
    """Tar+gzip the local xprof trace dir into /tmp and return the archive path."""
    archive_path = Path("/tmp") / f"{run_id}-profiler.tgz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(profile_dir, arcname="profiler")
    return archive_path


def _log_profiler_wandb(archive_path: Path, *, run_id: str) -> None:
    """Reattach to the finished W&B run (id == run_id) and log the trace as an artifact.

    `_run_grug_local` finishes the tracker before returning, so we resume by id to
    attach the profiler artifact to the same run.
    """
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=os.environ.get("WANDB_ENTITY"),
        id=run_id,
        resume="allow",
        job_type="profiler-upload",
    )
    artifact = wandb.Artifact(name=f"{run_id}-profiler", type="profiler")
    artifact.add_file(str(archive_path))
    run.log_artifact(artifact)
    run.finish()
    logger.info("Logged profiler artifact %s-profiler to W&B run %s", run_id, run_id)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    run_config = build_run_config(args)
    logger.info(
        "Standalone grug MoE: d=%d layers=%d experts=%dx top%d seq=%d batch=%d steps=%d "
        "mesh(expert=%s replica=%s) moe=%s attn=%s remat=%s",
        args.hidden_dim,
        args.num_layers,
        args.num_experts,
        args.top_k,
        args.seq_len,
        args.batch,
        args.steps,
        args.expert_axis,
        args.replica_axis,
        args.moe_impl,
        args.attn_impl,
        args.remat,
    )
    _run_grug_local(run_config)

    # Rank 0 wrote the xprof trace locally on the (ephemeral) pod; log it to W&B so it
    # survives. Only sink is W&B, so this is a no-op without --wandb (trace stays under
    # --output-dir).
    if args.wandb and args.profiler_steps > 0 and jax.process_index() == 0:
        profile_dir = Path(args.output_dir) / "logs" / args.run_id / "profiler"
        if not profile_dir.exists():
            logger.warning("Profiler outputs requested but trace dir is missing: %s", profile_dir)
        else:
            archive_path = _archive_profiler_dir(profile_dir, run_id=args.run_id)
            _log_profiler_wandb(archive_path, run_id=args.run_id)


if __name__ == "__main__":
    main()
