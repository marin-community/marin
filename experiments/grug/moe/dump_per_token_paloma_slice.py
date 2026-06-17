# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dump per-position losses (+ tokens, weights, segment_ids) for a single paloma slice.

Used to do fine-grained per-token analysis comparing two checkpoints on identical
data. The slice's validation dataset is iterated directly (not through the round-robin
TaggedEvaluator loader), so we score only the chosen slice and avoid wasted compute.

For each of the first ``max_batches`` batches of size ``batch_size`` at ``seq_len``:
- forward the model with ``next_token_loss(reduction="none")`` to get per-position loss
- save (loss, tokens, loss_weight, segment_ids) arrays to GCS as a single .npz

Two launchers pair this runner with the same paloma slice + seq_len + batch config,
just swapping the checkpoint + model config so we get matched-data dumps for:
  - the d=1280 EP=1 seq=4k baseline + SWA-only at seq=8k
  - the d=1280 EP=1 seq=8k from-scratch trained model

Offline alignment is then trivial: both .npz files have identical shape (max_batches,
batch_size, seq_len) and were drawn from the same sync-dataset indices.
"""

import dataclasses
import logging
import os
import posixpath
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.data.text import LmDataConfig

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.heuristic_v2 import MoeHeuristicV2
from experiments.grug.moe.model import GrugModelConfig

logger = logging.getLogger(__name__)


_SLIDING_WINDOW: int = 2048
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"


@dataclass(frozen=True)
class PerTokenDumpConfig:
    dim: int
    run_id: str
    output_path: str
    resources: ResourceConfig
    checkpoint_path: str
    slice_name: str  # e.g. "paloma/dolma_100_programing_languages"
    seq_len: int
    batch_size: int
    max_batches: int
    expert_parallel: int
    mp: str
    data: LmDataConfig
    # SWA-only extension config (use long_sliding_window for the long layers).
    # ``None`` keeps the baseline architecture (full attention on long layers).
    long_sliding_window: int | None = None
    seed: int = 0


def _build_model_config(dim: int, seq_len: int, long_sliding_window: int | None) -> GrugModelConfig:
    base = dataclasses.replace(
        MoeHeuristicV2().build_model_config(dim, seq_len=seq_len),
        sliding_window=_SLIDING_WINDOW,
    )
    if long_sliding_window is not None:
        base = dataclasses.replace(base, long_sliding_window=long_sliding_window)
    return base


def _run_eval_local(config: PerTokenDumpConfig) -> None:
    """TPU-side entrypoint: load checkpoint, iterate slice, dump per-token data."""
    import time
    from datetime import timedelta

    import equinox as eqx
    import fsspec
    import haliax as hax
    import jax
    import jax.numpy as jnp
    import jmp
    import numpy as np
    import wandb
    from haliax import Axis
    from jax.sharding import NamedSharding, reshard
    from jax.sharding import PartitionSpec as P
    from levanter.callbacks.profiler import ProfilerConfig
    from levanter.checkpoint import CheckpointerConfig
    from levanter.grug.attention import AttentionMask
    from levanter.tracker.wandb import WandbConfig
    from levanter.trainer import TrainerConfig
    from levanter.utils.mesh import MeshConfig
    from marin.training.training import temporary_checkpoint_base_path

    from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
    from experiments.grug.moe.model import Transformer
    from experiments.grug.moe.train import GrugTrainState, initial_state

    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=1,
        profiler=ProfilerConfig(),
        mp=jmp.get_policy(config.mp),
        tracker=WandbConfig(
            project="marin_moe",
            name=config.run_id,
            tags=["moe", "per_token_dump", f"d{config.dim}"],
            group="moe-may-per-token-dump",
        ),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": config.expert_parallel}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            temporary_base_path=temporary_checkpoint_base_path(config.output_path),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=None,
        ),
        load_checkpoint=False,
    )
    trainer.initialize()

    model_cfg = _build_model_config(config.dim, config.seq_len, config.long_sliding_window)
    h = MoeHeuristicV2()
    opt = h.build_muonh_config(config.batch_size, 1.0, config.dim, seq_len=config.seq_len).build(1)

    with trainer.use_device_mesh():

        @jax.jit
        def _init(key):
            return initial_state(model_cfg, optimizer=opt, mp=trainer.mp, key=key, ema_beta=None)

        state_template: GrugTrainState = _init(jax.random.PRNGKey(config.seed))
        loaded_state = restore_grug_state_from_checkpoint(
            state_template,
            checkpoint_search_paths=[config.checkpoint_path],
            load_checkpoint_setting=True,
            mesh=trainer.device_mesh,
            allow_partial=False,
        )
        model: Transformer = loaded_state.params  # type: ignore[assignment]
        logger.info("Checkpoint loaded: %s", config.checkpoint_path)

        # Grab the slice's validation dataset directly. Same data_config the
        # paloma TaggedEvaluator uses internally -- bypassing the tagged
        # round-robin so we score only the chosen slice.
        Pos = Axis("position", config.seq_len)
        val_sets = config.data.validation_sets(Pos)
        if config.slice_name not in val_sets:
            available = sorted(val_sets.keys())[:20]
            raise KeyError(f"slice {config.slice_name!r} not found. Available (first 20): {available}")
        slice_ds = val_sets[config.slice_name]
        sync_ds = slice_ds.as_sync_dataset()
        total_seqs = len(sync_ds)
        logger.info("[%s] %d sequences at seq=%d", config.slice_name, total_seqs, config.seq_len)

        batch_pspec = P(("data", "expert"), None)
        batch_sharding = NamedSharding(trainer.device_mesh, batch_pspec)

        @eqx.filter_jit
        def _score(model: Transformer, tokens, weight, segment_ids):
            tokens = reshard(tokens, batch_sharding)
            weight = reshard(weight, batch_sharding)
            segment_ids = reshard(segment_ids, batch_sharding)
            mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=(segment_ids, segment_ids))
            per_pos = model.next_token_loss(tokens, weight, mask=mask, reduction="none")
            return per_pos

        # Pull per-batch arrays, batch and score, accumulate.
        all_losses, all_tokens, all_weights, all_segments = [], [], [], []
        scored_sequences = 0
        for batch_idx in range(config.max_batches):
            start = batch_idx * config.batch_size
            if start >= total_seqs:
                logger.info("Slice exhausted at batch %d", batch_idx)
                break
            end = min(start + config.batch_size, total_seqs)
            real_count = end - start
            # Fetch examples one-by-one (sync dataset). Each is a NamedArray LmExample.
            examples = [sync_ds[i] for i in range(start, end)]
            tokens_np = np.stack([np.asarray(hax.ndarray(ex.tokens)).reshape(config.seq_len) for ex in examples])
            weight_np = np.stack([np.asarray(hax.ndarray(ex.loss_weight)).reshape(config.seq_len) for ex in examples])
            # segment_ids lives on attn_mask; pull from the tuple/array form.
            seg_arrays = []
            for ex in examples:
                seg = ex.attn_mask.segment_ids
                if isinstance(seg, tuple):
                    seg = seg[0]
                seg_arrays.append(np.asarray(hax.ndarray(seg)).reshape(config.seq_len))
            seg_np = np.stack(seg_arrays)
            # Pad to full batch_size if last batch is short -- weights stay 0 on the pad rows.
            if real_count < config.batch_size:
                pad_rows = config.batch_size - real_count
                tokens_np = np.concatenate([tokens_np, np.zeros((pad_rows, config.seq_len), dtype=tokens_np.dtype)])
                weight_np = np.concatenate([weight_np, np.zeros((pad_rows, config.seq_len), dtype=weight_np.dtype)])
                seg_np = np.concatenate([seg_np, np.zeros((pad_rows, config.seq_len), dtype=seg_np.dtype)])

            t0 = time.perf_counter()
            per_pos = _score(model, jnp.asarray(tokens_np), jnp.asarray(weight_np), jnp.asarray(seg_np))
            per_pos_np = np.asarray(per_pos)
            elapsed = time.perf_counter() - t0
            logger.info(
                "[batch %d/%d] scored %d sequences, elapsed=%.1fs, sum_loss=%.3f",
                batch_idx + 1,
                config.max_batches,
                real_count,
                elapsed,
                float((per_pos_np * weight_np).sum()),
            )
            all_losses.append(per_pos_np.astype(np.float32))
            all_tokens.append(tokens_np.astype(np.int32))
            all_weights.append(weight_np.astype(np.float32))
            all_segments.append(seg_np.astype(np.int32))
            scored_sequences += real_count

        losses_arr = np.stack(all_losses, axis=0)
        tokens_arr = np.stack(all_tokens, axis=0)
        weights_arr = np.stack(all_weights, axis=0)
        segments_arr = np.stack(all_segments, axis=0)
        slice_slug = config.slice_name.replace("/", "__")
        dump_path = posixpath.join(config.output_path, f"per_token_dump__{slice_slug}.npz")
        logger.info("Writing dump (%d batches, %d sequences) -> %s", len(all_losses), scored_sequences, dump_path)
        with fsspec.open(dump_path, "wb") as f:
            np.savez_compressed(
                f,
                losses=losses_arr,
                tokens=tokens_arr,
                weights=weights_arr,
                segment_ids=segments_arr,
                run_id=np.array(config.run_id),
                checkpoint_path=np.array(config.checkpoint_path),
                slice_name=np.array(config.slice_name),
                seq_len=np.array(config.seq_len),
                batch_size=np.array(config.batch_size),
                max_batches=np.array(config.max_batches),
                scored_sequences=np.array(scored_sequences),
                total_sequences_in_slice=np.array(total_seqs),
            )
        wandb.log(
            {
                "dump/path": dump_path,
                "dump/scored_sequences": scored_sequences,
                "dump/total_sequences_in_slice": total_seqs,
                "dump/total_loss": float((losses_arr * weights_arr).sum()),
                "dump/total_weighted_tokens": float(weights_arr.sum()),
            }
        )

    wandb.finish()


def run_per_token_dump(config: PerTokenDumpConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_eval_local,
        resources=config.resources,
    )
