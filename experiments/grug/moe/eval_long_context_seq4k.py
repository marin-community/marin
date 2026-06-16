# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-context PPL sweep at seq=4k over every saved checkpoint of an EP=1 baseline.

For each (dim, checkpoint_step) pair, build the baseline ``GrugModelConfig`` at
seq=4096 (no YaRN -- the May Recipe baselines trained at seq=4096), load the
checkpoint at EP=1, and score the three raw long-document slices from
``experiments/evals/long_context_ppl.py`` (PG19, GovReport, QuALITY contexts).
Each document is truncated to 4096 tokens, so this measures how the base model
scores the opening 4k of each long-form distribution -- a distribution-shift
baseline that the 32k-context-extended evals can be compared against.

Results land in one wandb run per dim, with metrics keyed by checkpoint step.

The companion per-dim launchers (``eval_long_context_seq4k_d{dim}.py``) wire
the checkpoint list and call this module's ``run_long_context_seq4k_eval``.
"""

import dataclasses
import logging
import math
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.grug.moe.model import GrugModelConfig

logger = logging.getLogger(__name__)


_SEQ: int = 4096
_SLIDING_WINDOW: int = 2048
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"
_MAX_DOCS_PER_DATASET: int = 32
_BATCH_SIZE: int = 16
_EXPERT_PARALLEL: int = 1


@dataclass(frozen=True)
class LongContextSweepEvalConfig:
    dim: int
    run_id: str
    output_path: str
    resources: ResourceConfig
    checkpoint_steps: tuple[int, ...]
    checkpoint_dir: str
    seq_len: int
    batch_size: int
    expert_parallel: int
    tokenizer_id: str
    max_docs_per_dataset: int
    mp: str
    seed: int = 0


def _build_baseline_model_config(dim: int, seq_len: int) -> GrugModelConfig:
    """Baseline May Recipe model at the given dim/seq_len. No YaRN; sliding_window=2048."""
    return dataclasses.replace(
        MoeMuonHHeuristic().build_model_config(dim, seq_len=seq_len),
        sliding_window=_SLIDING_WINDOW,
    )


def _run_eval_local(config: LongContextSweepEvalConfig) -> None:
    """TPU-side entrypoint: build model once, iterate checkpoints, score datasets per ckpt."""
    import time
    from datetime import timedelta

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jmp
    import wandb
    from datasets import load_dataset
    from jax.sharding import NamedSharding, reshard
    from jax.sharding import PartitionSpec as P
    from levanter.callbacks.profiler import ProfilerConfig
    from levanter.checkpoint import CheckpointerConfig
    from levanter.grug.attention import AttentionMask
    from levanter.tracker.wandb import WandbConfig
    from levanter.trainer import TrainerConfig
    from levanter.utils.mesh import MeshConfig
    from marin.training.training import temporary_checkpoint_base_path
    from transformers import AutoTokenizer

    from experiments.evals.long_context_ppl import long_context_raw_validation_sets
    from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
    from experiments.grug.moe.model import Transformer
    from experiments.grug.moe.train import GrugTrainState, initial_state

    def _load_doc_tokens(ds_cfg, tokenizer, max_docs: int, seq_len: int) -> list[list[int]]:
        ds = load_dataset(
            ds_cfg.hf_dataset_id,
            ds_cfg.hf_dataset_name,
            revision=ds_cfg.hf_dataset_revision,
            split=ds_cfg.split,
            streaming=True,
        )
        out: list[list[int]] = []
        for row in ds:
            text = row.get(ds_cfg.text_key)
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 2:
                continue
            out.append(ids[:seq_len])
            if len(out) >= max_docs:
                break
        return out

    def _pad_batch(tok_lists, seq_len):
        B = len(tok_lists)
        tokens = jnp.zeros((B, seq_len), dtype=jnp.int32)
        weight = jnp.zeros((B, seq_len), dtype=jnp.float32)
        for i, ids in enumerate(tok_lists):
            L = min(len(ids), seq_len)
            tokens = tokens.at[i, :L].set(jnp.asarray(ids[:L], dtype=jnp.int32))
            weight = weight.at[i, :L].set(1.0)
        return tokens, weight

    # Trainer is initialized once for the whole sweep; per-checkpoint we just reload state.
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
            tags=["moe", "long_context_eval", "seq4k", f"d{config.dim}"],
            group="moe-may-long-context-seq4k",
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

    model_cfg = _build_baseline_model_config(config.dim, config.seq_len)
    h = MoeMuonHHeuristic()
    opt = h.build_muonh_config(config.batch_size, 1.0, config.dim, seq_len=config.seq_len).build(1)

    with trainer.use_device_mesh():

        @jax.jit
        def _init(key):
            return initial_state(model_cfg, optimizer=opt, mp=trainer.mp, key=key, ema_beta=None)

        state_template: GrugTrainState = _init(jax.random.PRNGKey(config.seed))

        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)

        batch_pspec = P(("data", "expert"), None)
        batch_sharding = NamedSharding(trainer.device_mesh, batch_pspec)

        @eqx.filter_jit
        def _score_batch(model: Transformer, tokens, weight):
            tokens = reshard(tokens, batch_sharding)
            weight = reshard(weight, batch_sharding)
            seg = reshard(jnp.zeros(tokens.shape, dtype=jnp.int32), batch_sharding)
            mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=(seg, seg))
            per_pos = model.next_token_loss(tokens, weight, mask=mask, reduction="none")
            return jnp.sum(per_pos * weight), jnp.sum(weight)

        # Tokenize each dataset once; reuse across every checkpoint.
        ds_cfgs = long_context_raw_validation_sets()
        cached_batches: dict[str, list] = {}
        for ds_name, ds_cfg in ds_cfgs.items():
            tok_lists = _load_doc_tokens(ds_cfg, tokenizer, config.max_docs_per_dataset, config.seq_len)
            batches = []
            for i in range(0, len(tok_lists), config.batch_size):
                doc_batch = tok_lists[i : i + config.batch_size]
                real_count = len(doc_batch)
                if real_count < config.batch_size:
                    doc_batch = doc_batch + [[tokenizer.pad_token_id or 0]] * (config.batch_size - real_count)
                tokens, weight = _pad_batch(doc_batch, config.seq_len)
                batches.append((tokens, weight))
            cached_batches[ds_name] = (batches, len(tok_lists))
            logger.info("[%s] tokenized %d docs into %d batches", ds_name, len(tok_lists), len(batches))

        for ckpt_step in config.checkpoint_steps:
            ckpt_path = os.path.join(config.checkpoint_dir, f"step-{ckpt_step}")
            t_load = time.perf_counter()
            loaded_state = restore_grug_state_from_checkpoint(
                state_template,
                checkpoint_search_paths=[ckpt_path],
                load_checkpoint_setting=True,
                mesh=trainer.device_mesh,
                allow_partial=False,
            )
            model: Transformer = loaded_state.params  # type: ignore[assignment]
            logger.info("[step=%d] checkpoint loaded in %.1fs", ckpt_step, time.perf_counter() - t_load)

            for ds_name, (batches, total_docs) in cached_batches.items():
                t_start = time.perf_counter()
                total_loss, total_tokens = 0.0, 0
                for tokens, weight in batches:
                    loss_sum, tok_count = _score_batch(model, tokens, weight)
                    total_loss += float(loss_sum)
                    total_tokens += int(tok_count)
                avg_loss = total_loss / max(total_tokens, 1)
                ppl = math.exp(min(avg_loss, 50.0))
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "[step=%d %s] docs=%d tokens=%d avg_loss=%.5f ppl=%.3f elapsed=%.1fs",
                    ckpt_step,
                    ds_name,
                    total_docs,
                    total_tokens,
                    avg_loss,
                    ppl,
                    elapsed,
                )
                wandb.log(
                    {
                        f"long_context/{ds_name}/loss": avg_loss,
                        f"long_context/{ds_name}/ppl": ppl,
                        f"long_context/{ds_name}/tokens": total_tokens,
                        f"long_context/{ds_name}/docs": total_docs,
                    },
                    step=ckpt_step,
                )

    wandb.finish()


def run_long_context_seq4k_eval(config: LongContextSweepEvalConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_eval_local,
        resources=config.resources,
    )
