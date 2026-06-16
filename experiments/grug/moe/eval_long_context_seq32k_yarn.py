# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-context PPL sweep at seq=32k with YaRN over every checkpoint of an EP=1 baseline.

Loads each saved checkpoint of a May Recipe EP=1 baseline (trained at seq=4k),
applies YaRN NTK-by-parts inv_freq rescaling on the full-attention layers
(``long_yarn_old_seq_len=4096``, ``yarn_alpha=1``, ``yarn_beta=32``,
``mscale_coef=0.1``), and scores the three raw long-document slices from
``experiments/evals/long_context_ppl.py`` (PG19, GovReport, QuALITY contexts)
at seq=32768. Sliding-window layers keep ``sliding_window=2048`` -- YaRN
applies only to the long-context (full-attention) layers.

This is the zero-shot YaRN extension eval: the baseline never trained beyond
seq=4k, so the model has to generalize to 32k via inv_freq rescaling alone.

Results land in one wandb run per dim, keyed by checkpoint step.
"""

import dataclasses
import logging
import math
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.data.text import LmDataConfig
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.heuristic_v2 import MoeMuonHHeuristic
from experiments.grug.moe.model import GrugModelConfig

logger = logging.getLogger(__name__)


_SEQ: int = 32_768
_SLIDING_WINDOW: int = 2048
_YARN_OLD_SEQ_LEN: int = 4096
_YARN_ALPHA: int = 1
_YARN_BETA: int = 32
_YARN_MSCALE_COEF: float = 0.1
_TOKENIZER_ID: str = "marin-community/marin-tokenizer"
_MAX_DOCS_PER_DATASET: int = 32
_BATCH_SIZE: int = 16
_EXPERT_PARALLEL: int = 1


@dataclass(frozen=True)
class LongContextYarnSweepEvalConfig:
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
    yarn_old_seq_len: int
    yarn_alpha: int
    yarn_beta: int
    yarn_mscale_coef: float
    # When ``data`` is set, the standard paloma+tagged-eval suite is run per checkpoint
    # alongside the long-context PPL slices. Pass through the same LmDataConfig the
    # training launcher uses; the executor resolves any embedded InputName references
    # before this config reaches the Fray TPU job.
    data: LmDataConfig | None = None
    # Eval batch size used specifically for the paloma TaggedEvaluator. Defaults to
    # ``batch_size`` (the long-context scoring batch size). Override to match the
    # training-time launcher's ``GrugEvalConfig.eval_batch_size=128`` for apples-to-apples
    # comparison against training-time paloma -- the eval reads ``bs * max_eval_batches``
    # sequences sequentially from the slice's flat token stream, so a smaller bs reads
    # less of each slice and can bias toward whatever sits at the front of the file.
    paloma_eval_batch_size: int | None = None
    # When set, supervised target-only PPL slices are scored per checkpoint. Each value's
    # ``input_path``/``hf_dataset_id`` is resolved by the executor (so staged JSONL refs from
    # ``qasper_staged`` / ``booksum_*_staged`` ExecutorSteps become concrete GCS paths before
    # the Fray TPU job runs). Loss is masked: weight=0 over the rendered prompt, weight=1
    # over the target tokens. Shift-by-one is handled so per-position weight=1 selects
    # *predicting* a target token.
    supervised_datasets: dict[str, RawTextEvaluationDataset] | None = None
    # When set, applies a two-stage YaRN: first rescales (yarn_old_seq_len -> yarn_prior_seq_len)
    # to reproduce the training-time inv_freq of an already-extended checkpoint, then further
    # rescales (yarn_prior_seq_len -> seq_len). Leave None for single-stage extension.
    yarn_prior_seq_len: int | None = None
    seed: int = 0


def _build_yarn_model_config(
    dim: int,
    seq_len: int,
    yarn_old_seq_len: int,
    yarn_alpha: int,
    yarn_beta: int,
    yarn_mscale_coef: float,
    yarn_prior_seq_len: int | None = None,
) -> GrugModelConfig:
    """May Recipe model at given dim/seq, with YaRN applied on full-attention layers.

    Sliding-window layers (``sliding_window=2048``) are unchanged -- YaRN applies
    only to layers that see beyond the original training window. ``long_qk_mult``
    folds the YaRN m-scale into the attention-logit scaling on those layers.
    The m-scale uses the total compression ratio ``seq_len / yarn_old_seq_len`` --
    correct for both single-stage and two-stage rescaling, since the total
    inv_freq compression factor is the product across stages = old/seq_len.
    """
    from levanter.grug.attention import RotaryConfig

    base = dataclasses.replace(
        MoeMuonHHeuristic().build_model_config(dim, seq_len=seq_len),
        sliding_window=_SLIDING_WINDOW,
    )
    long_mscale = yarn_mscale_coef * math.log(seq_len / yarn_old_seq_len) + 1.0
    return dataclasses.replace(
        base,
        long_yarn_old_seq_len=yarn_old_seq_len,
        long_yarn_prior_seq_len=yarn_prior_seq_len,
        long_qk_mult=base.qk_mult * long_mscale,
        yarn_alpha=yarn_alpha,
        yarn_beta=yarn_beta,
        rope=RotaryConfig(theta=10000.0),
    )


def _run_eval_local(config: LongContextYarnSweepEvalConfig) -> None:
    """TPU-side entrypoint: build YaRN-extended model once, iterate checkpoints."""
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
    from levanter.eval import eval_model
    from levanter.grug.attention import AttentionMask
    from levanter.tracker.wandb import WandbConfig
    from levanter.trainer import TrainerConfig
    from levanter.utils.mesh import MeshConfig
    from marin.training.training import temporary_checkpoint_base_path
    from transformers import AutoTokenizer

    from experiments.evals.long_context_ppl import long_context_raw_validation_sets
    from experiments.grug.checkpointing import restore_grug_state_from_checkpoint
    from experiments.grug.moe.model import Transformer
    from experiments.grug.moe.train import GrugEvalConfig, GrugTrainState, build_tagged_evaluator, initial_state

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

    def _load_supervised_rows(ds_cfg, tokenizer, max_docs: int, seq_len: int) -> list[tuple[list[int], list[int]]]:
        """Return (input_ids, target_ids) per row. Input truncated from the LEFT if needed."""
        import gzip
        import json

        import fsspec

        input_key = ds_cfg.input_key or "input"
        target_key = ds_cfg.target_key or "target"

        def _row_iter():
            if ds_cfg.input_path is not None:
                with fsspec.open(ds_cfg.input_path, "rb") as raw:
                    decoded = gzip.GzipFile(fileobj=raw) if str(ds_cfg.input_path).endswith(".gz") else raw
                    for line in decoded:
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        if not line.strip():
                            continue
                        yield json.loads(line)
            else:
                hf_ds = load_dataset(
                    ds_cfg.hf_dataset_id,
                    ds_cfg.hf_dataset_name,
                    revision=ds_cfg.hf_dataset_revision,
                    split=ds_cfg.split,
                    streaming=True,
                )
                yield from hf_ds

        out: list[tuple[list[int], list[int]]] = []
        for row in _row_iter():
            input_text = row.get(input_key)
            target_text = row.get(target_key)
            if not isinstance(input_text, str) or not isinstance(target_text, str):
                continue
            input_text = input_text.strip()
            target_text = target_text.strip()
            if not target_text:
                continue
            input_ids = tokenizer.encode(input_text, add_special_tokens=False)
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)
            if len(target_ids) < 1:
                continue
            # Truncate target if it alone exceeds seq_len; otherwise drop oldest input tokens
            # so input+target fits. Target stays at the end so its loss positions are intact.
            if len(target_ids) >= seq_len:
                target_ids = target_ids[: seq_len - 1]
                input_ids = []
            else:
                max_input = seq_len - len(target_ids)
                if len(input_ids) > max_input:
                    input_ids = input_ids[-max_input:]
            out.append((input_ids, target_ids))
            if len(out) >= max_docs:
                break
        return out

    def _pad_supervised_batch(rows, seq_len, pad_token_id):
        """Pack (input, target) pairs into a batch. weight=1 at positions whose NEXT-token is target."""
        B = len(rows)
        tokens = jnp.zeros((B, seq_len), dtype=jnp.int32)
        weight = jnp.zeros((B, seq_len), dtype=jnp.float32)
        target_token_count = 0
        for i, (input_ids, target_ids) in enumerate(rows):
            li, lt = len(input_ids), len(target_ids)
            combined = list(input_ids) + list(target_ids)
            tokens = tokens.at[i, : li + lt].set(jnp.asarray(combined, dtype=jnp.int32))
            # next_token_loss: per_pos[j] = loss(logits[j], tokens[j+1]). To weight the loss
            # over target tokens (at absolute positions [li, li+lt)), we set weight at
            # positions [li-1, li+lt-1) -- length lt. Edge case li=0: start at 0.
            start = max(0, li - 1)
            end = li + lt - 1
            n = max(0, end - start)
            if n > 0:
                weight = weight.at[i, start:end].set(1.0)
            target_token_count += n
        return tokens, weight, target_token_count

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
            tags=["moe", "long_context_eval", "seq32k", "yarn", f"d{config.dim}"],
            group="moe-may-long-context-seq32k-yarn",
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

    model_cfg = _build_yarn_model_config(
        config.dim,
        config.seq_len,
        config.yarn_old_seq_len,
        config.yarn_alpha,
        config.yarn_beta,
        config.yarn_mscale_coef,
        yarn_prior_seq_len=config.yarn_prior_seq_len,
    )
    h = MoeMuonHHeuristic()
    opt = h.build_muonh_config(config.batch_size, 1.0, config.dim, seq_len=config.seq_len).build(1)

    with trainer.use_device_mesh():

        @jax.jit
        def _init(key):
            return initial_state(model_cfg, optimizer=opt, mp=trainer.mp, key=key, ema_beta=None)

        state_template: GrugTrainState = _init(jax.random.PRNGKey(config.seed))

        if config.data is not None:
            paloma_bs = config.paloma_eval_batch_size or config.batch_size
            paloma_eval_cfg = GrugEvalConfig(
                eval_batch_size=paloma_bs,
                steps_per_eval=1,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
            paloma_evaluator = build_tagged_evaluator(
                data_config=config.data,
                max_seq_len=config.seq_len,
                mesh=trainer.device_mesh,
                eval_cfg=paloma_eval_cfg,
            )
        else:
            paloma_evaluator = None

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

        cached_supervised: dict[str, tuple[list, int, int]] = {}
        if config.supervised_datasets:
            pad_id = tokenizer.pad_token_id or 0
            for ds_name, ds_cfg in config.supervised_datasets.items():
                rows = _load_supervised_rows(ds_cfg, tokenizer, config.max_docs_per_dataset, config.seq_len)
                batches = []
                total_target_tokens = 0
                for i in range(0, len(rows), config.batch_size):
                    row_batch = rows[i : i + config.batch_size]
                    real_count = len(row_batch)
                    if real_count < config.batch_size:
                        row_batch = row_batch + [([pad_id], [pad_id])] * (config.batch_size - real_count)
                    tokens, weight, target_tokens = _pad_supervised_batch(row_batch, config.seq_len, pad_id)
                    batches.append((tokens, weight))
                    total_target_tokens += target_tokens
                cached_supervised[ds_name] = (batches, len(rows), total_target_tokens)
                logger.info(
                    "[%s supervised] rows=%d batches=%d target_tokens=%d",
                    ds_name,
                    len(rows),
                    len(batches),
                    total_target_tokens,
                )

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

            if paloma_evaluator is not None:
                t_paloma = time.perf_counter()
                paloma_log = eval_model(paloma_evaluator, model, prefix="eval")
                logger.info(
                    "[step=%d paloma] macro_loss=%.5f macro_bpb=%.5f elapsed=%.1fs",
                    ckpt_step,
                    paloma_log.get("eval/paloma/macro_loss", float("nan")),
                    paloma_log.get("eval/paloma/macro_bpb", float("nan")),
                    time.perf_counter() - t_paloma,
                )
                wandb.log(paloma_log, step=ckpt_step)

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

            for ds_name, (batches, total_rows, _target_tokens_expected) in cached_supervised.items():
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
                    "[step=%d %s supervised] rows=%d target_tokens=%d avg_loss=%.5f ppl=%.3f elapsed=%.1fs",
                    ckpt_step,
                    ds_name,
                    total_rows,
                    total_tokens,
                    avg_loss,
                    ppl,
                    elapsed,
                )
                wandb.log(
                    {
                        f"supervised/{ds_name}/loss": avg_loss,
                        f"supervised/{ds_name}/ppl": ppl,
                        f"supervised/{ds_name}/target_tokens": total_tokens,
                        f"supervised/{ds_name}/rows": total_rows,
                    },
                    step=ckpt_step,
                )

    wandb.finish()


def run_long_context_seq32k_yarn_eval(config: LongContextYarnSweepEvalConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_eval_local,
        resources=config.resources,
    )
