# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from tqdm_loggable.auto import tqdm

import haliax as hax
from haliax.partitioning import ResourceMapping, set_mesh, named_jit

import levanter
from levanter.data.splice_dataset import SpliceMultiDocumentLMConfig
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.models.loss import next_token_loss
from levanter.utils.hf_utils import HfTokenizer
from levanter.tracker.histogram import Histogram


@dataclass
class PzMultiDocConfig:
    """Configuration for multi-document P(z) evaluation.

    Evaluates sliding windows for each selected document (as per the multi-doc config).
    """

    chunk_size: int = 100
    prompt_tokens: Optional[int] = None
    cursor_inc_tokens: int = 5
    eval_batch_size: int = 64
    verbose: bool = False
    max_docs: Optional[int] = None  # if set, limit to first K selected docs


def _eval_windows_for_doc(
    model: LmHeadModel,
    tokenizer: HfTokenizer,
    tokens_1d: jnp.ndarray,
    cfg: PzMultiDocConfig,
    axis_resources: ResourceMapping,
    *,
    device_mesh=None,
    mp=None,
):
    from levanter.books.util import chunk_token_ids_to_sliding_windows

    N = int(cfg.chunk_size)
    P = int(cfg.prompt_tokens if cfg.prompt_tokens is not None else N // 2)
    S = max(1, int(cfg.cursor_inc_tokens))

    # Time window creation
    window_start = time.perf_counter()
    token_list = list(map(int, jax.device_get(tokens_1d)))
    chunks = chunk_token_ids_to_sliding_windows(token_list, tokenizer, chunk_size=N, cursor_inc=S)
    window_creation_time = time.perf_counter() - window_start
    
    if len(chunks) == 0:
        timing_info = {"window_creation_time": window_creation_time, "num_windows": 0, "inference_time": 0.0}
        return None, timing_info

    Pos = model.Pos.resize(N)
    tokens_named_list = [hax.named(jnp.asarray(c["input_ids"], dtype=jnp.int32), Pos) for c in chunks]

    # Pre-cast model to compute dtype once (optimization #4)
    # Cast once before evaluation instead of casting inside _single_logprob for every example
    model_compute = mp.cast_to_compute(model) if mp else model

    def _single_logprob(mod: LmHeadModel, pos_axis: hax.Axis, prompt_len: int, tokens_1d_named):
        ex = LmExample.from_prompt_and_completion(pos_axis, tokens_1d_named, prompt_length=prompt_len)
        logits = mod(ex.tokens, attn_mask=ex.attn_mask)
        logits = logits.astype(jnp.float32)
        nll = next_token_loss(Pos=pos_axis, Vocab=mod.Vocab, logits=logits, true_ids=ex.tokens, loss_mask=ex.loss_mask, reduction=None)
        total_nll = hax.sum(nll, axis=pos_axis).array
        return -total_nll

    def _vmapped(mod: LmHeadModel, pos_axis: hax.Axis, prompt_len: int, tokens_2d_named):
        return hax.vmap(lambda t: _single_logprob(mod, pos_axis, prompt_len, t), tokens_2d_named.axes[0])(tokens_2d_named)

    total_windows = len(tokens_named_list)
    B = int(max(1, cfg.eval_batch_size))

    def _eval_chunk(mod: LmHeadModel, pos_axis: hax.Axis, prompt_len: int, chunk_tokens_2d):
        lp_chunk = _vmapped(mod, pos_axis, prompt_len, chunk_tokens_2d)
        return hax.exp(lp_chunk)

    # JIT compile _eval_chunk for reuse across batches (optimization #1)
    # This avoids recompilation overhead on every batch
    # Model must be passed as argument, not captured from closure, to avoid "closing over sharded arrays" error
    _eval_chunk_jit = named_jit(
        _eval_chunk,
        axis_resources=axis_resources,
        out_axis_resources=None,  # Output is extracted as .array anyway
    )

    # Time model inference
    inference_start = time.perf_counter()
    pz_chunks: list[jnp.ndarray] = []
    num_batches = (total_windows + B - 1) // B
    pbar = tqdm(range(0, total_windows, B), desc="P(z) multi-doc eval", total=num_batches, disable=not cfg.verbose, leave=False)
    for start in pbar:
        end = min(total_windows, start + B)
        sub_named_list = tokens_named_list[start:end]
        sub_Batch = hax.Axis("batch", len(sub_named_list))
        sub_tokens_2d = hax.stack(sub_Batch, sub_named_list)
        if device_mesh is not None:
            with set_mesh(device_mesh), hax.axis_mapping(axis_resources):
                pz_chunk = _eval_chunk_jit(model_compute, Pos, P, sub_tokens_2d)
        else:
            with hax.axis_mapping(axis_resources):
                pz_chunk = _eval_chunk_jit(model_compute, Pos, P, sub_tokens_2d)
        pz_chunks.append(pz_chunk.array if hasattr(pz_chunk, "array") else pz_chunk)
    inference_time = time.perf_counter() - inference_start

    pz_all = jnp.concatenate(pz_chunks, axis=0) if pz_chunks else jnp.zeros((0,), dtype=jnp.float32)
    timing_info = {
        "window_creation_time": window_creation_time,
        "num_windows": total_windows,
        "inference_time": inference_time,
    }
    return pz_all, timing_info


def pz_multi_doc_callback(
    cfg: PzMultiDocConfig,
    tokenizer: HfTokenizer,
    axis_resources: ResourceMapping,
    mp,
    data_config: SpliceMultiDocumentLMConfig,
    *,
    device_mesh=None,
):
    """Return a training hook that computes P(z) across the selected documents.

    If the data config is not SpliceMultiDocumentLMConfig, the callback is a no-op.
    """

    # Cache tokens for selected docs on first invocation
    doc_tokens_list: Optional[Sequence[jnp.ndarray]] = None
    doc_indices: Optional[Sequence[int]] = None

    def _maybe_get_docs(model_Pos):
        nonlocal doc_tokens_list
        nonlocal doc_indices
        if doc_tokens_list is not None:
            return doc_tokens_list, doc_indices
        if not isinstance(data_config, SpliceMultiDocumentLMConfig):
            return None, None
        try:
            caches = data_config.build_caches("train", monitors=False)
            # Deterministically re-run selection to recover token arrays
            docs_np, idxs = data_config._select_multiple_docs(caches, data_config.num_docs)  # type: ignore[attr-defined]
            # Convert to JAX arrays
            doc_tokens_list = [jnp.asarray(arr, dtype=jnp.int32) for arr in docs_np]
            doc_indices = idxs
            return doc_tokens_list, doc_indices
        except Exception:
            return None, None

    def cb(step, force: bool = False):
        if step.step == 0 and not force:
            return
        if not isinstance(data_config, SpliceMultiDocumentLMConfig):
            return

        # Time total callback duration
        callback_start = time.perf_counter()
        model = step.eval_model
        docs, idxs = _maybe_get_docs(model.Pos)
        if not docs:
            return

        limit = len(docs)
        if cfg.max_docs is not None:
            limit = min(limit, int(cfg.max_docs))

        # Track timing across all documents
        total_window_creation_time = 0.0
        total_inference_time = 0.0
        total_doc_eval_time = 0.0

        # Evaluate per document and log simple summaries
        all_vals = []
        for k in range(limit):
            doc_start = time.perf_counter()
            tokens_1d = docs[k]
            result = _eval_windows_for_doc(model, tokenizer, tokens_1d, cfg, axis_resources, device_mesh=device_mesh, mp=mp)
            
            # Handle tuple return (pz_vals, timing_info)
            if isinstance(result, tuple) and len(result) == 2:
                pz_vals, timing_info = result
            else:
                # Shouldn't happen with new code, but handle gracefully
                pz_vals = result
                timing_info = {}
            
            if pz_vals is None or (hasattr(pz_vals, "size") and pz_vals.size == 0):
                continue
                
            doc_eval_time = time.perf_counter() - doc_start
            total_doc_eval_time += doc_eval_time
            total_window_creation_time += timing_info.get("window_creation_time", 0.0)
            total_inference_time += timing_info.get("inference_time", 0.0)
            
            all_vals.append(pz_vals)
            mean_pz = float(jnp.mean(pz_vals))
            median_pz = float(jnp.median(pz_vals))
            max_pz = float(jnp.max(pz_vals))
            metrics = {
                f"pz_multi/doc_{k}/mean": mean_pz,
                f"pz_multi/doc_{k}/median": median_pz,
                f"pz_multi/doc_{k}/max": max_pz,
                f"pz_multi/doc_{k}/eval_time_sec": doc_eval_time,
                f"pz_multi/doc_{k}/window_creation_time_sec": timing_info.get("window_creation_time", 0.0),
                f"pz_multi/doc_{k}/inference_time_sec": timing_info.get("inference_time", 0.0),
                f"pz_multi/doc_{k}/num_windows": timing_info.get("num_windows", 0),
            }
            levanter.tracker.log(metrics, step=int(step.step))
            # Per-document histogram
            try:
                hist = Histogram.from_array(pz_vals, num_bins=31)
                levanter.tracker.log({f"pz_multi/doc_{k}/hist": hist}, step=int(step.step))
            except Exception:
                pass

        # Combined histogram across all docs
        if len(all_vals) > 0:
            try:
                cat = jnp.concatenate(all_vals, axis=0)
                hist_all = Histogram.from_array(cat, num_bins=41)
                levanter.tracker.log({"pz_multi/all/hist": hist_all}, step=int(step.step))
            except Exception:
                pass

        # Log total timing metrics
        callback_total_time = time.perf_counter() - callback_start
        timing_metrics = {
            "pz_multi/total_callback_time_sec": callback_total_time,
            "pz_multi/total_window_creation_time_sec": total_window_creation_time,
            "pz_multi/total_inference_time_sec": total_inference_time,
            "pz_multi/total_doc_eval_time_sec": total_doc_eval_time,
            "pz_multi/num_docs_evaluated": limit,
        }
        levanter.tracker.log(timing_metrics, step=int(step.step))

    return cb
