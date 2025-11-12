# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import re
import json
import time
import fsspec
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import tempfile
from tqdm_loggable.auto import tqdm

import haliax as hax
from haliax.partitioning import ResourceMapping, set_mesh

import levanter
from levanter.books.util import chunk_token_ids_to_sliding_windows, create_bar_plot
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfigBase
from levanter.data.splice_dataset import SpliceSingleDocumentLMConfig
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.models.loss import next_token_loss
from levanter.utils.hf_utils import HfTokenizer
from levanter.tracker.histogram import Histogram


@dataclass
class PzSingleDocConfig:
    """Configuration for single-document P(z) evaluation with 50/50 splits.

    Evaluates sliding windows of length ``chunk_size`` across one document. The
    prompt length defaults to half the chunk size.
    """

    chunk_size: int = 100
    prompt_tokens: Optional[int] = None
    cursor_inc_tokens: int = 5
    eval_batch_size: int = 64
    verbose: bool = False
    # Optional: also write the selected doc preview to a GCS path prefix (e.g., gs://bucket/path)
    gcp_prefix: Optional[str] = None
    # If true, also write a JSONL record to the same prefix for data browsers that expect JSONL
    gcp_jsonl: bool = False


def pz_single_doc_callback(
    cfg: PzSingleDocConfig,
    tokenizer: HfTokenizer,
    axis_resources: ResourceMapping,
    mp,
    data_config: Union[LMMixtureDatasetConfig, SingleDatasetLMConfigBase, SpliceSingleDocumentLMConfig],
    *,
    device_mesh=None,
):
    """Return a training hook that computes P(z) for a single document.

    This expects the training ``data_config`` to be a ``SpliceSingleDocumentLMConfig`` so the
    exact document used in training can be retrieved from the cache. If not, the callback
    becomes a no-op.
    """

    # Persistent state across invocations
    doc_tokens_state: Optional[jnp.ndarray] = None
    doc_info_state: Optional[dict] = None  # dataset_name, doc_index, length
    _doc_logged_once: bool = False
    if cfg.verbose:
        try:
            print(
                f"[pz_single_doc] configured: chunk_size={cfg.chunk_size} prompt_tokens={cfg.prompt_tokens or cfg.chunk_size // 2} "
                f"cursor_inc_tokens={cfg.cursor_inc_tokens} eval_batch_size={cfg.eval_batch_size}",
                flush=True,
            )
        except Exception:
            pass

    def _maybe_get_doc_tokens(model_Pos) -> Optional[jnp.ndarray]:
        nonlocal doc_tokens_state
        nonlocal doc_info_state
        if doc_tokens_state is not None:
            if cfg.verbose:
                print(f"[pz_single_doc] using cached doc tokens (len={int(doc_tokens_state.shape[0])})", flush=True)
            return doc_tokens_state
        if not isinstance(data_config, SpliceSingleDocumentLMConfig):
            if cfg.verbose:
                print(
                    f"[pz_single_doc] data_config is not SpliceSingleDocumentLMConfig (got {type(data_config).__name__}); skipping",
                    flush=True,
                )
            return None
        try:
            caches = data_config.build_caches("train", monitors=False)
            # Prefer index-aware selector if available (so we can report the true doc index)
            if hasattr(data_config, "_select_doc_tokens_and_index"):
                arr, idx = data_config._select_doc_tokens_and_index(caches, model_Pos)  # type: ignore[attr-defined]
            else:
                arr = data_config._select_doc_tokens(caches, model_Pos)
                idx = getattr(data_config, "doc_index", None)
            # Convert to JAX array without NumPy
            doc = jnp.asarray(arr, dtype=jnp.int32)
            doc_tokens_state = doc
            doc_info_state = {
                "dataset_name": getattr(data_config, "dataset_name", None),
                "doc_index": int(idx) if idx is not None else -1,
                "length": int(doc.shape[0]),
            }
            if cfg.verbose:
                print(
                    f"[pz_single_doc] loaded training doc: dataset_name={doc_info_state['dataset_name']} doc_index={doc_info_state['doc_index']} len={doc_info_state['length']}",
                    flush=True,
                )
            return doc
        except Exception as e:
            if cfg.verbose:
                print(f"[pz1] failed to read single doc tokens: {e}", flush=True)
            return None

    def _maybe_log_doc_preview(tokens_1d: jnp.ndarray):
        nonlocal _doc_logged_once
        if _doc_logged_once:
            return
        try:
            # Decode a bounded preview for readability
            max_preview = int(min(int(tokens_1d.shape[0]), 2048))
            preview_text = tokenizer.decode(list(map(int, jax.device_get(tokens_1d[:max_preview]))))
            header_lines = [
                "[pz_single_doc] Selected Training Document",
                f"dataset_name: {doc_info_state.get('dataset_name') if doc_info_state else None}",
                f"doc_index: {doc_info_state.get('doc_index') if doc_info_state else None}",
                f"doc_len: {doc_info_state.get('length') if doc_info_state else int(tokens_1d.shape[0])}",
                f"preview_tokens: {max_preview}",
                "",
                "===== BEGIN PREVIEW =====",
            ]
            content = "\n".join(header_lines) + "\n" + preview_text + "\n===== END PREVIEW =====\n"

            import tempfile
            from levanter import tracker as _trk

            with tempfile.NamedTemporaryFile(prefix="pz_single_doc_", suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
                f.write(content)
                tmp_path = f.name

            art_name = "pz_single_doc.txt"
            if doc_info_state and doc_info_state.get("dataset_name") is not None:
                raw_ds = str(doc_info_state["dataset_name"])  # may contain slashes
                safe_ds = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_ds)
                art_name = f"pz_single_{safe_ds}_{doc_info_state.get('doc_index', -1)}.txt"
            _trk.current_tracker().log_artifact(tmp_path, name=art_name, type="pz_single_doc")

            # If a GCP prefix is provided, write both JSONL record and full text
            if cfg.gcp_prefix:
                try:
                    safe_ds = re.sub(r"[^A-Za-z0-9._-]+", "_", str(doc_info_state.get("dataset_name", "doc")))

                    # Write JSONL record with preview
                    record = {
                        "tag": "pz_single_doc",
                        "dataset_name": doc_info_state.get("dataset_name"),
                        "doc_index": doc_info_state.get("doc_index"),
                        "doc_len": doc_info_state.get("length"),
                        "preview_tokens": max_preview,
                        "preview_text": preview_text,
                    }
                    jsonl_name = f"pz_single_{safe_ds}_{doc_info_state.get('doc_index', -1)}.jsonl"
                    jsonl_path = cfg.gcp_prefix.rstrip("/") + "/" + jsonl_name
                    with fsspec.open(jsonl_path, "w", compression="infer", encoding="utf-8") as gj:
                        gj.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if cfg.verbose:
                        print(f"[pz_single_doc] wrote JSONL to {jsonl_path}", flush=True)

                    # Write full document text to separate .txt file
                    full_text = tokenizer.decode(list(map(int, jax.device_get(tokens_1d))))
                    txt_name = f"pz_single_{safe_ds}_{doc_info_state.get('doc_index', -1)}_full.txt"
                    txt_path = cfg.gcp_prefix.rstrip("/") + "/" + txt_name
                    with fsspec.open(txt_path, "w", compression="infer", encoding="utf-8") as gf:
                        gf.write(full_text)
                    if cfg.verbose:
                        print(f"[pz_single_doc] wrote full text ({len(full_text)} chars) to {txt_path}", flush=True)

                except Exception as ge:
                    if cfg.verbose:
                        print(f"[pz_single_doc] failed to write GCS files: {ge}", flush=True)

            # Short console preview (optional)
            if cfg.verbose:
                short_preview = preview_text[:512]
                print(f"[pz_single_doc] doc={doc_info_state} preview(512ch):\n{short_preview}", flush=True)
            _doc_logged_once = True
        except Exception as e:
            if cfg.verbose:
                print(f"[pz_single_doc] failed to log/print doc preview: {e}", flush=True)

    def _eval_windows(model: LmHeadModel, tokens_1d: jnp.ndarray):
        # Build sliding windows of fixed length N
        N = int(cfg.chunk_size)
        P = int(cfg.prompt_tokens if cfg.prompt_tokens is not None else N // 2)
        S = max(1, int(cfg.cursor_inc_tokens))
        # Use a utility to generate sliding windows over token ids
        token_list = list(map(int, jax.device_get(tokens_1d)))
        chunks = chunk_token_ids_to_sliding_windows(token_list, tokenizer, chunk_size=N, cursor_inc=S)
        if len(chunks) == 0:
            if cfg.verbose:
                print("[pz_single_doc] no sliding windows (doc shorter than chunk)", flush=True)
            return None

        Pos = model.Pos.resize(N)
        tokens_named_list = [hax.named(jnp.asarray(c["input_ids"], dtype=jnp.int32), Pos) for c in chunks]

        def _single_logprob(tokens_1d_named):
            ex = LmExample.from_prompt_and_completion(Pos, tokens_1d_named, prompt_length=P)
            m = model
            if mp is not None:
                m = mp.cast_to_compute(m)
            logits = m(ex.tokens, attn_mask=ex.attn_mask)
            logits = logits.astype(jnp.float32)
            nll = next_token_loss(Pos=Pos, Vocab=m.Vocab, logits=logits, true_ids=ex.tokens, loss_mask=ex.loss_mask, reduction=None)
            total_nll = hax.sum(nll, axis=Pos).array
            return -total_nll

        def _vmapped(tokens_2d_named):
            # Use the actual batch axis on this tensor (may differ per chunk)
            return hax.vmap(_single_logprob, tokens_2d_named.axes[0])(tokens_2d_named)

        # Evaluate in batches and collect all pz values as JAX arrays
        total_windows = len(tokens_named_list)
        B = int(max(1, cfg.eval_batch_size))

        def _eval_chunk(chunk_tokens_2d):
            lp_chunk = _vmapped(chunk_tokens_2d)
            return hax.exp(lp_chunk)  # NamedArray or JAX array depending on vmapped output

        pz_chunks: list[jnp.ndarray] = []
        num_batches = (total_windows + B - 1) // B
        pbar = tqdm(
            range(0, total_windows, B),
            desc="P(z) eval batches",
            total=num_batches,
            disable=not cfg.verbose,
            leave=False,
        )
        for start in pbar:
            end = min(total_windows, start + B)
            sub_named_list = tokens_named_list[start:end]
            sub_Batch = hax.Axis("batch", len(sub_named_list))
            sub_tokens_2d = hax.stack(sub_Batch, sub_named_list)
            if device_mesh is not None:
                with set_mesh(device_mesh), hax.axis_mapping(axis_resources):
                    pz_chunk = _eval_chunk(sub_tokens_2d)
            else:
                with hax.axis_mapping(axis_resources):
                    pz_chunk = _eval_chunk(sub_tokens_2d)
            # Collect as plain JAX arrays regardless of NamedArray or ndarray
            pz_chunks.append(pz_chunk.array if hasattr(pz_chunk, "array") else pz_chunk)

        pz_all = jnp.concatenate(pz_chunks, axis=0) if pz_chunks else jnp.zeros((0,), dtype=jnp.float32)
        mean_pz_val = jnp.mean(pz_all) if pz_chunks else jnp.array(0.0)
        median_pz_val = jnp.median(pz_all) if pz_chunks else jnp.array(0.0)
        max_pz_val = jnp.max(pz_all) if pz_chunks else jnp.array(0.0)
        return {
            "num_windows": int(total_windows),
            "chunk_size": int(N),
            "prompt_tokens": int(P),
            "suffix_tokens": int(N - P),
            "mean_pz": float(mean_pz_val),
            "median_pz": float(median_pz_val),
            "max_pz": float(max_pz_val),
        }

    def cb(step, force: bool = False):
        # Skip at step 0 unless forced (to avoid counting uninitialized runs)
        if step.step == 0 and not force:
            return

        model = step.eval_model
        if cfg.verbose:
            print(f"[pz_single_doc] entering callback at step={int(step.step)}", flush=True)
        doc = _maybe_get_doc_tokens(model.Pos)
        if doc is None:
            # silently no-op if we can't resolve the training doc
            if cfg.verbose:
                print("[pz_single_doc] no training doc resolved; exiting", flush=True)
            return

        # Log/print the selected document once for visibility
        _maybe_log_doc_preview(doc)
        start_time = time.time()
        metrics = _eval_windows(model, doc)
        eval_time = time.time() - start_time
        if metrics is None:
            if cfg.verbose:
                print("[pz_single_doc] no metrics (likely no windows)", flush=True)
            return

        # Log under a dedicated namespace
        to_log = {f"pz_single/{k}": v for k, v in metrics.items()}
        to_log["pz_single/doc_len"] = int(doc.shape[0])
        to_log["pz_single/eval_time_sec"] = eval_time
        levanter.tracker.log(to_log, step=int(step.step))

        # Log histogram of P(z) values using the tracker Histogram
        # Recompute full P(z) array cheaply: regenerate windows then eval in batches
        N = int(cfg.chunk_size)
        S = max(1, int(cfg.cursor_inc_tokens))
        token_list = list(map(int, jax.device_get(doc)))
        chunks = chunk_token_ids_to_sliding_windows(token_list, tokenizer, chunk_size=N, cursor_inc=S)
        if chunks:
            Pos = model.Pos.resize(N)
            tokens_named_list = [hax.named(jnp.asarray(c["input_ids"], dtype=jnp.int32), Pos) for c in chunks]
            total_windows = len(tokens_named_list)
            B = int(max(1, cfg.eval_batch_size))

            def _single_logprob(tokens_1d_named):
                ex = LmExample.from_prompt_and_completion(Pos, tokens_1d_named, prompt_length=int(cfg.prompt_tokens or N // 2))
                m = model
                if mp is not None:
                    m = mp.cast_to_compute(m)
                logits = m(ex.tokens, attn_mask=ex.attn_mask)
                logits = logits.astype(jnp.float32)
                nll = next_token_loss(Pos=Pos, Vocab=m.Vocab, logits=logits, true_ids=ex.tokens, loss_mask=ex.loss_mask, reduction=None)
                total_nll = hax.sum(nll, axis=Pos).array
                return -total_nll

            BatchTmp = hax.Axis("batch_tmp", 1)  # placeholder; we restack per chunk below
            def _vmapped(tokens_2d_named):
                return hax.vmap(_single_logprob, tokens_2d_named.axes[0])(tokens_2d_named)

            pz_chunks: list[jnp.ndarray] = []
            num_batches = (total_windows + B - 1) // B
            pbar = tqdm(
                range(0, total_windows, B),
                desc="P(z) histogram batches",
                total=num_batches,
                disable=not cfg.verbose,
                leave=False,
            )
            for start in pbar:
                end = min(total_windows, start + B)
                sub_named_list = tokens_named_list[start:end]
                sub_Batch = hax.Axis("batch", len(sub_named_list))
                sub_tokens_2d = hax.stack(sub_Batch, sub_named_list)
                if device_mesh is not None:
                    with set_mesh(device_mesh), hax.axis_mapping(axis_resources):
                        lp_chunk = _vmapped(sub_tokens_2d)
                else:
                    with hax.axis_mapping(axis_resources):
                        lp_chunk = _vmapped(sub_tokens_2d)
                _pz = hax.exp(lp_chunk)
                pz_chunks.append(_pz.array if hasattr(_pz, "array") else _pz)

            pz_all = jnp.concatenate(pz_chunks, axis=0) if pz_chunks else jnp.zeros((0,), dtype=jnp.float32)
            # Restore original W&B histogram logging for window p(z)
            try:
                hist = Histogram.from_array(pz_all, num_bins=31)
                levanter.tracker.log({"pz_single/pz_hist": hist}, step=int(step.step))
                if cfg.verbose:
                    print(
                        f"[pz_single_doc] logged histogram: num={int(hist.num)} min={float(hist.min)} max={float(hist.max)}",
                        flush=True,
                    )
            except Exception as e:
                if cfg.verbose:
                    print(f"[pz_single_doc] failed to log W&B histogram: {e}", flush=True)

            # --- Additional: Save per-character max-P(z) bar plot PNG to GCS (no tracker logging) ---
            try:
                # Compute token->character span mapping by decoding each token piece
                token_ids_list = list(map(int, jax.device_get(doc)))
                pieces = [tokenizer.decode([tid]) for tid in tqdm(token_ids_list, desc="Decoding tokens", disable=not cfg.verbose, leave=False)]
                token_char_start = np.zeros(len(pieces), dtype=np.int64)
                token_char_end = np.zeros(len(pieces), dtype=np.int64)
                offset = 0
                for i, s in enumerate(pieces):
                    token_char_start[i] = offset
                    offset += len(s)
                    token_char_end[i] = offset
                doc_char_len = int(offset)

                # Build per-character max P(z) across windows
                max_per_char = np.zeros(doc_char_len, dtype=np.float32)
                pz_vals = np.asarray(jax.device_get(pz_all), dtype=np.float32)
                for pz_val, ch in zip(pz_vals, chunks):
                    s_tok = int(ch["start_token"])
                    e_tok = int(ch["end_token"])  # inclusive
                    start_char = int(token_char_start[s_tok])
                    end_char_excl = int(token_char_end[e_tok])
                    if end_char_excl > start_char:
                        cur = max_per_char[start_char:end_char_excl]
                        if cur.size:
                            np.maximum(cur, pz_val, out=cur)

                # Save bar plot image using existing util helper; title uses doc id
                if cfg.gcp_prefix:
                    doc_id = None
                    if doc_info_state is not None:
                        doc_id = doc_info_state.get("doc_index", None)
                    title = f"{doc_id}" if doc_id is not None else "doc"
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
                        tmp_path = tmp_png.name
                    try:
                        # Use create_bar_plot to render and save
                        create_bar_plot(
                            max_per_char,
                            save_path=tmp_path,
                            title=str(title),
                            xlabel="Book position (character)",
                            vmin=0.0,
                            vmax=1.0,
                        )
                        # Copy to GCS with step index in filename
                        step_idx = int(step.step)
                        safe_prefix = cfg.gcp_prefix.rstrip("/")
                        out_name = f"pz_single_char_bar_doc{title}_step{step_idx}.png"
                        gcs_path = f"{safe_prefix}/{out_name}"
                        with fsspec.open(gcs_path, "wb") as f_out, open(tmp_path, "rb") as f_in:
                            f_out.write(f_in.read())
                        if cfg.verbose:
                            print(f"[pz_single_doc] wrote char-hist PNG to {gcs_path}", flush=True)
                    finally:
                        try:
                            import os

                            os.unlink(tmp_path)
                        except Exception:
                            pass
                else:
                    if cfg.verbose:
                        print("[pz_single_doc] gcp_prefix not set; skipping PNG save", flush=True)
            except Exception as e:
                if cfg.verbose:
                    print(f"[pz_single_doc] failed char-hist PNG save: {e}", flush=True)

    return cb
