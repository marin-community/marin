# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import time
from dataclasses import dataclass
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

import haliax as hax
from haliax.partitioning import ResourceMapping, set_mesh

import levanter
from levanter.callbacks import StepInfo
from levanter.data.text import LMMixtureDatasetConfig, SingleDatasetLMConfigBase
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.models.loss import next_token_loss
from levanter.utils.hf_utils import HfTokenizer
from levanter.data._prp import PermType
from levanter.tracker.histogram import Histogram

# Progress bar for long PRP builds; fallback is a no-op if tqdm is unavailable
try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover - keep code runnable without tqdm

    def _tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable


# Debug helpers for inspecting mesh/device state when verbose=True.
def _describe_mesh(mesh) -> str:
    try:
        return f"Mesh(shape={mesh.shape}, axis_names={getattr(mesh, 'axis_names', None)})"
    except Exception:
        return repr(mesh)


@dataclass
class PzInnerLoopConfig:
    datasets: Optional[List[str]] = None
    doc_tokens: Optional[int] = None
    # Minimum document length (in tokens) required for selection. If set, a document
    # must have at least max(chunk_size, min_doc_tokens) tokens to be eligible.
    min_doc_tokens: Optional[int] = None
    chunk_size: int = 512
    prompt_tokens: Optional[int] = None
    cursor_inc_tokens: int = 1
    num_documents: int = 1
    mode: str = "sliding"  # one of: "sliding" (default), "first"
    eval_batch_size: Optional[int] = 64  # batch across docs in 'first' mode
    histogram: bool = False
    histogram_linear: bool = True
    pz_threshold: float = 1e-4
    pz_npz: bool = False
    # Only log histogram artifact when (global_step % histogram_every_steps == 0). If None, log whenever histogram=True
    histogram_every_steps: Optional[int] = None
    decode_preview: Optional[int] = None
    verify_treecache: bool = False
    # Verbose printing for debug; default false so configs need not set it
    verbose: bool = False
    # (no extra debug controls here; use verbose for lightweight logging)
    # Selection controls
    restrict_to_training_subset: bool = True
    initial_batch_size: Optional[int] = None  # required if max_train_batches is set
    doc_shuffle: bool = True
    doc_perm_type: PermType = "feistel"
    doc_perm_seed: Optional[int] = None


def pz_eval_callback(
    config: PzInnerLoopConfig,
    tokenizer: HfTokenizer,
    axis_resources: ResourceMapping,
    mp,
    data_config: Union[LMMixtureDatasetConfig, SingleDatasetLMConfigBase],
    *,
    device_mesh=None,
):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    # Persistent state across callback invocations (per process)
    caches_state = None  # set on first call
    selected_indices_by_ds: dict[str, List[int]] = {}
    # Track which datasets we've already printed preview decodes for (guard against spam)
    decode_printed_datasets: set[str] = set()

    def _ts():
        return time.strftime("%H:%M:%S", time.localtime())

    def _log(msg: str, *, all_hosts: bool = False):
        # Gate all console printing behind config.verbose (default False)
        if not config.verbose:
            return
        # this is good
        if all_hosts or jax.process_index() == 0:
            # Avoid printing the process index to keep logs/W&B clean
            print(f"[PZ] {msg}", flush=True)

    # Detailed context logging to help diagnose mesh/device mismatches during P(z)
    def _log_ctx(prefix: str = "ctx"):
        if not config.verbose:
            return
        try:
            devices = [(d.platform, getattr(d, "id", None)) for d in jax.devices()]
            mesh = hax.partitioning._get_mesh()
            mapping = hax.partitioning.current_thread_local_mapping()
            _log(
                f"{prefix}: default_backend={jax.default_backend()} devices={devices} "
                f"mesh={_describe_mesh(mesh)} axis_mapping={mapping}",
                all_hosts=True,
            )
        except Exception as e:
            _log(f"{prefix}: failed to log context: {e}", all_hosts=True)

    def _compute_logprob_for_tokens(model: LmHeadModel, tokens_1d: np.ndarray, prompt_len: int) -> float:
        N = int(tokens_1d.shape[0])
        Pos = model.Pos.resize(N)
        toks_named = hax.named(np.array(tokens_1d, dtype=np.int32), Pos)
        ex = LmExample.from_prompt_and_completion(Pos, toks_named, prompt_length=int(prompt_len), ignore_id=pad_id)
        m = model
        if mp is not None:
            m = mp.cast_to_compute(m)
        if device_mesh is not None:
            ctx = set_mesh(device_mesh)
        else:
            ctx = None
        # Ensure compute runs under the trainer mesh (if provided) and axis mapping
        if ctx is not None:
            with ctx, hax.axis_mapping(axis_resources):
                _log_ctx(prefix="before_scalar_forward")
                logits = m(ex.tokens, attn_mask=ex.attn_mask)
                logits = logits.astype(jnp.float32)
                nll = next_token_loss(
                    Pos=Pos, Vocab=m.Vocab, logits=logits, true_ids=ex.tokens, loss_mask=ex.loss_mask, reduction=None
                )
                total_nll = hax.sum(nll, axis=Pos).array
                _log_ctx(prefix="after_scalar_forward")
        else:
            with hax.axis_mapping(axis_resources):
                _log_ctx(prefix="before_scalar_forward")
                logits = m(ex.tokens, attn_mask=ex.attn_mask)
                logits = logits.astype(jnp.float32)
                nll = next_token_loss(
                    Pos=Pos, Vocab=m.Vocab, logits=logits, true_ids=ex.tokens, loss_mask=ex.loss_mask, reduction=None
                )
                total_nll = hax.sum(nll, axis=Pos).array
                _log_ctx(prefix="after_scalar_forward")
        return -float(np.array(total_nll))

    def _run_for_model(model: LmHeadModel, *, curr_step: int):
        # DEBUG print helper
        def _p(msg: str):
            # Debug print helper (gated by config.verbose)
            try:
                if config.verbose and jax.process_index() == 0:
                    print(f"[PZ][step={curr_step}] {msg}", flush=True)
            except Exception:
                pass

        # Only process 0 does tracker I/O. All hosts print phase timings for debugging.
        _log(f"step={curr_step} | BEGIN pz_eval inner-loop")
        _log(
            f"jax world: process_count={jax.process_count()} local_device_count={jax.local_device_count()} total_device_count={len(jax.devices())}",
            all_hosts=True,
        )
        _log_ctx(prefix="begin_run_for_model")

        nonlocal caches_state
        eval_start_time = time.time()
        if caches_state is None:
            _log(f"step={curr_step} | building caches(train)...", all_hosts=True)
            _p("building train caches…")
            t0_caches = time.perf_counter()
            caches_state = data_config.build_caches("train", monitors=False)
            t1_caches = time.perf_counter()
            _log(f"built caches in {t1_caches - t0_caches:.3f}s; datasets={list(caches_state.keys())}")
            _p(f"built caches in {t1_caches - t0_caches:.3f}s; datasets={list(caches_state.keys())}")
        else:
            _log("reusing previously built caches")
            _p("reusing previously built caches")
        caches = caches_state

        # Determine datasets to evaluate
        if config.datasets is None:
            selected = list(caches.keys())
        else:
            selected = [name for name in config.datasets if name in caches]

        _log(f"step={curr_step} | selected datasets: {selected}")
        if config.verbose and jax.process_index() == 0:
            try:
                print(
                    f"[PZ][step={curr_step}] config.datasets={config.datasets} | caches={list(caches.keys())} | selected={selected}",
                    flush=True,
                )
            except Exception:
                pass

        results = {}

        # Aggregates across all datasets for a 'total' series
        total_num_windows = 0
        total_sum_pz = 0.0
        total_sum_sq_pz = 0.0
        total_min_pz = float("inf")
        total_max_pz = float("-inf")
        total_counts = jnp.zeros(10, dtype=jnp.int32)
        total_edges = jnp.linspace(0.0, 1.0, 11, dtype=jnp.float32)
        total_docs_selected = 0

        # If evaluating across all datasets (datasets=None), build a pooled selection once
        if config.datasets is None:
            if config.verbose and jax.process_index() == 0:
                print(
                    f"[PZ] DATASETS=None → pooling across all datasets. Previously pinned={bool(selected_indices_by_ds)}",
                    flush=True,
                )
            # Only compute if not already pinned; else reuse
            if len(selected_indices_by_ds) == 0:
                if config.verbose and jax.process_index() == 0:
                    print(
                        f"[PZ] POOL START: pooling across {len(selected)} datasets: {sorted(selected)}",
                        flush=True,
                    )
                need_global = int(max(1, config.num_documents))
                seq_len_train = int(model.Pos.size)
                # Stable dataset order for determinism
                selected_sorted = sorted(selected)
                pool_ds_ids: List[int] = []
                pool_doc_idx: List[np.ndarray] = []
                ds_name_to_id = {name: i for i, name in enumerate(selected_sorted)}
                t_pool0 = time.perf_counter()
                total_eligible_pool = 0
                for ds_name_all in selected_sorted:
                    cache_all = caches[ds_name_all]
                    input_store_all = cache_all.store.tree["input_ids"]  # type: ignore[index]
                    num_rows_all = int(input_store_all.num_rows)
                    # Read offsets and compute eligibility for this dataset
                    t_off0 = time.perf_counter()
                    offsets_all = input_store_all.offsets[0 : num_rows_all + 1].read().result()
                    if len(offsets_all) > 0:
                        offsets_all = offsets_all.copy()
                        offsets_all[0] = 0
                    t_off1 = time.perf_counter()
                    if config.verbose and jax.process_index() == 0:
                        print(
                            f"[PZ] {ds_name_all}: offsets read {t_off1 - t_off0:.3f}s (rows={num_rows_all})",
                            flush=True,
                        )

                    last_off_all = int(offsets_all[-1] if len(offsets_all) > 0 else 0)
                    total_sequences_all = last_off_all // seq_len_train

                    effective_sequences_all = total_sequences_all
                    if isinstance(data_config, LMMixtureDatasetConfig):
                        if (
                            data_config.experiment_budget is not None
                            and data_config.target_budget is not None
                            and data_config.target_budget > 0
                        ):
                            ratio = float(data_config.experiment_budget) / float(data_config.target_budget)
                            effective_sequences_all = int(effective_sequences_all * ratio)
                        if (
                            data_config.num_validation_sequences is not None
                            and ds_name_all in data_config.num_validation_sequences
                        ):
                            effective_sequences_all = max(
                                0, effective_sequences_all - int(data_config.num_validation_sequences[ds_name_all])
                            )
                        if data_config.max_train_batches is not None and ds_name_all in data_config.max_train_batches:
                            if config.initial_batch_size is None:
                                raise ValueError(
                                    "P(z) restrict_to_training_subset requires initial_batch_size when max_train_batches is set."
                                )
                            limit = int(data_config.max_train_batches[ds_name_all]) * int(config.initial_batch_size)
                            effective_sequences_all = min(effective_sequences_all, limit)

                    tokens_cutoff_all = int(effective_sequences_all * seq_len_train)

                    if len(offsets_all) >= 2:
                        lens_all = offsets_all[1:] - offsets_all[:-1]
                        min_required_len = int(config.chunk_size)
                        if config.min_doc_tokens is not None:
                            min_required_len = max(min_required_len, int(config.min_doc_tokens))
                        eligible_len_all = lens_all >= int(min_required_len)
                        eligible_cut_all = (
                            offsets_all[1:] <= tokens_cutoff_all
                            if config.restrict_to_training_subset
                            else np.ones_like(lens_all, dtype=bool)
                        )
                        eligible_mask_all = np.logical_and(eligible_len_all, eligible_cut_all)
                        eligible_indices_all = np.nonzero(eligible_mask_all)[0]
                    else:
                        eligible_indices_all = np.zeros((0,), dtype=np.int64)

                    total_eligible = int(eligible_indices_all.shape[0])
                    total_eligible_pool += total_eligible

                    if total_eligible > 0:
                        ds_id = ds_name_to_id[ds_name_all]
                        pool_ds_ids.append(np.full(total_eligible, ds_id, dtype=np.int32))
                        pool_doc_idx.append(eligible_indices_all.astype(np.int32))

                if len(pool_doc_idx) == 0:
                    raise ValueError("P(z): no eligible documents across any dataset; check caps and chunk_size")

                ds_ids_np = np.concatenate(pool_ds_ids)
                doc_idx_np = np.concatenate(pool_doc_idx)

                # Deterministic global shuffle of pooled eligibles
                t_pool1 = time.perf_counter()
                seed = int(config.doc_perm_seed or 0)
                base_key = jax.random.PRNGKey(seed)
                # Fold in sorted dataset names for stability
                names_blob = ("|".join(selected_sorted)).encode("utf-8")
                name_sum = int(np.frombuffer(names_blob, dtype=np.uint8).sum())
                pkey_global = jax.random.fold_in(base_key, name_sum)

                N_pool = int(doc_idx_np.shape[0])
                if N_pool < need_global:
                    # Fail early at the global level instead of per-dataset later
                    raise ValueError(
                        f"P(z) pooled selection found only {N_pool} eligible documents across all datasets, "
                        f"but requested {need_global}."
                    )
                perm_positions = jax.random.permutation(pkey_global, jnp.arange(N_pool, dtype=jnp.int32))
                take = np.asarray(perm_positions[:need_global]).astype(np.int32)
                chosen_ds_ids = ds_ids_np[take]
                chosen_doc_idx = doc_idx_np[take]
                # Group back per dataset
                for i, ds_name_sorted in enumerate(selected_sorted):
                    mask = chosen_ds_ids == i
                    if np.any(mask):
                        sel = chosen_doc_idx[mask].astype(int).tolist()
                    else:
                        sel = []
                    selected_indices_by_ds[ds_name_sorted] = sel
                t_pool2 = time.perf_counter()
                elapsed_pool = t_pool2 - t_pool0
            else:
                if config.verbose and jax.process_index() == 0:
                    print(
                        "[PZ] POOL REUSE: reusing previously selected indices across all datasets.",
                        flush=True,
                    )

        # After pooling, if we have a global selection, record expected total documents
        try:
            total_docs_selected = int(sum(len(selected_indices_by_ds.get(n, [])) for n in selected))
        except Exception:
            total_docs_selected = 0

        for ds_name in selected:
            ds_start_time = time.perf_counter()

            cache = caches[ds_name]
            input_store = cache.store.tree["input_ids"]  # type: ignore[index]

            # Persist selection once using training-subset gating + doc-level shuffle
            if ds_name not in selected_indices_by_ds:
                need = int(max(1, config.num_documents))
                _log(
                    f"dataset={ds_name} | selecting {need} doc(s) with shuffle-first + training-subset gating",
                )
                scan_t0 = time.perf_counter()
                num_rows = int(input_store.num_rows)

                # Compute effective training subset in sequences to derive token cutoff
                seq_len_train = int(model.Pos.size)
                # Batch read offsets once
                t_off0 = time.perf_counter()
                offsets = input_store.offsets[0 : num_rows + 1].read().result()
                if len(offsets) > 0:
                    offsets = offsets.copy()
                    offsets[0] = 0
                t_off1 = time.perf_counter()
                _p(
                    f"dataset={ds_name}: read offsets[0:{num_rows + 1}] in {t_off1 - t_off0:.3f}s (num_rows={num_rows})"
                )
                last_off = int(offsets[-1] if len(offsets) > 0 else 0)
                total_sequences = last_off // seq_len_train
                _p(
                    f"dataset={ds_name}: total_tokens={last_off} seq_len={seq_len_train} total_sequences={total_sequences}"
                )

                effective_sequences = total_sequences
                if isinstance(data_config, LMMixtureDatasetConfig):
                    # Simulated budget
                    if (
                        data_config.experiment_budget is not None
                        and data_config.target_budget is not None
                        and data_config.target_budget > 0
                    ):
                        ratio = float(data_config.experiment_budget) / float(data_config.target_budget)
                        effective_sequences = int(effective_sequences * ratio)
                    # Reserve validation sequences (take head for training)
                    if (
                        data_config.num_validation_sequences is not None
                        and ds_name in data_config.num_validation_sequences
                    ):
                        effective_sequences = max(
                            0, effective_sequences - int(data_config.num_validation_sequences[ds_name])
                        )
                    # Max train batches
                    if data_config.max_train_batches is not None and ds_name in data_config.max_train_batches:
                        if config.initial_batch_size is None:
                            raise ValueError(
                                "P(z) restrict_to_training_subset requires initial_batch_size when max_train_batches is set."
                            )
                        limit = int(data_config.max_train_batches[ds_name]) * int(config.initial_batch_size)
                        effective_sequences = min(effective_sequences, limit)

                tokens_cutoff = int(effective_sequences * seq_len_train)
                _p(f"dataset={ds_name}: effective_sequences={effective_sequences} tokens_cutoff={tokens_cutoff}")

                # Precompute eligibility mask using batched offsets (can be done before PRP)
                if len(offsets) >= 2:
                    lens = offsets[1:] - offsets[:-1]
                    min_required_len = int(config.chunk_size)
                    if config.min_doc_tokens is not None:
                        min_required_len = max(min_required_len, int(config.min_doc_tokens))
                    eligible_len = lens >= int(min_required_len)
                    eligible_cut = (
                        offsets[1:] <= tokens_cutoff
                        if config.restrict_to_training_subset
                        else np.ones_like(lens, dtype=bool)
                    )
                    eligible_mask = np.logical_and(eligible_len, eligible_cut)
                    eligible_count = int(np.count_nonzero(eligible_mask))
                else:
                    eligible_mask = np.zeros((0,), dtype=bool)
                    eligible_count = 0
                _p(f"dataset={ds_name}: eligible docs pre-PRP count={eligible_count}")
                _p(f"dataset={ds_name}: eligible_mask.sum()={int(np.sum(eligible_mask))}")
                _p(f"dataset={ds_name}: eligible_mask.sum()={int(np.sum(eligible_mask))}")

                # Build doc-level PRP order (or identity)
                # NOTE: The full-domain PRP build below is retained for reference but commented out.
                #       We now use a fast path that permutes only eligible indices with JAX.
                # if config.doc_shuffle:
                #     seed = int(config.doc_perm_seed or 0)
                #     base_key = jax.random.PRNGKey(seed)
                #     name_sum = int(np.frombuffer(ds_name.encode("utf-8"), dtype=np.uint8).sum())
                #     pkey = jax.random.fold_in(base_key, name_sum)
                #     prp: Permutation = Permutation.make(config.doc_perm_type, num_rows, pkey)
                #     order = []
                #     # Timing around the PRP order build (with tqdm)
                #     t_prp0 = time.perf_counter()
                #     if jax.process_index() == 0:
                #         print(
                #             f"\n===== BEGIN PRP ORDER BUILD: dataset={ds_name} rows={num_rows} =====",
                #             flush=True,
                #         )
                #     # Show progress only on process 0 to avoid clutter in multi-host runs
                #     for i in _tqdm(
                #         range(num_rows),
                #         total=num_rows,
                #         desc=f"PRP order: {ds_name}",
                #         disable=(jax.process_index() != 0),
                #     ):
                #         order.append(int(prp(i)))
                #     t_prp1 = time.perf_counter()
                #     if jax.process_index() == 0:
                #         elapsed = t_prp1 - t_prp0
                #         rate = (num_rows / elapsed) if elapsed > 0 else float('inf')
                #         print(
                #             f"===== END PRP ORDER BUILD: dataset={ds_name} took {elapsed:.3f}s ({rate:,.0f} it/s) =====\n",
                #             flush=True,
                #         )
                # else:
                #     order = list(range(num_rows))

                # FAST PATH: Permute only the eligible indices with JAX and take first K
                selected_doc_indices: List[int] = []
                t_fast0 = time.perf_counter()
                # Compute eligible indices (numpy), then permute via jax on jnp arrays
                eligible_indices_np = np.nonzero(eligible_mask)[0]
                if config.doc_shuffle:
                    seed = int(config.doc_perm_seed or 0)
                    base_key = jax.random.PRNGKey(seed)
                    name_sum = int(np.frombuffer(ds_name.encode("utf-8"), dtype=np.uint8).sum())
                    pkey = jax.random.fold_in(base_key, name_sum)
                    eligible_jnp = jnp.asarray(eligible_indices_np, dtype=jnp.int32)
                    permuted = jax.random.permutation(pkey, eligible_jnp)
                    # Bring to host and slice first K
                    selected_doc_indices = list(np.asarray(permuted[:need]).astype(int))
                else:
                    # Identity order over eligible; take head
                    selected_doc_indices = eligible_indices_np[:need].astype(int).tolist()
                t_fast1 = time.perf_counter()
                if config.verbose and jax.process_index() == 0:
                    elapsed_fast = t_fast1 - t_fast0
                    rate_fast = (max(1, len(eligible_indices_np)) / elapsed_fast) if elapsed_fast > 0 else float("inf")
                    print(
                        f"===== FAST ELIGIBLE SHUFFLE: dataset={ds_name} took {elapsed_fast:.3f}s "
                        f"(eligible={len(eligible_indices_np)}, K={need}, ~{rate_fast:,.0f} el/s) =====",
                        flush=True,
                    )

                if len(selected_doc_indices) == 0:
                    # No eligible documents at all — hard fail as requested
                    raise ValueError(
                        f"P(z) selection failed for {ds_name}: no eligible documents (len>=chunk_size and within training subset). "
                        f"tokens_cutoff={tokens_cutoff}, chunk_size={int(config.chunk_size)}"
                    )
                if len(selected_doc_indices) < need:
                    # Not enough eligible documents to satisfy requested num_documents — hard fail
                    raise ValueError(
                        f"P(z) selection for {ds_name} found only {len(selected_doc_indices)} eligible documents, "
                        f"but requested {need}. Increase training subset or reduce num_documents. "
                        f"tokens_cutoff={tokens_cutoff}, chunk_size={int(config.chunk_size)}"
                    )
                selected_indices_by_ds[ds_name] = selected_doc_indices
                scan_t1 = time.perf_counter()
                _log(f"dataset={ds_name} | selected indices={selected_doc_indices} in {scan_t1 - scan_t0:.3f}s")
                _p(
                    f"dataset={ds_name}: selected {len(selected_doc_indices)}/{need} in {scan_t1 - scan_t0:.3f}s; eligible={eligible_count}"
                )
                # (Preview decode handled later with decode_once_state guard if configured)
            else:
                _log(f"dataset={ds_name} | reusing pinned indices={selected_indices_by_ds[ds_name]}")

            selected_doc_indices = selected_indices_by_ds[ds_name]
            # Read tokens for selected indices (minimally)
            selected_docs: List[np.ndarray] = []
            _p(f"dataset={ds_name}: reading {len(selected_doc_indices)} docs…")
            for _sel_idx in selected_doc_indices:
                try:
                    arr = np.asarray(input_store[_sel_idx], dtype=np.int32).reshape(-1)
                    selected_docs.append(arr)
                except Exception as e:
                    _log(f"[WARN] dataset={ds_name} | error reading doc {_sel_idx}: {e}")
                    _p(f"dataset={ds_name}: error reading doc {_sel_idx}: {e}")
                    continue
            _p(f"dataset={ds_name}: finished reading docs count={len(selected_docs)}")

            # Optional: decode and print a small preview of the first selected document(s)
            if (
                config.decode_preview is not None
                and jax.process_index() == 0
                and ds_name not in decode_printed_datasets
                and len(selected_docs) > 0
            ):
                try:
                    num_to_decode = int(max(0, int(config.decode_preview)))
                except Exception:
                    num_to_decode = 0
                num_to_decode = min(num_to_decode, len(selected_docs))
                if num_to_decode > 0:
                    print(
                        f"[PZ][preview] dataset={ds_name} decoding first {num_to_decode} selected doc(s)…",
                        flush=True,
                    )
                    for j in range(num_to_decode):
                        toks = selected_docs[j]
                        idx = selected_doc_indices[j]
                        # Cap preview tokens for readability
                        preview_tok_cap = int(min(len(toks), 1024))
                        text = tokenizer.decode(toks[:preview_tok_cap].tolist(), skip_special_tokens=False)
                        print(
                            f"[PZ][preview] dataset={ds_name} doc_idx={idx} len={len(toks)} preview_tokens={preview_tok_cap}\n{text}\n---",
                            flush=True,
                        )
                    decode_printed_datasets.add(ds_name)

            # Use config.chunk_size, not model.Pos.size, to avoid evaluating mostly padding
            N = int(config.chunk_size)
            P = int(config.prompt_tokens if config.prompt_tokens is not None else N // 2)
            S = max(1, int(config.cursor_inc_tokens))

            if N != model.Pos.size:
                _log(f"[WARN] chunk_size={N} != model.Pos.size={model.Pos.size}, may cause recompilation")

            pz_values: List[float] = []
            span_ranges: List[tuple[int, int]] = []
            doc_indices_for_windows: List[int] = []
            first_mode_windows: List[np.ndarray] = []
            first_mode_indices: List[int] = []
            first_mode_doc_lens: List[int] = []
            first_eval_len: Optional[int] = None

            # Iterate over selected documents and compute windowed P(z)
            for doc_sel_idx, first_ids in zip(selected_doc_indices, selected_docs):
                # Determine document slice
                if config.doc_tokens is None:
                    eval_len = int(first_ids.shape[0])
                else:
                    eval_len = int(min(int(config.doc_tokens), int(first_ids.shape[0])))
                if first_eval_len is None:
                    first_eval_len = eval_len

                if eval_len == 0:
                    _log(f"[WARN] dataset={ds_name} | eval_len=0 for doc {doc_sel_idx}, skipping")
                    continue

                # Skip documents that are shorter than chunk_size (would be mostly padding)
                if eval_len < N:
                    _log(
                        f"[WARN] dataset={ds_name} | doc_idx={doc_sel_idx} doc_len={eval_len} < chunk_size={N}, skipping (would be mostly padding)"
                    )
                    continue

                doc_slice = first_ids[:eval_len]

                # Windowing mode selection
                _mode = (config.mode or "sliding").lower()
                if _mode == "first":
                    starts = [0]
                else:  # default to sliding
                    starts = list(range(0, max(eval_len - N, 0) + 1, S))
                    if not starts:
                        starts = [0]

                _log(
                    f"dataset={ds_name} | mode={_mode} doc_idx={doc_sel_idx} doc_len={eval_len} N={N} P={P} S={S} num_starts={len(starts)}"
                )

                # Time the window evaluations; first window usually includes compilation
                doc_t0 = time.perf_counter()
                first_window_time = None
                window_count = 0

                if _mode == "first":
                    # Collect the first window for batched compute later
                    s = 0
                    window = doc_slice[s : s + N]
                    if window.shape[0] < N:
                        pad_len = N - window.shape[0]
                        window = np.concatenate([window, np.full((pad_len,), pad_id, dtype=np.int32)], axis=0)
                    first_mode_windows.append(window)
                    first_mode_indices.append(doc_sel_idx)
                    first_mode_doc_lens.append(eval_len)
                    window_count = 1
                else:
                    for s in starts:
                        window = doc_slice[s : s + N]
                        if window.shape[0] < N:
                            pad_len = N - window.shape[0]
                            window = np.concatenate([window, np.full((pad_len,), pad_id, dtype=np.int32)], axis=0)
                        w_t0 = time.perf_counter()
                        lp = _compute_logprob_for_tokens(model, window, P)
                        p = float(np.exp(lp))
                        pz_values.append(p)
                        span_ranges.append((s, min(s + N - 1, eval_len - 1)))
                        doc_indices_for_windows.append(doc_sel_idx)
                        w_t1 = time.perf_counter()
                        window_count += 1
                        if first_window_time is None:
                            first_window_time = w_t1 - w_t0
                            _log(
                                f"dataset={ds_name} | doc_idx={doc_sel_idx} | first_window_time={first_window_time:.3f}s (includes possible compile)"
                            )
                        elif window_count % max(1, len(starts) // 5) == 0:
                            _log(
                                f"dataset={ds_name} | doc_idx={doc_sel_idx} | window {window_count}/{len(starts)} took {w_t1 - w_t0:.3f}s"
                            )

                doc_t1 = time.perf_counter()
                _log(
                    f"dataset={ds_name} | doc_idx={doc_sel_idx} | evaluated {window_count} windows in {doc_t1 - doc_t0:.3f}s"
                )

            # Aggregate and log scalars (only on process 0)
            # If first mode, perform batched evaluate across docs now
            if (config.mode or "sliding").lower() == "first" and len(first_mode_windows) > 0:
                arr2d = np.stack(first_mode_windows, axis=0)

                def _vmapped_batch(m: LmHeadModel, toks_2d: jnp.ndarray, prompt_len: int):
                    # Use Haliax named axes and vmap instead of positional jax.vmap
                    Pos = m.Pos.resize(N)

                    # Define a named batch axis; prefer "batch" to respect any axis_resources mapping
                    B = int(toks_2d.shape[0])
                    Batch = hax.Axis("batch", B)

                    # Name the input as (Batch, Pos)
                    toks_named_2d = hax.named(toks_2d, (Batch, Pos))

                    def single_named(tokens_1d_named):
                        # tokens_1d_named has axes {Pos}
                        ex = LmExample.from_prompt_and_completion(
                            Pos, tokens_1d_named, prompt_length=int(prompt_len), ignore_id=pad_id
                        )
                        mm = m
                        if mp is not None:
                            mm = mp.cast_to_compute(mm)
                        logits = mm(ex.tokens, attn_mask=ex.attn_mask)
                        logits = logits.astype(jnp.float32)
                        nll = next_token_loss(
                            Pos=Pos,
                            Vocab=mm.Vocab,
                            logits=logits,
                            true_ids=ex.tokens,
                            loss_mask=ex.loss_mask,
                            reduction=None,
                        )
                        # Return negative total NLL (i.e., log-prob) as a scalar array
                        return -hax.sum(nll, axis=Pos).array

                    if device_mesh is not None:
                        ctx2 = set_mesh(device_mesh)
                    else:
                        ctx2 = None

                    if ctx2 is not None:
                        with ctx2, hax.axis_mapping(axis_resources):
                            _log_ctx(prefix="before_batched_forward")
                            out = hax.vmap(single_named, Batch)(toks_named_2d)
                            _log_ctx(prefix="after_batched_forward")
                            # hax.vmap returns a vector along Batch; expose as JAX array
                            return getattr(out, "array", out)
                    else:
                        with hax.axis_mapping(axis_resources):
                            _log_ctx(prefix="before_batched_forward")
                            out = hax.vmap(single_named, Batch)(toks_named_2d)
                            _log_ctx(prefix="after_batched_forward")
                            return getattr(out, "array", out)

                B = int(config.eval_batch_size) if config.eval_batch_size is not None else arr2d.shape[0]
                b0 = time.perf_counter()
                for i_b in range(0, arr2d.shape[0], B):
                    batch = jnp.asarray(arr2d[i_b : i_b + B], dtype=jnp.int32)
                    _log(f"dispatch batched forward: batch_shape={batch.shape}, B={B}")
                    t_b0 = time.perf_counter()
                    lp_vec = _vmapped_batch(model, batch, P)
                    t_b1 = time.perf_counter()
                    _log(f"batched forward done in {t_b1 - t_b0:.3f}s; lp_vec.shape={np.asarray(lp_vec).shape}")
                    for j, lp in enumerate(np.array(lp_vec)):
                        pz_values.append(float(np.exp(lp)))
                        idx = first_mode_indices[i_b + j]
                        doc_len = first_mode_doc_lens[i_b + j]
                        span_ranges.append((0, min(N - 1, doc_len - 1)))
                        doc_indices_for_windows.append(idx)
                b1 = time.perf_counter()
                _log(
                    f"dataset={ds_name} | batched first-mode forward for {arr2d.shape[0]} docs in {b1 - b0:.3f}s (batch_size={B})"
                )

            # Aggregate and log scalars only when there are evaluated windows.
            if len(pz_values) > 0:
                arr = np.asarray(pz_values, dtype=np.float64)
                mean_pz = float(np.mean(arr))
                median_pz = float(np.median(arr))
                max_pz = float(np.max(arr))

                ds_elapsed_time = time.perf_counter() - ds_start_time
                metrics = {
                    f"pz_eval/{ds_name}/num_windows": int(len(pz_values)),
                    f"pz_eval/{ds_name}/num_documents": int(len(set(doc_indices_for_windows))),
                    f"pz_eval/{ds_name}/mean_pz": mean_pz,
                    f"pz_eval/{ds_name}/median_pz": median_pz,
                    f"pz_eval/{ds_name}/max_pz": max_pz,
                    # Back-compat: log the first evaluated document length under doc_len
                    f"pz_eval/{ds_name}/doc_len": int(first_eval_len or 0),
                    f"pz_eval/{ds_name}/chunk_size": int(N),
                    f"pz_eval/{ds_name}/prompt_tokens": int(P),
                    f"pz_eval/{ds_name}/suffix_tokens": int(N - P),
                    f"pz_eval/{ds_name}/cursor_inc_tokens": int(S),
                    f"pz_eval/{ds_name}/eval_time_seconds": ds_elapsed_time,
                }
                # Log a value histogram of P(z) in [0, 1] with 10 bins of width 0.1
                # This mirrors the tracker Histogram used for entropy, but with fixed edges.
                arr32 = np.asarray(pz_values, dtype=jnp.float32)
                # Clip to [0, 1] for safety before binning
                arr32 = np.clip(arr32, 0.0, 1.0)
                # Fixed 10 bins from 0.0 to 1.0 inclusive (edges length = 11)
                edges = jnp.linspace(0.0, 1.0, 11, dtype=jnp.float32)
                counts, edges_out = jnp.histogram(arr32, bins=edges)
                counts = counts.astype(jnp.int32)

                # Populate Histogram fields
                h_min = float(np.min(arr32))
                h_max = float(np.max(arr32))
                h_num = int(arr32.size)
                h_sum = float(np.sum(arr32))
                h_sum_sq = float(np.sum(arr32**2))
                hist = jax.device_get(Histogram(h_min, h_max, h_num, h_sum, h_sum_sq, edges_out, counts))

                # Update totals safely
                total_num_windows += h_num
                total_sum_pz += h_sum
                total_sum_sq_pz += h_sum_sq
                total_min_pz = float(min(total_min_pz, h_min))
                total_max_pz = float(max(total_max_pz, h_max))
                total_counts = total_counts + counts

                # Log metrics and histogram (present for non-empty datasets)
                levanter.tracker.log(metrics, step=curr_step)
                _log(f"dataset={ds_name} | metrics: {metrics}")
                levanter.tracker.log({f"pz_eval/{ds_name}/pz_hist": hist}, step=curr_step)

                # Mark dataset as evaluated only when non-empty
                results[ds_name] = True

        # After processing all datasets, log aggregated 'total' series
        if total_num_windows > 0:
            total_mean_pz = float(total_sum_pz / total_num_windows)
            # Approximate median from aggregated histogram
            counts_np = np.asarray(total_counts)
            edges_np = np.asarray(total_edges)
            cumsum = np.cumsum(counts_np)
            half = total_num_windows / 2
            bin_idx = int(np.searchsorted(cumsum, half, side="left"))
            bin_idx = max(0, min(bin_idx, len(counts_np) - 1))
            prev = 0 if bin_idx == 0 else cumsum[bin_idx - 1]
            in_bin = counts_np[bin_idx]
            if in_bin > 0:
                frac = (half - prev) / in_bin
            else:
                frac = 0.0
            bin_lo = float(edges_np[bin_idx])
            bin_hi = float(edges_np[bin_idx + 1])
            total_median_pz = float(bin_lo + frac * (bin_hi - bin_lo))

            total_hist = jax.device_get(
                Histogram(
                    float(total_min_pz),
                    float(total_max_pz),
                    int(total_num_windows),
                    float(total_sum_pz),
                    float(total_sum_sq_pz),
                    total_edges,
                    total_counts.astype(jnp.int32),
                )
            )
        else:
            total_mean_pz = total_median_pz = 0.0
            total_hist = None

        # Total metrics
        total_metrics = {
            "pz_eval/total/num_windows": int(total_num_windows),
            "pz_eval/total/num_documents": int(total_docs_selected),
            "pz_eval/total/requested_num_documents": int(max(1, config.num_documents)),
            "pz_eval/total/mean_pz": float(total_mean_pz),
            "pz_eval/total/median_pz": float(total_median_pz),
        }
        levanter.tracker.log(total_metrics, step=curr_step)
        if total_hist is not None:
            levanter.tracker.log({"pz_eval/total/pz_hist": total_hist}, step=curr_step)

        eval_total_time = time.time() - eval_start_time
        levanter.tracker.log(
            {
                "pz_eval/total_eval_time_seconds": eval_total_time,
                "pz_eval/num_datasets_evaluated": len(results),
            },
            step=curr_step,
        )
        _log(f"total P(z) eval time: {eval_total_time:.2f}s for {len(results)} dataset(s)")

        return results

    def cb(step: StepInfo, force=False):
        if step.step == 0 and not force:
            return
        # CRITICAL FIX: Do NOT early-return on non-zero processes!
        # The model evaluation involves JAX collectives that require ALL hosts to participate.
        # Returning early on non-zero processes causes a deadlock.
        _log(f"entering P(z) callback for step={int(step.step)}", all_hosts=True)
        model = step.eval_model

        t0 = time.perf_counter()
        _run_for_model(model, curr_step=int(step.step))
        t1 = time.perf_counter()
        _log(f"finished P(z) callback in {t1 - t0:.3f}s")

    return cb


# Refactor
# - Replaced jax.vmap in first-mode batched evaluation with haliax.vmap over a named
#   Batch axis, avoiding positional vmap and aligning with Haliax’s named-tensor style.
# - Named the input batch tokens as a Haliax NamedArray with axes (batch, Pos), and
#   vectorized the per-example log-prob computation via haliax.vmap.
# - Preserved existing mesh and axis_mapping contexts and the scalar (per-window)
#   evaluation path.
