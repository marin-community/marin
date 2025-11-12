# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Single-book careless suffix likelihood (P(z)) evaluation.

This is a streamlined variant of ``eval_sliding_total`` that evaluates a single
book/text with a single model configuration. It loads the model once, slides a
window across the text (character- or token-based), computes per-window P(z),
logs simple metrics, and saves a histogram and NPZ outputs either to WandB or
to a cloud path via fsspec when configured.
"""

import dataclasses
import itertools
import logging
import pathlib
import tempfile
import time
from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax.experimental.multihost_utils import process_allgather

import fsspec
import haliax as hax
from haliax.nn import log_softmax
from haliax.partitioning import round_axis_for_partitioning

import levanter
import levanter.tracker
from levanter.books.util import (
    chunk_text_to_sliding_window_token_chunks,
    chunk_token_ids_to_sliding_windows,
    compute_max_extraction_rates,
    create_pz_histogram,
    create_pz_histogram_linear,
)
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.data import DataLoader
from levanter.data.dataset import ListAsyncDataset
from levanter.models.gpt2 import Gpt2Config
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device


logger = logging.getLogger(__name__)


@dataclass
class PzEvalConfig:
    """Configuration for single-book careless suffix likelihood evaluation."""

    # Checkpoint options
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None
    initialize_from_hf: Optional[RepoRef] = None
    use_hf_model_config: bool = False

    # Model
    model: LmConfig = field(default_factory=Gpt2Config)

    # Data / Sliding window
    txt_path: str | pathlib.Path = "src/levanter/data/books/gatsby.txt"
    chunk_size: int = 100
    slice_length: int = 2000
    prompt_tokens: int = 50
    cursor_inc_chars: int = 10
    token_mode: bool = False
    cursor_inc_tokens: int = 1

    # Tokenizer
    tokenizer_name: Optional[str] = None

    # Runtime / Trainer
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    max_examples: Optional[int] = None

    # Output
    output_base_path: str = "gs://marin-us-central2/books_evals/"
    plot_path: Optional[str] = None
    eval_batch_size: int = 32
    histogram_path: Optional[str] = None
    pz_threshold: float = 0.0001
    book_title: str = "Book"
    pz_data_path: Optional[str] = None

    # Misc/Performance
    use_dataloader: bool = True
    histogram_linear: bool = True
    allow_nondivisible_batch_size: bool = False

    # Logging/IO
    gcp_log: bool = False  # If True, save artifacts to GCP (or fsspec) instead of WandB

    # Optimizations (toggleable)
    opt_align_seq_len_128: bool = False
    opt_disable_checkpointing: bool = False
    opt_force_flash_block_128: bool = False
    opt_warmup_compile: bool = True


def _save_plot_with_wandb(fig, filename: str, dpi: int = 300):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        fig.savefig(tmp_file.name, dpi=dpi, bbox_inches="tight")
        tmp_path = tmp_file.name
    levanter.tracker.current_tracker().log_artifact(tmp_path, name=filename, type="plot")
    pathlib.Path(tmp_path).unlink(missing_ok=True)


def _save_data_with_wandb(data, filename: str, **kwargs):
    suffix = pathlib.Path(filename).suffix or ".npz"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        if suffix == ".npz":
            np.savez(tmp_file.name, **kwargs)
        else:
            np.save(tmp_file.name, data)
        tmp_path = tmp_file.name
    levanter.tracker.current_tracker().log_artifact(tmp_path, name=filename, type="data")
    pathlib.Path(tmp_path).unlink(missing_ok=True)


def _save_plot_to_gcp(fig, output_path: str, filename: str, book_title: str, dpi: int = 300):
    full_path = f"{output_path.rstrip('/')}/{book_title}/{filename}"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        fig.savefig(tmp_file.name, dpi=dpi, bbox_inches="tight")
        tmp_path = tmp_file.name
    with fsspec.open(full_path, "wb") as f_out:
        with open(tmp_path, "rb") as f_in:
            f_out.write(f_in.read())
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    logger.info("Saved plot to: %s", full_path)


def _save_data_to_gcp(data, output_path: str, filename: str, book_title: str, **kwargs):
    full_path = f"{output_path.rstrip('/')}/{book_title}/{filename}"
    suffix = pathlib.Path(filename).suffix or ".npz"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        if suffix == ".npz":
            np.savez(tmp_file.name, **kwargs)
        else:
            np.save(tmp_file.name, data)
        tmp_path = tmp_file.name
    with fsspec.open(full_path, "wb") as f_out:
        with open(tmp_path, "rb") as f_in:
            f_out.write(f_in.read())
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    logger.info("Saved data to: %s", full_path)


def main(cfg: PzEvalConfig):
    """Run careless suffix P(z) evaluation for a single book."""

    # Initialize tracker, mesh, etc.
    levanter.initialize(cfg)

    # Tokenizer
    if cfg.tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    else:
        tokenizer = getattr(cfg.model, "the_tokenizer", None)
    if tokenizer is None:
        raise ValueError("Tokenizer not provided: set tokenizer_name or ensure model.the_tokenizer exists")

    # Print optimization flags
    print(
        "perf/opts: align_seq_len_128=%s disable_checkpointing=%s force_flash_block_128=%s warmup_compile=%s"
        % (
            cfg.opt_align_seq_len_128,
            cfg.opt_disable_checkpointing,
            cfg.opt_force_flash_block_128,
            cfg.opt_warmup_compile,
        )
    )

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Build/load model once
    cmapping = cfg.trainer.compute_axis_mapping
    pmapping = cfg.trainer.parameter_axis_mapping

    with cfg.trainer.device_mesh, hax.axis_mapping(pmapping):
        key = jax.random.PRNGKey(0)
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(hax.Axis("vocab", vocab_size), cmapping)
        mp: jmp.Policy = cfg.trainer.mp

        # Apply configurable optimizations to cfg.model before building/loading when possible
        if cfg.checkpoint_path:
            # Pre-apply for checkpoint path
            if cfg.opt_align_seq_len_128 and hasattr(cfg.model, "seq_len"):
                try:
                    old_len = int(getattr(cfg.model, "seq_len"))
                    base = max(cfg.chunk_size, old_len)
                    new_len = int(math.ceil(base / 128.0) * 128)
                    if new_len != old_len:
                        cfg.model = dataclasses.replace(cfg.model, seq_len=new_len)
                        print(f"perf/opt: adjusted seq_len {old_len} -> {new_len} (align to multiple of 128)")
                except Exception as e:
                    print(f"perf/opt: seq_len alignment skipped due to error: {e}")
            if cfg.opt_disable_checkpointing and hasattr(cfg.model, "gradient_checkpointing"):
                try:
                    if getattr(cfg.model, "gradient_checkpointing"):
                        cfg.model = dataclasses.replace(cfg.model, gradient_checkpointing=False)
                        print("perf/opt: disabled gradient checkpointing for eval")
                except Exception as e:
                    print(f"perf/opt: disable checkpointing skipped due to error: {e}")
            if cfg.opt_force_flash_block_128 and hasattr(cfg.model, "flash_attention_block_size"):
                try:
                    cfg.model = dataclasses.replace(cfg.model, flash_attention_block_size=128)
                    print("perf/opt: set flash_attention_block_size=128")
                except Exception as e:
                    print(f"perf/opt: set flash block size skipped due to error: {e}")

        if cfg.checkpoint_path:
            with use_cpu_device():
                model = eqx.filter_eval_shape(cfg.model.build, Vocab, key=key)
                model = load_checkpoint(model, cfg.checkpoint_path, subpath="model")
            model = hax.shard_with_axis_mapping(model, pmapping)
        else:
            hf_ref = cfg.hf_checkpoint or cfg.initialize_from_hf
            if hf_ref is None:
                raise ValueError("Need --checkpoint-path or --hf-checkpoint/--initialize-from-hf")
            converter: HFCheckpointConverter = cfg.model.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=hf_ref, tokenizer=tokenizer)
            if cfg.use_hf_model_config:
                cfg.model = converter.config_from_hf_config(converter.default_hf_config)
            # Apply optimizations after config replacement
            if cfg.opt_align_seq_len_128 and hasattr(cfg.model, "seq_len"):
                try:
                    old_len = int(getattr(cfg.model, "seq_len"))
                    base = max(cfg.chunk_size, old_len)
                    new_len = int(math.ceil(base / 128.0) * 128)
                    if new_len != old_len:
                        cfg.model = dataclasses.replace(cfg.model, seq_len=new_len)
                        print(f"perf/opt: adjusted seq_len {old_len} -> {new_len} (align to multiple of 128)")
                except Exception as e:
                    print(f"perf/opt: seq_len alignment skipped due to error: {e}")
            if cfg.opt_disable_checkpointing and hasattr(cfg.model, "gradient_checkpointing"):
                try:
                    if getattr(cfg.model, "gradient_checkpointing"):
                        cfg.model = dataclasses.replace(cfg.model, gradient_checkpointing=False)
                        print("perf/opt: disabled gradient checkpointing for eval")
                except Exception as e:
                    print(f"perf/opt: disable checkpointing skipped due to error: {e}")
            if cfg.opt_force_flash_block_128 and hasattr(cfg.model, "flash_attention_block_size"):
                try:
                    cfg.model = dataclasses.replace(cfg.model, flash_attention_block_size=128)
                    print("perf/opt: set flash_attention_block_size=128")
                except Exception as e:
                    print(f"perf/opt: set flash block size skipped due to error: {e}")
            model = converter.load_pretrained(cfg.model.model_type, ref=hf_ref, dtype=mp.compute_dtype)

        # Now that cfg.model is finalized, set Pos
        Pos = cfg.model.Pos

        def sequence_log_prob(mod: LmHeadModel, batch: LmExample):
            mod = mp.cast_to_compute(mod)
            with hax.axis_mapping(cmapping):
                logits = mod(batch.tokens, attn_mask=batch.attn_mask)
                lp = log_softmax(logits, axis=mod.Vocab)
                targets = hax.roll(batch.tokens, -1, Pos)
                lp = hax.take(lp, mod.Vocab, targets)
                mask = batch.loss_mask * (targets != pad_id).astype(np.float32)
                lp = hax.sum(lp * mask, axis=Pos)
                return jnp.exp(lp.array)

        sequence_log_prob = hax.named_jit(sequence_log_prob, out_axis_resources=None)

    # Determine model name for titles
    model_name = "Unknown Model"
    if cfg.initialize_from_hf:
        model_path = str(cfg.initialize_from_hf)
        model_name = pathlib.Path(model_path).name.lower()
    elif cfg.hf_checkpoint:
        model_name = str(cfg.hf_checkpoint).split("/")[-1].lower()

    # Read text (supports local paths and gs:// via fsspec)
    with fsspec.open(str(cfg.txt_path), "r") as f:
        raw_text = f.read()

    # Build sliding windows
    token_ids = None
    if cfg.token_mode:
        token_ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"]
        chunks = chunk_token_ids_to_sliding_windows(
            token_ids,
            tokenizer,
            chunk_size=cfg.chunk_size,
            cursor_inc=cfg.cursor_inc_tokens,
        )
    else:
        chunks = chunk_text_to_sliding_window_token_chunks(
            raw_text,
            tokenizer,
            chunk_size=cfg.chunk_size,
            slice_length=cfg.slice_length,
            cursor_inc=cfg.cursor_inc_chars,
        )

    if not chunks:
        logger.warning("No chunks generated; text may be too short for window parameters. Skipping.")
        levanter.tracker.current_tracker().finish()
        return

    examples: List[LmExample] = []
    span_ranges_list: List[Tuple[int, int]] = []
    for ch in chunks:
        ids = ch["input_ids"]
        if len(ids) < Pos.size:
            ids = ids + [pad_id] * (Pos.size - len(ids))
        tokens_named = hax.named(np.array(ids, dtype=np.int32), Pos)
        ex = LmExample.from_prompt_and_completion(Pos, tokens_named, prompt_length=cfg.prompt_tokens, ignore_id=pad_id)
        examples.append(ex)
        if cfg.token_mode:
            span_ranges_list.append((ch["start_token"], ch["end_token"]))
        else:
            span_ranges_list.append((ch["start_idx"], ch["end_idx"]))

    batch_size = cfg.eval_batch_size if (cfg.eval_batch_size and cfg.eval_batch_size > 0) else 32

    # Perf: environment summary
    try:
        num_hosts = jax.process_count()
        num_devices = jax.device_count()
    except Exception:
        num_hosts = 1
        num_devices = 1
    print(
        f"perf/summary: num_hosts={num_hosts} num_devices={num_devices} "
        f"batch_size={batch_size} chunk_size={cfg.chunk_size} token_mode={cfg.token_mode}"
    )

    # Perf: compile warm-up with a representative batch
    compile_seconds = None
    if cfg.opt_warmup_compile and len(examples) > 0:
        warm_b = min(batch_size, len(examples))
        B_warm = hax.Axis(cfg.trainer.batch_axis, warm_b)
        tokens_b = hax.stack(B_warm, [examples[i].tokens for i in range(warm_b)])
        loss_b = hax.stack(B_warm, [examples[i].loss_mask for i in range(warm_b)])
        warm_batch = LmExample(tokens=tokens_b, loss_mask=loss_b, attn_mask=examples[0].attn_mask)
        t0c = time.time()
        warm_out = sequence_log_prob(model, warm_batch)
        try:
            _ = jax.block_until_ready(warm_out)
        except Exception:
            pass
        compile_seconds = time.time() - t0c
        print(f"perf/compile_seconds: {compile_seconds:.4f}")

    if cfg.use_dataloader:
        dataset = ListAsyncDataset(examples, is_complete=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            axis_resources=cfg.trainer.compute_axis_mapping,
            mesh=cfg.trainer.device_mesh,
            allow_nondivisible_batch_size=cfg.allow_nondivisible_batch_size,
        )
        iterator = ((batch, None) for batch in loader)
    else:

        def _batches():
            it = iter(zip(examples, span_ranges_list))
            while True:
                block = list(itertools.islice(it, batch_size))  # type: ignore[name-defined]
                if not block:
                    break
                exs, ranges = zip(*block)
                B = hax.Axis(cfg.trainer.batch_axis, len(exs))
                tokens_b = hax.stack(B, [e.tokens for e in exs])
                loss_b = hax.stack(B, [e.loss_mask for e in exs])
                batch = LmExample(tokens=tokens_b, loss_mask=loss_b, attn_mask=exs[0].attn_mask)
                yield batch, ranges

        iterator = _batches()

    pz_list: List[float] = []
    span_ranges: List[Tuple[int, int]] = []
    total_chunks = len(examples)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    example_offset = 0

    metric_prefix = cfg.book_title.replace(" ", "_")

    # Perf accumulators and flags
    PRINT_EVERY = 20
    device_compute_seconds_sum = 0.0
    host_batch_prep_seconds_sum = 0.0
    collective_seconds_sum = 0.0
    measured_batches = 0
    processed_windows = 0
    loop_start_time = time.time()
    last_end_time = loop_start_time
    multi_host = (num_hosts > 1) and (not cfg.allow_nondivisible_batch_size)
    attn_backend = getattr(cfg.model, "attn_backend", None)
    use_flash = getattr(cfg.model, "use_flash_attention", None)
    print(f"perf/attention_backend_config: backend={attn_backend} use_flash_attention={use_flash}")

    for idx, (batch_ex, ranges) in enumerate(iterator):
        # Host-side batch prep time (approx next() overhead): time since last iteration end
        t_iter_start = time.time()
        host_prep = t_iter_start - last_end_time
        if cfg.max_examples and idx * batch_size >= cfg.max_examples:
            break

        # Determine the local ranges for this batch (since DataLoader yields None for ranges)
        if cfg.use_dataloader:
            b = batch_ex.tokens.shape[cfg.trainer.batch_axis]
            local_ranges = span_ranges_list[example_offset : example_offset + b]
            example_offset += b
        else:
            local_ranges = list(ranges)  # type: ignore[arg-type]

        # Compute local P(z) for the batch
        t_comp_start = time.time()
        pz_local = sequence_log_prob(model, batch_ex)
        try:
            _ = jax.block_until_ready(pz_local)
        except Exception:
            pass
        device_time = time.time() - t_comp_start

        if multi_host:
            # Multi-host safe path: gather both pz and ranges so ordering aligns across hosts
            t_coll_start = time.time()
            pz_g = process_allgather(pz_local)
            ranges_arr = jnp.asarray(np.array(local_ranges, dtype=np.int32))
            ranges_g = process_allgather(ranges_arr)
            try:
                _ = jax.block_until_ready(pz_g)
                _ = jax.block_until_ready(ranges_g)
            except Exception:
                pass
            coll_time = time.time() - t_coll_start

            # Flatten and extend
            pz_list.extend(np.asarray(pz_g).reshape(-1).tolist())
            span_ranges.extend([tuple(x) for x in np.asarray(ranges_g).reshape(-1, 2).tolist()])
        else:
            # Ragged last batch allowed: avoid cross-host gather to prevent shape mismatch
            pz_list.extend(np.asarray(pz_local).reshape(-1).tolist())
            span_ranges.extend(local_ranges)
            coll_time = 0.0

        # Perf accounting
        measured_batches += 1
        processed_windows += len(local_ranges)
        device_compute_seconds_sum += device_time
        host_batch_prep_seconds_sum += host_prep
        collective_seconds_sum += coll_time
        last_end_time = time.time()

        if (idx + 1) % PRINT_EVERY == 0 or (idx + 1) == total_batches:
            elapsed = last_end_time - loop_start_time
            windows_per_sec = processed_windows / max(elapsed, 1e-6)
            tokens_per_sec = (processed_windows * cfg.chunk_size) / max(elapsed, 1e-6)
            avg_host = host_batch_prep_seconds_sum / max(measured_batches, 1)
            avg_dev = device_compute_seconds_sum / max(measured_batches, 1)
            avg_coll = collective_seconds_sum / max(measured_batches, 1)
            print(
                f"perf/progress: batch={idx+1}/{total_batches} windows={processed_windows}/{total_chunks} "
                f"wps={windows_per_sec:.2f} tps={tokens_per_sec:.2f} "
                f"host_avg_s={avg_host:.4f} dev_avg_s={avg_dev:.4f} coll_avg_s={avg_coll:.4f}"
            )

        done = min((idx + 1) * batch_size, total_chunks)
        pct = 100 * done / total_chunks
        levanter.tracker.log(
            {
                f"{metric_prefix}/eval/batch_number": idx + 1,
                f"{metric_prefix}/eval/total_batches": total_batches,
                f"{metric_prefix}/eval/windows_processed": done,
                f"{metric_prefix}/eval/total_windows": total_chunks,
                f"{metric_prefix}/eval/progress_percent": pct,
            },
            step=idx,
        )

    # Normalize to a flat 1-D array of probabilities for downstream metrics
    pz_values = np.asarray(pz_list, dtype=np.float64).reshape(-1)

    # Simple stats for logging
    stats = compute_max_extraction_rates(pz_values)
    logger.info("First few (n,p) extraction entries: %s", stats[0][:5] if stats else [])

    # Default filenames if not provided
    if cfg.plot_path is None:
        cfg.plot_path = f"bar_plot_max_pz_{cfg.book_title}.png"
    if cfg.histogram_path is None:
        cfg.histogram_path = f"pz_distribution_histogram_{cfg.book_title}.png"
    if cfg.pz_data_path is None:
        cfg.pz_data_path = f"pz_data_{cfg.book_title}.npz"

    # Create histogram image to temp file then save
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        temp_hist_path = tmp_file.name

    title_suffix = "Token Mode" if cfg.token_mode else "Character Mode"
    histogram_title = f"{cfg.book_title} - {model_name} ({title_suffix})"

    if cfg.histogram_linear:
        _ = create_pz_histogram_linear(
            pz_list=pz_values,
            threshold=cfg.pz_threshold,
            save_path=temp_hist_path,
            book_title=histogram_title,
        )
    else:
        _ = create_pz_histogram(
            pz_list=pz_values,
            threshold=cfg.pz_threshold,
            save_path=temp_hist_path,
            book_title=histogram_title,
        )

    if cfg.gcp_log:
        # Save histogram PNG to cloud
        with (
            open(temp_hist_path, "rb") as f_in,
            fsspec.open(f"{cfg.output_base_path.rstrip('/')}/{cfg.book_title}/{cfg.histogram_path}", "wb") as f_out,
        ):
            f_out.write(f_in.read())
    else:
        levanter.tracker.current_tracker().log_artifact(temp_hist_path, name=cfg.histogram_path, type="plot")
    pathlib.Path(temp_hist_path).unlink(missing_ok=True)

    # Build per-position max P(z)
    if cfg.token_mode:
        seq_len = len(token_ids) if token_ids is not None else 0
    else:
        seq_len = len(raw_text)
    max_vals = np.zeros(seq_len, dtype=np.float32)
    for pz, (s0, s1) in zip(pz_values, span_ranges):
        max_vals[s0 : s1 + 1] = np.maximum(max_vals[s0 : s1 + 1], pz)

    # Log a few aggregate stats
    if cfg.token_mode:
        levanter.tracker.log(
            {
                f"{metric_prefix}/token_analysis/mean_max_pz": float(np.mean(max_vals)) if len(max_vals) else 0.0,
                f"{metric_prefix}/token_analysis/median_max_pz": float(np.median(max_vals)) if len(max_vals) else 0.0,
                f"{metric_prefix}/token_analysis/max_max_pz": float(np.max(max_vals)) if len(max_vals) else 0.0,
                f"{metric_prefix}/token_analysis/total_tokens": int(len(max_vals)),
            },
            step=0,
        )
    else:
        levanter.tracker.log(
            {
                f"{metric_prefix}/char_analysis/mean_max_pz": float(np.mean(max_vals)) if len(max_vals) else 0.0,
                f"{metric_prefix}/char_analysis/median_max_pz": float(np.median(max_vals)) if len(max_vals) else 0.0,
                f"{metric_prefix}/char_analysis/max_max_pz": float(np.max(max_vals)) if len(max_vals) else 0.0,
                f"{metric_prefix}/char_analysis/total_chars": int(len(max_vals)),
            },
            step=0,
        )

    # Save NPZ payload (pz_values, ranges, max series, and a small config vector)
    if cfg.gcp_log:
        _save_data_to_gcp(
            None,
            cfg.output_base_path,
            cfg.pz_data_path,
            cfg.book_title,
            pz_values=np.array(pz_values),
            span_ranges=np.array(span_ranges),
            max_pz=max_vals,
            config_info=np.array(
                [
                    cfg.chunk_size,
                    cfg.prompt_tokens,
                    cfg.cursor_inc_tokens if cfg.token_mode else cfg.cursor_inc_chars,
                    (len(token_ids) if cfg.token_mode and token_ids is not None else len(raw_text)),
                ]
            ),
        )
    else:
        _save_data_with_wandb(
            None,
            cfg.pz_data_path,
            pz_values=np.array(pz_values),
            span_ranges=np.array(span_ranges),
            max_pz=max_vals,
            config_info=np.array(
                [
                    cfg.chunk_size,
                    cfg.prompt_tokens,
                    cfg.cursor_inc_tokens if cfg.token_mode else cfg.cursor_inc_chars,
                    (len(token_ids) if cfg.token_mode and token_ids is not None else len(raw_text)),
                ]
            ),
        )

    # Final perf summary
    total_elapsed = time.time() - loop_start_time
    total_windows = len(pz_values)
    total_tokens = total_windows * cfg.chunk_size
    wps = total_windows / max(total_elapsed, 1e-6)
    tps = total_tokens / max(total_elapsed, 1e-6)
    print(f"perf/summary_final: compile_seconds={compile_seconds if compile_seconds is not None else 0.0:.4f}")
    print(
        f"perf/summary_final: batches={measured_batches} windows={total_windows} tokens={total_tokens} "
        f"elapsed_seconds={total_elapsed:.2f} windows_per_sec={wps:.2f} tokens_per_sec={tps:.2f}"
    )

    levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
