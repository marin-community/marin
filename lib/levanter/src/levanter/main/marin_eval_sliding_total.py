# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-book careless suffix likelihood evaluation.

This script is a standalone variant of :mod:`eval_careless_lm` that evaluates
many books in a single run. The model and tracker are initialised once and the
same parameters are reused for every book. Each book logs metrics and artifacts
with its own prefix so progress starts at step zero for every title.
"""

import dataclasses
import itertools
import json
import logging
import math
import os
import pathlib
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from tqdm import tqdm

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

RUN_START_TIME = None


def fsspec_exists(path: str) -> bool:
    """Check if a file exists using fsspec (works with local and GCS paths)."""
    try:
        with fsspec.open(path, "r") as f:
            f.close()
            return True
    except FileNotFoundError:
        return False
    except Exception:
        # Handle other potential fsspec errors (permissions, etc.)
        return False


@dataclass
class BookConfig:
    """Overrides for a single book evaluation.

    ``plot_path``, ``histogram_path``, and ``pz_data_path`` are optional; if
    omitted they will be auto-generated from ``book_title`` and placed inside
    the per-book output directory.
    """

    txt_path: str
    plot_path: Optional[str] = None
    histogram_path: Optional[str] = None
    pz_data_path: Optional[str] = None
    book_title: str = "Book"
    chunk_size: Optional[int] = None
    slice_length: Optional[int] = None
    prompt_tokens: Optional[int] = None
    cursor_inc_chars: Optional[int] = None
    token_mode: Optional[bool] = None
    cursor_inc_tokens: Optional[int] = None
    eval_batch_size: Optional[int] = None


@dataclass
class EvalSlidingTotalConfig:
    """Configuration for careless suffix likelihood evaluation over many books."""

    # Checkpoint options ---------------------------------------------------
    checkpoint_path: Optional[str] = None
    hf_checkpoint: Optional[RepoRef] = None
    initialize_from_hf: Optional[RepoRef] = None
    use_hf_model_config: bool = False

    # Model ----------------------------------------------------------------
    model: LmConfig = field(default_factory=Gpt2Config)

    # Data -----------------------------------------------------------------
    txt_path: str | pathlib.Path = "src/levanter/data/books/gatsby.txt"
    chunk_size: int = 100
    slice_length: int = 2000
    prompt_tokens: int = 50
    cursor_inc_chars: int = 10
    token_mode: bool = False
    cursor_inc_tokens: int = 1

    # Tokenizer -------------------------------------------------------------
    tokenizer_name: Optional[str] = None

    # Runtime / Trainer ----------------------------------------------------
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    max_examples: Optional[int] = None

    # Output ---------------------------------------------------------------
    output_base_path: str = "gs://marin-us-central2/books_evals/"
    plot_path: Optional[str] = None
    eval_batch_size: int = 32
    histogram_path: Optional[str] = None
    pz_threshold: float = 0.0001
    book_title: str = "Book"
    pz_data_path: Optional[str] = None

    # Performance tweaks ---------------------------------------------------
    use_dataloader: bool = True
    histogram_linear: bool = True

    # Multi-book overrides -------------------------------------------------
    books: Dict[str, BookConfig] = field(default_factory=dict)

    # Logging options ------------------------------------------------------
    gcp_log: bool = False  # If True, save artifacts to GCP instead of WandB

    # DataLoader options ---------------------------------------------------
    allow_nondivisible_batch_size: bool = False  # Allow batch sizes not divisible by number of devices


def save_plot_with_wandb(fig, filename: str, dpi: int = 300):
    """Save plot using W&B tracker only."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        fig.savefig(tmp_file.name, dpi=dpi, bbox_inches="tight")
        tmp_path = tmp_file.name

    levanter.tracker.current_tracker().log_artifact(tmp_path, name=filename, type="plot")
    os.unlink(tmp_path)


def save_data_with_wandb(data, filename: str, **kwargs):
    """Save data using W&B tracker only."""
    suffix = pathlib.Path(filename).suffix or ".npz"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        if suffix == ".npz":
            np.savez(tmp_file.name, **kwargs)
        else:
            np.save(tmp_file.name, data)
        tmp_path = tmp_file.name

    levanter.tracker.current_tracker().log_artifact(tmp_path, name=filename, type="data")
    os.unlink(tmp_path)


def save_plot_to_gcp(fig, output_path: str, filename: str, book_title: str, dpi: int = 300):
    """Save plot directly to GCP using fsspec with book subfolder."""
    full_path = f"{output_path.rstrip('/')}/{book_title}/{filename}"
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        fig.savefig(tmp_file.name, dpi=dpi, bbox_inches="tight")
        tmp_path = tmp_file.name

    # Copy to GCP using fsspec
    with fsspec.open(full_path, "wb") as f_out:
        with open(tmp_path, "rb") as f_in:
            f_out.write(f_in.read())

    os.unlink(tmp_path)
    logger.info(f"📊 Saved plot to: {full_path}")


def save_data_to_gcp(data, output_path: str, filename: str, book_title: str, **kwargs):
    """Save data directly to GCP using fsspec with book subfolder."""
    full_path = f"{output_path.rstrip('/')}/{book_title}/{filename}"
    suffix = pathlib.Path(filename).suffix or ".npz"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        if suffix == ".npz":
            np.savez(tmp_file.name, **kwargs)
        else:
            np.save(tmp_file.name, data)
        tmp_path = tmp_file.name

    # Copy to GCP using fsspec
    with fsspec.open(full_path, "wb") as f_out:
        with open(tmp_path, "rb") as f_in:
            f_out.write(f_in.read())

    os.unlink(tmp_path)
    logger.info(f"💾 Saved data to: {full_path}")


# ---------------------------------------------------------------------------
# Per-book evaluation
# ---------------------------------------------------------------------------


def evaluate_book(
    cfg: EvalSlidingTotalConfig,
    model: LmHeadModel,
    tokenizer,
    sequence_log_prob,
    pad_id: int,
    Pos: hax.Axis,
    model_name: str,
    main_cfg: EvalSlidingTotalConfig,  # Pass main config for gcp_log flag
):
    """Run careless evaluation on a single book using a pre-loaded model."""

    # Use fsspec to read text files (supports both local and remote paths like gs://)
    with fsspec.open(cfg.txt_path, "r") as f:
        raw_text = f.read()

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

    # Check if chunks are empty and handle gracefully
    if not chunks:
        logger.warning(f"⚠️  WARNING: No chunks generated for book '{cfg.book_title}' (txt_path: {cfg.txt_path})")
        logger.warning("⚠️  Book may be too short for sliding window parameters:")
        logger.warning(f"⚠️    - chunk_size: {cfg.chunk_size}")
        if cfg.token_mode:
            logger.warning(f"⚠️    - cursor_inc_tokens: {cfg.cursor_inc_tokens}")
            logger.warning(f"⚠️    - token_ids length: {len(token_ids) if token_ids else 0} tokens")
        else:
            logger.warning(f"⚠️    - slice_length: {cfg.slice_length}")
            logger.warning(f"⚠️    - cursor_inc_chars: {cfg.cursor_inc_chars}")
        logger.warning(f"⚠️    - raw_text length: {len(raw_text)} characters")
        logger.warning("⚠️  Skipping evaluation for this book.")
        return

    examples: List[LmExample] = []
    span_ranges_list: List[Tuple[int, int]] = []
    print(f"🔍 DEBUG: Processing {len(chunks)} chunks into examples...", flush=True)
    for chunk in chunks:
        ids = chunk["input_ids"]
        if len(ids) < Pos.size:
            ids = ids + [pad_id] * (Pos.size - len(ids))
        tokens_named = hax.named(np.array(ids, dtype=np.int32), Pos)
        ex = LmExample.from_prompt_and_completion(Pos, tokens_named, prompt_length=cfg.prompt_tokens, ignore_id=pad_id)
        examples.append(ex)
        if cfg.token_mode:
            span_ranges_list.append((chunk["start_token"], chunk["end_token"]))
        else:
            span_ranges_list.append((chunk["start_idx"], chunk["end_idx"]))

    total_chunks = len(examples)
    batch_size = cfg.eval_batch_size if (cfg.eval_batch_size and cfg.eval_batch_size > 0) else 32

    if cfg.use_dataloader:
        dataset = ListAsyncDataset(examples, is_complete=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            axis_resources=cfg.trainer.compute_axis_mapping,
            mesh=cfg.trainer.device_mesh,
            allow_nondivisible_batch_size=cfg.allow_nondivisible_batch_size,
        )
        batches = ((batch, None) for batch in loader)
    else:
        examples_iter = iter(zip(examples, span_ranges_list))

        def batches(it):
            while True:
                block = list(itertools.islice(it, batch_size))
                if not block:
                    break
                exs, ranges = zip(*block)
                B = hax.Axis(cfg.trainer.batch_axis, len(exs))
                tokens_b = hax.stack(B, [e.tokens for e in exs])
                loss_b = hax.stack(B, [e.loss_mask for e in exs])
                batch = LmExample(tokens=tokens_b, loss_mask=loss_b, attn_mask=exs[0].attn_mask)
                yield batch, ranges

    pz_list: List[float] = []
    span_ranges: List[Tuple[int, int]] = []

    total_batches = math.ceil(total_chunks / batch_size)
    example_offset = 0

    iterator = batches if cfg.use_dataloader else batches(examples_iter)

    metric_prefix = cfg.book_title.replace(" ", "_")

    for idx, (batch_ex, ranges) in enumerate(iterator):
        if cfg.max_examples and idx * batch_size >= cfg.max_examples:
            break

        if cfg.use_dataloader:
            b = batch_ex.tokens.shape[cfg.trainer.batch_axis]
            ranges = span_ranges_list[example_offset : example_offset + b]
            example_offset += b

        pz = process_allgather(sequence_log_prob(model, batch_ex))
        pz_list.extend(np.array(pz).tolist())
        span_ranges.extend(ranges)

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

    stats = compute_max_extraction_rates(pz_list)
    logger.info("First few (n,p) extraction entries: %s", stats[0][:5])

    mode_suffix = "Token Mode" if cfg.token_mode else "Character Mode"
    histogram_book_title = f"{cfg.book_title} - {model_name} ({mode_suffix})"

    # Create histogram and save via W&B only
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        temp_histogram_path = tmp_file.name

    if cfg.histogram_linear:
        hist_stats = create_pz_histogram_linear(
            pz_list=pz_list,
            threshold=cfg.pz_threshold,
            save_path=temp_histogram_path,
            book_title=histogram_book_title,
            model_name=model_name,
        )
    else:
        hist_stats = create_pz_histogram(
            pz_list=pz_list,
            threshold=cfg.pz_threshold,
            save_path=temp_histogram_path,
            book_title=histogram_book_title,
        )

    if hist_stats:
        # Save histogram - choose method based on gcp_log flag
        if main_cfg.gcp_log:
            # Copy histogram to GCP using fsspec with book subfolder
            histogram_gcp_path = f"{main_cfg.output_base_path.rstrip('/')}/{cfg.book_title}/{cfg.histogram_path}"
            with fsspec.open(histogram_gcp_path, "wb") as f_out:
                with open(temp_histogram_path, "rb") as f_in:
                    f_out.write(f_in.read())
            logger.info(f"📊 Saved histogram to: {histogram_gcp_path}")
        else:
            levanter.tracker.current_tracker().log_artifact(temp_histogram_path, name=cfg.histogram_path, type="plot")
    os.unlink(temp_histogram_path)

    if cfg.token_mode:
        seq_len = len(token_ids)
        max_vals = np.zeros(seq_len, dtype=np.float32)
    else:
        seq_len = len(raw_text)
        max_vals = np.zeros(seq_len, dtype=np.float32)

    for pz, (s0, s1) in zip(pz_list, span_ranges):
        max_vals[s0 : s1 + 1] = np.maximum(max_vals[s0 : s1 + 1], pz)

    if cfg.token_mode:
        levanter.tracker.log(
            {
                f"{metric_prefix}/token_analysis/mean_max_pz": float(np.mean(max_vals)),
                f"{metric_prefix}/token_analysis/median_max_pz": float(np.median(max_vals)),
                f"{metric_prefix}/token_analysis/max_max_pz": float(np.max(max_vals)),
                f"{metric_prefix}/token_analysis/tokens_above_0.5": int(np.sum(max_vals > 0.5)),
                f"{metric_prefix}/token_analysis/tokens_above_0.9": int(np.sum(max_vals > 0.9)),
                f"{metric_prefix}/token_analysis/total_tokens": len(max_vals),
            },
            step=0,
        )
    else:
        levanter.tracker.log(
            {
                f"{metric_prefix}/char_analysis/mean_max_pz": float(np.mean(max_vals)),
                f"{metric_prefix}/char_analysis/median_max_pz": float(np.median(max_vals)),
                f"{metric_prefix}/char_analysis/max_max_pz": float(np.max(max_vals)),
                f"{metric_prefix}/char_analysis/chars_above_0.5": int(np.sum(max_vals > 0.5)),
                f"{metric_prefix}/char_analysis/chars_above_0.9": int(np.sum(max_vals > 0.9)),
                f"{metric_prefix}/char_analysis/total_chars": len(max_vals),
            },
            step=0,
        )

    # Save pz data - choose method based on gcp_log flag
    if main_cfg.gcp_log:
        save_data_to_gcp(
            None,
            main_cfg.output_base_path,
            cfg.pz_data_path,
            cfg.book_title,
            pz_values=np.array(pz_list),
            span_ranges=np.array(span_ranges),
            max_pz=max_vals,
            config_info=np.array(
                [
                    cfg.chunk_size,
                    cfg.prompt_tokens,
                    cfg.cursor_inc_tokens if cfg.token_mode else cfg.cursor_inc_chars,
                    len(token_ids) if cfg.token_mode else len(raw_text),
                ]
            ),
        )
    else:
        save_data_with_wandb(
            None,
            cfg.pz_data_path,
            pz_values=np.array(pz_list),
            span_ranges=np.array(span_ranges),
            max_pz=max_vals,
            config_info=np.array(
                [
                    cfg.chunk_size,
                    cfg.slice_length,
                    cfg.prompt_tokens,
                    cfg.cursor_inc_tokens if cfg.token_mode else cfg.cursor_inc_chars,
                    len(token_ids) if cfg.token_mode else len(raw_text),
                ]
            ),
        )

    fig, ax = plt.subplots(figsize=(14, 2))
    im = ax.imshow(
        max_vals[np.newaxis, :],
        cmap="Blues",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    if cfg.token_mode:
        ax.set_title(f"{cfg.book_title}: Maximum per-token probability for {model_name}")
        ax.set_xlabel("Book position (token)")
    else:
        ax.set_title(f"{cfg.book_title}: Maximum per-character probability for {model_name}")
        ax.set_xlabel("Book position (character)")
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Max. probability")

    plt.tight_layout()
    # Save plot and data - choose method based on gcp_log flag
    if main_cfg.gcp_log:
        save_plot_to_gcp(fig, main_cfg.output_base_path, cfg.plot_path, cfg.book_title, dpi=300)
        npy_filename = pathlib.Path(cfg.plot_path).with_suffix(".npy").name
        save_data_to_gcp(max_vals, main_cfg.output_base_path, npy_filename, cfg.book_title)
    else:
        save_plot_with_wandb(fig, cfg.plot_path, dpi=300)
        npy_filename = pathlib.Path(cfg.plot_path).with_suffix(".npy").name
        save_data_with_wandb(max_vals, npy_filename)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(cfg: EvalSlidingTotalConfig):
    """Run careless suffix evaluation for each book listed in ``cfg.books``.

    The model and tracker are initialised once; each book reuses the same
    parameters but logs metrics and artifacts with a book-specific prefix and
    output directory.  If ``plot_path``/``histogram_path``/``pz_data_path`` are
    omitted for a book, filenames are derived from the book title to ensure
    uniqueness.
    """

    global RUN_START_TIME
    RUN_START_TIME = time.time()

    levanter.initialize(cfg)

    # Tokenizer -------------------------------------------------------------
    if cfg.tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    else:
        tokenizer = getattr(cfg.model, "the_tokenizer", None)

    if tokenizer is None:
        raise ValueError("Tokenizer not provided: set tokenizer_name in config or ensure model.the_tokenizer exists")

    Pos = cfg.model.Pos
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # Build / load model once ----------------------------------------------
    cmapping = cfg.trainer.compute_axis_mapping
    pmapping = cfg.trainer.parameter_axis_mapping

    with cfg.trainer.device_mesh, hax.axis_mapping(pmapping):
        key = jax.random.PRNGKey(0)
        vocab_size = len(tokenizer)
        Vocab = round_axis_for_partitioning(hax.Axis("vocab", vocab_size), cmapping)
        mp: jmp.Policy = cfg.trainer.mp

        if cfg.checkpoint_path:
            with use_cpu_device():
                model = eqx.filter_eval_shape(cfg.model.build, Vocab, key=key)
                model = load_checkpoint(model, cfg.checkpoint_path, subpath="model")
            model = hax.shard_with_axis_mapping(model, pmapping)
        else:
            hf_ref = cfg.hf_checkpoint or cfg.initialize_from_hf
            if hf_ref is None:
                raise ValueError("Need --checkpoint-path or --hf-checkpoint")
            converter: HFCheckpointConverter = cfg.model.hf_checkpoint_converter()
            converter = converter.replaced(reference_checkpoint=hf_ref, tokenizer=tokenizer)
            if cfg.use_hf_model_config:
                cfg.model = converter.config_from_hf_config(converter.default_hf_config)
            model = converter.load_pretrained(cfg.model.model_type, ref=hf_ref, dtype=mp.compute_dtype)

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
        if "--" in model_path:
            parts = model_path.split("--")
            if len(parts) >= 2:
                extracted_name = parts[1].lower().replace("-", " ").title()  # llama-13b -> Llama 13B
                print(f"🔍 DEBUG: Found '--' in path, extracted_name = {extracted_name}", flush=True)
                model_name = extracted_name
            else:
                extracted_name = parts[-1].lower()
                print(f"🔍 DEBUG: Fallback to last part, extracted_name = {extracted_name}", flush=True)
                model_name = extracted_name
        else:
            model_name = pathlib.Path(model_path).name.lower()
    elif cfg.hf_checkpoint:
        model_name = str(cfg.hf_checkpoint).split("/")[-1].lower().replace("-", "-")

    # Set up progress bar for book evaluation
    book_items = list(cfg.books.items())
    total_books = len(book_items)

    for name, book in tqdm(book_items, desc="📚 Evaluating books", unit="book"):
        book_cfg = dataclasses.replace(cfg)
        for field_name, value in dataclasses.asdict(book).items():
            if value is not None:
                setattr(book_cfg, field_name, value)

        if book_cfg.book_title == "Book":
            book_cfg.book_title = name

        if book_cfg.plot_path is None:
            book_cfg.plot_path = f"bar_plot_max_pz_{book_cfg.book_title}.png"
        if book_cfg.histogram_path is None:
            book_cfg.histogram_path = f"pz_distribution_histogram_{book_cfg.book_title}.png"
        if book_cfg.pz_data_path is None:
            book_cfg.pz_data_path = f"pz_data_{book_cfg.book_title}.npz"

        # Check if this book evaluation is already completed (caching logic)
        success_file_path = f"{cfg.output_base_path.rstrip('/')}/{book_cfg.book_title}.success"

        if fsspec_exists(success_file_path):
            logger.info(f"✅ Book '{book_cfg.book_title}' already completed, skipping")
            continue

        # No filesystem directory needed - all files saved via W&B
        logger.info(f"🔥 Starting evaluation for: {book_cfg.book_title}")
        start_time = time.time()

        try:
            evaluate_book(book_cfg, model, tokenizer, sequence_log_prob, pad_id, Pos, model_name, cfg)

            elapsed_time = time.time() - start_time
            logger.info(f"✅ Completed {book_cfg.book_title} in {elapsed_time:.1f}s")

            # Write success file after successful completion
            success_metadata = {
                "book_title": book_cfg.book_title,
                "book_path": book_cfg.txt_path,
                "completed_at": datetime.utcnow().isoformat(),
                "elapsed_time_seconds": elapsed_time,
                "status": "success",
            }

            with fsspec.open(success_file_path, "w") as f:
                f.write(json.dumps(success_metadata, indent=2))

            logger.info(f"📝 Success marker written to: {success_file_path}")

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ Failed {book_cfg.book_title} after {elapsed_time:.1f}s: {e}")
            # Don't write success file on failure, allowing retry on restart
            raise  # Re-raise to trigger job restart

    logger.info(f"🎉 Completed evaluation of all {total_books} books!")

    levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
