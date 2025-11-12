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
import logging
import math
import os
import pathlib
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental.multihost_utils import process_allgather

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


# ---------------------------------------------------------------------------
# Helpers copied from eval_careless_lm
# ---------------------------------------------------------------------------


def upload_hlo_dumps_to_wandb():
    """Upload HLO dumps created during the run to WandB."""

    global RUN_START_TIME
    xla_dump_dir = "/tmp/xla_dumps"
    if not os.path.exists(xla_dump_dir):
        return

    current_time = time.time()
    run_start_cutoff = current_time - (1 * 60 * 60)  # 1 hour

    recent_dump_files = []
    for root, dirs, files in os.walk(xla_dump_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.getmtime(filepath) > run_start_cutoff:
                recent_dump_files.append(filepath)

    if not recent_dump_files:
        return

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = tmp_file.name

    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for filepath in recent_dump_files:
                arcname = os.path.relpath(filepath, os.path.dirname(xla_dump_dir))
                tar.add(filepath, arcname=arcname)

        levanter.tracker.current_tracker().log_artifact(tar_path, name="hlo_dumps.tar.gz", type="hlo_analysis")
    finally:
        os.unlink(tar_path)


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
):
    """Run careless evaluation on a single book using a pre-loaded model."""

    raw_text = pathlib.Path(cfg.txt_path).read_text()

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

    examples: List[LmExample] = []
    span_ranges_list: List[Tuple[int, int]] = []
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
        )
    else:
        hist_stats = create_pz_histogram(
            pz_list=pz_list,
            threshold=cfg.pz_threshold,
            save_path=temp_histogram_path,
            book_title=histogram_book_title,
        )

    if hist_stats:
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

    save_data_with_wandb(
        None,
        cfg.pz_data_path,
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
        ax.set_title(f"{cfg.book_title}: Maximum per-token probability")
        ax.set_xlabel("Book position (token)")
    else:
        ax.set_title(f"{cfg.book_title}: Maximum per-character probability")
        ax.set_xlabel("Book position (character)")
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Max. probability")

    plt.tight_layout()
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

    import sys

    print("🔥 EVAL_SLIDING_TOTAL main: Function started!", flush=True)
    sys.stdout.flush()

    global RUN_START_TIME
    RUN_START_TIME = time.time()

    print("🔥 EVAL_SLIDING_TOTAL main: About to call levanter.initialize", flush=True)
    sys.stdout.flush()

    levanter.initialize(cfg)

    print("🔥 EVAL_SLIDING_TOTAL main: levanter.initialize completed!", flush=True)
    sys.stdout.flush()

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
            model_name = model_path.split("--")[-1].lower().replace("-", "-")
        else:
            model_name = pathlib.Path(model_path).name.lower()
    elif cfg.hf_checkpoint:
        model_name = str(cfg.hf_checkpoint).split("/")[-1].lower().replace("-", "-")

    for name, book in cfg.books.items():
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

        # No filesystem directory needed - all files saved via W&B

        evaluate_book(book_cfg, model, tokenizer, sequence_log_prob, pad_id, Pos, model_name)

    upload_hlo_dumps_to_wandb()
    levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
