# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""src.levanter.main.eval_careless_lm
===================================
On-the-fly suffix‐likelihood evaluation over a plain-text file.

Compared with ``eval_sliding_lm.py`` this script:
1.  Generates ``LmExample``s directly from a **text cursor** (no cache).
2.  Slides one or more *characters* at a time through the book.
3.  After scoring all suffix windows, produces a *character-level* plot
    of the **maximum** suffix probability covering each character.

It also calls ``levanter.books.util.compute_max_extraction_rates`` to
print the (n, p) discoverability statistics used in memorisation studies.
"""

import itertools
import logging
import math
import os
import pathlib
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import equinox as eqx
import fsspec
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

# Helpers -----------------------------------------------------------------
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


jax.config.update("jax_default_matmul_precision", "highest")
# -----------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)

RUN_START_TIME = None


@dataclass
class EvalCarelessLmConfig:
    """YAML/CLI config for the script (mirrors Levanter style)."""

    # Checkpoint options -------------------------------------------------------
    checkpoint_path: Optional[str] = None  # local Levanter checkpoint
    hf_checkpoint: Optional[RepoRef] = None  # e.g. "openai/gpt2"
    initialize_from_hf: Optional[RepoRef] = None
    use_hf_model_config: bool = False

    # Model architecture -------------------------------------------------------
    model: LmConfig = field(default_factory=Gpt2Config)

    # Data ---------------------------------------------------------------------
    txt_path: str | pathlib.Path = "src/levanter/data/books/gatsby.txt"
    chunk_size: int = 100
    slice_length: int = 2000
    prompt_tokens: int = 50
    cursor_inc_chars: int = 10  # stride in characters
    token_mode: bool = False
    cursor_inc_tokens: int = 1  # stride in tokens when token_mode is true

    # Tokenizer ----------------------------------------------------------------
    tokenizer_name: Optional[str] = None  # e.g. "meta-llama/Llama-3.1-8B"

    # Runtime / Trainer --------------------------------------------------------
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    max_examples: Optional[int] = None  # cap for quick debug; None → all

    # Output -------------------------------------------------------------------
    output_base_path: str = "gs://marin-us-central2/books_evals/"  # Base path for all outputs
    plot_path: str = "bar_plot_char_max_pz_70b.png"
    eval_batch_size: int = 32
    histogram_path: str = "pz_distribution_histogram.png"
    pz_threshold: float = 0.0001
    book_title: str = "Book"
    pz_data_path: str = "pz_data.npz"

    # Performance tweaks -------------------------------------------------------
    use_dataloader: bool = True  # DataLoader keeps devices busy but uses additional host memory
    histogram_linear: bool = True


# -----------------------------------------------------------------------------------------
# Main ------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------


def upload_hlo_dumps_to_wandb():
    """Upload HLO dumps to WandB as artifacts (only from current run)"""

    global RUN_START_TIME

    xla_dump_dir = "/tmp/xla_dumps"
    if not os.path.exists(xla_dump_dir):
        logger.warning("No XLA dumps found at /tmp/xla_dumps")
        return

    # Record current time as the cutoff - only files newer than this run's start
    # We'll use a time shortly before model initialization as the cutoff
    current_time = time.time()
    # Assume run started at most 1 hour ago (adjust based on your typical run times)
    run_start_cutoff = current_time - (1 * 60 * 60)  # 1 hour ago

    # Collect files that were created during this run
    recent_dump_files = []
    all_files = []

    for root, dirs, files in os.walk(xla_dump_dir):
        for file in files:
            filepath = os.path.join(root, file)
            all_files.append(filepath)

            # Check file modification time
            file_mtime = os.path.getmtime(filepath)
            if file_mtime > run_start_cutoff:
                recent_dump_files.append(filepath)

    logger.info(
        f"Found {len(all_files)} total files, {len(recent_dump_files)} recent files (modified after"
        f" {time.ctime(run_start_cutoff)})"
    )

    if not recent_dump_files:
        logger.warning("No recent XLA dump files found from this run")
        return

    # Create a tar archive with only recent files
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = tmp_file.name

    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for filepath in recent_dump_files:
                # Add each file individually to maintain directory structure
                arcname = os.path.relpath(filepath, os.path.dirname(xla_dump_dir))
                tar.add(filepath, arcname=arcname)

        # Upload to WandB
        levanter.tracker.current_tracker().log_artifact(tar_path, name="hlo_dumps.tar.gz", type="hlo_analysis")
        logger.info(f"Successfully uploaded {len(recent_dump_files)} recent HLO dumps to WandB")

        # Log statistics
        recent_file_sizes = [os.path.getsize(f) for f in recent_dump_files]
        levanter.tracker.log(
            {
                "hlo_analysis/total_dump_files": len(recent_dump_files),
                "hlo_analysis/total_files_in_dir": len(all_files),
                "hlo_analysis/dump_dir_size_mb": sum(recent_file_sizes) / (1024 * 1024),
                "hlo_analysis/run_start_cutoff": run_start_cutoff,
            },
            step=0,
        )

    except Exception as e:
        logger.error(f"Failed to upload HLO dumps: {e}")
    finally:
        # Clean up temporary file
        os.unlink(tar_path)


def get_full_output_path(cfg: EvalCarelessLmConfig, filename: str) -> str:
    """Construct full output path by joining base path with filename."""
    if cfg.output_base_path.endswith("/"):
        return cfg.output_base_path + filename
    else:
        return cfg.output_base_path + "/" + filename


def save_plot_with_fsspec(fig, output_path: str, dpi: int = 300):
    """Save matplotlib figure using fsspec for cloud storage compatibility."""
    if output_path.startswith("gs://"):
        # For GCS paths, save to temporary file then copy
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            fig.savefig(tmp_file.name, dpi=dpi, bbox_inches="tight")
            tmp_path = tmp_file.name

        # Copy to GCS using fsspec
        with fsspec.open(tmp_path, "rb") as local_file:
            with fsspec.open(output_path, "wb") as remote_file:
                remote_file.write(local_file.read())

        # Clean up temp file
        os.unlink(tmp_path)
    else:
        # Local path - save directly
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")


def save_data_with_fsspec(data, output_path: str, **kwargs):
    """Save numpy data using fsspec for cloud storage compatibility."""
    if output_path.startswith("gs://"):
        # For GCS paths, save to temporary file then copy
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp_file:
            if output_path.endswith(".npz"):
                np.savez(tmp_file.name, **kwargs)
            else:
                np.save(tmp_file.name, data)
            tmp_path = tmp_file.name

        # Copy to GCS using fsspec
        with fsspec.open(tmp_path, "rb") as local_file:
            with fsspec.open(output_path, "wb") as remote_file:
                remote_file.write(local_file.read())

        # Clean up temp file
        os.unlink(tmp_path)
    else:
        # Local path - save directly
        if output_path.endswith(".npz"):
            np.savez(output_path, **kwargs)
        else:
            np.save(output_path, data)


def main(cfg: EvalCarelessLmConfig):
    global RUN_START_TIME
    RUN_START_TIME = time.time()

    print(f"[TIMING] Starting levanter initialization at {time.time():.3f}", flush=True)
    start_time = time.time()
    levanter.initialize(cfg)
    print(f"[TIMING] Levanter initialize took {time.time() - start_time:.3f}s", flush=True)

    # Extract model name from HF path for plot titles
    model_name = "Unknown Model"
    if cfg.initialize_from_hf:
        # Extract model name from paths like "/opt/gcsfuse_mount/models/meta-llama--Llama-3-1-70B"
        model_path = str(cfg.initialize_from_hf)
        if "--" in model_path:
            model_name = model_path.split("--")[-1].lower().replace("-", "-")
        else:
            model_name = pathlib.Path(model_path).name.lower()
    elif cfg.hf_checkpoint:
        # Handle RepoRef objects
        model_name = str(cfg.hf_checkpoint).split("/")[-1].lower().replace("-", "-")

    # Construct full output paths
    full_plot_path = get_full_output_path(cfg, cfg.plot_path)
    full_histogram_path = get_full_output_path(cfg, cfg.histogram_path)
    full_pz_data_path = get_full_output_path(cfg, cfg.pz_data_path)
    # Tokenizer & axes ---------------------------------------------------------
    print(f"[TIMING] Starting tokenizer setup at {time.time():.3f}", flush=True)
    start_time = time.time()
    if cfg.tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
    else:
        # fall back to model-provided tokenizer (works for GPT-style configs)
        tokenizer = getattr(cfg.model, "the_tokenizer", None)

    if tokenizer is None:
        raise ValueError("Tokenizer not provided: set tokenizer_name in config or ensure model.the_tokenizer exists")

    Pos = cfg.model.Pos
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    print(f"[TIMING] Tokenizer setup took {time.time() - start_time:.3f}s", flush=True)

    # Build / load model --------------------------------------------------------
    print(f"[TIMING] Starting model loading at {time.time():.3f}", flush=True)
    start_time = time.time()
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

        # Sequence log-probability function ------------------------------------
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

    print(f"[TIMING] Model loading took {time.time() - start_time:.3f}s", flush=True)

    # Data stream ---------------------------------------------------------------
    print(f"[TIMING] Starting data preprocessing at {time.time():.3f}", flush=True)
    start_time = time.time()
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

    examples: list[LmExample] = []
    span_ranges_list: list[Tuple[int, int]] = []
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
    print(f"Total sliding windows: {total_chunks}", flush=True)
    print(f"[TIMING] Data preprocessing took {time.time() - start_time:.3f}s", flush=True)

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

    # Evaluation loop -----------------------------------------------------------
    print(f"[TIMING] Starting evaluation loop at {time.time():.3f}", flush=True)
    eval_start_time = time.time()
    pz_list: List[float] = []
    span_ranges: List[Tuple[int, int]] = []

    total_batches = math.ceil(total_chunks / batch_size)
    example_offset = 0

    iterator = batches if cfg.use_dataloader else batches(examples_iter)

    # Timing accumulators
    forward_time_total = 0.0
    transfer_time_total = 0.0
    batch_count = 0

    for idx, (batch_ex, ranges) in enumerate(iterator):
        if cfg.max_examples and idx * batch_size >= cfg.max_examples:
            break

        if cfg.use_dataloader:
            b = batch_ex.tokens.shape[cfg.trainer.batch_axis]
            ranges = span_ranges_list[example_offset : example_offset + b]
            example_offset += b

        # Time forward pass
        forward_start = time.time()
        pz_device = sequence_log_prob(model, batch_ex)
        forward_time = time.time() - forward_start
        forward_time_total += forward_time

        # Time host-device transfer
        transfer_start = time.time()
        pz = process_allgather(pz_device)
        transfer_time = time.time() - transfer_start
        transfer_time_total += transfer_time

        pz_list.extend(np.array(pz).tolist())
        span_ranges.extend(ranges)
        batch_count += 1

        done = min((idx + 1) * batch_size, total_chunks)
        pct = 100 * done / total_chunks

        # Print detailed timing every 10 batches
        if (idx + 1) % 10 == 0 or idx == 0:
            avg_forward = forward_time_total / batch_count
            avg_transfer = transfer_time_total / batch_count
            print(
                f"[TIMING] Batch {idx+1}: Forward={forward_time:.3f}s, Transfer={transfer_time:.3f}s, Avg:"
                f" Forward={avg_forward:.3f}s, Transfer={avg_transfer:.3f}s",
                flush=True,
            )

        print(f"Batch {idx+1}/{total_batches} – {done}/{total_chunks} windows ({pct:.1f} %)", flush=True)
        levanter.tracker.log(
            {
                "eval/batch_number": idx + 1,
                "eval/total_batches": total_batches,
                "eval/windows_processed": done,
                "eval/total_windows": total_chunks,
                "eval/progress_percent": pct,
            },
            step=idx,
        )

    eval_total_time = time.time() - eval_start_time
    print(f"[TIMING] Evaluation loop completed in {eval_total_time:.3f}s", flush=True)
    print(
        f"[TIMING] Total forward time: {forward_time_total:.3f}s ({forward_time_total/eval_total_time*100:.1f}%)",
        flush=True,
    )
    print(
        f"[TIMING] Total transfer time: {transfer_time_total:.3f}s ({transfer_time_total/eval_total_time*100:.1f}%)",
        flush=True,
    )
    print(
        f"[TIMING] Average per batch: Forward={forward_time_total/batch_count:.3f}s,"
        f" Transfer={transfer_time_total/batch_count:.3f}s",
        flush=True,
    )

    # Extraction statistics -----------------------------------------------------
    print(f"[TIMING] Starting extraction statistics at {time.time():.3f}", flush=True)
    start_time = time.time()
    stats = compute_max_extraction_rates(pz_list)
    logger.info("First few (n,p) extraction entries: %s", stats[0][:5])

    # Create and save histogram ------------------------------------------------
    mode_suffix = "Token Mode" if cfg.token_mode else "Character Mode"
    histogram_book_title = f"{cfg.book_title} - {model_name} ({mode_suffix})"
    if cfg.histogram_linear:
        hist_stats = create_pz_histogram_linear(
            pz_list=pz_list, threshold=cfg.pz_threshold, save_path=full_histogram_path, book_title=histogram_book_title
        )
    else:
        hist_stats = create_pz_histogram(
            pz_list=pz_list, threshold=cfg.pz_threshold, save_path=full_histogram_path, book_title=histogram_book_title
        )
    if hist_stats:
        logger.info("P_z histogram statistics: %s", hist_stats)
        # Log the histogram as an artifact
        levanter.tracker.current_tracker().log_artifact(full_histogram_path, name=cfg.histogram_path, type="plot")

    # Character- or token-level max-P(z) curve --------------------------------
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
                "token_analysis/mean_max_pz": float(np.mean(max_vals)),
                "token_analysis/median_max_pz": float(np.median(max_vals)),
                "token_analysis/max_max_pz": float(np.max(max_vals)),
                "token_analysis/tokens_above_0.5": int(np.sum(max_vals > 0.5)),
                "token_analysis/tokens_above_0.9": int(np.sum(max_vals > 0.9)),
                "token_analysis/total_tokens": len(max_vals),
            },
            step=0,
        )
    else:
        levanter.tracker.log(
            {
                "char_analysis/mean_max_pz": float(np.mean(max_vals)),
                "char_analysis/median_max_pz": float(np.median(max_vals)),
                "char_analysis/max_max_pz": float(np.max(max_vals)),
                "char_analysis/chars_above_0.5": int(np.sum(max_vals > 0.5)),
                "char_analysis/chars_above_0.9": int(np.sum(max_vals > 0.9)),
                "char_analysis/total_chars": len(max_vals),
            },
            step=0,
        )

    # Save pz_list and related data as npz file
    save_data_with_fsspec(
        None,  # not used for npz
        full_pz_data_path,
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
    levanter.tracker.current_tracker().log_artifact(full_pz_data_path, name=cfg.pz_data_path, type="data")
    # ------------------------------------------------------------------
    # Visualization: show a *single-row* heat-map so each character
    # column is a vertical bar whose colour encodes max-P(z).
    # Darker = closer to 1, lighter = closer to 0 (see Blues colormap).
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 2))
    im = ax.imshow(
        max_vals[np.newaxis, :],  # shape (1, seq_len)
        cmap="Blues",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    if cfg.token_mode:
        ax.set_title(f"{cfg.book_title} - {model_name}: Maximum per-token probability (Token Mode)")
        ax.set_xlabel("Book position (token)")
    else:
        ax.set_title(f"{cfg.book_title} - {model_name}: Maximum per-character probability (Character Mode)")
        ax.set_xlabel("Book position (character)")
    ax.set_yticks([])  # Hide y-axis (only one row)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Max. probability")

    plt.tight_layout()
    save_plot_with_fsspec(fig, full_plot_path, dpi=300)
    levanter.tracker.current_tracker().log_artifact(full_plot_path, name=cfg.plot_path, type="plot")

    npy_filename = pathlib.Path(cfg.plot_path).with_suffix(".npy").name
    full_npy_path = get_full_output_path(cfg, npy_filename)
    save_data_with_fsspec(max_vals, full_npy_path)
    levanter.tracker.current_tracker().log_artifact(full_npy_path, name=npy_filename, type="array")

    upload_hlo_dumps_to_wandb()
    levanter.tracker.current_tracker().finish()


if __name__ == "__main__":
    levanter.config.main(main)()
