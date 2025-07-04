from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict
from enum import Enum, auto

import ray
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import fsspec
import datasets
import pyarrow as pa
import pyarrow.parquet as pq

from marin.processing.classification.inference import write_dataset
from marin.utils import remove_tpu_lockfile_on_exit, fsspec_mkdirs


def chunk_text_to_sliding_window_token_chunks(
    text: str,
    tokenizer,
    *,
    chunk_size: int = 100,
    slice_length: int = 2000,
    cursor_inc: int = 10,
) -> list[Dict[str, Any]]:
    """Tokenise *text* into overlapping `chunk_size`-token windows.

    Replicates the logic in ``careless.py`` almost verbatim but drops the
    torch-specific bits.  Returns a list of dictionaries with keys:

    ``input_ids``          – *list[int]* of length ``chunk_size``
    ``start_idx``          – start character index in *text*
    ``end_idx``            – end character index (inclusive) in *text*
    ``text``               – decoded chunk text (useful for debugging)
    ``attention_mask``     – list[int] same length as ``input_ids``
    ``text_len``           – length of decoded text in characters
    """

    all_chunks: list[Dict[str, Any]] = []
    text_cursor = 0
    text_len = len(text)

    progress_markers = {i for i in range(10, 101, 10)}

    while text_cursor < text_len:
        start_idx = text_cursor
        end_idx_plus_one = min(text_cursor + slice_length, text_len)
        text_slice = text[start_idx:end_idx_plus_one]

        enc = tokenizer(text_slice, add_special_tokens=False, return_attention_mask=True)
        input_ids: list[int] = enc["input_ids"][:chunk_size]
        attention_mask: list[int] = enc.get("attention_mask", [1] * len(input_ids))[:chunk_size]

        if len(input_ids) == chunk_size:
            decoded_chunk = tokenizer.decode(input_ids, skip_special_tokens=True)
            decoded_len = len(decoded_chunk)
            all_chunks.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "start_idx": start_idx,
                    "end_idx": start_idx + decoded_len - 1,
                    "text_len": decoded_len,
                    "text": decoded_chunk,
                }
            )

        text_cursor += cursor_inc
        pct_complete = int(100 * text_cursor / text_len)
        if pct_complete in progress_markers:
            logging.getLogger(__name__).info("Sliding-window progress: %s%%", pct_complete)
            progress_markers.remove(pct_complete)

    return all_chunks


# ---------------------------------------------------------------------------
# Logging & Enums
# ---------------------------------------------------------------------------
logger = logging.getLogger("ray")


class Precision(Enum):
    FLOAT16 = auto()
    FLOAT32 = auto()


@dataclass
class SlidingLogitsConfig:
    """Configuration for sliding-window forward-pass logging."""

    model_name: str
    input_path: str  # path to raw txt (local or gs://)
    output_dir: str  # directory where parquet + plot will be written

    # Runtime / batching --------------------------------------------------
    batch_size: int = 8
    memory_gb: int = 10

    # Chunk parameters ----------------------------------------------------
    chunk_size: int = 100
    slice_length: int = 2000
    cursor_inc: int = 10

    # Tokeniser / model ---------------------------------------------------
    max_length: int = 100  # ensure model input not longer than chunk

    # Prompt / suffix split ----------------------------------------------
    # Number of tokens treated as the prompt; if None, defaults to
    # `chunk_size // 2` (50 / 50 split).
    prompt_tokens: int | None = None

    # Numerical precision for model weights + saved logits.
    precision: Precision = Precision.FLOAT32

    # TPU device count (set TPU_NUM_DEVICES). If None, use all visible cores.
    num_devices: int | None = None


# Decorator to ensure TPU lockfile cleanup in case of errors
@remove_tpu_lockfile_on_exit
def compute_sliding_logits(cfg: SlidingLogitsConfig) -> None:
    """Run causal-LM forward pass over sliding windows and save outputs."""

    logger.info(
        "Computing sliding-window logits for %s using %s",
        cfg.input_path,
        cfg.model_name,
    )   

    # Ensure output directory exists (works for GCS/local)
    fsspec_mkdirs(cfg.output_dir)

    # ------------------------------------------------------------------
    # Configure TPU device visibility *before* importing torch_xla.
    # ------------------------------------------------------------------
    if cfg.num_devices is not None:
        os.environ["TPU_NUM_DEVICES"] = str(cfg.num_devices)
        os.environ["PJRT_DEVICE_COUNT"] = str(cfg.num_devices)
        os.environ.pop("X_NUM_DEVICES", None)
        logger.info("Set TPU_NUM_DEVICES=%s", cfg.num_devices)
    else:
        # Default: expose all chips on the host.  Overwrite any preset value
        # (cluster base image often sets TPU_NUM_DEVICES=1).
        os.environ.pop("X_NUM_DEVICES", None)
        if "TPU_NUM_DEVICES" in os.environ:
            logger.info("Clearing pre-existing TPU_NUM_DEVICES=%s", os.environ["TPU_NUM_DEVICES"])
        os.environ.pop("TPU_NUM_DEVICES", None)
        os.environ.pop("PJRT_DEVICE_COUNT", None)

    # Lazy import AFTER env vars are settled -----------------------------------
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.runtime as xr

    # Force PJRT runtime to initialise now so that xr.world_size() reflects
    # the topology.  This *must* happen after we set TPU_NUM_DEVICES.
    world_size = xr.world_size()  # triggers runtime init if not yet initialised
    logger.info("Parent process sees XR world_size=%d", world_size)

    # ------------------------------------------------------------------
    # All heavy lifting happens in _sliding_logits_worker defined at module
    # scope (so it is picklable by multiprocessing).
    # ------------------------------------------------------------------
    xmp.spawn(_sliding_logits_worker, args=(cfg,), nprocs=world_size, start_method="fork")


# ---------------------------------------------------------------------------
# Low-level worker -----------------------------------------------------------
# ---------------------------------------------------------------------------


def _sliding_logits_worker(index: int, cfg: "SlidingLogitsConfig") -> None:  # type: ignore
    """Per-XLA-core worker. Runs inside torch-xla xmp.spawn process."""

    # Import torch_xla *inside* worker process, after PJRT runtime decided on
    # device topology.
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch

    # ------------------------------------------------------------------
    # 1. Load raw text --------------------------------------------------
    # ------------------------------------------------------------------
    fs_file = fsspec.open(cfg.input_path, "r")
    with fs_file as f:
        full_text: str = f.read()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    chunks = chunk_text_to_sliding_window_token_chunks(
        full_text,
        tokenizer,
        chunk_size=cfg.chunk_size,
        slice_length=cfg.slice_length,
        cursor_inc=cfg.cursor_inc,
    )
    logger.info("[Core %d] Total generated windows: %d", index, len(chunks))

    desired_dtype = torch.float16 if cfg.precision == Precision.FLOAT16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=desired_dtype)

    # Shard across world size so each XLA core gets a slice.
    world_size = xr.world_size()
    chunks_shard = chunks[index :: world_size]
    logger.info("[Core %d] Shard size: %d windows", index, len(chunks_shard))

    device = xm.xla_device()
    model.to(device)
    model.eval()

    shard_path = os.path.join(cfg.output_dir, f"sliding_logits_{index}.parquet")

    # Build PyArrow schema (logits as list<list<float16|float32>>)
    value_type = pa.float16() if cfg.precision == Precision.FLOAT16 else pa.float32()
    schema = pa.schema(
        [
            ("input_ids", pa.list_(pa.int32())),
            ("start_idx", pa.int32()),
            ("end_idx", pa.int32()),
            ("text_len", pa.int32()),
            ("text", pa.string()),
            ("logits", pa.list_(pa.list_(value_type))),
            ("pz", pa.float32()),
        ]
    )

    # Open ParquetWriter on remote path via Arrow filesystem abstraction
    filesystem, path_within_fs = pa.fs.FileSystem.from_uri(shard_path)
    writer = pq.ParquetWriter(path_within_fs, schema, filesystem=filesystem, compression="zstd")

    prompt_len = cfg.prompt_tokens if cfg.prompt_tokens is not None else cfg.chunk_size // 2

    # Per-core character-level max-prob array
    text_len = len(full_text)
    char_max_local = np.zeros(text_len, dtype=np.float32)

    total_batches = (len(chunks_shard) + cfg.batch_size - 1) // cfg.batch_size
    start_time = time.time()
    
    for batch_idx, batch_start in enumerate(range(0, len(chunks_shard), cfg.batch_size)):
        batch_start_time = time.time()
        
        batch_chunks = chunks_shard[batch_start : batch_start + cfg.batch_size]
        texts = [c["text"] for c in batch_chunks]
        
        # Calculate progress and timing estimates
        progress_percent = (batch_idx + 1) / total_batches * 100
        elapsed_time = time.time() - start_time
        
        if batch_idx > 0:  # Skip time estimate for first batch
            avg_time_per_batch = elapsed_time / batch_idx
            remaining_batches = total_batches - batch_idx - 1
            eta_seconds = avg_time_per_batch * remaining_batches
            
            # Format time estimates
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"
                
            print(f"[Core {index}] Processing batch {batch_idx + 1}/{total_batches} ({progress_percent:.1f}%) - ETA: {eta_str}", flush=True)
        else:
            print(f"[Core {index}] Processing batch {batch_idx + 1}/{total_batches} ({progress_percent:.1f}%)", flush=True)
        
        # Tokenization timing
        tokenize_start = time.time()
        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        tokenize_time = time.time() - tokenize_start
        print(f"[Core {index}] Tokenization: {tokenize_time:.2f}s", flush=True)

        # Forward pass timing
        forward_start = time.time()
        with torch.no_grad():
            outputs = model(**tokens)
        forward_time = time.time() - forward_start
        print(f"[Core {index}] Forward pass: {forward_time:.2f}s", flush=True)

        # Logits processing timing
        logits_start = time.time()
        logits = outputs.logits.to(desired_dtype)
        logits_time = time.time() - logits_start
        print(f"[Core {index}] Logits processing: {logits_time:.2f}s", flush=True)

        # P(z) computation timing
        pz_start = time.time()
        shift_logits = logits[:, :-1, :]
        shift_labels = tokens["input_ids"][:, 1:]
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        suffix_start = max(0, prompt_len - 1)
        if suffix_start < token_lp.size(1):
            suffix_lp = token_lp[:, suffix_start:].sum(dim=-1)
            pz_batch = torch.exp(suffix_lp).cpu().tolist()
        else:
            pz_batch = [0.0] * len(batch_chunks)
        pz_time = time.time() - pz_start
        print(f"[Core {index}] P(z) computation: {pz_time:.2f}s", flush=True)

        # Data preparation timing
        prep_start = time.time()
        logits_np = logits.cpu().numpy()
        input_ids_col = pa.array([c["input_ids"] for c in batch_chunks], type=pa.list_(pa.int32()))
        start_idx_col = pa.array([c["start_idx"] for c in batch_chunks], type=pa.int32())
        end_idx_col = pa.array([c["end_idx"] for c in batch_chunks], type=pa.int32())
        text_len_col = pa.array([c["text_len"] for c in batch_chunks], type=pa.int32())
        text_col = pa.array([c["text"] for c in batch_chunks], type=pa.string())
        logits_col = pa.array(logits_np.tolist(), type=pa.list_(pa.list_(value_type)))
        pz_col = pa.array(pz_batch, type=pa.float32())

        batch = pa.record_batch(
            [
                input_ids_col,
                start_idx_col,
                end_idx_col,
                text_len_col,
                text_col,
                logits_col,
                pz_col,
            ],
            schema=schema,
        )

        for ch, pz_val in zip(batch_chunks, pz_batch):
            c0, c1 = ch["start_idx"], ch["end_idx"]
            char_max_local[c0 : c1 + 1] = np.maximum(char_max_local[c0 : c1 + 1], pz_val)

        prep_time = time.time() - prep_start
        print(f"[Core {index}] Data preparation: {prep_time:.2f}s", flush=True)

        # Table building and writing timing
        table_start = time.time()
        writer.write_batch(batch)
        table_time = time.time() - table_start
        print(f"[Core {index}] Table build/write: {table_time:.2f}s", flush=True)

        # Cleanup timing
        cleanup_start = time.time()
        del logits, tokens, outputs, batch, logits_np
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        xm.mark_step()
        cleanup_time = time.time() - cleanup_start
        print(f"[Core {index}] Cleanup: {cleanup_time:.2f}s", flush=True)
        
        # Total batch time
        batch_total_time = time.time() - batch_start_time
        print(f"[Core {index}] Total batch time: {batch_total_time:.2f}s", flush=True)
        print("---", flush=True)

    # Final timing summary
    total_time = time.time() - start_time
    avg_time_per_batch = total_time / total_batches
    
    if total_time < 60:
        total_time_str = f"{total_time:.1f}s"
    elif total_time < 3600:
        total_time_str = f"{total_time/60:.1f}m"
    else:
        total_time_str = f"{total_time/3600:.1f}h"
        
    if avg_time_per_batch < 60:
        avg_time_str = f"{avg_time_per_batch:.1f}s"
    elif avg_time_per_batch < 3600:
        avg_time_str = f"{avg_time_per_batch/60:.1f}m"
    else:
        avg_time_str = f"{avg_time_per_batch/3600:.1f}h"
    
    print(f"[Core {index}] Completed {total_batches} batches in {total_time_str} (avg: {avg_time_str}/batch)", flush=True)
    
    writer.close()
    logger.info("[Core %d] Finished writing shard to %s", index, shard_path)

    # Write per-core char_max array directly to GCS
    cm_part_path = os.path.join(cfg.output_dir, f"char_max_part_{index}.npy")
    with fsspec.open(cm_part_path, "wb") as fo:
        np.save(fo, char_max_local)
    logger.info("[Core %d] Wrote char_max part to %s", index, cm_part_path)


# ---------------------------------------------------------------------------
# Ray remote wrapper ---------------------------------------------------------
# ---------------------------------------------------------------------------
# When running under the Marin Executor, the step function is executed in a
# generic Ray task that does **not** request TPU resources.  We provide an
# explicit remote version that _does_ request the TPU so that the scheduler
# places the work on the TPU host and libtpu is visible.
#
# Usage from client code / experiment:
#   from marin.generation.sliding_logits import compute_sliding_logits_remote as compute_sliding_logits
#   ExecutorStep(fn=compute_sliding_logits_remote, ...)

compute_sliding_logits_remote = ray.remote(
    # Rough memory estimate (adjust if OOM)
    memory=16 * 1024 * 1024 * 1024,  # 16 GB
    resources={"TPU": 4, "TPU-v4-8-head": 1},
)(compute_sliding_logits)


if __name__ == "__main__":
    import draccus

    @draccus.wrap()
    def main(cfg: SlidingLogitsConfig):  # pragma: no cover
        compute_sliding_logits(cfg)

    main() 