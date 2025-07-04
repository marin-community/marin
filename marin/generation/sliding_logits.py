from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
import threading
import queue

import fsspec
import numpy as np
import ray
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin.utils import fsspec_mkdirs, remove_tpu_lockfile_on_exit


def chunk_text_to_sliding_window_token_chunks(
    text: str,
    tokenizer,
    *,
    chunk_size: int = 100,
    slice_length: int = 2000,
    cursor_inc: int = 10,
) -> list[dict[str, Any]]:
    """Tokenise *text* into overlapping `chunk_size`-token windows.

    Replicates the logic in ``careless.py`` almost verbatim but drops the
    torch-specific bits.  Returns a list of dictionaries with keys:

    ``input_ids``          - *list[int]* of length ``chunk_size``
    ``start_idx``          - start character index in *text*
    ``end_idx``            - end character index (inclusive) in *text*
    ``text``               - decoded chunk text (useful for debugging)
    ``attention_mask``     - list[int] same length as ``input_ids``
    ``text_len``           - length of decoded text in characters
    """

    all_chunks: list[dict[str, Any]] = []
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
    output_dir: str  # directory where output shards (.npz/.npy) and plot will be written

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
    # `chunk_size // 2` (50/50 split).
    prompt_tokens: int | None = None

    # Numerical precision for model weights + saved logits.
    precision: Precision = Precision.FLOAT32

    # TPU device count (set TPU_NUM_DEVICES). If None, use all visible cores.
    num_devices: int | None = None

    # If True, write uncompressed .npy files with np.save instead of
    # compressed .npz archives. Allows faster writes at the cost of larger
    # output files.
    uncompress: bool = False

    # Block size for fsspec writes in bytes. This controls the chunk size
    # used when streaming data to remote filesystems (e.g., GCS).
    #TODO: figure out what's reasonable for this
    block_size: int = 64 * 1024 * 1024

    # Number of batches to accumulate in memory before writing to disk.
    # Larger values reduce write overhead but increase peak memory usage.
    batches_per_save: int = 1

    # If True, batches are handed off to background threads for writing so that
    # the forward-pass loop does not block on I/O. When False, writes happen
    # synchronously as before.
    background_queue: bool = False

    # Number of background writer threads. Only used when
    # ``background_queue`` is True.
    num_background_writers: int = 1


def _writer_loop(batch_queue: queue.Queue, cfg: SlidingLogitsConfig, error_list: list[Exception]) -> None:
    """Background thread function to write batches to disk.

    Parameters are shared via a queue to avoid blocking the main TPU worker.
    Any exception is stored in ``error_list`` so the caller can re-raise it
    after all threads join.
    """
    try:
        for payload in iter(batch_queue.get, None):
            data_dict, batch_path = payload
            try:
                with fsspec.open(batch_path, "wb", block_size=cfg.block_size) as fo:
                    if cfg.uncompress:
                        np.save(
                            fo,
                            data_dict,
                            allow_pickle=True,
                            pickle_kwargs={"protocol": 4},
                        )
                    else:
                        np.savez_compressed(fo, **data_dict)
            finally:
                batch_queue.task_done()
    except Exception as exc:  # pragma: no cover - background thread
        error_list.append(exc)


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


def _sliding_logits_worker(index: int, cfg: SlidingLogitsConfig) -> None:  # type: ignore
    """Per-XLA-core worker. Runs inside torch-xla xmp.spawn process."""

    # Import torch_xla *inside* worker process, after PJRT runtime decided on
    # device topology.
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

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
    chunks_shard = chunks[index::world_size]
    logger.info("[Core %d] Shard size: %d windows", index, len(chunks_shard))

    device = xm.xla_device()
    model.to(device)
    model.eval()

    shard_path_prefix = os.path.join(cfg.output_dir, f"sliding_logits_{index}")

    prompt_len = cfg.prompt_tokens if cfg.prompt_tokens is not None else cfg.chunk_size // 2

    # Per-core character-level max-prob array
    text_len = len(full_text)
    char_max_local = np.zeros(text_len, dtype=np.float32)

    batch_queue: queue.Queue | None = None
    writer_threads: list[threading.Thread] = []
    writer_errors: list[Exception] = []
    if cfg.background_queue:
        queue_size = cfg.num_background_writers * 2
        batch_queue = queue.Queue(maxsize=queue_size)
        # Launch a pool of writer threads. Batches are distributed roughly
        # evenly as each thread pulls work from the queue.
        for _ in range(cfg.num_background_writers):
            t = threading.Thread(
                target=_writer_loop,
                args=(batch_queue, cfg, writer_errors),
                daemon=True,
            )
            t.start()
            writer_threads.append(t)

    total_batches = (len(chunks_shard) + cfg.batch_size - 1) // cfg.batch_size
    start_time = time.time()

    save_counter = 0
    accum_batches = 0

    accum_logits: list[np.ndarray] = []
    accum_input_ids: list[np.ndarray] = []
    accum_start_idx: list[np.ndarray] = []
    accum_end_idx: list[np.ndarray] = []
    accum_text_len: list[np.ndarray] = []
    accum_text: list[np.ndarray] = []
    accum_pz: list[np.ndarray] = []

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

            print(
                f"[Core {index}] Processing batch {batch_idx + 1}/{total_batches} "
                f"({progress_percent:.1f}%) - ETA: {eta_str}",
                flush=True,
            )
        else:
            print(
                f"[Core {index}] Processing batch {batch_idx + 1}/{total_batches} " f"({progress_percent:.1f}%)",
                flush=True,
            )

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

        # P(z) computation timing (keep on device for speed)
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

        batch_input_ids = np.array([ch["input_ids"] for ch in batch_chunks], dtype=np.int32)
        batch_start_idx = np.array([ch["start_idx"] for ch in batch_chunks], dtype=np.int32)
        batch_end_idx = np.array([ch["end_idx"] for ch in batch_chunks], dtype=np.int32)
        batch_text_len = np.array([ch["text_len"] for ch in batch_chunks], dtype=np.int32)
        batch_text = np.array([ch["text"] for ch in batch_chunks], dtype=object)
        batch_pz = np.array(pz_batch, dtype=np.float32)

        for i, ch in enumerate(batch_chunks):
            c0, c1 = ch["start_idx"], ch["end_idx"]
            char_max_local[c0 : c1 + 1] = np.maximum(char_max_local[c0 : c1 + 1], batch_pz[i])

        prep_time = time.time() - prep_start
        print(f"[Core {index}] Data preparation: {prep_time:.2f}s", flush=True)

        accum_logits.append(logits_np)
        accum_input_ids.append(batch_input_ids)
        accum_start_idx.append(batch_start_idx)
        accum_end_idx.append(batch_end_idx)
        accum_text_len.append(batch_text_len)
        accum_text.append(batch_text)
        accum_pz.append(batch_pz)
        accum_batches += 1

        table_time = 0.0
        if accum_batches >= cfg.batches_per_save:
            table_start = time.time()
            out_logits = np.concatenate(accum_logits, axis=0)
            out_input_ids = np.concatenate(accum_input_ids, axis=0)
            out_start_idx = np.concatenate(accum_start_idx, axis=0)
            out_end_idx = np.concatenate(accum_end_idx, axis=0)
            out_text_len = np.concatenate(accum_text_len, axis=0)
            out_text = np.concatenate(accum_text, axis=0)
            out_pz = np.concatenate(accum_pz, axis=0)

            ext = "npy" if cfg.uncompress else "npz"
            batch_path = f"{shard_path_prefix}_part{save_counter}.{ext}"
            data_dict = {
                "input_ids": out_input_ids,
                "start_idx": out_start_idx,
                "end_idx": out_end_idx,
                "text_len": out_text_len,
                "text": out_text,
                "logits": out_logits,
                "pz": out_pz,
            }

            if cfg.background_queue:
                assert batch_queue is not None
                batch_queue.put((data_dict, batch_path))
            else:
                with fsspec.open(batch_path, "wb", block_size=cfg.block_size) as fo:
                    if cfg.uncompress:
                        np.save(
                            fo,
                            data_dict,
                            allow_pickle=True,
                            pickle_kwargs={"protocol": 4},
                        )
                    else:
                        np.savez_compressed(fo, **data_dict)
            table_time = time.time() - table_start
            print(f"[Core {index}] Table build/write: {table_time:.2f}s", flush=True)
            save_counter += 1
            accum_batches = 0
            accum_logits.clear()
            accum_input_ids.clear()
            accum_start_idx.clear()
            accum_end_idx.clear()
            accum_text_len.clear()
            accum_text.clear()
            accum_pz.clear()

        # Cleanup timing
        cleanup_start = time.time()
        del logits, tokens, outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        xm.mark_step()
        cleanup_time = time.time() - cleanup_start
        print(f"[Core {index}] Cleanup: {cleanup_time:.2f}s", flush=True)

        # Total batch time
        batch_total_time = time.time() - batch_start_time
        print(f"[Core {index}] Total batch time: {batch_total_time:.2f}s", flush=True)
        print("---", flush=True)

    if accum_batches > 0:
        table_start = time.time()
        out_logits = np.concatenate(accum_logits, axis=0)
        out_input_ids = np.concatenate(accum_input_ids, axis=0)
        out_start_idx = np.concatenate(accum_start_idx, axis=0)
        out_end_idx = np.concatenate(accum_end_idx, axis=0)
        out_text_len = np.concatenate(accum_text_len, axis=0)
        out_text = np.concatenate(accum_text, axis=0)
        out_pz = np.concatenate(accum_pz, axis=0)

        ext = "npy" if cfg.uncompress else "npz"
        batch_path = f"{shard_path_prefix}_part{save_counter}.{ext}"
        data_dict = {
            "input_ids": out_input_ids,
            "start_idx": out_start_idx,
            "end_idx": out_end_idx,
            "text_len": out_text_len,
            "text": out_text,
            "logits": out_logits,
            "pz": out_pz,
        }

        if cfg.background_queue:
            assert batch_queue is not None
            batch_queue.put((data_dict, batch_path))
        else:
            with fsspec.open(batch_path, "wb", block_size=cfg.block_size) as fo:
                if cfg.uncompress:
                    np.save(
                        fo,
                        data_dict,
                        allow_pickle=True,
                        pickle_kwargs={"protocol": 4},
                    )
                else:
                    np.savez_compressed(fo, **data_dict)
        table_time = time.time() - table_start
        print(f"[Core {index}] Table build/write: {table_time:.2f}s", flush=True)

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

    print(
        f"[Core {index}] Completed {total_batches} batches in {total_time_str} "
        f"(avg: {avg_time_str}/batch)",
        flush=True,
    )

    if cfg.background_queue and batch_queue is not None:
        for _ in writer_threads:
            batch_queue.put(None)
        batch_queue.join()
        for t in writer_threads:
            t.join()
        if writer_errors:
            raise writer_errors[0]

    logger.info("[Core %d] Finished writing shard files with prefix %s", index, shard_path_prefix)

    # Write per-core char_max array directly to GCS
    cm_part_path = os.path.join(cfg.output_dir, f"char_max_part_{index}.npy")
    with fsspec.open(cm_part_path, "wb") as fo:
        np.save(fo, char_max_local, allow_pickle=True, pickle_kwargs={'protocol': 4})
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
