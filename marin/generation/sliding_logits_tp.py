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
import gc
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
class SlidingLogitsTPConfig:
    """Configuration for tensor-parallel sliding-window forward-pass logging."""

    model_name: str
    input_path: str  # path to raw txt (local or gs://)
    output_dir: str  # directory where output shards (.npz/.npy) and plot will be written

    # Runtime / batching --------------------------------------------------
    # For tensor parallel, we process one chunk at a time
    batch_size: int = 1

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

    # Tensor parallel specific parameters
    # Mesh shape for tensor parallelism - typically (1, num_devices) for model parallel
    mesh_shape: tuple[int, int] | None = None


def _writer_loop(batch_queue: queue.Queue, cfg: SlidingLogitsTPConfig, error_list: list[Exception]) -> None:
    """Background thread function to write batches to disk.

    Parameters are shared via a queue to avoid blocking the main TPU worker.
    Any exception is stored in ``error_list`` so the caller can re-raise it
    after all threads join.
    """
    import threading
    thread_id = threading.get_ident()
    print(f"[Writer {thread_id}] Background writer thread started", flush=True)
    
    try:
        write_count = 0
        while True:
            payload = batch_queue.get()
            
            # Handle shutdown signal
            if payload is None:
                print(f"[Writer {thread_id}] Received shutdown signal", flush=True)
                batch_queue.task_done()  # Mark shutdown signal as done
                break
            
            write_start_time = time.time()
            data_dict, batch_path = payload
            
            print(f"[Writer {thread_id}] Starting write #{write_count} to {batch_path}", flush=True)
            print(f"[Writer {thread_id}] Queue size: {batch_queue.qsize()}", flush=True)
            
            try:
                # Time the file opening
                open_start = time.time()
                with fsspec.open(batch_path, "wb", block_size=cfg.block_size) as fo:
                    open_time = time.time() - open_start
                    print(f"[Writer {thread_id}] File opened in {open_time:.2f}s", flush=True)
                    
                    # Time the actual write
                    write_data_start = time.time()
                    if cfg.uncompress:
                        np.save(
                            fo,
                            data_dict,
                            allow_pickle=True,
                        )
                    else:
                        np.savez_compressed(fo, **data_dict)
                    write_data_time = time.time() - write_data_start
                    print(f"[Writer {thread_id}] Data written in {write_data_time:.2f}s", flush=True)
                    
            except Exception as write_exc:
                print(f"[Writer {thread_id}] ERROR during write: {write_exc}", flush=True)
                raise write_exc
            finally:
                batch_queue.task_done()  # Mark work item as done
            
            total_write_time = time.time() - write_start_time
            print(f"[Writer {thread_id}] Completed write #{write_count} in {total_write_time:.2f}s total", flush=True)
            write_count += 1
            
    except Exception as exc:  # pragma: no cover - background thread
        print(f"[Writer {thread_id}] FATAL ERROR: {exc}", flush=True)
        error_list.append(exc)
    
    print(f"[Writer {thread_id}] Background writer thread exiting after {write_count} writes", flush=True)


def _apply_tensor_parallel_sharding(model, mesh):
    """Apply tensor parallel sharding to model parameters."""
    import torch_xla.distributed.spmd as xs
    
    print(f"[TP] Applying tensor parallel sharding to model parameters", flush=True)
    print(f"[TP] Mesh shape: {mesh.shape()}", flush=True)
    
    # Get all named parameters
    param_count = 0
    sharded_count = 0
    
    for name, param in model.named_parameters():
        param_count += 1
        param_shape = param.shape
        print(f"[TP] Parameter {name}: shape={param_shape}, numel={param.numel()}", flush=True)
        
        # Apply sharding based on parameter type and shape
        if len(param_shape) >= 2:
            # For 2D+ tensors, shard along the last dimension (typical for linear layers)
            if param_shape[-1] >= mesh.shape()['model']:
                # Only shard if the dimension is large enough
                partition_spec = tuple(None for _ in range(len(param_shape) - 1)) + ('model',)
                xs.mark_sharding(param, mesh, partition_spec)
                sharded_count += 1
                print(f"[TP] Sharded {name} with spec {partition_spec}", flush=True)
            else:
                print(f"[TP] Replicated {name} (dimension too small for sharding)", flush=True)
        else:
            # For 1D tensors (biases, etc.), replicate across all devices
            print(f"[TP] Replicated {name} (1D tensor)", flush=True)
    
    print(f"[TP] Applied sharding to {sharded_count}/{param_count} parameters", flush=True)
    return model


# Decorator to ensure TPU lockfile cleanup in case of errors
@remove_tpu_lockfile_on_exit
def compute_sliding_logits_tp(cfg: SlidingLogitsTPConfig) -> None:
    """Run tensor-parallel causal-LM forward pass over sliding windows and save outputs."""

    logger.info(
        "Computing tensor-parallel sliding-window logits for %s using %s",
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
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.distributed.spmd import Mesh

    # Enable SPMD mode for tensor parallelism
    print(f"[TP] Enabling SPMD mode", flush=True)
    xr.use_spmd()

    # Get device information
    num_devices = xr.global_runtime_device_count()
    print(f"[TP] Total devices available: {num_devices}", flush=True)
    
    device_attrs = xr.global_runtime_device_attributes()
    print(f"[TP] Device attributes:", flush=True)
    for i, attr in enumerate(device_attrs):
        print(f"[TP]   Device {i}: {attr}", flush=True)

    # Create mesh for tensor parallelism
    mesh_shape = cfg.mesh_shape if cfg.mesh_shape is not None else (1, num_devices)
    print(f"[TP] Creating mesh with shape: {mesh_shape}", flush=True)
    
    device_ids = np.array(range(num_devices))
    print(f"[TP] Device IDs: {device_ids}", flush=True)
    
    mesh = Mesh(device_ids, mesh_shape, ('data', 'model'))
    print(f"[TP] Created mesh:", flush=True)
    print(f"[TP]   Mesh shape: {mesh.shape()}", flush=True)
    print(f"[TP]   Logical mesh:\n{mesh.get_logical_mesh()}", flush=True)

    # ------------------------------------------------------------------
    # Load text and create chunks
    # ------------------------------------------------------------------
    print(f"[TP] Loading text from {cfg.input_path}", flush=True)
    fs_file = fsspec.open(cfg.input_path, "r")
    with fs_file as f:
        full_text: str = f.read()
    
    print(f"[TP] Loaded text with {len(full_text)} characters", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[TP] Loaded tokenizer", flush=True)

    chunks = chunk_text_to_sliding_window_token_chunks(
        full_text,
        tokenizer,
        chunk_size=cfg.chunk_size,
        slice_length=cfg.slice_length,
        cursor_inc=cfg.cursor_inc,
    )
    print(f"[TP] Created {len(chunks)} sliding window chunks", flush=True)

    # ------------------------------------------------------------------
    # Load and shard model
    # ------------------------------------------------------------------
    desired_dtype = torch.float16 if cfg.precision == Precision.FLOAT16 else torch.float32
    print(f"[TP] Loading model {cfg.model_name} with dtype {desired_dtype}", flush=True)
    
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=desired_dtype)
    print(f"[TP] Model loaded successfully", flush=True)

    # Move model to XLA device
    device = xm.xla_device()
    print(f"[TP] Moving model to XLA device: {device}", flush=True)
    model.to(device)
    model.eval()

    # Apply tensor parallel sharding
    model = _apply_tensor_parallel_sharding(model, mesh)
    print(f"[TP] Model sharding applied successfully", flush=True)

    # ------------------------------------------------------------------
    # Setup output and processing
    # ------------------------------------------------------------------
    shard_path_prefix = os.path.join(cfg.output_dir, f"sliding_logits_tp")
    prompt_len = cfg.prompt_tokens if cfg.prompt_tokens is not None else cfg.chunk_size // 2

    # Character-level max-prob array
    text_len = len(full_text)
    char_max_local = np.zeros(text_len, dtype=np.float32)
    print(f"[TP] Initialized char_max array with length {text_len}", flush=True)

    # Setup background queue if enabled
    batch_queue: queue.Queue | None = None
    writer_threads: list[threading.Thread] = []
    writer_errors: list[Exception] = []
    if cfg.background_queue:
        queue_size = cfg.num_background_writers * 3
        batch_queue = queue.Queue(maxsize=queue_size)
        print(f"[TP] Setting up {cfg.num_background_writers} background writer threads", flush=True)
        for i in range(cfg.num_background_writers):
            t = threading.Thread(
                target=_writer_loop,
                args=(batch_queue, cfg, writer_errors),
                daemon=True,
            )
            t.start()
            writer_threads.append(t)
            print(f"[TP] Started background writer thread {i+1}", flush=True)

    # ------------------------------------------------------------------
    # Process chunks sequentially (tensor parallel processes same data)
    # ------------------------------------------------------------------
    total_chunks = len(chunks)
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

    print(f"[TP] Starting processing of {total_chunks} chunks", flush=True)

    for chunk_idx, chunk in enumerate(chunks):
        chunk_start_time = time.time()
        
        # Calculate progress and timing estimates
        progress_percent = (chunk_idx + 1) / total_chunks * 100
        elapsed_time = time.time() - start_time

        if chunk_idx > 0:
            avg_time_per_chunk = elapsed_time / chunk_idx
            remaining_chunks = total_chunks - chunk_idx - 1
            eta_seconds = avg_time_per_chunk * remaining_chunks

            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds/3600:.1f}h"

            print(
                f"[TP] Processing chunk {chunk_idx + 1}/{total_chunks} "
                f"({progress_percent:.1f}%) - ETA: {eta_str}",
                flush=True,
            )
        else:
            print(
                f"[TP] Processing chunk {chunk_idx + 1}/{total_chunks} "
                f"({progress_percent:.1f}%)",
                flush=True,
            )

        # Tokenization timing
        tokenize_start = time.time()
        text = chunk["text"]
        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        
        # Move tokens to device and mark as replicated across all devices
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        # In tensor parallel, input data is replicated across all devices
        for k, v in tokens.items():
            xs.mark_sharding(v, mesh, (None, None))  # Replicate batch and sequence dims
        
        tokenize_time = time.time() - tokenize_start
        print(f"[TP] Tokenization: {tokenize_time:.2f}s", flush=True)

        # Forward pass timing
        forward_start = time.time()
        with torch.no_grad():
            outputs = model(**tokens)
        forward_time = time.time() - forward_start
        print(f"[TP] Forward pass: {forward_time:.2f}s", flush=True)

        # Logits processing timing
        logits_start = time.time()
        logits = outputs.logits.to(desired_dtype)
        logits_time = time.time() - logits_start
        print(f"[TP] Logits processing: {logits_time:.2f}s", flush=True)

        # P(z) computation timing
        pz_start = time.time()
        shift_logits = logits[:, :-1, :]
        shift_labels = tokens["input_ids"][:, 1:]
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        suffix_start = max(0, prompt_len - 1)
        if suffix_start < token_lp.size(1):
            suffix_lp = token_lp[:, suffix_start:].sum(dim=-1)
            pz_value = torch.exp(suffix_lp).cpu().item()
        else:
            pz_value = 0.0
        pz_time = time.time() - pz_start
        print(f"[TP] P(z) computation: {pz_time:.2f}s, pz={pz_value:.6f}", flush=True)

        del shift_logits, shift_labels, log_probs, token_lp, suffix_lp
        gc.collect()

        # Data preparation timing
        prep_start = time.time()
        logits_np = logits.cpu().numpy()

        # Create arrays for this single chunk
        chunk_input_ids = np.array([chunk["input_ids"]], dtype=np.int32)
        chunk_start_idx = np.array([chunk["start_idx"]], dtype=np.int32)
        chunk_end_idx = np.array([chunk["end_idx"]], dtype=np.int32)
        chunk_text_len = np.array([chunk["text_len"]], dtype=np.int32)
        chunk_text = np.array([chunk["text"]], dtype=object)
        chunk_pz = np.array([pz_value], dtype=np.float32)

        # Update character-level max probabilities
        c0, c1 = chunk["start_idx"], chunk["end_idx"]
        char_max_local[c0 : c1 + 1] = np.maximum(char_max_local[c0 : c1 + 1], pz_value)

        prep_time = time.time() - prep_start
        print(f"[TP] Data preparation: {prep_time:.2f}s", flush=True)

        # Accumulate data
        accum_logits.append(logits_np)
        accum_input_ids.append(chunk_input_ids)
        accum_start_idx.append(chunk_start_idx)
        accum_end_idx.append(chunk_end_idx)
        accum_text_len.append(chunk_text_len)
        accum_text.append(chunk_text)
        accum_pz.append(chunk_pz)
        accum_batches += 1

        # Save accumulated data when reaching batch limit
        if accum_batches >= cfg.batches_per_save:
            save_start = time.time()
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
                print(f"[TP] Queuing data for background write. Queue size: {batch_queue.qsize()}", flush=True)
                batch_queue.put((data_dict, batch_path))
                del data_dict
                gc.collect()
            else:
                with fsspec.open(batch_path, "wb", block_size=cfg.block_size) as fo:
                    if cfg.uncompress:
                        np.save(fo, data_dict, allow_pickle=True)
                    else:
                        np.savez_compressed(fo, **data_dict)

            save_time = time.time() - save_start
            print(f"[TP] Save operation: {save_time:.2f}s", flush=True)
            
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
        gc.collect()
        cleanup_time = time.time() - cleanup_start
        print(f"[TP] Cleanup: {cleanup_time:.2f}s", flush=True)

        # Total chunk time
        chunk_total_time = time.time() - chunk_start_time
        print(f"[TP] Total chunk time: {chunk_total_time:.2f}s", flush=True)
        print("---", flush=True)

    # Save any remaining accumulated data
    if accum_batches > 0:
        save_start = time.time()
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
            print(f"[TP] Queuing final batch for background write. Queue size: {batch_queue.qsize()}", flush=True)
            batch_queue.put((data_dict, batch_path))
        else:
            with fsspec.open(batch_path, "wb", block_size=cfg.block_size) as fo:
                if cfg.uncompress:
                    np.save(fo, data_dict, allow_pickle=True)
                else:
                    np.savez_compressed(fo, **data_dict)
        
        save_time = time.time() - save_start
        print(f"[TP] Final save operation: {save_time:.2f}s", flush=True)

    # Final timing summary
    total_time = time.time() - start_time
    avg_time_per_chunk = total_time / total_chunks

    if total_time < 60:
        total_time_str = f"{total_time:.1f}s"
    elif total_time < 3600:
        total_time_str = f"{total_time/60:.1f}m"
    else:
        total_time_str = f"{total_time/3600:.1f}h"

    if avg_time_per_chunk < 60:
        avg_time_str = f"{avg_time_per_chunk:.1f}s"
    elif avg_time_per_chunk < 3600:
        avg_time_str = f"{avg_time_per_chunk/60:.1f}m"
    else:
        avg_time_str = f"{avg_time_per_chunk/3600:.1f}h"

    print(
        f"[TP] Completed {total_chunks} chunks in {total_time_str} "
        f"(avg: {avg_time_str}/chunk)",
        flush=True,
    )

    # Shutdown background writers
    if cfg.background_queue and batch_queue is not None:
        print(f"[TP] Shutting down {len(writer_threads)} background writers...", flush=True)
        print(f"[TP] Queue size at shutdown: {batch_queue.qsize()}", flush=True)
        
        for i in range(len(writer_threads)):
            print(f"[TP] Sending shutdown signal to writer {i+1}/{len(writer_threads)}", flush=True)
            batch_queue.put(None)
        
        print(f"[TP] Waiting for all queued work to complete...", flush=True)
        batch_queue.join()
        print(f"[TP] All queued work completed", flush=True)
        
        print(f"[TP] Waiting for writer threads to exit...", flush=True)
        for i, t in enumerate(writer_threads):
            t.join()
            print(f"[TP] Writer thread {i+1}/{len(writer_threads)} exited", flush=True)
        
        if writer_errors:
            print(f"[TP] ERROR: {len(writer_errors)} writer errors occurred", flush=True)
            raise writer_errors[0]
        else:
            print(f"[TP] All background writers shut down successfully", flush=True)

    print(f"[TP] Finished writing shard files with prefix {shard_path_prefix}", flush=True)

    # Write character max array
    print(f"[TP] About to write char_max array...", flush=True)
    cm_path = os.path.join(cfg.output_dir, f"char_max_tp.npy")
    print(f"[TP] Opening file: {cm_path}", flush=True)
    
    try:
        with fsspec.open(cm_path, "wb") as fo:
            print(f"[TP] File opened successfully, writing data...", flush=True)
            np.save(fo, char_max_local, allow_pickle=True)
            print(f"[TP] Data written successfully", flush=True)
    except Exception as e:
        print(f"[TP] ERROR writing char_max: {e}", flush=True)
        raise
    
    print(f"[TP] About to log completion...", flush=True)
    logger.info("[TP] Wrote char_max to %s", cm_path)
    print(f"[TP] Tensor parallel sliding logits processing completed successfully", flush=True)


# ---------------------------------------------------------------------------
# Ray remote wrapper ---------------------------------------------------------
# ---------------------------------------------------------------------------
compute_sliding_logits_tp_remote = ray.remote(
    # Rough memory estimate for 70B model (adjust if OOM)
    memory=128 * 1024 * 1024 * 1024,  # 128 GB
    resources={"TPU": 8, "TPU-v6e-8-head": 1},
)(compute_sliding_logits_tp)


if __name__ == "__main__":
    import draccus

    @draccus.wrap()
    def main(cfg: SlidingLogitsTPConfig):  # pragma: no cover
        compute_sliding_logits_tp(cfg)

    main() 