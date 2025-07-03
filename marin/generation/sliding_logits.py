from __future__ import annotations

import logging
import os
import tempfile
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

    for batch_start in range(0, len(chunks_shard), cfg.batch_size):
        batch_chunks = chunks_shard[batch_start : batch_start + cfg.batch_size]
        texts = [c["text"] for c in batch_chunks]

        tokens = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)

        logits = outputs.logits.to(desired_dtype).cpu()

        # Compute P(z) for each example
        shift_logits = logits[:, :-1, :]
        shift_labels = tokens["input_ids"][:, 1:].cpu()
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        suffix_start = max(0, prompt_len - 1)
        if suffix_start < token_lp.size(1):
            suffix_lp = token_lp[:, suffix_start:].sum(dim=-1)
            pz_batch = torch.exp(suffix_lp).tolist()
        else:
            pz_batch = [0.0] * len(batch_chunks)

        # Build PyArrow rows
        rows = []
        for i, ch in enumerate(batch_chunks):
            rows.append(
                {
                    "input_ids": ch["input_ids"],
                    "start_idx": ch["start_idx"],
                    "end_idx": ch["end_idx"],
                    "text_len": ch["text_len"],
                    "text": ch["text"],
                    "logits": (
                        [[np.float16(v) for v in row] for row in logits[i].tolist()]
                        if cfg.precision == Precision.FLOAT16
                        else [[float(v) for v in row] for row in logits[i].tolist()]
                    ),
                    "pz": pz_batch[i],
                }
            )

            # update char_max_local for this window
            c0, c1 = ch["start_idx"], ch["end_idx"]
            char_max_local[c0 : c1 + 1] = np.maximum(char_max_local[c0 : c1 + 1], pz_batch[i])

        table = pa.Table.from_pylist(rows, schema=schema)
        writer.write_table(table, row_group_size=len(rows))

        # free tensors ASAP
        del logits, tokens, outputs, rows, table
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        xm.mark_step()

    writer.close()
    logger.info("[Core %d] Finished writing shard to %s", index, shard_path)

    # Write per-core char_max array directly to GCS
    cm_part_path = os.path.join(cfg.output_dir, f"char_max_part_{index}.npy")
    with fsspec.open(cm_part_path, "wb") as fo:
        np.save(fo, char_max_local)
    logger.info("[Core %d] Wrote char_max part to %s", index, cm_part_path)


if __name__ == "__main__":
    import draccus

    @draccus.wrap()
    def main(cfg: SlidingLogitsConfig):  # pragma: no cover
        compute_sliding_logits(cfg)

    main() 