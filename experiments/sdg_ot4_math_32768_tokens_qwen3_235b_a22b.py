"""Parallel incremental version of SDG regeneration.

This version:
1. Uses Ray to parallelize across workers
2. Each worker processes one row at a time and writes immediately
3. Uses row-level output files for fine-grained resumability
4. On restart, skips already-processed rows by checking existing output files
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import ray

from experiments.defaults import default_download
from marin.execution.executor import executor_main, this_output_path, ExecutorStep

logger = logging.getLogger(__name__)

ot4_math_qwen_235b_annotated = default_download(
    name="raw/marin-community/open-thoughts-4-30k-math-qwen3-235b-a22b-annotated",
    hf_dataset_id="marin-community/open-thoughts-4-30k-math-qwen3-235b-a22b-annotated",
    revision="c3d5f3b",
)

MODEL_NAME = "Qwen/Qwen3-235B-A22B-fp8-tput"
GENERATED_TEXT_COLUMN = "qwen235b_generated_text"
MAX_TOKENS = 32768
MAX_CONCURRENT = 16  # Low concurrency to avoid timeouts


def is_response_complete(text: str) -> bool:
    """A response is complete if it contains a closed thinking block and a boxed answer."""
    return "</think>" in text and "\\boxed{" in text


def serialize_value(value):
    """Serialize a single value for parquet compatibility."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def serialize_record(record: dict) -> dict:
    """Convert a record to be parquet-compatible by serializing complex nested structures."""
    return {key: serialize_value(value) for key, value in record.items()}


@ray.remote(max_retries=3)
def process_and_save_row(
    row: dict,
    output_path: str,
    row_index: int,
    model_name: str = MODEL_NAME,
    column: str = GENERATED_TEXT_COLUMN,
    temperature: float = 0.8,
    max_tokens: int = MAX_TOKENS,
    max_retries: int = 5,
    initial_backoff: float = 30.0,
) -> dict[str, Any]:
    """Process a single row and save immediately. Returns status dict."""
    ms_id = row.get("ms_id", f"idx_{row_index}")
    filename = f"row_{row_index:06d}_{ms_id}.parquet"

    # Construct output file path properly for GCS
    if output_path.startswith("gs://"):
        output_file = f"{output_path.rstrip('/')}/{filename}"
    else:
        output_file = os.path.join(output_path, filename)

    # Check if already processed
    fs, fs_path = fsspec.core.url_to_fs(output_file)
    if fs.exists(fs_path):
        return {"ms_id": ms_id, "status": "skipped", "file": output_file}

    # Check if regeneration needed
    existing = row.get(column, "")
    if existing and is_response_complete(existing):
        # No regeneration needed, just save
        processed_row = row
    else:
        # Need to regenerate - use httpx directly to avoid together SDK dependency issues
        import httpx

        logging.info("Re-generating truncated response for row %s", ms_id)
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")

        last_exception = None
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=600.0) as client:
                    response = client.post(
                        "https://api.together.xyz/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": row["instruction_seed"]}],
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()
                    generated_text = result["choices"][0]["message"]["content"]
                processed_row = {**row, column: generated_text}
                break
            except Exception as e:
                last_exception = e
                error_name = type(e).__name__
                error_str = str(e).lower()
                # Retry on timeout OR rate limit errors
                if "timeout" in error_str or "Timeout" in error_name or "429" in str(e) or "rate" in error_str:
                    backoff = initial_backoff * (2 ** attempt)
                    logging.warning(
                        "Attempt %d/%d failed for row %s (retryable error), retrying in %.1fs: %s",
                        attempt + 1, max_retries, ms_id, backoff, e
                    )
                    time.sleep(backoff)
                else:
                    raise
        else:
            logging.error("All %d retries exhausted for row %s", max_retries, ms_id)
            raise last_exception

    # Serialize and write
    serialized_row = serialize_record(processed_row)
    table = pa.Table.from_pylist([serialized_row])

    # Write to temp file first, then rename for atomicity
    temp_file = f"{output_file}.tmp"
    temp_fs, temp_fs_path = fsspec.core.url_to_fs(temp_file)

    # Ensure parent directory exists
    parent_path = os.path.dirname(fs_path)
    if not fs.exists(parent_path):
        fs.makedirs(parent_path, exist_ok=True)

    with fs.open(temp_fs_path, 'wb') as f:
        pq.write_table(table, f)
    fs.mv(temp_fs_path, fs_path)

    return {"ms_id": ms_id, "status": "processed", "file": output_file}


def load_all_input_rows(input_path: str) -> list[dict]:
    """Load all rows from input parquet files."""
    fs, path = fsspec.core.url_to_fs(input_path)
    parquet_files = fs.glob(os.path.join(path, "**/*.parquet"))

    all_rows = []
    for pf in parquet_files:
        # Use fsspec to read directly - don't reconstruct paths
        logger.info("Loading input file: %s", pf)
        with fs.open(pf, 'rb') as f:
            table = pq.read_table(f)
        all_rows.extend(table.to_pylist())

    logger.info("Loaded %d total input rows", len(all_rows))
    return all_rows


def get_processed_indices(output_path: str) -> set[int]:
    """Get set of already-processed row indices from existing output files."""
    fs, path = fsspec.core.url_to_fs(output_path)

    try:
        if not fs.exists(path):
            return set()
        output_files = fs.glob(os.path.join(path, "row_*.parquet"))
    except Exception:
        return set()

    processed_indices = set()
    for f in output_files:
        # Parse index from filename: row_000123_ms_id.parquet
        basename = os.path.basename(f)
        if basename.startswith("row_") and ".parquet" in basename:
            try:
                idx_str = basename.split("_")[1]
                processed_indices.add(int(idx_str))
            except (IndexError, ValueError):
                pass

    logger.info("Found %d already-processed rows", len(processed_indices))
    return processed_indices


@dataclass
class RegenerateConfig:
    input_path: str
    output_path: str


def run_regeneration_parallel(config: RegenerateConfig):
    """Run regeneration with parallel workers and incremental saves."""
    # Initialize Ray if not already
    if not ray.is_initialized():
        ray.init()

    # Load all input rows
    all_rows = load_all_input_rows(config.input_path)

    # Get already-processed row indices
    processed_indices = get_processed_indices(config.output_path)

    # Build list of (idx, row) pairs to process
    rows_to_process = [
        (idx, row) for idx, row in enumerate(all_rows)
        if idx not in processed_indices
    ]

    logger.info("Tasks to process: %d (%d already processed)",
                len(rows_to_process), len(processed_indices))

    if not rows_to_process:
        logger.info("All rows already processed!")
        return

    # Process with limited parallelism to respect rate limits
    completed = 0
    total = len(rows_to_process)
    pending_tasks = {}  # Map from task ref to row index
    row_iter = iter(rows_to_process)

    # Submit initial batch
    for _ in range(min(MAX_CONCURRENT, total)):
        try:
            idx, row = next(row_iter)
            task = process_and_save_row.remote(
                row=row,
                output_path=config.output_path,
                row_index=idx,
            )
            pending_tasks[task] = idx
        except StopIteration:
            break

    logger.info("Started with %d concurrent tasks (max %d)", len(pending_tasks), MAX_CONCURRENT)

    # Process tasks, submitting new ones as others complete
    while pending_tasks:
        # Wait for at least one task to complete
        ready, _ = ray.wait(list(pending_tasks.keys()), num_returns=1, timeout=120)

        for ref in ready:
            row_idx = pending_tasks.pop(ref)
            try:
                result = ray.get(ref)
                completed += 1
                if completed % 50 == 0 or completed == total:
                    logger.info("Progress: %d/%d (%.1f%%) - Last: row %d (%s)",
                               completed, total, 100 * completed / total,
                               row_idx, result.get("status", "?"))
            except Exception as e:
                logger.error("Task failed for row %d: %s", row_idx, e)
                completed += 1

            # Submit next task if available
            try:
                idx, row = next(row_iter)
                task = process_and_save_row.remote(
                    row=row,
                    output_path=config.output_path,
                    row_index=idx,
                )
                pending_tasks[task] = idx
            except StopIteration:
                pass  # No more rows to process

    logger.info("Completed! Processed %d rows", completed)

    logger.info("Completed! Processed %d rows", completed)


def combine_output_files(output_path: str, final_output: str):
    """Combine individual row files into final parquet files."""
    fs, path = fsspec.core.url_to_fs(output_path)
    row_files = sorted(fs.glob(os.path.join(path, "row_*.parquet")))

    if not row_files:
        logger.warning("No row files found to combine")
        return

    logger.info("Combining %d row files into final output", len(row_files))

    # Read all and combine
    all_rows = []
    for rf in row_files:
        full_path = f"{fs.protocol}://{rf}" if fs.protocol != "file" else rf
        table = pq.read_table(full_path, filesystem=fs)
        all_rows.extend(table.to_pylist())

    # Write combined output
    combined_table = pa.Table.from_pylist(all_rows)
    pq.write_table(combined_table, final_output)
    logger.info("Wrote combined output with %d rows to %s", len(all_rows), final_output)


open_thoughts_4_math_qwen3_235b_a22b_fp8_tpu_annotated_parallel = ExecutorStep(
    name="documents/open-thoughts-4-30k-math-qwen3-235b-a22b-fp8-tput-annotated-32768-parallel",
    fn=run_regeneration_parallel,
    config=RegenerateConfig(
        input_path=ot4_math_qwen_235b_annotated,
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[open_thoughts_4_math_qwen3_235b_a22b_fp8_tpu_annotated_parallel]
    )
