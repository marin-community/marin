# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.inference \
    --config_path marin/processing/classification/config/dclm_fasttext.yaml
"""

import logging
import os

import draccus
import pandas as pd
import ray
from ray.util.queue import Queue
import datetime
from marin.core.runtime import cached_or_construct_output
from marin.processing.classification.classifier import (
    AutoClassifierRayActor,
)
from marin.processing.classification.config.inference_config import DatasetSchemaConfig, InferenceConfig
from marin.utils import (
    fsspec_glob,
    fsspec_mkdirs,
    rebase_file_path,
)
from marin.processing.classification.checkpoint_utils import get_finished_ids, get_id_from_row
from marin.processing.classification.autoscaler import AutoscalingActorPool, AutoscalingActorPoolConfig

logger = logging.getLogger("ray")


def read_dataset_streaming(input_filename: str, columns: list[str] | None = None):
    """Read in a dataset as a streaming iterator using datasets library

    Args:
        input_filename: str
            The path to the input file. Currently supports .jsonl.gz, .jsonl.zst, and .parquet

    Returns:
        Iterator: An iterator over the dataset rows
    """
    import datasets

    # Disable caching for streaming
    datasets.disable_caching()
    datasets.logging.set_verbosity_warning()

    # Determine file type and load with streaming
    if input_filename.endswith((".jsonl.gz", ".jsonl.zst", ".jsonl")):
        # Load as JSON lines with streaming
        dataset = datasets.load_dataset("json", data_files=input_filename, streaming=True, split="train")
    elif input_filename.endswith(".parquet"):
        # Load parquet with streaming
        dataset = datasets.load_dataset("parquet", data_files=input_filename, streaming=True, split="train")
    else:
        raise ValueError(f"Unsupported filetype: {input_filename}")

    # Filter columns if specified
    if columns:
        dataset = dataset.select_columns(columns)

    # Yield rows from the streaming dataset
    yield from dataset


# TODO(chris): Consolidate this with other make json serializable functions
def make_json_serializable(row: dict) -> dict:
    """Make a row JSON serializable"""
    for key, value in row.items():
        if isinstance(value, dict):
            row[key] = make_json_serializable(value)
        if isinstance(value, datetime.datetime):
            row[key] = value.isoformat()
    return row


def convert_batch_dict_to_output_rows(batch_dict: dict, output_column_names: list[str], batch_size: int) -> list[dict]:
    output_rows = []
    for i in range(batch_size):
        output_row = {}
        for col in output_column_names:
            if col in batch_dict:
                output_row[col] = batch_dict[col][i]
        output_rows.append(output_row)

    return output_rows


def write_dataset_streaming(rows_iterator, output_filename: str, append: bool = False):
    """Writes rows to a file in streaming fashion

    JSONL behavior:
      - For JSONL(.gz/.zst), write to a deterministic temp file under /tmp in append-mode,
        then upload the full file to the destination (e.g., gs://...). This provides safe
        append semantics for object stores.
      - Checkpoint restoration should continue to read from the remote path.
    """
    import json
    import hashlib
    import os
    import shutil

    import fsspec

    mode = "ab" if append else "wb"

    if ".jsonl" in output_filename:
        # Build deterministic temp path for this output file
        file_hash = hashlib.sha256(output_filename.encode("utf-8")).hexdigest()
        if output_filename.endswith(".jsonl.gz"):
            tmp_path = f"/tmp/marin_{file_hash}.jsonl.gz"
        elif output_filename.endswith(".jsonl.zst"):
            tmp_path = f"/tmp/marin_{file_hash}.jsonl.zst"
        else:
            tmp_path = f"/tmp/marin_{file_hash}.jsonl"

        # If appending and local temp doesn't exist, hydrate it from remote (if present)
        if append and not os.path.exists(tmp_path):
            fs, _ = fsspec.core.url_to_fs(output_filename)
            if fs.exists(output_filename):
                with fsspec.open(output_filename, "rb") as src, open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        # Turn on compression inference to have fsspec auto-compress files according to the suffix
        with fsspec.open(tmp_path, mode, compression="infer") as f:
            for row in rows_iterator:
                row = make_json_serializable(row)
                f.write((json.dumps(row) + "\n").encode("utf-8"))

        # Upload temp file to destination (overwrite remote with full content)
        with fsspec.open(output_filename, "wb") as dst, open(tmp_path, "rb") as src:
            shutil.copyfileobj(src, dst)
        return
    if output_filename.endswith(".parquet"):
        # For parquet, we need to collect rows and write in batches
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = list(rows_iterator)
        if rows:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)

            fs, _ = fsspec.core.url_to_fs(output_filename)
            if append and fs.exists(output_filename):
                # Read existing parquet and append
                with fsspec.open(output_filename, "rb") as f:
                    existing_table = pq.read_table(f)
                table = pa.concat_tables([existing_table, table])

            with fsspec.open(output_filename, "wb") as f:
                pq.write_table(table, f)
        return

    raise ValueError(f"Unsupported filetype: {output_filename}")


@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_with_quality_classifier_streaming(
    input_filename: str,
    output_filename: str,
    model_name_or_path: str,
    attribute_name: str,
    model_type: str,
    dataset_schema: DatasetSchemaConfig,
    autoscaling_actor_pool_config: AutoscalingActorPoolConfig,
    batch_size: int = 512,
    resume: bool = True,
):
    """Process a file with streaming I/O and resumption capability"""
    print(f"[*] Processing {input_filename} to {output_filename}")

    # Check if we should resume from existing progress by loading finished IDs
    finished_ids = set()
    if resume:
        finished_ids = get_finished_ids(output_filename, dataset_schema.id_column)
        if finished_ids:
            print(f"[*] Resuming: found {len(finished_ids)} already processed IDs")
        else:
            print(f"[*] No existing IDs found in {output_filename}")

    # Create streaming iterator
    row_iterator = read_dataset_streaming(input_filename, dataset_schema.input_columns)

    # Initialize for batch processing
    append_mode = len(finished_ids) > 0

    # Initialize batch
    batch = []
    total_processed = len(finished_ids)
    total_skipped = 0

    task_queue = Queue()
    result_queue = Queue()

    pool = AutoscalingActorPool(
        AutoClassifierRayActor,
        model_name_or_path,
        attribute_name,
        model_type,
        task_queue,
        result_queue,
        autoscaler_config=autoscaling_actor_pool_config,
    )

    num_batches = 0
    for row in row_iterator:
        # Skip rows that have already been processed
        row_id = get_id_from_row(row, dataset_schema.id_column)
        if row_id is None:
            logger.warning(f"No ID found in row: {row} for ID path: {dataset_schema.id_column} - skipping")
            continue
        if row_id in finished_ids:
            total_skipped += 1
            continue

        batch.append(row)

        if len(batch) >= batch_size:
            # Process batch
            batch_dict = {}
            for key in dataset_schema.input_columns:
                batch_dict[key] = [row.get(key, "") for row in batch]

            num_batches += 1
            # Apply classifier
            task_queue.put(batch_dict)
            batch = []

    # Final batch that might not be of size batch_size
    if batch:
        batch_dict = {}
        for key in dataset_schema.input_columns:
            batch_dict[key] = [row.get(key, "") for row in batch]
        num_batches += 1

        task_queue.put(batch_dict)

    num_collected_batches = 0
    while num_collected_batches < num_batches:
        processed_batch = result_queue.get()
        num_collected_batches += 1
        output_rows = convert_batch_dict_to_output_rows(
            processed_batch, dataset_schema.output_columns, len(processed_batch[dataset_schema.output_columns[0]])
        )
        write_dataset_streaming(output_rows, output_filename, append=append_mode or total_processed > 0)
        total_processed += len(processed_batch)
        logger.info(f"[*] Processed {total_processed} rows (skipped {total_skipped}) from {input_filename}")

    pool.shutdown()

    print(
        f"[*] Completed processing {input_filename} - \
        Total rows: {total_processed} (skipped {total_skipped} already finished)"
    )


@ray.remote(max_calls=1)
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_ray(
    input_filename: str,
    output_filename: str,
    model_name_or_path: str,
    attribute_name: str,
    model_type: str | None,
    filetype: str,
    autoscaling_actor_pool_config: AutoscalingActorPoolConfig,
    dataset_schema: DatasetSchemaConfig,
    batch_size: int = 512,
    resume: bool = True,
):
    process_file_with_quality_classifier_streaming(
        input_filename,
        output_filename,
        model_name_or_path,
        attribute_name,
        model_type,
        dataset_schema,
        autoscaling_actor_pool_config,
        batch_size,
        resume,
    )


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def run_inference(inference_config: InferenceConfig):
    logger.info(f"Running inference for {inference_config.input_path} to {inference_config.output_path}")
    filepaths = fsspec_glob(os.path.join(inference_config.input_path, f"**/*.{inference_config.filetype}"))

    if len(filepaths) == 0:
        pattern = f"**/*.{inference_config.filetype}"
        raise FileNotFoundError(f"No files found in {inference_config.input_path} with pattern {pattern}")

    input_path = inference_config.input_path
    output_path = inference_config.output_path

    # Resilient wait/get with per-task retries to tolerate preemptions.
    max_in_flight = inference_config.task.max_in_flight
    max_retries_per_file = 100

    options_kwargs = {
        "memory": inference_config.runtime.memory_limit_gb * 1024 * 1024 * 1024,
    }

    pending_refs: dict = {}
    attempt_count: dict[str, int] = {}

    def submit(input_fp: str):
        output_fp = rebase_file_path(input_path, input_fp, output_path)
        fsspec_mkdirs(os.path.dirname(output_fp))
        ref = process_file_ray.options(**options_kwargs).remote(
            input_fp,
            output_fp,
            inference_config.model_name,
            inference_config.attribute_name,
            inference_config.model_type,
            inference_config.filetype,
            inference_config.autoscaling_actor_pool_config,
            inference_config.dataset_schema,
            inference_config.batch_size,
            inference_config.resume,
        )
        pending_refs[ref] = input_fp

    for input_filepath in filepaths:
        # Throttle submissions
        while len(pending_refs) >= max_in_flight:
            ready_refs, _ = ray.wait(list(pending_refs.keys()), num_returns=1)
            ready_ref = ready_refs[0]
            file_for_ref = pending_refs.pop(ready_ref)
            try:
                ray.get(ready_ref)
                logger.info(f"Completed: {file_for_ref}")
            except Exception as e:
                # Log and resubmit up to max retries (tolerate spot preemptions)
                count = attempt_count.get(file_for_ref, 0) + 1
                attempt_count[file_for_ref] = count
                logger.warning(f"Task failed for {file_for_ref} (attempt {count}): {e}")
                if count < max_retries_per_file:
                    submit(file_for_ref)
                else:
                    logger.error(f"Giving up after {count} attempts for {file_for_ref}")

        submit(input_filepath)

    # Drain remaining tasks
    while len(pending_refs) > 0:
        ready_refs, _ = ray.wait(list(pending_refs.keys()), num_returns=1)
        ready_ref = ready_refs[0]
        file_for_ref = pending_refs.pop(ready_ref)
        try:
            ray.get(ready_ref)
            logger.info(f"Completed: {file_for_ref}")
        except Exception as e:
            count = attempt_count.get(file_for_ref, 0) + 1
            attempt_count[file_for_ref] = count
            logger.warning(f"Task failed for {file_for_ref} (attempt {count}): {e}")
            if count < max_retries_per_file:
                submit(file_for_ref)
            else:
                logger.error(f"Giving up after {count} attempts for {file_for_ref}")


@draccus.wrap()
def main(inference_config: InferenceConfig):
    ray.get(run_inference.remote(inference_config))


if __name__ == "__main__":
    main()
