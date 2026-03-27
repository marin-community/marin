# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Classification inference using Fray for job dispatch and actor management.

Usage:

python -m marin.processing.classification.inference \
    --config_path marin/processing/classification/config/dclm_fasttext.yaml
"""

import logging
import os
import queue
import time

import draccus

from fray.v2 import Entrypoint, JobRequest, JobStatus, ResourceConfig, current_client

from marin.core.runtime import cached_or_construct_output
from marin.processing.classification.autoscaler import AutoscalingActorPool, AutoscalingActorPoolConfig
from marin.processing.classification.checkpoint_utils import get_finished_ids, get_id_from_row
from marin.processing.classification.classifier import AutoClassifier
from marin.processing.classification.config.inference_config import DatasetSchemaConfig, InferenceConfig
from marin.processing.classification.dataset_utils import read_dataset_streaming, write_dataset_streaming
from marin.utils import (
    fsspec_glob,
    fsspec_mkdirs,
    rebase_file_path,
)

logger = logging.getLogger(__name__)


def convert_batch_dict_to_output_rows(batch_dict: dict, output_column_names: list[str], batch_size: int) -> list[dict]:
    output_rows = []
    for i in range(batch_size):
        output_row = {}
        for col in output_column_names:
            if col in batch_dict:
                output_row[col] = batch_dict[col][i]
        output_rows.append(output_row)

    return output_rows


@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_with_quality_classifier_streaming(
    input_filename: str,
    output_filename: str,
    model_name_or_path: str,
    attribute_name: str,
    model_type: str,
    dataset_schema: DatasetSchemaConfig,
    autoscaling_actor_pool_config: AutoscalingActorPoolConfig,
    num_batches_per_upload: int,
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

    task_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()

    pool = AutoscalingActorPool(
        AutoClassifier,
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
    output_rows_buffer = []
    while num_collected_batches < num_batches:
        processed_batch = result_queue.get()
        num_collected_batches += 1
        output_rows = convert_batch_dict_to_output_rows(
            processed_batch, dataset_schema.output_columns, len(processed_batch[dataset_schema.output_columns[0]])
        )
        output_rows_buffer.extend(output_rows)

        if num_collected_batches % num_batches_per_upload == 0:
            write_dataset_streaming(output_rows_buffer, output_filename, append=append_mode or total_processed > 0)
            output_rows_buffer = []

        total_processed += len(processed_batch)
        logger.info(f"[*] Processed {total_processed} rows (skipped {total_skipped}) from {input_filename}")

    if output_rows_buffer:
        write_dataset_streaming(output_rows_buffer, output_filename, append=append_mode or total_processed > 0)

    pool.shutdown()

    print(
        f"[*] Completed processing {input_filename} - \
        Total rows: {total_processed} (skipped {total_skipped} already finished)"
    )


def run_inference(inference_config: InferenceConfig):
    logger.info(f"Running inference for {inference_config.input_path} to {inference_config.output_path}")
    filepaths = fsspec_glob(os.path.join(inference_config.input_path, f"**/*.{inference_config.filetype}"))

    if len(filepaths) == 0:
        pattern = f"**/*.{inference_config.filetype}"
        raise FileNotFoundError(f"No files found in {inference_config.input_path} with pattern {pattern}")

    input_path = inference_config.input_path
    output_path = inference_config.output_path

    max_in_flight = inference_config.task.max_in_flight
    max_retries_per_file = 100

    client = current_client()

    pending_handles: dict = {}
    attempt_count: dict[str, int] = {}

    memory_gb = inference_config.runtime.memory_limit_gb
    ram_str = f"{memory_gb}g" if memory_gb >= 1 else f"{int(memory_gb * 1024)}m"

    def submit(input_fp: str):
        output_fp = rebase_file_path(input_path, input_fp, output_path)
        fsspec_mkdirs(os.path.dirname(output_fp))

        # Sanitize name for Fray job naming (no spaces, no special chars)
        job_name = f"classify-{os.path.basename(input_fp).replace('.', '-')}"
        job_name = job_name.replace(" ", "-")[:60]

        handle = client.submit(
            JobRequest(
                name=job_name,
                entrypoint=Entrypoint.from_callable(
                    process_file_with_quality_classifier_streaming,
                    args=[
                        input_fp,
                        output_fp,
                        inference_config.model_name,
                        inference_config.attribute_name,
                        inference_config.model_type,
                        inference_config.dataset_schema,
                        inference_config.autoscaling_actor_pool_config,
                        inference_config.num_batches_per_upload,
                        inference_config.batch_size,
                        inference_config.resume,
                    ],
                ),
                resources=ResourceConfig(ram=ram_str),
            )
        )
        pending_handles[handle] = input_fp

    def drain_one_completed():
        """Wait for one pending job, handle its result, and retry on failure."""
        while True:
            for handle in list(pending_handles):
                status = handle.status()
                if JobStatus.finished(status):
                    file_for_handle = pending_handles.pop(handle)
                    if status == JobStatus.SUCCEEDED:
                        logger.info(f"Completed: {file_for_handle}")
                    else:
                        count = attempt_count.get(file_for_handle, 0) + 1
                        attempt_count[file_for_handle] = count
                        logger.warning(f"Task failed for {file_for_handle} (attempt {count})")
                        if count < max_retries_per_file:
                            submit(file_for_handle)
                        else:
                            logger.error(f"Giving up after {count} attempts for {file_for_handle}")
                    return
            time.sleep(0.1)

    for input_filepath in filepaths:
        while max_in_flight is not None and len(pending_handles) >= max_in_flight:
            drain_one_completed()
        submit(input_filepath)

    while pending_handles:
        drain_one_completed()


@draccus.wrap()
def main(inference_config: InferenceConfig):
    run_inference(inference_config)


if __name__ == "__main__":
    main()
