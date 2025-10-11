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
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.queue import Queue
import datetime
import threading
import queue
from marin.core.runtime import cached_or_construct_output
from marin.processing.classification.classifier import (
    AutoClassifier,
    BaseClassifier,
    AutoClassifierRayActor,
)
from marin.processing.classification.cleanup import Reaper
from marin.processing.classification.config.inference_config import DatasetSchemaConfig, InferenceConfig
from marin.utils import (
    fsspec_glob,
    fsspec_mkdirs,
    rebase_file_path,
)
from marin.processing.classification.autoscaler import AutoscalingActorPool

logger = logging.getLogger("ray")


def _kill_processes_by_regex_if_no_vllm(pattern: str) -> None:
    """Hacky cleanup: kill processes whose name/cmdline matches pattern
    only if there is no process that mentions "VLLM".

    This is intended to mitigate cases where accelerator lock files are held
    by crashed or orphaned Ray workers.
    """
    try:
        import os as _os
        import re as _re
        import signal as _signal
        import psutil as _psutil  # type: ignore
    except Exception:
        # If psutil or other deps aren't available, silently skip
        return

    try:
        # Check if any VLLM process is present; if so, do nothing
        has_vllm = False
        for proc in _psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            try:
                info = proc.info
                name = info.get("name") or ""
                cmdline = " ".join(info.get("cmdline") or [])
                haystack = f"{name} {cmdline}"
                if _re.search(r"VLLM", haystack):
                    has_vllm = True
                    break
            except (_psutil.NoSuchProcess, _psutil.AccessDenied, _psutil.ZombieProcess) as e:
                print(f"Error getting process info for {proc.pid}: {e}")
                continue

        if has_vllm:
            return

        # No VLLM processes found; kill processes matching the provided pattern
        for proc in _psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            try:
                info = proc.info
                name = info.get("name") or ""
                cmdline = " ".join(info.get("cmdline") or [])
                haystack = f"{name} {cmdline}"
                if _re.search(pattern, haystack, flags=_re.IGNORECASE):
                    if proc.pid == _os.getpid():
                        continue
                    _os.kill(proc.pid, _signal.SIGKILL)
            except (_psutil.NoSuchProcess, _psutil.AccessDenied, _psutil.ZombieProcess) as e:
                print(f"Error killing process {proc.pid}: {e}")
                continue
    except Exception:
        # Best-effort cleanup; ignore any unexpected errors
        pass


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


def count_existing_rows(output_filename: str) -> int:
    """Count the number of rows already processed in the output file

    Uses fsspec to support remote filesystems (e.g., gs://) instead of os.path.exists.
    """
    import datasets
    import fsspec

    try:
        # Check if file exists on local or remote filesystem
        fs, _ = fsspec.core.url_to_fs(output_filename)
        if not fs.exists(output_filename):
            return 0

        # Use datasets library to count rows efficiently
        datasets.disable_caching()
        datasets.logging.set_verbosity_warning()

        if output_filename.endswith((".jsonl.gz", ".jsonl.zst", ".jsonl")):
            # Load as JSON lines with streaming to count
            # dataset = datasets.load_dataset("json", data_files=output_filename, streaming=True, split="train")
            with fsspec.open(output_filename, "rt", compression="infer") as f:
                count = 0
                for _ in f:
                    count += 1

            return count
        elif output_filename.endswith(".parquet"):
            # Load parquet with streaming to count
            import pyarrow.dataset as ds

            def count_rows_dataset(path: str) -> int:
                dataset = ds.dataset(path, format="parquet")  # local, s3://..., gs://..., etc.
                # Arrow ≥12 has count_rows(); for older Arrow, fall back to sum of file metadata
                try:
                    return dataset.count_rows()  # uses metadata; very fast
                except AttributeError:
                    return sum(ds.parquet_dataset_factory(path).finish().count_rows())

            return count_rows_dataset(output_filename)
        else:
            return 0

        # Count rows by iterating through the streaming dataset
        # count = 0
        # for _ in dataset:
        #     count += 1

        # return count
    except (FileNotFoundError, Exception):
        return 0


def get_id_from_row(row: dict, id_column: str | dict[str, str]) -> str:
    """Get the ID from a row

    Args:
        row: The data row
        id_column: Either a string column name, or a dict with nested column access
                   e.g., {"metadata": "id"} means row["metadata"]["id"]

    Returns:
        The ID value from the row
    """
    if isinstance(id_column, dict):
        # Handle nested column access
        parent_key = next(iter(id_column.keys()))
        child_key = next(iter(id_column.values()))
        return row[parent_key][child_key]
    else:
        return row[id_column]


def has_id_column(row: dict, id_column: str | dict[str, str]) -> bool:
    """Check if a row has the required id column

    Args:
        row: The data row
        id_column: Either a string column name, or a dict with nested column access

    Returns:
        True if the id column exists in the row
    """
    if isinstance(id_column, dict):
        parent_key = next(iter(id_column.keys()))
        child_key = next(iter(id_column.values()))
        return parent_key in row and isinstance(row[parent_key], dict) and child_key in row[parent_key]
    else:
        return id_column in row


def get_finished_ids(output_filename: str, id_column: str | dict[str, str]) -> set:
    """Get the set of IDs that have already been processed in the output file

    Args:
        output_filename: Path to the output file
        id_column: Name of the column containing the ID

    Returns:
        Set of IDs that have already been processed
    """
    import json
    import fsspec

    finished_ids = set()

    try:
        # Check if file exists on local or remote filesystem
        fs, _ = fsspec.core.url_to_fs(output_filename)
        if not fs.exists(output_filename):
            return finished_ids

        if output_filename.endswith((".jsonl.gz", ".jsonl.zst", ".jsonl")):
            # Read JSON lines and extract IDs
            with fsspec.open(output_filename, "rt", compression="infer") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        if has_id_column(row, id_column):
                            finished_ids.add(get_id_from_row(row, id_column))
                    except json.JSONDecodeError:
                        continue
        elif output_filename.endswith(".parquet"):
            # Read parquet and extract IDs
            import pyarrow.parquet as pq

            with fsspec.open(output_filename, "rb") as f:
                if isinstance(id_column, dict):
                    # Handle nested column access
                    parent_key = next(iter(id_column.keys()))
                    child_key = next(iter(id_column.values()))
                    table = pq.read_table(f, columns=[parent_key])
                    # Extract child values from nested dicts
                    for row in table.to_pylist():
                        if parent_key in row and isinstance(row[parent_key], dict) and child_key in row[parent_key]:
                            finished_ids.add(row[parent_key][child_key])
                else:
                    # Simple column access
                    table = pq.read_table(f, columns=[id_column])
                    finished_ids = set(table[id_column].to_pylist())

        return finished_ids
    except (FileNotFoundError, Exception) as e:
        print(f"[!] Error reading finished IDs from {output_filename}: {e}")
        return finished_ids


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
            elif col in batch_dict:
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
            else:
                # Ensure file exists for append
                open(tmp_path, "wb").close()

        # Write rows to temp with append semantics using fsspec with compression inference
        with fsspec.open(tmp_path, mode, compression="infer") as f:
            for row in rows_iterator:
                row = make_json_serializable(row)
                f.write((json.dumps(row) + "\n").encode("utf-8"))

        # Upload temp file to destination (overwrite remote with full content)
        with fsspec.open(output_filename, "wb") as dst, open(tmp_path, "rb") as src:
            shutil.copyfileobj(src, dst)
    elif output_filename.endswith(".parquet"):
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
    else:
        raise ValueError(f"Unsupported filetype: {output_filename}")


class AsyncJSONLWriter:
    """Asynchronous JSONL writer that writes to a deterministic local temp file and uploads on close.

    This preserves the existing append semantics against remote object stores by writing locally
    and then overwriting the remote object at the end of the job.
    """

    def __init__(
        self,
        output_filename: str,
        append: bool = False,
        max_queue_size: int = 128,
        export_every_n_batches: int = 3,
    ):
        import hashlib
        import fsspec

        self.output_filename = output_filename
        self.append = append
        self.max_queue_size = max_queue_size
        self.export_every_n_batches = export_every_n_batches
        self._q: queue.Queue | None = queue.Queue(maxsize=max_queue_size)
        self._exc: Exception | None = None
        self._stop_sentinel = object()
        self._batches_since_export: int = 0

        file_hash = hashlib.sha256(output_filename.encode("utf-8")).hexdigest()
        if output_filename.endswith(".jsonl.gz"):
            self.tmp_path = f"/tmp/marin_{file_hash}.jsonl.gz"
        elif output_filename.endswith(".jsonl.zst"):
            self.tmp_path = f"/tmp/marin_{file_hash}.jsonl.zst"
        else:
            self.tmp_path = f"/tmp/marin_{file_hash}.jsonl"

        # Hydrate local temp if appending and temp doesn't exist
        # It is possible that the file exists at a lesser state of completion due to preemption.
        # Imagine Node A completes 1 batch then preempted, Node B then completes 2 batches and gets preempted.
        # Now, Node A should resume from the state that the file at global path states it is at.
        if append and not os.path.exists(self.tmp_path):
            fs, _ = fsspec.core.url_to_fs(output_filename)
            if fs.exists(output_filename):
                with fsspec.open(output_filename, "rb") as src, open(self.tmp_path, "wb") as dst:
                    import shutil as _shutil

                    _shutil.copyfileobj(src, dst)
            else:
                open(self.tmp_path, "wb").close()

        self._thread = threading.Thread(target=self._run_writer, daemon=True)
        self._thread.start()

    def submit_rows(self, rows: list[dict]) -> None:
        if self._exc:
            raise self._exc
        assert self._q is not None
        self._q.put(rows)

    def close(self) -> None:
        import fsspec

        assert self._q is not None
        self._q.put(self._stop_sentinel)
        self._thread.join()
        if self._exc:
            raise self._exc
        # Final snapshot upload
        with fsspec.open(self.output_filename, "wb") as dst, open(self.tmp_path, "rb") as src:
            import shutil as _shutil

            _shutil.copyfileobj(src, dst)

    def _run_writer(self) -> None:
        import json
        import fsspec

        mode = "ab" if self.append else "wb"
        try:
            with fsspec.open(self.tmp_path, mode, compression="infer") as f:
                while True:
                    item = self._q.get()
                    if item is self._stop_sentinel:
                        break
                    for row in item:
                        row = make_json_serializable(row)
                        f.write((json.dumps(row) + "\n").encode("utf-8"))
                    # Count one submitted batch and optionally export snapshot
                    if self.export_every_n_batches and self.export_every_n_batches > 0:
                        self._batches_since_export += 1
                        if self._batches_since_export >= self.export_every_n_batches:
                            # Flush current temp file and upload snapshot
                            f.flush()
                            try:
                                import shutil as _shutil

                                with fsspec.open(self.output_filename, "wb") as dst, open(self.tmp_path, "rb") as src:
                                    _shutil.copyfileobj(src, dst)
                            except Exception as e:
                                # Non-fatal; next cycles or close will retry final upload
                                print(f"Error uploading snapshot to {self.output_filename}: {e}")
                            self._batches_since_export = 0
                f.flush()
        except Exception as e:
            self._exc = e
        finally:
            try:
                # Drain remaining items to unblock producers if any
                while not self._q.empty():
                    self._q.get_nowait()
            except Exception:
                pass


def read_dataset(input_filename: str, columns: list[str] | None = None):
    """Read in a dataset and return as a Huggingface Dataset"""
    import datasets

    datasets.disable_caching()
    datasets.logging.set_verbosity_warning()

    if input_filename.endswith(".jsonl.gz"):
        df = pd.read_json(input_filename, compression="gzip", lines=True)
    elif input_filename.endswith(".jsonl.zst"):
        df = pd.read_json(input_filename, compression="zstd", lines=True)
    elif input_filename.endswith(".parquet"):
        df = pd.read_parquet(input_filename)
    else:
        raise ValueError(f"Unsupported filetype: {input_filename}")

    if columns:
        df = df[columns]

    return datasets.Dataset.from_pandas(df)


def write_dataset(dataset, output_filename: str):
    """Writes a Huggingface Dataset to a file (remote or local) - kept for backward compatibility"""
    if output_filename.endswith(".jsonl.gz"):
        dataset.to_json(output_filename, compression="gzip")
    elif output_filename.endswith(".jsonl.zst"):
        df_pandas = dataset.to_pandas()
        df_pandas.to_json(output_filename, orient="records", compression="zstd", lines=True)
    elif output_filename.endswith(".parquet"):
        dataset.to_parquet(output_filename)
    else:
        raise ValueError(f"Unsupported filetype: {output_filename}")


def get_input_dataset_column_names(dataset_schema: DatasetSchemaConfig | None = None) -> list[str]:
    schema = dataset_schema or DatasetSchemaConfig()
    return schema.input_columns


def get_output_dataset_column_names(dataset_schema: DatasetSchemaConfig | None = None) -> list[str]:
    schema = dataset_schema or DatasetSchemaConfig()
    return schema.output_columns


@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_with_quality_classifier_streaming(
    input_filename: str,
    output_filename: str,
    # quality_classifier: BaseClassifier,
    model_name_or_path: str,
    attribute_name: str,
    model_type: str,
    classifier_kwargs: dict,
    batch_size: int = 512,
    resume: bool = True,
    use_autoscaling_actor_pool: bool = False,
    classifier_actor_options: dict | None = None,
    dataset_schema: DatasetSchemaConfig | None = None,
):
    """Process a file with streaming I/O and resumption capability"""
    print(f"[*] Processing {input_filename} to {output_filename}")

    # Initialize mutable defaults
    if classifier_actor_options is None:
        classifier_actor_options = {}

    # Get column names
    input_column_names = get_input_dataset_column_names(dataset_schema)
    output_column_names = get_output_dataset_column_names(dataset_schema)

    # Check if we should resume from existing progress by loading finished IDs
    finished_ids = set()
    if resume:
        finished_ids = get_finished_ids(output_filename, dataset_schema.id_column)
        if finished_ids:
            print(f"[*] Resuming: found {len(finished_ids)} already processed IDs")
        else:
            print(f"[*] No existing IDs found in {output_filename}")

    # Create streaming iterator
    row_iterator = read_dataset_streaming(input_filename, input_column_names)

    # Initialize for batch processing
    append_mode = len(finished_ids) > 0

    # For parquet, collect batches; for JSONL, stream asynchronously
    # if output_filename.endswith(".parquet"):
    #     parquet_batches = []
    # async_writer = None
    # else:
    # async_writer = AsyncJSONLWriter(output_filename, append=append_mode)

    batch = []
    total_processed = len(finished_ids)
    total_skipped = 0

    task_queue = Queue()
    result_queue = Queue()

    if use_autoscaling_actor_pool:
        pool = AutoscalingActorPool(
            AutoClassifierRayActor,
            model_name_or_path,
            attribute_name,
            model_type,
            task_queue,
            result_queue,
            actor_kwargs=classifier_kwargs,
            actor_options=classifier_actor_options,
        )
    else:
        quality_classifier = AutoClassifier.from_model_path(
            model_name_or_path, attribute_name, model_type, **classifier_kwargs
        )

    num_batches = 0
    for row in row_iterator:
        # Skip rows that have already been processed
        row_id = get_id_from_row(row, dataset_schema.id_column)
        if row_id in finished_ids:
            total_skipped += 1
            continue

        batch.append(row)

        if len(batch) >= batch_size:
            # Process batch
            batch_dict = {}
            for key in input_column_names:
                batch_dict[key] = [row.get(key, "") for row in batch]

            num_batches += 1
            # Apply classifier
            if use_autoscaling_actor_pool:
                task_queue.put(batch_dict)
            else:
                processed_batch = quality_classifier(batch_dict)
                output_rows = convert_batch_dict_to_output_rows(processed_batch, output_column_names, len(batch))
                write_dataset_streaming(output_rows, output_filename, append=append_mode or total_processed > 0)
                total_processed += len(batch)
                logger.info(f"[*] Processed {total_processed} rows (skipped {total_skipped}) from {input_filename}")

            batch = []

    # Final batch that might not be of size batch_size
    if batch:
        batch_dict = {}
        for key in input_column_names:
            batch_dict[key] = [row.get(key, "") for row in batch]
        num_batches += 1

        if use_autoscaling_actor_pool:
            task_queue.put(batch_dict)
        else:
            processed_batch = quality_classifier(batch_dict)
            output_rows = convert_batch_dict_to_output_rows(processed_batch, output_column_names, len(batch))
            write_dataset_streaming(output_rows, output_filename, append=append_mode or total_processed > 0)
            total_processed += len(batch)
            logger.info(f"[*] Processed {total_processed} rows (skipped {total_skipped}) from {input_filename}")

    if use_autoscaling_actor_pool:
        num_collected_batches = 0
        while num_collected_batches < num_batches:
            processed_batch = result_queue.get()

            num_collected_batches += 1

            batch_size = len(processed_batch["text"])
            output_rows = convert_batch_dict_to_output_rows(processed_batch, output_column_names, batch_size)
            write_dataset_streaming(output_rows, output_filename, append=append_mode or total_processed > 0)
            total_processed += batch_size
            logger.info(f"[*] Processed {total_processed} rows (skipped {total_skipped}) from {input_filename}")

        pool.shutdown()

        # Write batch
    #         if output_filename.endswith(".parquet"):
    #             parquet_batches.extend(output_rows)
    #             # Write parquet in larger chunks to be efficient
    #             if len(parquet_batches) >= batch_size * 10:
    #                 _write_parquet_batch(parquet_batches, output_filename, append=append_mode or total_processed > 0)
    #                 parquet_batches = []
    #         else:
    #             # Enqueue JSONL write to overlap with next compute batch
    #             # async_writer.submit_rows(output_rows)
    #             write_dataset_streaming(output_rows, output_filename, append=append_mode or total_processed > 0)

    #         total_processed += len(batch)
    #         print(f"[*] Processed {total_processed} rows from {input_filename}")
    #         batch = []

    # # Process remaining rows in final batch
    # if batch:
    #     batch_dict = {}
    #     for key in input_column_names:
    #         batch_dict[key] = [row.get(key, "") for row in batch]

    #     processed_batch = quality_classifier(batch_dict)

    #     output_rows = []
    #     for i in range(len(batch)):
    #         output_row = {}
    #         for col in output_column_names:
    #             if col in processed_batch:
    #                 output_row[col] = processed_batch[col][i]
    #             elif col in batch_dict:
    #                 output_row[col] = batch_dict[col][i]
    #         output_rows.append(output_row)

    #     # Write final batch
    #     if output_filename.endswith(".parquet"):
    #         parquet_batches.extend(output_rows)
    #     else:
    #         # Enqueue final JSONL rows
    #         # async_writer.submit_rows(output_rows)
    #         write_dataset_streaming(output_rows, output_filename, append=append_mode or total_processed > 0)

    #     total_processed += len(batch)

    # Flush remaining writes
    # if output_filename.endswith(".parquet"):
    #     if parquet_batches:
    #         _write_parquet_batch(parquet_batches, output_filename, append_mode or total_processed > rows_to_skip)
    # else:
    # Finalize JSONL: join background writer and upload to destination
    # if async_writer is not None:
    #     async_writer.close()

    print(
        f"[*] Completed processing {input_filename} - \
        Total rows: {total_processed} (skipped {total_skipped} already finished)"
    )


def _write_parquet_batch(rows: list, output_filename: str, append: bool):
    """Helper function to write a batch of rows to parquet"""
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return

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


@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_with_quality_classifier(input_filename: str, output_filename: str, quality_classifier: BaseClassifier):
    """Legacy function - kept for backward compatibility"""
    process_file_with_quality_classifier_streaming(input_filename, output_filename, quality_classifier)


# @ray.remote(max_calls=1)
# @remove_tpu_lockfile_on_exit
@ray.remote(max_calls=1)
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_ray(
    input_filename: str,
    output_filename: str,
    model_name_or_path: str,
    attribute_name: str,
    model_type: str | None,
    filetype: str,
    classifier_kwargs: dict,
    batch_size: int = 512,
    resume: bool = True,
    queue: Queue | None = None,
    use_autoscaling_actor_pool: bool = False,
    classifier_actor_options: dict | None = None,
    dataset_schema: DatasetSchemaConfig | None = None,
):
    try:
        # Initialize mutable defaults
        if classifier_actor_options is None:
            classifier_actor_options = {}

        # quality_classifier = AutoClassifier.from_model_path(
        #     model_name_or_path, attribute_name, model_type, **classifier_kwargs
        # )

        process_file_with_quality_classifier_streaming(
            input_filename,
            output_filename,
            model_name_or_path,
            attribute_name,
            model_type,
            classifier_kwargs,
            batch_size,
            resume,
            use_autoscaling_actor_pool,
            classifier_actor_options,
            dataset_schema=dataset_schema,
        )

    except Exception:
        # On failure, try to clear out stray Ray worker processes if safe
        # _kill_processes_by_regex_if_no_vllm(r"process_file_ray")
        raise
    finally:
        if queue is not None:
            queue.put(
                {
                    "pid": os.getpid(),
                    "node_id": ray.get_runtime_context().get_node_id(),
                }
            )


# @ray.remote(max_calls=1)
# @remove_tpu_lockfile_on_exit
# @ray.remote(max_calls=1)
# @cached_or_construct_output(success_suffix="SUCCESS")
# def _process_dir(
#     input_path: str,
#     output_path: str,
#     model_name_or_path: str,
#     attribute_name: str,
#     model_type: str | None,
#     filetype: str,
#     classifier_kwargs: dict,
#     batch_size: int = 512,
#     resume: bool = True,
#     queue: Queue | None = None,
#     use_autoscaling_actor_pool: bool = False,
# ):
#     """Perform quality classification on a directory of files

#     We assume that the input_path is a directory of files. Using _process_dir is more
#     efficient than process_file_ray because it avoids the overhead of spawning a new
#     Ray task for each file and instead processes all files in a single task.
#     """

#     files = fsspec_glob(os.path.join(input_path, f"*.{filetype}"))

#     if len(files) == 0:
#         logger.error(f"No files found in {input_path} with pattern {filetype}!!! This is likely an error.")
#         return

#     quality_classifier = AutoClassifier.from_model_path(
#         model_name_or_path, attribute_name, model_type, **classifier_kwargs
#     )

#     for input_filename in files:
#         output_filename = rebase_file_path(input_path, input_filename, output_path)
#         process_file_with_quality_classifier_streaming(
#             input_filename, output_filename, quality_classifier, batch_size, resume, use_autoscaling_actor_pool
#         )


def get_process_filepath_func(subdirectories: list[str]):
    # if len(subdirectories) > 0:
    #     return _process_dir
    # else:
    return process_file_ray


def get_filepaths_and_process_filepath_func(inference_config: InferenceConfig):
    # NOTE(chris): Maximize parallelism by doing one task per file. If this is too high
    # then we can use _process_dir to process multiple files in a single task.
    process_filepath_func = process_file_ray
    filepaths = fsspec_glob(os.path.join(inference_config.input_path, f"**/*.{inference_config.filetype}"))

    return filepaths, process_filepath_func


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def cleanup_tpu_processes(queue: Queue):
    killed_pids = []
    while True:
        info = queue.get()

        if info is None:
            break
        reaper = Reaper.options(scheduling_strategy=NodeAffinitySchedulingStrategy(info["node_id"], soft=False)).remote()
        results = ray.get(reaper.kill_if_holding_accel.remote(info["pid"]))
        killed_pids.extend(results["killed"])

    return killed_pids


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def run_inference(inference_config: InferenceConfig):
    logger.info(f"Running inference for {inference_config.input_path} to {inference_config.output_path}")
    filepaths, process_filepath_func = get_filepaths_and_process_filepath_func(inference_config)

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

    if not inference_config.classifier_actor_options:
        options_kwargs["resources"] = inference_config.runtime.resources

    pending_refs: dict = {}
    attempt_count: dict[str, int] = {}

    queue = Queue(actor_options={"num_cpus": 0, "resources": {"head_node": 0.001}})

    def submit(input_fp: str):
        output_fp = rebase_file_path(input_path, input_fp, output_path)
        fsspec_mkdirs(os.path.dirname(output_fp))
        ref = process_filepath_func.options(**options_kwargs).remote(
            input_fp,
            output_fp,
            inference_config.model_name,
            inference_config.attribute_name,
            inference_config.model_type,
            inference_config.filetype,
            inference_config.classifier_kwargs,
            inference_config.batch_size,
            inference_config.resume,
            queue,
            inference_config.use_autoscaling_actor_pool,
            inference_config.classifier_actor_options,
            inference_config.dataset_schema,
        )
        pending_refs[ref] = input_fp

    cleanup_task = cleanup_tpu_processes.remote(queue)

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
                # Hacky cleanup on failure to clear accelerator locks held by stray workers
                _kill_processes_by_regex_if_no_vllm(r"process_file_ray")
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
            # Hacky cleanup on failure to clear accelerator locks held by stray workers
            _kill_processes_by_regex_if_no_vllm(r"process_file_ray")
            if count < max_retries_per_file:
                submit(file_for_ref)
            else:
                logger.error(f"Giving up after {count} attempts for {file_for_ref}")

    # Stop sentinel for cleanup since all done
    queue.put(None)
    killed_pids = ray.get(cleanup_task)
    print(f"Killed PIDs: {killed_pids}")


@draccus.wrap()
def main(inference_config: InferenceConfig):
    ray.get(run_inference.remote(inference_config))


if __name__ == "__main__":
    main()
