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

import datetime
import threading
import queue
from marin.core.runtime import cached_or_construct_output
from marin.processing.classification.classifier import (
    AutoClassifier,
    BaseClassifier,
)
from marin.processing.classification.config.inference_config import InferenceConfig
from marin.utils import (
    fsspec_glob,
    fsspec_mkdirs,
    rebase_file_path,
    remove_tpu_lockfile_on_exit,
)

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


def make_json_serializable(row: dict) -> dict:
    """Make a row JSON serializable"""
    for key, value in row.items():
        if isinstance(value, dict):
            row[key] = make_json_serializable(value)
        if isinstance(value, datetime.datetime):
            row[key] = value.isoformat()
    return row


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


def get_input_dataset_column_names(input_filename: str) -> list[str]:
    if "fineweb" in input_filename.lower():
        return ["text", "id"]
    elif "dclm" in input_filename.lower():
        return ["text", "metadata"]
    else:
        logger.warning("We are assuming the input dataset has the following columns: text, id")
        return ["text", "id"]


def get_output_dataset_column_names(input_filename: str) -> list[str]:
    if "fineweb" in input_filename.lower():
        return ["id", "attributes"]
    elif "dclm" in input_filename.lower():
        # HACK(chris): Either standardize or make the user specify the output columns
        return ["metadata", "attributes", "generated_text", "text"]
    else:
        logger.warning("We are assuming the output dataset has the following columns: id, attributes")
        return ["id", "attributes", "generated_text", "text"]


def process_file_with_quality_classifier_streaming(
    input_filename: str,
    output_filename: str,
    quality_classifier: BaseClassifier,
    batch_size: int = 512,
    resume: bool = True,
):
    """Process a file with streaming I/O and resumption capability"""
    print(f"[*] Processing {input_filename} to {output_filename}")

    # Check if we should resume from existing progress
    rows_to_skip = 0
    if resume:
        rows_to_skip = count_existing_rows(output_filename)
        if rows_to_skip > 0:
            print(f"[*] Resuming from row {rows_to_skip}")
        else:
            print(f"[*] No existing rows found in {output_filename}")

    # Get column names
    input_column_names = get_input_dataset_column_names(input_filename)
    output_column_names = get_output_dataset_column_names(input_filename)

    # Create streaming iterator
    row_iterator = read_dataset_streaming(input_filename, input_column_names)

    # Skip already processed rows
    for _ in range(rows_to_skip):
        try:
            next(row_iterator)
        except StopIteration:
            print(f"[*] File already fully processed: {input_filename}")
            return

    # Initialize for batch processing
    append_mode = rows_to_skip > 0

    # For parquet, collect batches; for JSONL, stream asynchronously
    if output_filename.endswith(".parquet"):
        parquet_batches = []
        # async_writer = None
    # else:
    # async_writer = AsyncJSONLWriter(output_filename, append=append_mode)

    batch = []
    total_processed = rows_to_skip

    for row in row_iterator:
        batch.append(row)

        if len(batch) >= batch_size:
            # Process batch
            batch_dict = {}
            for key in input_column_names:
                batch_dict[key] = [row.get(key, "") for row in batch]

            # Apply classifier
            processed_batch = quality_classifier(batch_dict)

            # Prepare output rows
            output_rows = []
            for i in range(len(batch)):
                output_row = {}
                for col in output_column_names:
                    if col in processed_batch:
                        output_row[col] = processed_batch[col][i]
                    elif col in batch_dict:
                        output_row[col] = batch_dict[col][i]
                output_rows.append(output_row)

            # Write batch
            if output_filename.endswith(".parquet"):
                parquet_batches.extend(output_rows)
                # Write parquet in larger chunks to be efficient
                if len(parquet_batches) >= batch_size * 10:
                    _write_parquet_batch(parquet_batches, output_filename, append=append_mode or total_processed > 0)
                    parquet_batches = []
            else:
                # Enqueue JSONL write to overlap with next compute batch
                # async_writer.submit_rows(output_rows)
                write_dataset_streaming(output_rows, output_filename, append=append_mode or total_processed > 0)

            total_processed += len(batch)
            print(f"[*] Processed {total_processed} rows from {input_filename}")
            batch = []

    # Process remaining rows in final batch
    if batch:
        batch_dict = {}
        for key in input_column_names:
            batch_dict[key] = [row.get(key, "") for row in batch]

        processed_batch = quality_classifier(batch_dict)

        output_rows = []
        for i in range(len(batch)):
            output_row = {}
            for col in output_column_names:
                if col in processed_batch:
                    output_row[col] = processed_batch[col][i]
                elif col in batch_dict:
                    output_row[col] = batch_dict[col][i]
            output_rows.append(output_row)

        # Write final batch
        if output_filename.endswith(".parquet"):
            parquet_batches.extend(output_rows)
        else:
            # Enqueue final JSONL rows
            # async_writer.submit_rows(output_rows)
            write_dataset_streaming(output_rows, output_filename, append=append_mode or total_processed > 0)

        total_processed += len(batch)

    # Flush remaining writes
    # if output_filename.endswith(".parquet"):
    #     if parquet_batches:
    #         _write_parquet_batch(parquet_batches, output_filename, append_mode or total_processed > rows_to_skip)
    # else:
    # Finalize JSONL: join background writer and upload to destination
    # if async_writer is not None:
    #     async_writer.close()

    print(f"[*] Completed processing {input_filename} - Total rows: {total_processed}")


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


@ray.remote(max_calls=1)
@remove_tpu_lockfile_on_exit
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
):
    quality_classifier = AutoClassifier.from_model_path(
        model_name_or_path, attribute_name, model_type, **classifier_kwargs
    )

    process_file_with_quality_classifier_streaming(
        input_filename, output_filename, quality_classifier, batch_size, resume
    )


@ray.remote(max_calls=1)
@remove_tpu_lockfile_on_exit
@cached_or_construct_output(success_suffix="SUCCESS")
def _process_dir(
    input_path: str,
    output_path: str,
    model_name_or_path: str,
    attribute_name: str,
    model_type: str | None,
    filetype: str,
    classifier_kwargs: dict,
    batch_size: int = 512,
    resume: bool = True,
):
    """Perform quality classification on a directory of files

    We assume that the input_path is a directory of files. Using _process_dir is more
    efficient than process_file_ray because it avoids the overhead of spawning a new
    Ray task for each file and instead processes all files in a single task.
    """

    files = fsspec_glob(os.path.join(input_path, f"*.{filetype}"))

    if len(files) == 0:
        logger.error(f"No files found in {input_path} with pattern {filetype}!!! This is likely an error.")
        return

    quality_classifier = AutoClassifier.from_model_path(
        model_name_or_path, attribute_name, model_type, **classifier_kwargs
    )

    for input_filename in files:
        output_filename = rebase_file_path(input_path, input_filename, output_path)
        process_file_with_quality_classifier_streaming(
            input_filename, output_filename, quality_classifier, batch_size, resume
        )


def get_process_filepath_func(subdirectories: list[str]):
    if len(subdirectories) > 0:
        return _process_dir
    else:
        return process_file_ray


def get_filepaths_and_process_filepath_func(inference_config: InferenceConfig):
    # NOTE(chris): Maximize parallelism by doing one task per file. If this is too high
    # then we can use _process_dir to process multiple files in a single task.
    process_filepath_func = process_file_ray
    filepaths = fsspec_glob(os.path.join(inference_config.input_path, f"**/*.{inference_config.filetype}"))

    return filepaths, process_filepath_func


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
        "resources": inference_config.runtime.resources,
    }

    pending_refs: dict = {}
    attempt_count: dict[str, int] = {}

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
