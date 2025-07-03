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
    for row in dataset:
        yield row


def count_existing_rows(output_filename: str) -> int:
    """Count the number of rows already processed in the output file"""
    import os

    import datasets

    try:
        # Check if file exists
        if not os.path.exists(output_filename):
            return 0

        # Use datasets library to count rows efficiently
        datasets.disable_caching()
        datasets.logging.set_verbosity_warning()

        if output_filename.endswith((".jsonl.gz", ".jsonl.zst", ".jsonl")):
            # Load as JSON lines with streaming to count
            dataset = datasets.load_dataset("json", data_files=output_filename, streaming=True, split="train")
        elif output_filename.endswith(".parquet"):
            # Load parquet with streaming to count
            dataset = datasets.load_dataset("parquet", data_files=output_filename, streaming=True, split="train")
        else:
            return 0

        # Count rows by iterating through the streaming dataset
        count = 0
        for _ in dataset:
            count += 1

        return count
    except (FileNotFoundError, Exception):
        return 0


def write_dataset_streaming(rows_iterator, output_filename: str, append: bool = False):
    """Writes rows to a file in streaming fashion"""
    import json

    import fsspec

    mode = "a" if append else "w"

    if ".jsonl" in output_filename:
        with fsspec.open(output_filename, mode, compression="infer") as f:
            for row in rows_iterator:
                f.write(json.dumps(row) + "\n")
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
        return ["metadata", "attributes"]
    else:
        logger.warning("We are assuming the output dataset has the following columns: id, attributes")
        return ["id", "attributes", "generated_text"]


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

    # For parquet, collect batches; for JSONL, write immediately
    if output_filename.endswith(".parquet"):
        parquet_batches = []

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

            # Write batch immediately
            if output_filename.endswith(".parquet"):
                parquet_batches.extend(output_rows)
                # Write parquet in larger chunks to be efficient
                if len(parquet_batches) >= batch_size * 10:
                    _write_parquet_batch(parquet_batches, output_filename, append=append_mode or total_processed > 0)
                    parquet_batches = []
            else:
                # Use existing streaming write function for JSONL files
                write_dataset_streaming(iter(output_rows), output_filename, append=append_mode or total_processed > 0)

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
            # Use existing streaming write function for JSONL files
            write_dataset_streaming(iter(output_rows), output_filename, append=append_mode or total_processed > 0)

        total_processed += len(batch)

    # Write any remaining parquet data
    if output_filename.endswith(".parquet") and parquet_batches:
        _write_parquet_batch(parquet_batches, output_filename, append_mode or total_processed > rows_to_skip)

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


@ray.remote(num_cpus=3)
@remove_tpu_lockfile_on_exit
def run_inference(inference_config: InferenceConfig):
    logger.info(f"Running inference for {inference_config.input_path} to {inference_config.output_path}")
    filepaths, process_filepath_func = get_filepaths_and_process_filepath_func(inference_config)

    input_path = inference_config.input_path
    output_path = inference_config.output_path
    responses = []
    for input_filepath in filepaths:
        if len(responses) > inference_config.task.max_in_flight:
            ready_refs, responses = ray.wait(responses, num_returns=1)
            ray.get(ready_refs)

        output_filepath = rebase_file_path(input_path, input_filepath, output_path)
        fsspec_mkdirs(os.path.dirname(output_filepath))

        result_ref = process_filepath_func.options(
            memory=inference_config.runtime.memory_limit_gb * 1024 * 1024 * 1024,
            resources=inference_config.runtime.resources,
        ).remote(
            input_filepath,
            output_filepath,
            inference_config.model_name,
            inference_config.attribute_name,
            inference_config.model_type,
            inference_config.filetype,
            inference_config.classifier_kwargs,
            inference_config.batch_size,
            inference_config.resume,
        )

        responses.append(result_ref)

    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")
        raise e


@draccus.wrap()
def main(inference_config: InferenceConfig):
    ray.get(run_inference.remote(inference_config))


if __name__ == "__main__":
    main()
