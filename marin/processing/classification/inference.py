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
    fsspec_get_atomic_directories,
    fsspec_glob,
    fsspec_mkdirs,
    rebase_file_path,
)

logger = logging.getLogger("ray")


def read_dataset(input_filename: str, columns: list[str] | None = None):
    """Read in a dataset and return as a Huggingface Dataset

    Args:
        input_filename: str
            The path to the input file. Currently supports .jsonl.gz and .parquet

    Returns:
        datasets.Dataset: A Huggingface Dataset in-memory without using the disk
    """
    import datasets

    datasets.disable_caching()
    datasets.logging.set_verbosity_warning()
    # We use pandas to read in the file so that we don't have to materialize
    # the entire dataset in disk since we have limited disk space.
    # Huggingface datasets loads the dataset into disk first and mmaps.
    if input_filename.endswith(".jsonl.gz"):
        df = pd.read_json(input_filename, compression="gzip", lines=True)
        dataset = datasets.Dataset.from_pandas(df)
        return dataset
    elif input_filename.endswith(".parquet"):
        df = pd.read_parquet(input_filename, columns=columns)
        dataset = datasets.Dataset.from_pandas(df)
        return dataset
    else:
        raise ValueError(f"Unsupported filetype: {input_filename}")


def write_dataset(dataset, output_filename: str):
    """Writes a Huggingface Dataset to a file (remote or local)"""
    if output_filename.endswith(".jsonl.gz"):
        dataset.to_json(output_filename, compression="gzip")
    elif output_filename.endswith(".parquet"):
        dataset.to_parquet(output_filename)
    else:
        raise ValueError(f"Unsupported filetype: {output_filename}")


@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_with_quality_classifier(input_filename: str, output_filename: str, quality_classifier: BaseClassifier):
    print(f"[*] Processing {input_filename} to {output_filename}")
    dataset = read_dataset(input_filename)

    dataset = dataset.select_columns(["text", "id"])
    dataset = dataset.map(lambda batch: quality_classifier(batch), batched=True, batch_size=512)
    dataset = dataset.select_columns(["id", "attributes"])

    write_dataset(dataset, output_filename)


@ray.remote
def process_file_ray(
    input_filename: str,
    output_filename: str,
    model_name_or_path: str,
    attribute_name: str,
    model_type: str | None,
    filetype: str,
):
    quality_classifier = AutoClassifier.from_model_path(model_name_or_path, attribute_name, model_type=model_type)

    process_file_with_quality_classifier(input_filename, output_filename, quality_classifier)


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def _process_dir(
    input_path: str,
    output_path: str,
    model_name_or_path: str,
    attribute_name: str,
    model_type: str | None,
    filetype: str,
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

    quality_classifier = AutoClassifier.from_model_path(model_name_or_path, attribute_name, model_type=model_type)

    for input_filename in files:
        output_filename = rebase_file_path(input_path, input_filename, output_path)
        process_file_with_quality_classifier(input_filename, output_filename, quality_classifier)


def get_process_filepath_func(subdirectories: list[str]):
    if len(subdirectories) > 0:
        return _process_dir
    else:
        return process_file_ray


def get_filepaths_and_process_filepath_func(inference_config: InferenceConfig):
    filepaths = fsspec_get_atomic_directories(inference_config.input_path)
    process_filepath_func = get_process_filepath_func(filepaths)

    # This is the case where the directory has no subdirectories. So, we are iterating through files and not directories
    if len(filepaths) == 0:
        filepaths = fsspec_glob(os.path.join(inference_config.input_path, f"**/*.{inference_config.filetype}"))

    return filepaths, process_filepath_func


@ray.remote
def run_inference(inference_config: InferenceConfig):
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
        )

        responses.append(result_ref)

    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")
        raise


@draccus.wrap()
def main(inference_config: InferenceConfig):
    ray.get(run_inference.remote(inference_config))


if __name__ == "__main__":
    main()
