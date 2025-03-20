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
    elif input_filename.endswith(".jsonl.zst"):
        df = pd.read_json(input_filename, compression="zstd", lines=True)
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
    elif output_filename.endswith(".jsonl.zst"):
        df_pandas = dataset.to_pandas()
        df_pandas.to_json(output_filename, orient="records", compression="zstd", lines=True)
        # dataset.to_json(output_filename, to_json_kwargs={"compression": "zstd", "lines": True})
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
        return ["id", "attributes"]


@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_with_quality_classifier(input_filename: str, output_filename: str, quality_classifier: BaseClassifier):
    print(f"[*] Processing {input_filename} to {output_filename}")
    dataset = read_dataset(input_filename)

    # TODO(chris): Add support for more types of columns.
    input_column_names = get_input_dataset_column_names(input_filename)
    output_column_names = get_output_dataset_column_names(input_filename)
    dataset = dataset.select_columns(input_column_names)
    dataset = dataset.map(lambda batch: quality_classifier(batch), batched=True, batch_size=512)
    dataset = dataset.select_columns(output_column_names)

    write_dataset(dataset, output_filename)


@ray.remote
def process_file_ray(
    input_filename: str,
    output_filename: str,
    model_name_or_path: str,
    attribute_name: str,
    model_type: str | None,
    filetype: str,
    classifier_kwargs: dict,
):
    quality_classifier = AutoClassifier.from_model_path(
        model_name_or_path, attribute_name, model_type, **classifier_kwargs
    )

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
    classifier_kwargs: dict,
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
        process_file_with_quality_classifier(input_filename, output_filename, quality_classifier)


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
            inference_config.classifier_kwargs,
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
