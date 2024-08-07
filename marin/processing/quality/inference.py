"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.quality.inference --config marin/processing/quality/embedding/fineweb.yaml
"""

import argparse
import datetime
import json
import io
import os
from typing import List

import datasets
import fsspec
import ray
from ray.data.datasource import FilenameProvider
from ray.runtime_env import RuntimeEnv

from marin.core.runtime import cached_or_construct_output, map_files_in_directory, map_directories_in_directory
from marin.processing.quality.config.inference_config import InferenceConfig, StorageConfig
from marin.processing.quality.classifier import (
    AutoClassifier,
    BaseQualityClassifier,
)
from marin.processing.quality.utils import (
    download_huggingface_file_with_backoff,
    download_gcs_file_with_backoff,
    is_json_serializable,
    make_serializable,
)
from marin.utils import (
    fsspec_glob,
    fsspec_mkdirs,
    rebase_file_path,
    fsspec_isdir,
    fsspec_get_curr_subdirectories,
    fsspec_get_atomic_directories,
)


class JsonFilenameProvider(FilenameProvider):

    def __init__(self, files: List[str], input_dir: str):
        self.files = files
        self.input_dir = input_dir

    def get_filename_for_block(self, block, task_index, block_index):
        input_filename = self.files[task_index]
        output_filename = os.path.basename(input_filename)
        return output_filename


@ray.remote
def process_file_using_actor_pool(input_dir: str, output_dir: str, model_name: str):
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True

    print(f"[*] Reading in dataset {input_dir}")
    print(f"[*] Output directory is {output_dir}")

    files = fsspec_glob(os.path.join(input_dir, "**/*.jsonl.gz"))

    ds = (
        ray.data.read_json(
            files,
            arrow_open_stream_args={"compression": "gzip"},
            override_num_blocks=len(files),
        )
        .map_batches(
            AutoClassifier,
            # concurrency=(1,16),
            concurrency=(1, len(files)),
            fn_constructor_args=(model_name),
            batch_size=None,
        )
        .write_json(
            output_dir,
            filename_provider=JsonFilenameProvider(files, input_dir),
            arrow_open_stream_args={"compression": "gzip"},
        )
    )


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_ray(input_filename: str, output_filename: str, model_name: str, attribute_name: str):
    print(f"[*] Read in dataset {input_filename}")

    quality_classifier = AutoClassifier.from_model_path(model_name, attribute_name)

    json_list = []
    with fsspec.open(input_filename, "rt", compression="gzip") as f_in:
        for line in f_in:
            json_list.append(json.loads(line))

    dataset = datasets.Dataset.from_list(json_list)

    dataset = dataset.select_columns(["text", "id", "source"])
    predicted_dataset = dataset.map(lambda batch: quality_classifier(batch), batched=True, batch_size=1024)

    with fsspec.open(output_filename, "wt", compression="gzip") as f_out:
        for row in predicted_dataset:
            res = {"id": row["id"], "source": row["source"], "attributes": row["attributes"]}
            json_row = json.dumps(res)
            f_out.write(json_row + "\n")


@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_with_quality_classifier(
    input_filename: str, output_filename: str, quality_classifier: BaseQualityClassifier
):
    json_list = []
    with fsspec.open(input_filename, "rt", compression="gzip") as f_in:
        for line in f_in:
            json_list.append(json.loads(line))

    dataset = datasets.Dataset.from_list(json_list)

    dataset = dataset.select_columns(["text", "id", "source"])
    predicted_dataset = dataset.map(lambda batch: quality_classifier(batch), batched=True, batch_size=512)

    with fsspec.open(output_filename, "wt", compression="gzip") as f_out:
        for row in predicted_dataset:
            res = {"id": row["id"], "source": row["source"], "attributes": row["attributes"]}
            json_row = json.dumps(res)
            f_out.write(json_row + "\n")


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def process_dir(input_dir: str, output_dir: str, model_name: str, attribute_name: str):
    files = fsspec_glob(os.path.join(input_dir, "**/*.jsonl.gz"))

    quality_classifier = AutoClassifier.from_model_path(model_name, attribute_name)

    for input_filename in files:
        output_filename = rebase_file_path(input_dir, input_filename, output_dir)
        process_file_with_quality_classifier(input_filename, output_filename, quality_classifier)


def get_process_filepath_func(subdirectories: List[str]):
    if len(subdirectories) > 0:
        return process_dir
    else:
        return process_file_ray


def get_filepaths_and_process_filepath_func(inference_config: InferenceConfig):
    filepaths = fsspec_get_atomic_directories(inference_config.input_dir)
    process_filepath_func = get_process_filepath_func(filepaths)

    # This is the case where the directory has no subdirectories. So, we are iterating through files and not directories
    if len(filepaths) == 0:
        filepaths = fsspec_glob(os.path.join(inference_config.input_dir, "**/*.jsonl.gz"))

    return filepaths, process_filepath_func


def main(inference_config: InferenceConfig):
    ray.init()

    filepaths, process_filepath_func = get_filepaths_and_process_filepath_func(inference_config)

    responses = []
    for input_filepath in filepaths:
        if len(responses) > inference_config.task.max_in_flight:
            ready_refs, responses = ray.wait(responses, num_returns=1)
            ray.get(ready_refs)

        output_filepath = rebase_file_path(inference_config.input_dir, input_filepath, inference_config.output_dir)
        fsspec_mkdirs(output_filepath)

        result_ref = process_filepath_func.options(
            memory=inference_config.runtime.memory_limit_gb * 1024 * 1024 * 1024,
            runtime_env=RuntimeEnv(
                pip=inference_config.runtime.requirements_filepath,
            ),
            resources=inference_config.runtime.tpu_resources_per_task,
        ).remote(input_filepath, output_filepath, inference_config.model_name, inference_config.attribute_name)

        responses.append(result_ref)

    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to convert fineweb html to markdown.")
    parser.add_argument("--config", type=str, help="Path to the config file", required=True)

    args = parser.parse_args()

    inference_config = InferenceConfig.from_yaml(args.config)

    main(inference_config=inference_config)
