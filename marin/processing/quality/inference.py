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
from ray.runtime_env import RuntimeEnv

from marin.core.runtime import cached_or_construct_output, map_files_in_directory, map_directories_in_directory
from marin.processing.quality.config.inference_config import InferenceConfig, StorageConfig
from marin.processing.quality.classifier import (
    AutoClassifier,
    BatchFasttextQualityClassifier,
)
from marin.processing.quality.utils import (
    download_huggingface_file_with_backoff,
    download_gcs_file_with_backoff,
    is_json_serializable,
    make_serializable,
)
from marin.utils import fsspec_glob, fsspec_mkdirs, rebase_file_path, fsspec_isdir, fsspec_get_curr_subdirectories, fsspec_get_atomic_directories

from ray.data.datasource import FilenameProvider

class JsonFilenameProvider(FilenameProvider):

    def __init__(self, files: List[str], input_dir: str):
        self.files = files
        self.input_dir = input_dir

    def get_filename_for_block(self, block, task_index, block_index):
        input_filename = self.files[task_index]
        output_filename = os.path.basename(input_filename)
        return output_filename

@ray.remote
def process_file_using_actor_pool(input_dir: str, output_dir: str, pattern: str, model_ref: ray.ObjectRef, model_local_filepath: str):
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True

    print(f"[*] Reading in dataset {input_dir}")
    print(f"[*] Output directory is {output_dir}")

    files = fsspec_glob(os.path.join(input_dir, pattern))

    ds = ray.data.read_json(
        files,
        arrow_open_stream_args={"compression": "gzip"},
        override_num_blocks=len(files),
    ).map_batches(
        AutoClassifier,
        # concurrency=(1,16),
        concurrency=(1, len(files)),
        fn_constructor_args=(model_ref, model_local_filepath),
        batch_size=None,
    ).write_json(output_dir, filename_provider=JsonFilenameProvider(files, input_dir), arrow_open_stream_args={"compression": "gzip"})


def print_tpu_driver_info():
    import subprocess

    # Command to run
    # command = ["sudo", "rm", "/tmp/libtpu_lockfile"]  # Example: list files in current directory
    command = ["cat", "/tmp/tpu_logs/tpu_driver.INFO"]

    # Run the command
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Wait for the command to complete and get the output
    stdout, stderr = process.communicate()

    # Check the return code
    return_code = process.returncode

    # Print the results
    print(f"Return Code: {return_code}")
    print(f"Standard Output:\n{stdout}")
    print(f"Standard Error:\n{stderr}")

@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_ray(input_filename: str, output_filename: str, model_ref: ray.ObjectRef, model_path: str):
    # TODO(chris): remove this code when we are sure that TPU is working
    if "fineweb" in model_path:
        try:
            import jax

            print(f"TPU DEVICE COUNT: {jax.device_count('tpu')}")
        except Exception as e:
            print(e)
            print_tpu_driver_info()

    print(f"[*] Read in dataset {input_filename}")

    quality_classifier = AutoClassifier.from_model_path(model_ref, model_path)

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
def process_file_with_quality_classifier(input_filename: str, output_filename: str, quality_classifier: BaseQualityClassifier):
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
def process_dir(input_dir: str, output_dir: str, pattern: str, model_name: str):
    if "fineweb" in model_name:
        try:
            import jax
            import jaxlib

            print(f"Jaxlib version: {jaxlib.__version__}")
            print(f"TPU ID: {ray.get_runtime_context().get_accelerator_ids()['TPU']}")
            print(f"TPU DEVICE COUNT: {jax.device_count('tpu')}")
        except Exception as e:
            # print(e)
            # print_tpu_driver_info()
            pass

    files = fsspec_glob(os.path.join(input_dir, pattern))

    quality_classifier = AutoClassifier.from_model_path(model_name)

    for input_filename in files:
        output_filename = rebase_file_path(input_dir, input_filename, output_dir)
        process_file_with_quality_classifier(input_filename, output_filename, quality_classifier)


def place_model_in_memory(storage_config: StorageConfig):
    byte_buffer = io.BytesIO()
    directory, basename = os.path.dirname(storage_config.local_filepath), os.path.basename(storage_config.local_filepath)
    os.makedirs(directory, exist_ok=True)

    try:
        try:
            print("Download using huggingface start.")
            download_huggingface_file_with_backoff(
                storage_config.hf_repo_id, storage_config.hf_filename, directory, storage_config.local_filepath
            )
            print("Downloaded from huggingface.")
        except Exception as e:
            print(e, flush=True)

            try:
                print("Download using GCS start.")
                download_gcs_file_with_backoff(
                    storage_config.gcs_bucket_name, storage_config.gcs_blob_name, storage_config.local_filepath
                )
                print("Downloaded from GCS.")
            except Exception as e:
                print(e, flush=True)

        with open(storage_config.local_filepath, "rb") as f:
            byte_buffer.write(f.read())
        byte_buffer.seek(0)
    except Exception as e:
        raise e

    model_ref = ray.put(byte_buffer)
    return model_ref


def main(inference_config: InferenceConfig):
    ray.init()

    # TODO(Chris): Cleanup this and put into the runtime config
    if inference_config.runtime.tpu_resources_per_task > 0:
        resources = {"TPU": inference_config.runtime.tpu_resources_per_task}
    else:
        resources = {}

    # model_local_filepath = inference_config.storage.local_filepath if inference_config.storage is not None else model_ref
    if inference_config.use_ray_data:
        # subdirectories = fsspec_get_curr_subdirectories(inference_config.input_dir)
        subdirectories = fsspec_get_atomic_directories(inference_config.input_dir)
        responses = []
        for input_subdir in subdirectories:
            if len(responses) > inference_config.task.max_in_flight:
                ready_refs, responses = ray.wait(responses, num_returns=1)
                ray.get(ready_refs)

            output_subdir = rebase_file_path(inference_config.input_dir, input_subdir, inference_config.output_dir)
            fsspec_mkdirs(output_subdir)

            # TODO(chris): Change to actor pool when done testing with process_dir
            result_ref = process_dir.options(
                memory=inference_config.runtime.memory_limit_gb * 1024 * 1024 * 1024,
                runtime_env=RuntimeEnv(
                    pip=inference_config.runtime.requirements_filepath,
                ),
                resources=resources,
            ).remote(input_subdir, output_subdir, "**/*.jsonl.gz", inference_config.model_name)
            
            responses.append(result_ref)
    else:
        responses = map_files_in_directory(
            process_file_ray.options(
                memory=inference_config.runtime.memory_limit_gb * 1024 * 1024 * 1024,
                runtime_env=RuntimeEnv(
                    pip=inference_config.runtime.requirements_filepath,
                ),
                resources=resources,
            ).remote,
            inference_config.input_dir,
            "**/*.jsonl.gz",
            inference_config.output_dir,
            inference_config.task,
            inference_config.model_name,
        )

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
