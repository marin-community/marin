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

import datasets
import fsspec
import ray
from ray.runtime_env import RuntimeEnv

from marin.core.runtime import cached_or_construct_output, map_files_in_directory
from marin.processing.quality.config.inference_config import InferenceConfig, StorageConfig
from marin.processing.quality.classifier import (
    BatchFasttextQualityClassifier,
    DummyQualityClassifier,
    FasttextQualityClassifier,
    BERTQualityClassifier,
)
from marin.processing.quality.utils import (
    download_huggingface_file_with_backoff,
    download_gcs_file_with_backoff,
    is_json_serializable,
    make_serializable,
)


@ray.remote
def process_file_using_actor_pool(input_filename, output_filename, model_ref):
    print(f"[*] Reading in dataset {input_filename}")

    ds = ray.data.read_json(
        input_filename,
        arrow_open_stream_args={"compression": "gzip"},
    )

    # print(f"[*] Finished reading in dataset {input_filename}")

    # FIXME(chris): Could we parallelize using map_batches?
    # There are sometimes some issues with initialization when concurrency increases
    ds = ds.map_batches(
        BatchFasttextQualityClassifier,
        # concurrency=(1,16),
        concurrency=(1, 16),
        fn_constructor_args=(model_ref,),
        batch_size=1024,
    )

    # only one file
    ds = ds.repartition(1)

    # TODO(CHRIS): override filename, batch writes everything??
    ds.write_json(output_filename, arrow_open_stream_args={"compression": "gzip"})


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
def process_file(input_filename, output_filename, model_ref):
    try:
        import jax

        print(f"TPU DEVICE COUNT: {jax.device_count('tpu')}")
    except Exception as e:
        print(e)
        print_tpu_driver_info()

    print(f"[*] Read in dataset {input_filename}")

    quality_classifier = BERTQualityClassifier(model_ref)

    json_list = []
    with fsspec.open(input_filename, "rt", compression="gzip") as f_in:
        for line in f_in:
            json_list.append(json.loads(line))

    dataset = datasets.Dataset.from_list(json_list)

    dataset = dataset.select_columns(["text", "id", "source"])
    predicted_dataset = dataset.map(quality_classifier, batched=True, batch_size=512)

    with fsspec.open(output_filename, "wt", compression="gzip") as f_out:
        for row in predicted_dataset:
            res = {"id": row["id"], "source": row["source"], "attributes": row["attributes"]}
            json_row = json.dumps(res)
            f_out.write(json_row + "\n")


def place_model_in_memory(storage_config: StorageConfig):
    byte_buffer = io.BytesIO()
    directory, basename = os.path.dirname(model_path), os.path.basename(model_path)
    os.makedirs(directory, exist_ok=True)

    try:
        try:
            print("Download using huggingface start.")
            download_huggingface_file_with_backoff(storage_config.hf_repo_id, storage_config.hf_filename, directory)
            print("Downloaded from huggingface.")
        except Exception as e:
            print(e, flush=True)

            try:
                print("Download using GCS start.")
                download_gcs_file_with_backoff(storage_config.gcs_bucket_name, storage_config.gcs_blob_name, model_path)
                print("Downloaded from GCS.")
            except Exception as e:
                print(e, flush=True)

        with open(model_path, "rb") as f:
            byte_buffer.write(f.read())
        byte_buffer.seek(0)
    except Exception as e:
        raise e

    model_ref = ray.put(byte_buffer)
    return model_ref


def main(inference_config: InferenceConfig):
    ray.init()

    if inference_config.storage is not None:
        model_ref = place_model_in_memory(inference_config.storage)
    else:
        model_ref = inference_config.model_name

    runtime_env = RuntimeEnv(
        memory=inference_config.runtime.memory_limit_gb * 1024 * 1024 * 1024,
        pip=inference_config.runtime.requirements_filepath,
        resources={"TPU": inference_config.runtime.tpu_resources_per_task},
    )

    if inference_config.use_ray_data:
        responses = process_file_using_actor_pool.options(runtime_env=runtime_env).remote(
            inference_config.input_dir, inference_config.output_dir, model_ref
        )
    else:
        responses = map_files_in_directory(
            process_file.options(runtime_env=runtime_env).remote,
            inference_config.input_dir,
            "**/*.jsonl.gz",
            inference_config.output_dir,
            None,
            model_ref,
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
