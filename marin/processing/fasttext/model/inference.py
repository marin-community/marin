"""
Usage:
ray job submit --working-dir . --no-wait -- \
python -m marin.processing.fasttext.model.inference --input_dir gs://marin-data/processed/fineweb/fw-v1.0/md/CC-MAIN-2023-50/  --output_dir gs://marin-data/attributes/fasttext-quality/fineweb/fw-v1.0/md/CC-MAIN-2024-10/
"""
import argparse
import datetime
import json
import io
import os

import fsspec
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory
from marin.processing.fasttext.model.classifier import BatchFasttextQualityClassifier, DummyQualityClassifier, FasttextQualityClassifier
from marin.processing.utils import download_huggingface_file_with_backoff, download_gcs_file_with_backoff

class StorageConfig:
    def __init__(self, gcs_bucket_name, gcs_blob_name, hf_repo_id, hf_filename):
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_blob_name = gcs_blob_name
        self.hf_repo_id = hf_repo_id
        self.hf_filename = hf_filename

def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def make_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(element) for element in obj)
    elif not is_json_serializable(obj):
        return str(obj)
    return obj


# TODO(chris): fix this code, perhaps having fasttext dependency on head node can make it easier to load model ref to workers
@ray.remote(memory= 16 * 1024 * 1024 * 1024, runtime_env={"pip": ["fasttext"]})
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_using_actor_pool(input_filename, output_filename, model_ref):
    ds = ray.data.read_json(input_filename, arrow_open_stream_args={"compression": "gzip"},)

    print(f"[*] Read in dataset {input_filename}")
    
    # FIXME(chris): Could we parallelize using map_batches?
    # There are sometimes some issues with initialization when concurrency increases
    results = ds.map_batches(
        BatchFasttextQualityClassifier,
        # concurrency=(1,16),
        concurrency=1,
        fn_constructor_args=(model_ref,),
        batch_size=1024,
    )

    print("[*] Finished quality classification")
    
    # with fsspec.open(input_filename, "rt", compression="gzip") as f_in:
    with fsspec.open(output_filename, "wt", compression="gzip") as f_out:
        for row in results.iter_rows():                
            json_row = json.dumps(make_serializable(row))
            f_out.write(json_row + "\n")

@ray.remote(memory= 16 * 1024 * 1024 * 1024, runtime_env={"pip": ["fasttext"]})
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file(input_filename, output_filename, model_ref):
    print(f"[*] Read in dataset {input_filename}")
    
    quality_classifier = FasttextQualityClassifier(model_ref)

    with fsspec.open(input_filename, "rt", compression="gzip") as f_in:
        with fsspec.open(output_filename, "wt", compression="gzip") as f_out:
            for line in f_in:
                data = json.loads(line)
                processed_data = quality_classifier(data)
                res = {
                    "id": processed_data["id"],
                    "source": processed_data["source"],
                    "attributes": processed_data["attributes"]
                }          
                json_row = json.dumps(res)
                f_out.write(json_row + "\n")


def main(input_dir: str, output_dir: str, model_path: str, storage_config: StorageConfig):
    ray.init()

    byte_buffer = io.BytesIO()
    directory, basename = os.path.dirname(model_path), os.path.basename(model_path)
    os.makedirs(directory, exist_ok=True)
    
    # NOTE(chris): 1 try theoretically should work but there are sometimes some SystemExceptions from either FastText loading or Huggingface about metadata
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
    # responses = map_files_in_directory(process_file_using_actor_pool.remote, input_dir, "**/*_md.jsonl.gz", output_dir, None, model_ref)
    # responses = map_files_in_directory(process_file_using_actor_pool.remote, input_dir, "**/0_*_md.jsonl.gz", output_dir, None, model_ref)
    responses = map_files_in_directory(process_file.remote, input_dir, "**/*_md.jsonl.gz", output_dir, None, model_ref)


    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to convert fineweb html to markdown.")
    parser.add_argument('--input_dir', type=str, help='Path to the unprocessed data directory', required=True)
    parser.add_argument('--output_dir', type=str, help='Path to store data updated with quality scores directory', required=True)
    parser.add_argument('--model-path', type=str, default="", help="Model path of fasttext classifier")
    parser.add_argument('--quantized', type=str)

    args = parser.parse_args()

    if args.model_path == "":
        if args.quantized:
            args.model_path = os.path.expanduser("~/dolma_fasttext_model/model_quantized.bin")
        else:
            args.model_path = os.path.expanduser("~/dolma_fasttext_model/model.bin")

    if args.quantized:
        storage_config = StorageConfig(
            gcs_bucket_name="marin-data",
            gcs_blob_name="scratch/chrisc/dolma_fasttext_model/model_quantized.bin",
            hf_repo_id="BabyChou/dolma-fasttext-model-quantized",
            hf_filename="model.bin"
        )
    else:
        storage_config = StorageConfig(
            gcs_bucket_name="marin-data",
            gcs_blob_name="scratch/chrisc/dolma_fasttext_model/model.bin",
            hf_repo_id="allenai/dolma-1_7-fasttext-quality-filter",
            hf_filename="model.bin"
        )

    main(input_dir=args.input_dir, output_dir=args.output_dir, model_path=args.model_path, storage_config=storage_config)