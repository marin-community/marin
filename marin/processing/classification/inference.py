"""
Usage:

ray job submit --working-dir . --no-wait -- \
python -m marin.processing.classification.inference \
    --config_path marin/processing/classification/config/dclm_fasttext.yaml
"""

import json
import os

import draccus
import fsspec
import ray

# TODO(Chris): Can we remove this import, it needs pyarrow and pandas
from ray.data.datasource import FilenameProvider
from ray.runtime_env import RuntimeEnv

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


class JsonFilenameProvider(FilenameProvider):

    def __init__(self, files: list[str], input_path: str):
        self.files = files
        self.input_path = input_path

    def get_filename_for_block(self, block, task_index, block_index):
        input_filename = self.files[task_index]
        output_filename = os.path.basename(input_filename)
        return output_filename


@ray.remote
def process_file_using_actor_pool(input_path: str, output_path: str, model_name_or_path: str):
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.preserve_order = True

    print(f"[*] Reading in dataset {input_path}")
    print(f"[*] Output directory is {output_path}")

    files = fsspec_glob(os.path.join(input_path, "**/*.jsonl.gz"))

    ray.data.read_json(
        files,
        arrow_open_stream_args={"compression": "gzip"},
        override_num_blocks=len(files),
    ).map_batches(
        AutoClassifier,
        # concurrency=(1,16),
        concurrency=(1, len(files)),
        fn_constructor_args=(model_name_or_path),
        batch_size=None,
    ).write_json(
        output_path,
        filename_provider=JsonFilenameProvider(files, input_path),
        arrow_open_stream_args={"compression": "gzip"},
    )


@ray.remote
@cached_or_construct_output(success_suffix="SUCCESS")
def process_file_ray(
    input_filename: str, output_filename: str, model_name_or_path: str, attribute_name: str, model_type: str | None
):
    import datasets

    print(f"[*] Read in dataset {input_filename}")

    quality_classifier = AutoClassifier.from_model_path(model_name_or_path, attribute_name, model_type=model_type)

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
def process_file_with_quality_classifier(input_filename: str, output_filename: str, quality_classifier: BaseClassifier):
    import datasets

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
def process_dir(input_path: str, output_path: str, model_name_or_path: str, attribute_name: str, model_type: str | None):
    files = fsspec_glob(os.path.join(input_path, "**/*.jsonl.gz"))

    quality_classifier = AutoClassifier.from_model_path(model_name_or_path, attribute_name, model_type=model_type)

    for input_filename in files:
        output_filename = rebase_file_path(input_path, input_filename, output_path)
        process_file_with_quality_classifier(input_filename, output_filename, quality_classifier)


def get_process_filepath_func(subdirectories: list[str]):
    if len(subdirectories) > 0:
        return process_dir
    else:
        return process_file_ray


def get_filepaths_and_process_filepath_func(inference_config: InferenceConfig):
    filepaths = fsspec_get_atomic_directories(inference_config.input_path)
    process_filepath_func = get_process_filepath_func(filepaths)

    # This is the case where the directory has no subdirectories. So, we are iterating through files and not directories
    if len(filepaths) == 0:
        filepaths = fsspec_glob(os.path.join(inference_config.input_path, "**/*.jsonl.gz"))

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
            runtime_env=RuntimeEnv(
                pip=inference_config.runtime.requirements_filepath,
            ),
            resources=inference_config.runtime.resources,
        ).remote(
            input_filepath,
            output_filepath,
            inference_config.model_name,
            inference_config.attribute_name,
            inference_config.model_type,
        )

        responses.append(result_ref)

    try:
        ray.get(responses)
    except Exception as e:
        print(f"Error processing: {e}")


@draccus.wrap()
def main(inference_config: InferenceConfig):
    ray.get(run_inference.remote(inference_config))


if __name__ == "__main__":
    main()
