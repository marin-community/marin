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

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import fsspec
import ray
from fray.cluster import ResourceConfig
from fray.cluster.ray import get_scheduling_strategy
from marin.generation.chunk_utils import ChunkStrategy
from marin.generation.pipeline import vLLMTextGeneration
from marin.utils import fsspec_glob
from ray.data import DataContext
from ray.data.datasource import FilenameProvider


@dataclass
class TextGenerationInferenceConfig:
    # IO specific
    input_path: str
    output_path: str

    # Model specific
    model_name: str
    engine_kwargs: dict[str, Any]
    generation_kwargs: dict[str, Any]

    # Prompting specific
    template: str | None = None
    template_path: str | None = None
    apply_chat_template: bool = True
    save_templated_prompt: bool = False
    max_doc_tokens: int = 7000
    chunk_strategy: ChunkStrategy | None = None
    chunk_size: int | None = None

    # Ray data specific
    num_instances: tuple[int, int] = (1, 1)
    batch_size: int = 32
    tensor_parallel_size: int = 1
    preserve_order: bool = False
    one_to_one_input_output_mapping: bool = False

    # File specific
    filetype: str = "jsonl.gz"
    # If none, then we use the same filetype as the input if possible, if not then we use json.
    output_filetype_override: str | None = None
    prompt_column: str = "text"

    # Hardware specific
    resource_config: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v6e-8"))
    generated_text_column_name: str = "text"

    # Checkpoint specific
    # This checkpoint id column is the "key" in the file that will be used to uniquely identify an input example.
    # This is used for deduplicating work in the case when we are resuming from a checkpoint.
    checkpoint_id_column: str | None = None


class OneToOneFilenameProvider(FilenameProvider):
    def __init__(self, files: list[str], input_path: str):
        self.files = files
        self.input_path = input_path

    def get_filename_for_block(self, block, task_index, block_index, file_format=None):
        input_filename = self.files[task_index]
        output_filename = os.path.basename(input_filename)
        return output_filename


class OverwriteOutputFiletypeFilenameProvider(FilenameProvider):
    def __init__(self, file_format: str):
        self.file_format = file_format
        self.dataset_id = str(uuid.uuid4())

    def get_filename_for_block(self, block, task_index, block_index, file_format=None):
        return f"{self.dataset_id}_{task_index:06}_{block_index:06}.{self.file_format}"


def set_ray_data_config(config: TextGenerationInferenceConfig):
    ctx = DataContext.get_current()

    # This means that the input of the data will not be shuffled. This means that the output will be written in order.
    # This is important because we assume that our outputs will have the same name as our input per Dolma.
    ctx.execution_options.preserve_order = config.preserve_order

    # This allows us to run long-running tasks even if there is preemption.
    ctx.max_errored_blocks = -1

    # Helps with debugging
    ctx.log_internal_stack_trace_to_stdout = True

    # This is the amount of time to wait for the actors to be created.
    # We increase the default timeout since model loading
    # for large models can take awhile.
    ctx.wait_for_min_actors_s = 60 * 10 * config.tensor_parallel_size


def ray_resources_kwarg(config: TextGenerationInferenceConfig):
    # Clear JAX_PLATFORMS so TPU devices are detected correctly
    runtime_env = {"env_vars": {"JAX_PLATFORMS": ""}}

    if config.tensor_parallel_size == 1:
        return {"resources": {"TPU": 1}, "max_restarts": -1, "runtime_env": runtime_env}
    else:

        def scheduling_strategy_dict_fn():
            return dict(
                scheduling_strategy=get_scheduling_strategy(config.resource_config),
                runtime_env=runtime_env,
            )

        return {"ray_remote_args_fn": scheduling_strategy_dict_fn}


def get_ray_data_read_kwargs(config: TextGenerationInferenceConfig):
    ray_data_read_kwargs = {}
    if config.filetype == "jsonl.gz":
        ray_data_read_kwargs["arrow_open_stream_args"] = {"compression": "gzip"}
    elif config.filetype == "jsonl.zst":
        ray_data_read_kwargs["arrow_open_stream_args"] = {"compression": "zstd"}

    if config.one_to_one_input_output_mapping:
        files = fsspec_glob(os.path.join(config.input_path, f"**/*.{config.filetype}"))
        ray_data_read_kwargs["override_num_blocks"] = len(files)

    return ray_data_read_kwargs


def get_ray_data_write_kwargs(config: TextGenerationInferenceConfig):
    ray_data_write_kwargs = {}
    if config.one_to_one_input_output_mapping:
        files = fsspec_glob(os.path.join(config.input_path, f"**/*.{config.filetype}"))
        ray_data_write_kwargs["filename_provider"] = OneToOneFilenameProvider(files, config.input_path)
    elif config.output_filetype_override:
        ray_data_write_kwargs["filename_provider"] = OverwriteOutputFiletypeFilenameProvider(
            config.output_filetype_override
        )

    return ray_data_write_kwargs


@ray.remote
def _find_finished_ids_for_file(checkpoint_filepath: str, id_column: str | dict[str, str]):
    from marin.processing.classification.dataset_utils import read_dataset

    if isinstance(id_column, dict):
        dataset_column = next(iter(id_column.keys()))
        metadata_key_column = next(iter(id_column.values()))
    else:
        dataset_column = id_column
        metadata_key_column = None

    # TODO(chris): replace columns with user input
    df = read_dataset(checkpoint_filepath, columns=[dataset_column])
    finished_ids = set()

    if metadata_key_column is None:
        finished_ids = set(df[dataset_column])
    else:
        for metadata in df[dataset_column]:
            if metadata is not None and metadata_key_column in metadata:
                finished_ids.add(metadata[metadata_key_column])

    return finished_ids


def find_all_finished_ids(checkpoint_path: str, filetype: str, id_column: str | dict[str, str]):
    import concurrent.futures

    import tqdm

    files = fsspec_glob(os.path.join(checkpoint_path, f"**/*.{filetype}"))
    finished_ids = set()

    refs = [_find_finished_ids_for_file.remote(file, id_column) for file in files]
    futures = [ref.future() for ref in refs]
    for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Finding finished IDs"):
        finished_ids.update(future.result())

    return finished_ids


@ray.remote(
    # Run on the head node because the head node is not preemptible.
    # This makes sure that the inference itself is not preempted.
    # If it is preemptible, then we would have to
    # run this entire pipeline again.
    # scheduling_strategy=NodeAffinitySchedulingStrategy(
    #     node_id=ray.get_runtime_context().get_node_id(),
    #     soft=False,
    # )
    num_cpus=0,
    resources={"head_node": 0.001},
)
def run_inference(config: TextGenerationInferenceConfig):
    set_ray_data_config(config)

    ray_data_read_kwargs = get_ray_data_read_kwargs(config)
    if config.filetype == "parquet":
        ds = ray.data.read_parquet(config.input_path, **ray_data_read_kwargs)
    else:
        ds = ray.data.read_json(config.input_path, **ray_data_read_kwargs)

    assert (config.template_path or config.template) and not (
        config.template_path and config.template
    ), "Must provide either a template or a template path, but not both"

    if config.template_path:
        with fsspec.open(config.template_path, "r", compression="infer") as f:
            template = str(f.read())
    elif config.template:
        template = config.template

    if config.checkpoint_id_column:
        if config.output_filetype_override:
            output_filetype = config.output_filetype_override
        else:
            output_filetype = config.filetype
        finished_ids = find_all_finished_ids(config.output_path, output_filetype, config.checkpoint_id_column)
        if len(finished_ids) > 0:
            if isinstance(config.checkpoint_id_column, dict):
                dataset_column = next(iter(config.checkpoint_id_column.keys()))
                metadata_key_column = next(iter(config.checkpoint_id_column.values()))
                ds = ds.filter(lambda x: x[dataset_column][metadata_key_column] not in finished_ids)
            else:
                ds = ds.filter(lambda x: x[config.checkpoint_id_column] not in finished_ids)
            print("Dataset count after checkpoint filter:", ds.count())

    ds = ds.map_batches(  # Apply batch inference for all input data.
        vLLMTextGeneration,
        # Set the concurrency to the number of LLM instances.
        concurrency=config.num_instances,
        # Specify the batch size for inference.
        batch_size=config.batch_size,
        fn_constructor_kwargs={
            "model_name": config.model_name,
            "engine_kwargs": config.engine_kwargs,
            "generation_kwargs": config.generation_kwargs,
            "template": template,
            "prompt_column": config.prompt_column,
            "save_templated_prompt": config.save_templated_prompt,
            "apply_chat_template": config.apply_chat_template,
            "max_doc_tokens": config.max_doc_tokens,
            "generated_text_column_name": config.generated_text_column_name,
        },
        **ray_resources_kwarg(config),
    )

    output_filetype = config.output_filetype_override or config.filetype
    if output_filetype == "parquet":
        ds = ds.write_parquet(config.output_path, **get_ray_data_write_kwargs(config))
    else:
        ds = ds.write_json(config.output_path, **get_ray_data_write_kwargs(config))
