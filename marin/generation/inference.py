import os
from dataclasses import dataclass
from typing import Any

import ray
from ray.data import DataContext
from ray.data.datasource import FilenameProvider
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from marin.generation.pipeline import vLLMTextGeneration
from marin.utils import fsspec_glob


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
    template: str

    # Ray data specific
    num_instances: tuple[int, int] = (1, 4)
    batch_size: int = 32
    tensor_parallel_size: int = 1
    preserve_order: bool = True
    one_to_one_input_output_mapping: bool = True

    # File specific
    filetype: str = "jsonl.gz"
    prompt_column: str = "text"


class OneToOneFilenameProvider(FilenameProvider):
    def __init__(self, files: list[str], input_path: str):
        self.files = files
        self.input_path = input_path

    def get_filename_for_block(self, block, task_index, block_index):
        input_filename = self.files[task_index]
        output_filename = os.path.basename(input_filename)
        return output_filename


def set_ray_data_config(config: TextGenerationInferenceConfig):
    ctx = DataContext.get_current()

    # This means that the input of the data will not be shuffled. This means that the output will be written in order.
    # This is important because we assume that our outputs will have the same name as our input per Dolma.
    ctx.execution_options.preserve_order = config.preserve_order

    # This allows us to run long-running tasks even if there is preemption.
    ctx.max_errored_blocks = -1

    # This is the amount of time to wait for the actors to be created.
    # We increase the default timeout since model loading
    # for large models can take awhile.
    ctx.wait_for_min_actors_s = 60 * 10 * config.tensor_parallel_size


def get_scheduling_strategy_fn(config: TextGenerationInferenceConfig):
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{"TPU": 1, "CPU": 1}] * config.tensor_parallel_size,
            strategy="PACK",  # STRICT_PACK means same node, PACK means different node possible
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(pg, placement_group_capture_child_tasks=True))

    return scheduling_strategy_fn


def ray_resources_kwarg(config: TextGenerationInferenceConfig):
    if config.tensor_parallel_size == 1:
        return {"resources": {"TPU": 1}}
    else:
        return {"ray_remote_args_fn": get_scheduling_strategy_fn(config)}


def get_ray_data_read_kwargs(config: TextGenerationInferenceConfig):
    ray_data_read_kwargs = {}
    if config.filetype == "jsonl.gz":
        ray_data_read_kwargs["arrow_open_stream_args"] = {"compression": "gzip"}

    if config.one_to_one_input_output_mapping:
        files = fsspec_glob(os.path.join(config.input_path, f"**/*.{config.filetype}"))
        ray_data_read_kwargs["override_num_blocks"] = len(files)

    return ray_data_read_kwargs


def get_ray_data_write_kwargs(config: TextGenerationInferenceConfig):
    ray_data_write_kwargs = {}
    if config.one_to_one_input_output_mapping:
        files = fsspec_glob(os.path.join(config.input_path, f"**/*.{config.filetype}"))
        ray_data_write_kwargs["filename_provider"] = OneToOneFilenameProvider(files, config.input_path)
    return ray_data_write_kwargs


@ray.remote
def run_inference(config: TextGenerationInferenceConfig):
    set_ray_data_config(config)

    ray_data_read_kwargs = get_ray_data_read_kwargs(config)
    ds = ray.data.read_json(config.input_path, **ray_data_read_kwargs)

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
            "template": config.template,
            "prompt_column": config.prompt_column,
        },
        **ray_resources_kwarg(config),
    )
    ds = ds.write_json(config.output_path, **get_ray_data_write_kwargs(config))
