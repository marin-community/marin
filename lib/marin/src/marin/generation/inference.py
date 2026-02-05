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

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import fsspec
import pyarrow as pa
import ray
from fray.cluster import ResourceConfig
from fray.cluster.base import TpuConfig
from marin.generation.chunk_utils import ChunkStrategy

logger = logging.getLogger(__name__)
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

    # Disable high memory usage warnings (vLLM uses lots of memory for model weights)
    ctx.issue_detectors_config.high_memory_detector_config.detection_time_interval_s = -1


def ray_resources_kwarg(config: TextGenerationInferenceConfig):
    # Clear JAX_PLATFORMS so TPU devices are detected correctly
    runtime_env = {"env_vars": {"JAX_PLATFORMS": "tpu"}}

    # Request TPU resources - use the tensor_parallel_size from config
    # which should match the number of TPU chips needed
    resources: dict[str, float] = {"TPU": config.tensor_parallel_size}

    # Also request the specific TPU type resource to ensure scheduling on the correct node type.
    # The cluster config defines resources like "TPU-v5p-8-head: 1" for each TPU type.
    if config.resource_config and config.resource_config.device:
        tpu_variant = config.resource_config.device.variant
        tpu_head_resource = f"TPU-{tpu_variant}-head"
        resources[tpu_head_resource] = 1

    return {
        "resources": resources,
        "max_restarts": -1,
        "runtime_env": runtime_env,
    }


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


def _get_column_types_from_schema(schema) -> dict[str, pa.DataType]:
    """Extract column names and their PyArrow types from a schema.

    Args:
        schema: Either a ray.data.Schema or a PyArrow schema.

    Returns:
        Dict mapping column names to their PyArrow types.
    """
    if hasattr(schema, "base_schema"):
        # Handle Ray Data Schema wrapper - extract the underlying PyArrow schema
        arrow_schema = schema.base_schema
    else:
        arrow_schema = schema
    column_types = {}
    for schema_field in arrow_schema:
        column_types[schema_field.name] = schema_field.type
    return column_types


@ray.remote(num_cpus=0, max_retries=0)
def _find_finished_ids_for_file(checkpoint_filepath: str, id_column: str | dict[str, str]):
    import pandas as pd
    import sys

    try:
        if isinstance(id_column, dict):
            dataset_column = next(iter(id_column.keys()))
            metadata_key_column = next(iter(id_column.values()))
        else:
            dataset_column = id_column
            metadata_key_column = None

        df = pd.read_parquet(checkpoint_filepath, columns=[dataset_column])
        finished_ids = set()

        if metadata_key_column is None:
            finished_ids = set(df[dataset_column])
        else:
            for metadata in df[dataset_column]:
                if metadata is not None and metadata_key_column in metadata:
                    finished_ids.add(metadata[metadata_key_column])

        return finished_ids
    except Exception as e:
        print(f"Error processing {checkpoint_filepath}: {e}", file=sys.stderr, flush=True)
        raise


def find_all_finished_ids(checkpoint_path: str, filetype: str, id_column: str | dict[str, str]):
    import sys

    files = fsspec_glob(os.path.join(checkpoint_path, f"**/*.{filetype}"))
    print(f"[CHECKPOINT] Found {len(files)} checkpoint files to scan", flush=True)
    logger.info(f"Found {len(files)} checkpoint files to scan for finished IDs")

    if len(files) == 0:
        return set()

    finished_ids = set()

    # Process files in smaller batches to avoid overwhelming the scheduler
    batch_size = 20
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(files) + batch_size - 1) // batch_size
        print(f"[CHECKPOINT] Processing batch {batch_num}/{total_batches}", flush=True)

        refs = [_find_finished_ids_for_file.remote(file, id_column) for file in batch_files]

        # Use ray.wait with timeout to track progress
        try:
            ready_refs, remaining_refs = ray.wait(refs, num_returns=len(refs), timeout=120)
            if remaining_refs:
                print(f"[CHECKPOINT] WARNING: {len(remaining_refs)} tasks not completed after 2 min timeout", flush=True)
                # Try to get what we can
                for ref in remaining_refs:
                    ray.cancel(ref, force=True)

            for ref in ready_refs:
                try:
                    result = ray.get(ref)
                    finished_ids.update(result)
                except Exception as e:
                    print(f"[CHECKPOINT] Error getting result: {e}", flush=True)

            print(f"[CHECKPOINT] Batch {batch_num} complete. Total IDs: {len(finished_ids)}", flush=True)
        except Exception as e:
            print(f"[CHECKPOINT] Error in batch {batch_num}: {e}", flush=True)
            raise

    print(f"[CHECKPOINT] Scan complete. Found {len(finished_ids)} finished IDs", flush=True)
    logger.info(f"Checkpoint scan complete. Found {len(finished_ids)} finished IDs")
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

    # Capture column types from input schema to ensure consistent types when writing.
    # This prevents schema unification errors when batches have all-null columns
    # (which PyArrow may infer as float instead of their correct type).
    input_schema = ds.schema()
    column_types = _get_column_types_from_schema(input_schema) if input_schema else {}

    assert (config.template_path or config.template) and not (config.template_path and config.template), (
        "Must provide either a template or a template path, but not both"
    )

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

        # Check how many checkpoint files exist before scanning
        checkpoint_files = fsspec_glob(os.path.join(config.output_path, f"**/*.{output_filetype}"))
        logger.info(f"Found {len(checkpoint_files)} existing checkpoint files")

        if len(checkpoint_files) > 0:
            finished_ids = find_all_finished_ids(config.output_path, output_filetype, config.checkpoint_id_column)
            logger.info(f"Found {len(finished_ids)} already-processed IDs to filter out")
            if len(finished_ids) > 0:
                if isinstance(config.checkpoint_id_column, dict):
                    dataset_column = next(iter(config.checkpoint_id_column.keys()))
                    metadata_key_column = next(iter(config.checkpoint_id_column.values()))
                    ds = ds.filter(lambda x: x[dataset_column][metadata_key_column] not in finished_ids)
                else:
                    ds = ds.filter(lambda x: x[config.checkpoint_id_column] not in finished_ids)
                # Note: Removed ds.count() call here - it was blocking and causing jobs to hang
                # The filter will still work lazily during inference
        else:
            logger.info("No checkpoint files found, skipping checkpoint filtering")

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
            "column_types": column_types,
        },
        **ray_resources_kwarg(config),
    )

    output_filetype = config.output_filetype_override or config.filetype
    if output_filetype == "parquet":
        ds = ds.write_parquet(config.output_path, **get_ray_data_write_kwargs(config))
    else:
        ds = ds.write_json(config.output_path, **get_ray_data_write_kwargs(config))
