import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import fsspec
import ray
from ray.data import DataContext
from ray.data.datasource import FilenameProvider
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from experiments.evals.resource_configs import TPU_V6E_8_STRICT_PACK, ResourceConfig
from marin.generation.pipeline import vLLMTextGeneration
from marin.generation.ray_utils import get_ray_remote_args_scheduling_strategy_fn
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
    template: str | None = None
    template_path: str | None = None
    apply_chat_template: bool = True
    save_templated_prompt: bool = False
    max_doc_tokens: int = 7000

    # Ray data specific
    num_instances: tuple[int, int] = (1, 4)
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
    resource_config: ResourceConfig = field(default_factory=lambda: TPU_V6E_8_STRICT_PACK)
    generated_text_column_name: str = "text"

    # Checkpoint specific
    checkpoint_id_column: str | None = None


class OneToOneFilenameProvider(FilenameProvider):
    def __init__(self, files: list[str], input_path: str):
        self.files = files
        self.input_path = input_path

    def get_filename_for_block(self, block, task_index, block_index):
        input_filename = self.files[task_index]
        output_filename = os.path.basename(input_filename)
        return output_filename


class OverwriteOutputFiletypeFilenameProvider(FilenameProvider):
    def __init__(self, file_format: str):
        self.file_format = file_format
        self.dataset_id = str(uuid.uuid4())

    def get_filename_for_block(self, block, task_index, block_index):
        return f"{self.dataset_id}_{task_index:06}_{block_index:06}" f".{self.file_format}"


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
    if config.tensor_parallel_size == 1:
        return {"resources": {"TPU": 1}}
    else:
        return {
            "ray_remote_args_fn": get_ray_remote_args_scheduling_strategy_fn(
                config.resource_config.num_tpu, config.resource_config.strategy
            )
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


def fix_warc_truncated_schema(batch: dict[str, Any]) -> dict[str, Any]:
    """Fix WARC-Truncated field in metadata to maintain schema consistency.

    NOTE(chris):
    Fix WARC-Truncated schema issue. In DCLM, some input rows will have the column "WARC-Truncated" with a string
    value there while others will not. When the vLLMTextGeneration concatenates blocks together after its outputs,
    this will lead to an error where it tries to concatenate blocks with inconsistent schemas. That's why we call
    this function before the vLLM text generation function to ensure a consistent schema.
    For example, the input schema is:
    Column                                                           Type
    ------                                                           ----
    bff_contained_ngram_count_before_dedupe                          int64
    language_id_whole_page_fasttext                                  struct<en: double>
    metadata                                                         struct<Content-Length: string, Content-Type:
                                                                     string, WARC-Block-Digest: string,
                                                                     WARC-Concurrent-To: string, WARC-Date:
                                                                     timestamp[s], WARC-IP-Address: string,
                                                                     WARC-Identified-Payload-Type: string,
                                                                     WARC-Payload-Digest: string, WARC-Record-ID:
                                                                     string, WARC-Target-URI: string, WARC-Type:
                                                                     string, WARC-Warcinfo-ID: string,
                                                                     WARC-Truncated: string>
    previous_word_count                                              int64
    text                                                             string
    url                                                              string
    warcinfo                                                         string
    fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob  double

    However, since only some columns have the WARC-Truncated nested column in metadata, it leads the output schema
    to set it to null. Here is an example of the broken output schema:
    Column                                                           Type
    ------                                                           ----
    bff_contained_ngram_count_before_dedupe                          int64
    language_id_whole_page_fasttext                                  struct<en: double>
    metadata                                                         struct<Content-Length: string, Content-Type:
                                                                     string, WARC-Block-Digest: string,
                                                                     WARC-Concurrent-To: string, WARC-Date:
                                                                     timestamp[us], WARC-IP-Address: string,
                                                                     WARC-Identified-Payload-Type: string,
                                                                     WARC-Payload-Digest: string, WARC-Record-ID:
                                                                     string, WARC-Target-URI: string,
                                                                     WARC-Truncated: null, WARC-Type: string,
                                                                     WARC-Warcinfo-ID: string>
    previous_word_count                                              int64
    text                                                             string
    url                                                              string
    warcinfo                                                         string
    fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob  double
    generated_text                                                   string

    We fix this by checking if the row contains the "WARC-Truncated" key and if it doesn't then, we set the default
    value to an empty string to have a consistent schema.
    """
    if "metadata" in batch:
        metadata_list = batch["metadata"]
        for metadata in metadata_list:
            if metadata is not None:
                if "WARC-Truncated" in metadata:
                    if metadata["WARC-Truncated"] is None:
                        metadata["WARC-Truncated"] = ""
                else:
                    metadata["WARC-Truncated"] = ""

    return batch


@ray.remote
def _find_finished_ids_for_file(checkpoint_filepath: str, id_column: str | dict[str, str]):
    from marin.processing.classification.inference import read_dataset

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
    scheduling_strategy=NodeAffinitySchedulingStrategy(
        node_id=ray.get_runtime_context().get_node_id(),
        soft=False,
    )
)
def run_inference(config: TextGenerationInferenceConfig):
    set_ray_data_config(config)

    ray_data_read_kwargs = get_ray_data_read_kwargs(config)
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

    ds = ds.write_json(config.output_path, **get_ray_data_write_kwargs(config))
