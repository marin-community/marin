import os
import json
import numpy as np
from functools import partial
from dataclasses import dataclass
from zephyr import Dataset, create_backend, load_parquet, load_file
from experiments.defaults import default_download
from marin.execution.executor import executor_main, this_output_path, ExecutorStep
from marin.generation.llm_generation import TogetherAIProvider

ot4_math_qwen_32b_annotated = default_download(
    name="raw/marin-community/open-thoughts-4-math-qwen3-32b-annotated",
    hf_dataset_id="marin-community/open-thoughts-4-math-qwen3-32b-annotated",
    revision="1bd0352",
)

TOTAL_MATH_PROMPTS_DESIRED = 30_000
TOTAL_MATH_FILES = 83

TOTAL_MATH_PROMPTS_PER_FILE = TOTAL_MATH_PROMPTS_DESIRED // TOTAL_MATH_FILES

def serialize_value(value):
    """Serialize a single value for parquet compatibility."""
    if isinstance(value, np.ndarray):
        # Convert numpy arrays to lists
        return value.tolist()
    else:
        # Primitive types (str, int, float, bool, None)
        return value

def serialize_record(record: dict) -> dict:
    """Convert a record to be parquet-compatible by serializing complex nested structures."""
    return {key: serialize_value(value) for key, value in record.items()}

@dataclass
class SubsamplingConfig:
    input_path: str
    output_path: str
    rows_per_file_desired: int

def llm_output(row: dict, model_name: str, generated_text_column_name, temperature: float=0.8, max_tokens: int = 16000):
    provider = TogetherAIProvider(model_name, generation_kwargs={
        "temperature": temperature,
        "max_tokens": max_tokens,
    })
    [generated_text] = provider.generate([row["instruction_seed"]])
    updated_row = {**row, generated_text_column_name: generated_text}
    return updated_row

def run_subsampling(config: SubsamplingConfig):
    # Read, transform, write
    backend = create_backend("ray", max_parallelism=100)
    pipeline = (
        Dataset.from_files(os.path.join(config.input_path, "**/*.parquet"))
        .flat_map(load_parquet)
        # .take_per_shard(1)
        .take_per_shard(config.rows_per_file_desired)
        .map(lambda x: partial(llm_output, model_name="Qwen/Qwen3-235B-A22B-fp8-tput", generated_text_column_name="qwen235b_generated_text")(x))
        .map(lambda x: serialize_record(x))
        # For some reason the list of messages which are List[Dict[str, str]] gets detected as a numpy.ndarray
        # so we need to convert these to list
        .write_parquet(os.path.join(config.output_path, "data-{shard:05d}-of-{total:05d}.parquet"))
    )
    list(backend.execute(pipeline))

open_thoughts_4_math_qwen3_235b_a22b_fp8_tpu_annotated = ExecutorStep(
    name="documents/open-thoughts-4-30k-math-qwen3-235b-a22b-fp8-tput-annotated",
    fn=run_subsampling,
    config=SubsamplingConfig(
        input_path=ot4_math_qwen_32b_annotated,
        output_path=this_output_path(),
        rows_per_file_desired=TOTAL_MATH_PROMPTS_PER_FILE
    ),
    pip_dependency_groups=["together"]
)

if __name__ == "__main__":
    executor_main(
        steps=[open_thoughts_4_math_qwen3_235b_a22b_fp8_tpu_annotated]
    )
