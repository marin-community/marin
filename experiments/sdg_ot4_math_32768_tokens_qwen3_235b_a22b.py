import logging
import os
import subprocess
import sys
import time

import numpy as np
from dataclasses import dataclass
from functools import partial

from experiments.defaults import default_download
from marin.execution.executor import executor_main, this_output_path, ExecutorStep
from zephyr import Backend, Dataset, load_parquet

logger = logging.getLogger(__name__)

ot4_math_qwen_235b_annotated = default_download(
    name="raw/marin-community/open-thoughts-4-30k-math-qwen3-235b-a22b-annotated",
    hf_dataset_id="marin-community/open-thoughts-4-30k-math-qwen3-235b-a22b-annotated",
    revision="c3d5f3b",
)

MODEL_NAME = "Qwen/Qwen3-235B-A22B-fp8-tput"
GENERATED_TEXT_COLUMN = "qwen235b_generated_text"
MAX_TOKENS = 32768


def is_response_complete(text: str) -> bool:
    """A response is complete if it contains a closed thinking block and a boxed answer."""
    return "</think>" in text and "\\boxed{" in text


def serialize_value(value):
    """Serialize a single value for parquet compatibility."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def serialize_record(record: dict) -> dict:
    """Convert a record to be parquet-compatible by serializing complex nested structures."""
    return {key: serialize_value(value) for key, value in record.items()}


def regenerate_if_truncated(
    row: dict,
    model_name: str,
    column: str,
    temperature: float = 0.8,
    max_tokens: int = MAX_TOKENS,
    max_retries: int = 5,
    initial_backoff: float = 30.0,
):
    """Re-generate the response from scratch if the existing one appears truncated.

    Uses exponential backoff to handle transient API timeouts.
    """
    existing = row.get(column, "")
    if existing and is_response_complete(existing):
        return row

    # Install together on worker nodes (runs once per worker due to caching)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "together"])
    from marin.generation.llm_generation import TogetherAIProvider

    logger.info("Re-generating truncated response for row %s", row.get("ms_id", "?"))
    provider = TogetherAIProvider(model_name, generation_kwargs={
        "temperature": temperature,
        "max_tokens": max_tokens,
    })

    # Retry with exponential backoff for transient API errors
    last_exception = None
    for attempt in range(max_retries):
        try:
            [generated_text] = provider.generate([row["instruction_seed"]])
            return {**row, column: generated_text}
        except Exception as e:
            last_exception = e
            error_name = type(e).__name__
            # Check if it's a timeout or transient error worth retrying
            if "timeout" in str(e).lower() or "Timeout" in error_name:
                backoff = initial_backoff * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed for row %s with timeout, retrying in %.1fs: %s",
                    attempt + 1, max_retries, row.get("ms_id", "?"), backoff, e
                )
                time.sleep(backoff)
            else:
                # Non-timeout error, raise immediately
                raise

    # All retries exhausted
    logger.error("All %d retries exhausted for row %s", max_retries, row.get("ms_id", "?"))
    raise last_exception


@dataclass
class RegenerateConfig:
    input_path: str
    output_path: str


def run_regeneration(config: RegenerateConfig):
    pipeline = (
        Dataset.from_files(os.path.join(config.input_path, "**/*.parquet"))
        .flat_map(load_parquet)
        .map(lambda x: partial(
            regenerate_if_truncated,
            model_name=MODEL_NAME,
            column=GENERATED_TEXT_COLUMN,
        )(x))
        .map(serialize_record)
        .write_parquet(os.path.join(config.output_path, "data-{shard:05d}-of-{total:05d}.parquet"))
    )
    Backend.execute(pipeline, max_parallelism=100)


open_thoughts_4_math_qwen3_235b_a22b_fp8_tpu_annotated = ExecutorStep(
    name="documents/open-thoughts-4-30k-math-qwen3-235b-a22b-fp8-tput-annotated-32768",
    fn=run_regeneration,
    config=RegenerateConfig(
        input_path=ot4_math_qwen_235b_annotated,
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[open_thoughts_4_math_qwen3_235b_a22b_fp8_tpu_annotated]
    )
