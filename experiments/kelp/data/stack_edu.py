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

"""Stack-Edu Python dataset integration for Kelp tree diffusion.

Integrates with the existing Marin Stack-Edu Python dataset pipeline,
processing the data for tree diffusion training.
"""

import logging
from dataclasses import dataclass
from typing import Iterator

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array

from marin.execution.executor import ExecutorStep, this_output_path

from experiments.kelp.tree.parser import extract_functions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StackEduProcessingConfig:
    """Configuration for Stack-Edu processing."""

    input_path: str
    """Path to Stack-Edu Python filtered data."""

    output_path: str
    """Output path for processed data."""

    min_docstring_len: int = 10
    """Minimum docstring length to include."""

    max_function_len: int = 2048
    """Maximum function length in characters."""

    max_examples: int | None = None
    """Maximum number of examples to extract (None for all)."""


def process_stack_edu_file(
    content: str,
    min_docstring_len: int = 10,
    max_function_len: int = 2048,
) -> list[dict]:
    """Process a single Stack-Edu file.

    Args:
        content: Python file content.
        min_docstring_len: Minimum docstring length.
        max_function_len: Maximum function length.

    Returns:
        List of extracted function examples.
    """
    functions = extract_functions(content)

    examples = []
    for func in functions:
        if not func["docstring"]:
            continue
        if len(func["docstring"]) < min_docstring_len:
            continue
        if len(func["full_code"]) > max_function_len:
            continue

        prompt = f'"""{func["docstring"]}"""\n{func["signature"]}'

        examples.append(
            {
                "prompt": prompt,
                "code": func["full_code"],
                "docstring": func["docstring"],
                "signature": func["signature"],
                "body": func["body"],
            }
        )

    return examples


def process_stack_edu(config: StackEduProcessingConfig) -> dict:
    """Process Stack-Edu dataset for tree diffusion.

    This function is called by the executor framework.

    Args:
        config: Processing configuration.

    Returns:
        Dictionary with processing results.
    """
    import json
    import gzip

    import fsspec

    fs = fsspec.filesystem(fsspec.utils.get_protocol(config.input_path))

    file_pattern = f"{config.input_path}/stack-edu-*.json.gz"
    files = fs.glob(file_pattern)

    logger.info(f"Found {len(files)} Stack-Edu files")

    all_examples = []
    total_files_processed = 0
    total_functions_extracted = 0

    for file_path in files:
        if config.max_examples and len(all_examples) >= config.max_examples:
            break

        try:
            with fs.open(file_path, "rb") as f:
                with gzip.open(f, "rt", encoding="utf-8") as gz:
                    for line in gz:
                        if config.max_examples and len(all_examples) >= config.max_examples:
                            break

                        record = json.loads(line)
                        content = record.get("text", "")

                        if not content:
                            continue

                        examples = process_stack_edu_file(
                            content,
                            min_docstring_len=config.min_docstring_len,
                            max_function_len=config.max_function_len,
                        )

                        all_examples.extend(examples)
                        total_functions_extracted += len(examples)

            total_files_processed += 1

            if total_files_processed % 100 == 0:
                logger.info(f"Processed {total_files_processed} files, {len(all_examples)} examples")

        except Exception as e:
            logger.warning(f"Error processing {file_path}: {e}")
            continue

    out_fs = fsspec.filesystem(fsspec.utils.get_protocol(config.output_path))
    out_fs.makedirs(config.output_path, exist_ok=True)

    output_file = f"{config.output_path}/stack_edu_functions.jsonl.gz"
    with out_fs.open(output_file, "wb") as f:
        with gzip.open(f, "wt", encoding="utf-8") as gz:
            for example in all_examples:
                gz.write(json.dumps(example) + "\n")

    logger.info(f"Wrote {len(all_examples)} examples to {output_file}")

    return {
        "status": "completed",
        "files_processed": total_files_processed,
        "examples_extracted": len(all_examples),
        "output_path": output_file,
    }


def stack_edu_processing_step(
    max_examples: int | None = None,
) -> ExecutorStep:
    """Create ExecutorStep for Stack-Edu processing.

    Args:
        max_examples: Maximum examples to extract.

    Returns:
        ExecutorStep for processing.
    """
    from experiments.midtraining_datasets import stackv2_edu_filtered_python

    return ExecutorStep(
        name="kelp/data/stack_edu_functions",
        fn=process_stack_edu,
        config=StackEduProcessingConfig(
            input_path=stackv2_edu_filtered_python,  # type: ignore
            output_path=this_output_path(),
            max_examples=max_examples,
        ),
    )


def load_processed_stack_edu(path: str) -> list[dict]:
    """Load processed Stack-Edu data.

    Args:
        path: Path to processed data file.

    Returns:
        List of example dicts.
    """
    import json
    import gzip

    import fsspec

    fs = fsspec.filesystem(fsspec.utils.get_protocol(path))

    examples = []
    with fs.open(path, "rb") as f:
        with gzip.open(f, "rt", encoding="utf-8") as gz:
            for line in gz:
                examples.append(json.loads(line))

    return examples


class LlamaTokenizer:
    """Wrapper for LLaMA-3 tokenizer."""

    def __init__(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        self.mask_token_id = self.vocab_size - 1  # Use last token as mask

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def create_stack_edu_data_iter(
    batch_size: int,
    max_seq_len: int,
    key: jax.Array,
    data_path: str | None = None,
    use_llama_tokenizer: bool = True,
) -> Iterator[dict[str, Array]]:
    """Create an iterator over Stack-Edu dataset batches.

    Args:
        batch_size: Batch size.
        max_seq_len: Maximum sequence length.
        key: PRNG key.
        data_path: Path to processed data (uses default if None).
        use_llama_tokenizer: Whether to use LLaMA tokenizer.

    Yields:
        Batches with 'tokens' and 'prefix_len' keys.
    """
    if use_llama_tokenizer:
        tokenizer = LlamaTokenizer()
    else:
        from experiments.kelp.tokenizer import SimpleTokenizer

        tokenizer = SimpleTokenizer()

    if data_path is None:
        data_path = "gs://marin-us-central2/kelp/data/stack_edu_functions.jsonl.gz"

    examples = load_processed_stack_edu(data_path)
    logger.info(f"Loaded {len(examples)} examples from Stack-Edu")

    encoded_examples = []
    for item in examples:
        prompt_ids = tokenizer.encode(item["prompt"])
        code_ids = tokenizer.encode(item["code"])

        if len(code_ids) <= max_seq_len:
            encoded_examples.append(
                {
                    "tokens": code_ids,
                    "prefix_len": min(len(prompt_ids), max_seq_len),
                }
            )

    logger.info(f"Filtered to {len(encoded_examples)} examples within max_seq_len")

    if not encoded_examples:
        raise ValueError("No valid examples found in Stack-Edu dataset")

    num_examples = len(encoded_examples)

    while True:
        key, shuffle_key = random.split(key)
        indices = random.permutation(shuffle_key, jnp.arange(num_examples))

        for i in range(0, num_examples, batch_size):
            batch_indices = indices[i : i + batch_size]

            if len(batch_indices) < batch_size:
                key, extra_key = random.split(key)
                extra_indices = random.randint(extra_key, (batch_size - len(batch_indices),), 0, num_examples)
                batch_indices = jnp.concatenate([batch_indices, extra_indices])

            batch_tokens = []
            batch_prefix_lens = []

            for idx in batch_indices:
                example = encoded_examples[int(idx)]
                tokens = example["tokens"]

                padded = tokens + [tokenizer.pad_token_id] * (max_seq_len - len(tokens))
                padded = padded[:max_seq_len]

                batch_tokens.append(padded)
                batch_prefix_lens.append(example["prefix_len"])

            yield {
                "tokens": jnp.array(batch_tokens),
                "prefix_len": jnp.array(batch_prefix_lens),
            }


if __name__ == "__main__":
    from marin.execution.executor import executor_main

    executor_main(steps=[stack_edu_processing_step(max_examples=1000)])
