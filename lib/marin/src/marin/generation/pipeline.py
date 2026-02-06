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
import re
from typing import Any, Callable

import pyarrow as pa

from marin.generation.llm_generation import BaseLLMProvider, vLLMProvider
from marin.generation.templates import STEP_BY_STEP_TEMPLATE

logger = logging.getLogger(__name__)

# Default static check pattern for boxed answers
_BOXED_PATTERN = re.compile(r"\\boxed\{[^}]+\}")

try:
    from vllm.inputs.data import TokensPrompt
except ImportError:
    logger.warning("vLLM is not installed, so we will not be able to generate text.")
    TokensPrompt = Any


def _is_all_null(arr) -> bool:
    """Check if all values in an array are null/None/NaN.

    Handles Python lists, numpy arrays, and PyArrow arrays.
    """
    import numpy as np

    if isinstance(arr, pa.Array):
        return arr.null_count == len(arr)
    elif isinstance(arr, pa.ChunkedArray):
        return arr.null_count == len(arr)
    elif isinstance(arr, np.ndarray):
        if arr.dtype == object:
            return all(v is None for v in arr)
        elif np.issubdtype(arr.dtype, np.floating):
            return np.all(np.isnan(arr))
        else:
            return False
    elif isinstance(arr, list):
        return all(v is None for v in arr)
    return False


def _normalize_batch_schema(batch: dict[str, Any], column_types: dict[str, pa.DataType]) -> dict[str, Any]:
    """
    Normalize schema for all columns in a batch based on expected types.

    When PyArrow encounters a batch where all values in a column are null,
    it may infer the type as float64 instead of the correct type. This causes
    schema unification errors in Ray Data. This function ensures that columns
    have their correct types even when all values are null.

    This function ALWAYS converts columns with all-null values to PyArrow arrays
    with explicit types to ensure consistent schema across all batches.

    Args:
        batch: The batch of data to normalize.
        column_types: Dict mapping column names to their expected PyArrow types.

    Returns:
        The batch with corrected column types.
    """
    import numpy as np

    for col, expected_type in column_types.items():
        if col not in batch:
            continue

        arr = batch[col]

        # Check if all values are null/None/NaN
        if _is_all_null(arr):
            # Get the length of the array
            if isinstance(arr, (pa.Array, pa.ChunkedArray)):
                length = len(arr)
            elif isinstance(arr, np.ndarray):
                length = len(arr)
            elif isinstance(arr, list):
                length = len(arr)
            else:
                continue

            # Create a PyArrow array with the correct type and all null values
            batch[col] = pa.array([None] * length, type=expected_type)

    return batch


class TextGeneration:
    def __init__(
        self,
        llm: BaseLLMProvider,
        template: list[str] | str | None = None,
        num_generations: int = 1,
        prompt_column: str = "text",
        save_templated_prompt: bool = False,
        generated_text_column_name: str = "generated_text",
    ):
        """Initializes a text generation pipeline that takes an input batch and generates text for each example.

        Inputs:
            llm: The LLM provider to use for the pipeline.
            template: The template to use for the pipeline. This can be a string or a list of strings.
                If it is a string, it will be used for all examples.
                If it is a list of strings, we will zip the template with each example
                (i.e. iterate through template, example in zip(templates, examples))
            num_generations: The number of generations to generate for each example.
            prompt_column: The column name of the prompt in the input batch.
            save_templated_prompt: Whether to save the templated prompt.
            generated_text_column_name: The column name of the generated text in the output batch.
        """
        self.llm = llm

        # Normalize template to a list for consistent handling at call time
        # Accept either a single string template or a list of per-example templates.
        base_template = template or STEP_BY_STEP_TEMPLATE
        self.templates: list[str] = base_template if isinstance(base_template, list) else [base_template]
        self.num_generations = num_generations
        self.prompt_column = prompt_column
        self.save_templated_prompt = save_templated_prompt
        self.generated_text_column_name = generated_text_column_name

    def _update_batch(self, batch: dict[str, Any], generated_text: list[str], prompts: list[str]) -> dict[str, Any]:
        batch.update({self.generated_text_column_name: generated_text})

        if self.save_templated_prompt:
            batch.update({"prompt": prompts})

        return batch

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Generate a batch of text using an LLM where the example text is in dolma format in the "text" column."""

        examples = batch[self.prompt_column]

        if len(self.templates) == 1:
            # Broadcast single template to all examples
            prompts = [self.templates[0].format(example=example) for example in examples]
        else:
            assert len(self.templates) == len(examples), "The number of templates must match the number of examples."
            prompts = [
                template.format(example=example) for template, example in zip(self.templates, examples, strict=False)
            ]

        generated_text = self.llm.generate(prompts)

        return self._update_batch(batch, generated_text, prompts)


def _validate_tpu_available():
    """Validate that TPU is available, fail fast if not.

    This prevents silent CPU fallback which would cause errors later
    (e.g., 'TpuDevice' object has no attribute 'coords').
    """
    import jax

    devices = jax.devices()
    tpu_devices = [d for d in devices if "tpu" in str(d.platform).lower()]

    if not tpu_devices:
        device_info = [f"{d.platform}:{d.device_kind}" for d in devices]
        raise RuntimeError(
            f"TPU required but not available. This worker will be restarted on a different node. "
            f"Available devices: {device_info}"
        )

    logger.info(f"TPU validation passed. Found {len(tpu_devices)} TPU device(s): {tpu_devices}")


class vLLMTextGeneration(TextGeneration):
    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        template: list[str] | str | None = None,
        num_generations: int = 1,
        num_instances: tuple[int, int] = (1, 4),
        prompt_column: str = "text",
        apply_chat_template: bool = True,
        save_templated_prompt: bool = False,
        max_doc_tokens: int = 7000,
        generated_text_column_name: str = "generated_text",
        column_types: dict[str, pa.DataType] | None = None,
        save_samples_as_list: bool = False,
    ):
        """Initialize vLLM text generation pipeline.

        Args:
            model_name: The name of the model to use.
            engine_kwargs: Keyword arguments for the vLLM engine.
            generation_kwargs: Keyword arguments for generation (can include `n` for multi-sample).
            template: Template(s) for formatting prompts.
            num_generations: Number of generations (typically 1).
            num_instances: Number of parallel instances.
            prompt_column: The column containing prompts.
            apply_chat_template: Whether to apply chat template.
            save_templated_prompt: Whether to save the templated prompt.
            max_doc_tokens: Maximum document tokens.
            generated_text_column_name: Column name for output.
            column_types: Expected column types for schema normalization.
            save_samples_as_list: If True and n>1 in generation_kwargs, saves samples as a list
                instead of joining with space. Useful for multi-sample voting/validation.
        """
        # Validate TPU is available before proceeding. This prevents silent CPU
        # fallback which would cause cryptic errors later in tpu_inference.
        _validate_tpu_available()

        # Initialize the LLM Provider here for the pipeline since we need the model
        # to be placed in the same placement group as the pipeline
        llm = vLLMProvider(model_name, engine_kwargs, generation_kwargs)

        super().__init__(
            llm, template, num_generations, prompt_column, save_templated_prompt, generated_text_column_name
        )
        self.apply_chat_template = apply_chat_template
        self.max_doc_tokens = max_doc_tokens
        self.tokenizer = self.llm.llm.get_tokenizer()
        self.column_types = column_types or {}
        self.save_samples_as_list = save_samples_as_list

    def _truncate_example(self, example: str) -> str:
        example_tokens = self.tokenizer.encode(example)
        if len(example_tokens) > self.max_doc_tokens:
            example_tokens = example_tokens[: self.max_doc_tokens]
            example = self.tokenizer.decode(example_tokens)

        return example

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        prompts = []

        examples = batch[self.prompt_column]

        if len(self.templates) == 1:
            single_template = self.templates[0]
            for example in examples:
                example = self._truncate_example(example)
                if self.apply_chat_template:
                    try:
                        chat_example = [{"role": "user", "content": single_template.format(example=example)}]
                    except Exception as e:
                        logger.error(f"Error formatting template: {e}")
                        logger.error(f"Template: {single_template}")
                        logger.error(f"Example: {example}")
                        raise e
                    prompts.append(
                        self.tokenizer.apply_chat_template(chat_example, tokenize=False, add_generation_prompt=True)
                    )
                else:
                    prompts.append(example)
        else:
            assert len(self.templates) == len(examples), "The number of templates must match the number of examples."
            for template, example in zip(self.templates, examples, strict=False):
                example = self._truncate_example(example)
                if self.apply_chat_template:
                    try:
                        chat_example = [{"role": "user", "content": template.format(example=example)}]
                    except Exception as e:
                        logger.error(f"Error formatting template: {e}")
                        logger.error(f"Template: {template}")
                        logger.error(f"Example: {example}")
                        raise e
                    prompts.append(
                        self.tokenizer.apply_chat_template(chat_example, tokenize=False, add_generation_prompt=True)
                    )
                else:
                    prompts.append(example)

        if self.save_samples_as_list:
            # Use multi-sample generation and save as list
            all_samples = self.llm.generate_multi_sample(prompts)
            batch[self.generated_text_column_name] = all_samples
            if self.save_templated_prompt:
                batch["prompt"] = prompts
        else:
            # Standard generation (joins samples with space if n>1)
            generated_text = self.llm.generate(prompts)
            batch = self._update_batch(batch, generated_text, prompts)

        # Normalize schema for all columns to prevent type mismatches when
        # batches have all-null values (which PyArrow may infer as float).
        if self.column_types:
            batch = _normalize_batch_schema(batch, self.column_types)

        return batch


def default_static_check(text: str) -> bool:
    """Default static validation: requires \\boxed{} and no non-English letters.

    Args:
        text: The generated text to validate.

    Returns:
        True if the text passes all checks, False otherwise.
    """
    if not text:
        return False
    # Must have boxed answer
    if not _BOXED_PATTERN.search(text):
        return False
    # Must not have non-English letters (non-ASCII alphabetic characters)
    if any(ch.isalpha() and ord(ch) > 127 for ch in text):
        return False
    return True


def boxed_only_static_check(text: str) -> bool:
    """Static validation that only requires \\boxed{}.

    This is more permissive than default_static_check and allows non-English
    characters (e.g., Chinese, Greek math symbols). Useful for multilingual
    models or math problems that use Greek letters.

    Args:
        text: The generated text to validate.

    Returns:
        True if the text contains \\boxed{}, False otherwise.
    """
    if not text:
        return False
    # Only require boxed answer
    return bool(_BOXED_PATTERN.search(text))


class vLLMTextGenerationWithSelection(vLLMTextGeneration):
    """vLLM text generation with multi-sample generation and selection.

    This class generates multiple samples per prompt (using the `n` parameter in
    generation_kwargs), applies a static validation check to each sample, and
    selects the longest valid sample. This is useful for reasoning tasks where
    longer, more detailed responses tend to be higher quality.

    The selection process:
    1. Generate `n` samples per prompt
    2. Sort samples by length (longest first)
    3. Apply static check to each sample
    4. Select the first (longest) sample that passes validation
    5. If no sample passes, mark the row with selected_text=None

    Usage:
        config = TextGenerationInferenceConfig(
            ...
            generation_kwargs={"temperature": 0.8, "max_tokens": 30000, "n": 4},
            ...
        )
        # Use vLLMTextGenerationWithSelection instead of vLLMTextGeneration
    """

    def __init__(
        self,
        model_name: str,
        engine_kwargs: dict[str, Any] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        template: list[str] | str | None = None,
        num_generations: int = 1,
        num_instances: tuple[int, int] = (1, 4),
        prompt_column: str = "text",
        apply_chat_template: bool = True,
        save_templated_prompt: bool = False,
        max_doc_tokens: int = 7000,
        generated_text_column_name: str = "generated_text",
        column_types: dict[str, pa.DataType] | None = None,
        static_check_fn: Callable[[str], bool] | None = None,
        save_all_samples: bool = False,
        all_samples_column_name: str = "all_samples",
        selection_strategy: str = "longest",
    ):
        """Initialize the multi-sample generation pipeline with selection.

        Args:
            model_name: The name of the model to use.
            engine_kwargs: Keyword arguments for the vLLM engine.
            generation_kwargs: Keyword arguments for generation (should include `n` for multi-sample).
            template: Template(s) for formatting prompts.
            num_generations: Number of generations (typically 1 for multi-sample selection).
            num_instances: Number of parallel instances.
            prompt_column: The column containing prompts.
            apply_chat_template: Whether to apply chat template.
            save_templated_prompt: Whether to save the templated prompt.
            max_doc_tokens: Maximum document tokens.
            generated_text_column_name: Column name for selected output.
            column_types: Expected column types for schema normalization.
            static_check_fn: Function to validate samples. Defaults to requiring \\boxed{}
                and no non-English characters. Returns True if sample is valid.
            save_all_samples: Whether to save all generated samples (for debugging).
            all_samples_column_name: Column name for all samples if save_all_samples=True.
            selection_strategy: Strategy for selecting from valid samples. Options:
                - "longest": select longest valid sample (default)
                - "first": select first valid sample (no sorting, faster)
        """
        super().__init__(
            model_name=model_name,
            engine_kwargs=engine_kwargs,
            generation_kwargs=generation_kwargs,
            template=template,
            num_generations=num_generations,
            num_instances=num_instances,
            prompt_column=prompt_column,
            apply_chat_template=apply_chat_template,
            save_templated_prompt=save_templated_prompt,
            max_doc_tokens=max_doc_tokens,
            generated_text_column_name=generated_text_column_name,
            column_types=column_types,
        )
        self.static_check_fn = static_check_fn or default_static_check
        self.save_all_samples = save_all_samples
        self.all_samples_column_name = all_samples_column_name
        self.selection_strategy = selection_strategy

    def _select_best_sample(self, samples: list[str]) -> str | None:
        """Select the best sample that passes static validation.

        Args:
            samples: List of generated samples for a single prompt.

        Returns:
            The selected valid sample, or None if no sample passes validation.
            Selection depends on self.selection_strategy:
            - "longest": returns longest valid sample
            - "first": returns first valid sample (no sorting)
        """
        if self.selection_strategy == "first":
            # Return first valid sample (no sorting)
            for sample in samples:
                if self.static_check_fn(sample):
                    return sample
        else:
            # Default: sort by length descending (prefer longer responses)
            sorted_samples = sorted(samples, key=len, reverse=True)
            for sample in sorted_samples:
                if self.static_check_fn(sample):
                    return sample

        return None

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Generate multiple samples per prompt and select the best one."""
        prompts = []
        examples = batch[self.prompt_column]

        # Build prompts (same logic as parent class)
        if len(self.templates) == 1:
            single_template = self.templates[0]
            for example in examples:
                example = self._truncate_example(example)
                if self.apply_chat_template:
                    try:
                        chat_example = [{"role": "user", "content": single_template.format(example=example)}]
                    except Exception as e:
                        logger.error(f"Error formatting template: {e}")
                        logger.error(f"Template: {single_template}")
                        logger.error(f"Example: {example}")
                        raise e
                    prompts.append(
                        self.tokenizer.apply_chat_template(chat_example, tokenize=False, add_generation_prompt=True)
                    )
                else:
                    prompts.append(example)
        else:
            assert len(self.templates) == len(examples), "The number of templates must match the number of examples."
            for template, example in zip(self.templates, examples, strict=False):
                example = self._truncate_example(example)
                if self.apply_chat_template:
                    try:
                        chat_example = [{"role": "user", "content": template.format(example=example)}]
                    except Exception as e:
                        logger.error(f"Error formatting template: {e}")
                        logger.error(f"Template: {template}")
                        logger.error(f"Example: {example}")
                        raise e
                    prompts.append(
                        self.tokenizer.apply_chat_template(chat_example, tokenize=False, add_generation_prompt=True)
                    )
                else:
                    prompts.append(example)

        # Generate multiple samples per prompt
        all_samples = self.llm.generate_multi_sample(prompts)

        # Select best sample for each prompt
        selected_texts: list[str | None] = []
        for samples in all_samples:
            best = self._select_best_sample(samples)
            selected_texts.append(best)

        # Update batch with selected text
        batch[self.generated_text_column_name] = selected_texts

        if self.save_templated_prompt:
            batch["prompt"] = prompts

        if self.save_all_samples:
            batch[self.all_samples_column_name] = all_samples

        # Normalize schema for all columns
        if self.column_types:
            batch = _normalize_batch_schema(batch, self.column_types)

        return batch
