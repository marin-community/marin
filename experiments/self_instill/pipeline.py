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
"""
Self-Instill Pipeline: Ray Data transformations for synthetic data generation.

This module provides Ray Data-compatible batch processing classes that can be
used in marin's ExecutorStep framework:

1. SelfInstillGeneration: Generate multiple samples per prompt with validation
2. SelfInstillSummarization: Condense long reasoning into clean explanations
3. SelfInstillValidation: LLM-based quality validation

These classes are designed to work with Ray Data's map_batches API and support
distributed execution on TPU clusters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from experiments.self_instill.prompts import (
    REASONING_LONG_INSTRUCTION,
    REASONING_INSTRUCTION,
    format_final_output,
    format_summarization_prompt,
)
from experiments.self_instill.validation import (
    StaticCheckStrategy,
    ValidationPipeline,
    create_default_validation_pipeline,
)

# Disable multiprocessing for vLLM when running on Ray
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
except ImportError:
    logger.warning("vLLM is not installed. Generation functionality will not be available.")
    LLM = None
    SamplingParams = None


# =============================================================================
# DATA CONFIGURATIONS
# =============================================================================

@dataclass
class SelfInstillGenerationConfig:
    """Configuration for self-instill generation step.

    This config controls how multiple response candidates are generated
    and how the best candidate is selected.
    """
    # Model configuration
    model_name: str
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    # Generation configuration
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 32768
    num_samples: int = 4  # Number of candidates to generate per prompt

    # Prompt configuration
    prompt_column: str = "instruction_seed"  # Column containing the prompt
    use_long_instruction: bool = True  # Use detailed reasoning instruction
    apply_chat_template: bool = False  # Apply chat template (for instruct models)

    # Selection strategy
    selection_strategy: str = "longest_valid"  # "longest_valid", "first_valid", "longest"

    # Output configuration
    generated_text_column: str = "generated_text"

    # Static validation (fast filtering)
    require_boxed: bool = True
    reject_non_english: bool = True


@dataclass
class SelfInstillSummarizationConfig:
    """Configuration for summarization step.

    The summarizer takes long-form reasoning and produces condensed explanations.
    """
    # Model configuration
    model_name: str
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    # Generation configuration
    temperature: float = 0.4
    top_p: float = 0.95
    max_tokens: int = 10000
    num_samples: int = 3  # Generate multiple summaries, pick best

    # Input/output columns
    input_column: str = "generated_text"
    output_column: str = "summary"

    # Validation
    require_boxed: bool = True
    reject_non_english: bool = True


@dataclass
class SelfInstillValidationConfig:
    """Configuration for LLM-based validation step.

    This configures which validation strategies to use and their parameters.
    """
    # Model configuration
    model_name: str
    engine_kwargs: dict[str, Any] = field(default_factory=dict)

    # Validation configuration
    temperature: float = 0.6
    top_p: float = 0.95
    num_samples: int = 3  # Samples per validation prompt
    require_unanimous: bool = True  # All samples must agree

    # Which validation stages to include
    include_cycle_consistency: bool = True
    include_factual_error: bool = True
    include_total_correctness: bool = True

    # Input/output columns
    question_column: str = "instruction_seed"
    answer_column: str = "generated_text"
    output_column: str = "is_valid"


# =============================================================================
# GENERATION BATCH PROCESSOR
# =============================================================================

class SelfInstillGeneration:
    """
    Ray Data batch processor for generating multiple candidate responses.

    This class is designed to work with Ray Data's map_batches API:
    ```python
    ds.map_batches(
        SelfInstillGeneration,
        fn_constructor_kwargs={"config": generation_config},
        ...
    )
    ```

    For each input prompt, it:
    1. Generates multiple candidate responses (num_samples)
    2. Applies static validation (boxed format, language check)
    3. Selects the best candidate based on selection_strategy
    """

    def __init__(self, config: SelfInstillGenerationConfig):
        """Initialize the generation processor with vLLM."""
        self.config = config

        # Initialize vLLM
        engine_kwargs = {"tensor_parallel_size": 1, **config.engine_kwargs}
        self.llm = LLM(model=config.model_name, **engine_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        # Sampling parameters for generation
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            n=config.num_samples,  # Generate multiple samples
        )

        # Static validation for quick filtering
        self.static_checker = StaticCheckStrategy(
            require_boxed=config.require_boxed,
            reject_non_english=config.reject_non_english,
        )

    def _format_prompt(self, prompt: str) -> str:
        """Format the prompt with reasoning instructions."""
        instruction = REASONING_LONG_INSTRUCTION if self.config.use_long_instruction else REASONING_INSTRUCTION
        formatted = f"{prompt}\n{instruction}\n"

        if self.config.apply_chat_template:
            chat_example = [{"role": "user", "content": formatted}]
            formatted = self.tokenizer.apply_chat_template(
                chat_example, tokenize=False, add_generation_prompt=True
            )

        return formatted

    def _select_best_output(self, outputs: list[str]) -> str | None:
        """
        Select the best output based on the configured strategy.

        Args:
            outputs: List of candidate outputs

        Returns:
            The selected output, or None if no valid output found
        """
        strategy = self.config.selection_strategy

        if strategy == "longest":
            # Just pick the longest, no validation
            return max(outputs, key=len) if outputs else None

        # For "longest_valid" and "first_valid", we need to validate
        # Sort by length (longest first) for "longest_valid"
        if strategy == "longest_valid":
            outputs = sorted(outputs, key=len, reverse=True)

        for output in outputs:
            result = self.static_checker.validate("", output)
            if result.is_accepted:
                return output

        # No valid output found
        return None

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Process a batch of prompts and generate responses.

        Args:
            batch: Dictionary with column arrays, must include prompt_column

        Returns:
            Batch with added generated_text_column
        """
        prompts = batch[self.config.prompt_column]

        # Format all prompts
        formatted_prompts = [self._format_prompt(p) for p in prompts]

        # Generate with vLLM
        vllm_outputs = self.llm.generate(formatted_prompts, self.sampling_params)

        # Process outputs and select best for each prompt
        generated_texts = []
        for output in vllm_outputs:
            # Get all samples for this prompt
            candidates = [o.text for o in output.outputs]
            # Select the best
            selected = self._select_best_output(candidates)
            generated_texts.append(selected if selected else "")

        batch[self.config.generated_text_column] = generated_texts
        return batch


# =============================================================================
# SUMMARIZATION BATCH PROCESSOR
# =============================================================================

class SelfInstillSummarization:
    """
    Ray Data batch processor for summarizing long-form reasoning.

    This class takes verbose reasoning output and produces:
    - A clear, concise explanation of the solution approach
    - Key reasoning steps
    - Final answer in boxed format
    """

    def __init__(self, config: SelfInstillSummarizationConfig):
        """Initialize the summarization processor with vLLM."""
        self.config = config

        # Initialize vLLM
        engine_kwargs = {"tensor_parallel_size": 1, **config.engine_kwargs}
        self.llm = LLM(model=config.model_name, **engine_kwargs)

        # Sampling parameters for summarization
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            n=config.num_samples,
        )

        # Static validation for summaries
        self.static_checker = StaticCheckStrategy(
            require_boxed=config.require_boxed,
            reject_non_english=config.reject_non_english,
        )

    def _select_best_summary(self, summaries: list[str]) -> str | None:
        """Select the first valid summary."""
        for summary in summaries:
            result = self.static_checker.validate("", summary)
            if result.is_accepted:
                return summary.strip()
        return None

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Process a batch of reasoning outputs and generate summaries.

        Args:
            batch: Dictionary with column arrays, must include input_column

        Returns:
            Batch with added summary column
        """
        inputs = batch[self.config.input_column]

        # Format summarization prompts
        prompts = [format_summarization_prompt(text) for text in inputs]

        # Generate summaries
        vllm_outputs = self.llm.generate(prompts, self.sampling_params)

        # Select best summary for each
        summaries = []
        for output in vllm_outputs:
            candidates = [o.text for o in output.outputs]
            selected = self._select_best_summary(candidates)
            summaries.append(selected if selected else "")

        batch[self.config.output_column] = summaries
        return batch


# =============================================================================
# VALIDATION BATCH PROCESSOR
# =============================================================================

class SelfInstillValidation:
    """
    Ray Data batch processor for LLM-based validation.

    Runs a configurable validation pipeline including:
    - Cycle consistency (does answer address the question?)
    - Factual error check (any logical/mathematical errors?)
    - Total correctness (complete and correct solution?)
    """

    def __init__(self, config: SelfInstillValidationConfig):
        """Initialize the validation processor with vLLM."""
        self.config = config

        # Initialize vLLM
        engine_kwargs = {"tensor_parallel_size": 1, **config.engine_kwargs}
        self.llm = LLM(model=config.model_name, **engine_kwargs)

        # Sampling parameters for validation
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=50,  # Short responses for validation
        )

        # Create validation pipeline
        self.pipeline = create_default_validation_pipeline(
            num_samples=config.num_samples,
            require_unanimous=config.require_unanimous,
            include_static=False,  # Assume static already done
            include_cycle=config.include_cycle_consistency,
            include_fact=config.include_factual_error,
            include_correctness=config.include_total_correctness,
        )

    def _llm_fn(self, prompt: str, num_samples: int = 1) -> list[str]:
        """LLM inference function for validation strategies."""
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=50,
            n=num_samples,
        )
        outputs = self.llm.generate([prompt], sampling_params)
        return [o.text for o in outputs[0].outputs]

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Process a batch and validate each answer.

        Args:
            batch: Dictionary with question and answer columns

        Returns:
            Batch with added validation column
        """
        questions = batch[self.config.question_column]
        answers = batch[self.config.answer_column]

        # Validate each answer
        is_valid_list = []
        for question, answer in zip(questions, answers):
            if not answer:  # Skip empty answers
                is_valid_list.append(False)
                continue

            is_valid, _ = self.pipeline.validate(question, answer, self._llm_fn)
            is_valid_list.append(is_valid)

        batch[self.config.output_column] = is_valid_list
        return batch


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

class SelfInstillFormatOutput:
    """
    Ray Data batch processor for formatting final output.

    Combines reasoning and summary into the final format:
    <think>
    {reasoning}
    </think>

    {summary}
    """

    def __init__(
        self,
        reasoning_column: str = "generated_text",
        summary_column: str = "summary",
        original_prompt_column: str = "instruction_seed",
        output_column: str = "messages",
    ):
        """
        Initialize output formatter.

        Args:
            reasoning_column: Column containing the long-form reasoning
            summary_column: Column containing the summary
            original_prompt_column: Column containing the original prompt
            output_column: Column name for the formatted conversation
        """
        self.reasoning_column = reasoning_column
        self.summary_column = summary_column
        self.original_prompt_column = original_prompt_column
        self.output_column = output_column

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Format outputs into conversation format.

        Args:
            batch: Dictionary with reasoning and summary columns

        Returns:
            Batch with added messages column in OpenAI format
        """
        reasonings = batch[self.reasoning_column]
        summaries = batch[self.summary_column]
        prompts = batch[self.original_prompt_column]

        messages_list = []
        for prompt, reasoning, summary in zip(prompts, reasonings, summaries):
            # Format the assistant response
            assistant_content = format_final_output(reasoning, summary)

            # Create conversation in OpenAI format
            user_prompt = f"{prompt.strip()}\n{REASONING_INSTRUCTION}\n"
            conversation = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_content},
            ]
            messages_list.append(conversation)

        batch[self.output_column] = messages_list
        return batch
