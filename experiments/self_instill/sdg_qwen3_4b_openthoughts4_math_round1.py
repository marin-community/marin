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
Self-Instill: OpenThoughts4 Math Experiment (Qwen3-4B Instruction-Tuned)

This experiment generates high-quality synthetic reasoning data from the
marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-32768-tokens dataset
using the self-instill pipeline.

NOTE: This is for INSTRUCTION-TUNED models (Qwen3-4B), which output in the format:
    <think> REASONING </think> SUMMARY
Unlike base models, we do NOT need a separate summarization step.

Pipeline:
1. Preprocess: Extract user message from conversations[0]["value"]
2. Generate 4 samples per prompt, select longest that passes static check
   - Static check: must have <think>...</think> format AND \\boxed{}
3. (SKIPPED for instruction-tuned) Summarization - model already outputs summary
4. Validate via LLM (using longer prompts for thinking models):
   - Cycle consistency: Infer question from answer, compare to original
   - Factual error: Check for math/logic errors
   - Total correctness: Verify complete and correct solution
5. Format final output

Usage:
    python experiments/self_instill/sdg_qwen3_4b_openthoughts4_math_round1.py
"""

from dataclasses import dataclass, replace

import ray

from fray.cluster import ResourceConfig

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.export.hf_upload import upload_dir_to_hf
from marin.generation.inference import TextGenerationInferenceConfig
from marin.generation.inference import run_inference as run_generation_inference
from experiments.self_instill.prompts import REASONING_INSTRUCTION

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Model HuggingFace ID (using HF Hub directly instead of local gcsfuse path)
QWEN3_4B_HF_ID = "Qwen/Qwen3-4B"

# TPU resource type for inference steps
RESOURCE_TYPE = "v5p-16"

# Batch size for inference steps (smaller = less work lost on preemption)
# With n=4 samples, batch_size=16 means 64 sequences per batch
BATCH_SIZE = 16

# Flag to indicate this is an instruction-tuned model
IS_INSTRUCTION_TUNED = True

# =============================================================================
# ROUND CONFIGURATION
# =============================================================================
# Change this to run different rounds (e.g., "round1", "round2", etc.)
ROUND = "round1"
BASE_PATH = f"documents/self-instill/qwen3-4b-openthoughts4/{ROUND}"


# =============================================================================
# PROMPTS FOR INSTRUCTION-TUNED MODELS (THINKING MODELS)
# =============================================================================
# These prompts are longer and allow the model to think before giving a verdict.
# The model will output reasoning and then [[Y]] or [[N]] at the end.
# =============================================================================

# Cycle consistency - Step 1: Generate inferred question from answer
# Used for: instruction-tuned models that can reason before responding
CYCLE_QUESTION_GENERATION_PROMPT_INSTRUCT = """Given an answer, please generate the most likely question that would have prompted this answer. Focus on inferring the core question that this answer is addressing. Output only the inferred question, without any additional explanation.

Answer:
{answer}

Inferred Question:"""

# Cycle consistency - Step 2: Compare original and inferred questions
# Used for: instruction-tuned models that can reason before responding
CYCLE_COMPARISON_PROMPT_INSTRUCT = """You are evaluating whether an answer is relevant to the original question and touches the core of the question by comparing the original question with an inferred question derived only from the answer.

Original Question: {original_question}
Inferred Question: {inferred_question}

Compare the two questions and determine:
1. If the original question and inferred question are asking about the same core topic
2. If the original question and inferred question share the same key elements and requirements
3. If answering one question would effectively address the other question

After your analysis, provide your decision: [[Y]] if the questions are semantically equivalent and address the same core problem, or [[N]] if they are asking about different things."""

# Factual error check prompt
# Used for: instruction-tuned models that can reason before responding
FACTUAL_ERROR_PROMPT_INSTRUCT = """Please act as an impartial judge and carefully analyze the following answer for any factual errors, logical flaws, or misleading information.

Question: {question}
Answer: {answer}

Consider the credibility of the claims made in the answer and determine if they align with established knowledge. Evaluate:
1. Are there any incorrect facts, dates, numbers, formulas, or claims?
2. Is there any faulty logic, reasoning, or problem-solving approach?
3. Are there any misleading, incomplete, or ambiguous explanations?
4. Does the answer introduce any misconceptions or propagate common errors?

Minor typos or grammatical errors are acceptable. But be strict about any factual error, calculation error, or logical flaw. When unsure, lean toward accepting statements unless they contain clear errors.

After a thorough analysis, provide your decision: [[Y]] if the answer has no factual errors or major flaws, or [[N]] if it contains important factual errors or logical flaws that would mislead the user."""

# Total correctness check prompt
# Used for: instruction-tuned models that can reason before responding
TOTAL_CORRECTNESS_PROMPT_INSTRUCT = """Please act as an impartial judge and evaluate whether the response is completely correct in both process and conclusion.

Question: {question}
Answer: {answer}

Consider correctness, usefulness, completeness and depth in your assessment. Consider whether this answer completely solves the question.

You should rely on your own reasoning to form a reference solution and compare the answer to your reasoning.

Begin your evaluation by giving a brief summary of your thoughts on the response. Focus on whether it is accurate, addresses the question well, and is reasonably detailed. Be precise about any errors or gaps you notice.

Notes:
1. If the answer is partial, high-level, or just states that this is an open problem, you should not accept it.
2. If the answer lacks details or is not comprehensive, you should not accept it.
3. If the answer contains any errors, you should not accept it.
4. You should only accept the answer if it is at least 95% correct and solves the question.

After providing your explanation, decide whether this answer is correct. Think twice about whether this answer solves the question.
Format: Accepted: [[Y]] if you accept the answer or Accepted: [[N]] if you do not accept."""


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Define dataset config directly
OT4_MATH_HF_ID = "marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-32768-tokens"
OT4_MATH_REVISION = "6a05237"

download_ot4_math = ExecutorStep(
    name="raw/marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-32768-tokens",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=OT4_MATH_HF_ID,
        revision=versioned(OT4_MATH_REVISION),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path=f"raw/marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-32768-tokens-{OT4_MATH_REVISION}",
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 0: PREPROCESS - Extract user message from conversations format
# =============================================================================


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing OpenThoughts4 data."""
    input_path: str
    output_path: str


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def preprocess_ot4_data(config: PreprocessConfig):
    """Extract user message from conversations[0]['value'] and add instruction_seed column."""
    import hashlib
    import ray.data

    def extract_user_message(row):
        """Extract user message and create instruction_seed column."""
        # OpenThoughts4 uses "conversations" with "from"/"value" format
        conversations = row.get("conversations", [])
        if conversations and len(conversations) > 0:
            # First message should be from "human"
            user_msg = conversations[0].get("value", "")
        else:
            user_msg = ""

        # Create a unique ID based on the user message (use existing ms_id if available)
        ms_id = row.get("ms_id", hashlib.md5(user_msg.encode()).hexdigest()[:16])

        row["instruction_seed"] = user_msg
        row["ms_id"] = ms_id
        # Drop the original conversations column to avoid PyArrow serialization issues
        # (nested dicts are not supported by PyArrow)
        if "conversations" in row:
            del row["conversations"]
        return row

    # Read and transform
    ds = ray.data.read_parquet(config.input_path)
    ds = ds.map(extract_user_message)

    # Filter out empty messages
    ds = ds.filter(lambda x: x["instruction_seed"] and len(x["instruction_seed"]) > 0)

    print(f"Preprocessed rows: {ds.count()}")

    # Write output
    ds.write_parquet(config.output_path)


preprocess_ot4 = ExecutorStep(
    name=f"{BASE_PATH}/preprocessed",
    fn=preprocess_ot4_data,
    config=PreprocessConfig(
        input_path=download_ot4_math,
        output_path=this_output_path(),
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 1: GENERATION WITH MULTI-SAMPLE SELECTION
# =============================================================================
# Generate 4 samples per prompt, select longest that passes static validation
# For instruction-tuned models: must have <think>...</think> format AND \boxed{}
# For base models: must have \boxed{} and no non-English characters

generate_with_selection = ExecutorStep(
    name=f"{BASE_PATH}/generated",
    fn=run_generation_inference,
    config=TextGenerationInferenceConfig(
        # IO
        input_path=preprocess_ot4,
        output_path=this_output_path(),

        # Model
        model_name=QWEN3_4B_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768,
        },
        generation_kwargs={
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 32768,
            "n": 4,  # Generate 4 samples per prompt
        },

        # Prompting - use instruction_seed column with reasoning instruction
        template="{example}\n" + REASONING_INSTRUCTION + "\n",
        prompt_column="instruction_seed",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 128),
        batch_size=BATCH_SIZE,
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),

        # Output column
        generated_text_column_name="generated_text",

        # Checkpointing
        checkpoint_id_column="ms_id_sample",

        # Multi-sample selection
        enable_multi_sample_selection=True,
        save_all_samples=True,  # Save for debugging
        all_samples_column_name="all_samples",
        static_check_type="boxed_only",  # Only require \boxed{}, allow non-English chars
        selection_strategy="first",  # Pick first valid sample (no sorting by length)

        # Retry settings
        max_task_retries=10,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 1.5: FILTER - Remove rows where no sample passed static check
# =============================================================================
# For instruction-tuned models: check for <think>...</think> format AND \boxed{}
# This ensures the model output is in the expected reasoning format.


@dataclass
class FilterConfig:
    """Configuration for filtering rows."""
    input_path: str
    output_path: str
    is_instruction_tuned: bool = True  # Whether this is an instruction-tuned model


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def filter_valid_generations(config: FilterConfig):
    """
    Filter samples from all_samples that pass static checks, then explode into separate rows.

    For instruction-tuned models:
        - Must have <think>...</think> format
        - Must have \\boxed{} for final answer

    For base models:
        - Must have \\boxed{}
        - Must not have non-English characters

    This creates one row per valid sample, with sample_idx to track which sample it was.
    The UQ validation will run on all valid samples, then we select the first passing one.
    """
    import re
    import ray.data

    # Regex patterns
    THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
    BOXED_PATTERN = re.compile(r"\\boxed\{[^}]+\}")

    def passes_static_check(text: str, is_instruction_tuned: bool) -> bool:
        """Check if text passes static validation."""
        if not text or len(text) == 0:
            return False

        has_boxed = bool(BOXED_PATTERN.search(text))

        if is_instruction_tuned:
            # Instruction-tuned: must have <think>...</think> AND \boxed{}
            has_think_format = bool(THINK_PATTERN.search(text))
            return has_think_format and has_boxed
        else:
            # Base model: must have \boxed{} and no non-English characters
            has_non_english = any(ch.isalpha() and ord(ch) > 127 for ch in text)
            return has_boxed and not has_non_english

    is_instruct = config.is_instruction_tuned

    def explode_valid_samples(row):
        """
        For each sample in all_samples that passes static check,
        yield a separate row with sample_idx and generated_text set to that sample.
        """
        all_samples = row.get("all_samples", [])
        ms_id = row.get("ms_id", "")
        instruction_seed = row.get("instruction_seed", "")

        valid_rows = []
        for idx, sample in enumerate(all_samples):
            if passes_static_check(sample, is_instruct):
                new_row = {
                    "ms_id": ms_id,
                    "instruction_seed": instruction_seed,
                    "generated_text": sample,
                    "sample_idx": idx,
                    # Create unique ID for checkpointing in UQ steps
                    "ms_id_sample": f"{ms_id}_sample{idx}",
                }
                valid_rows.append(new_row)

        return valid_rows

    ds = ray.data.read_parquet(config.input_path)

    # Explode: each valid sample becomes its own row
    ds = ds.flat_map(explode_valid_samples)

    total_rows = ds.count()
    # Count unique ms_ids to see how many original samples have at least one valid
    unique_ms_ids = ds.unique("ms_id")
    num_unique = unique_ms_ids.count()

    print(f"Exploded to {total_rows} rows from {num_unique} unique samples (is_instruction_tuned={is_instruct})")
    print(f"Average valid samples per original: {total_rows / num_unique:.2f}" if num_unique > 0 else "No valid samples")

    ds.write_parquet(config.output_path)


filter_valid = ExecutorStep(
    name=f"{BASE_PATH}/filtered",
    fn=filter_valid_generations,
    config=FilterConfig(
        input_path=generate_with_selection,
        output_path=this_output_path(),
        is_instruction_tuned=IS_INSTRUCTION_TUNED,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 2: SUMMARIZATION (SKIPPED FOR INSTRUCTION-TUNED MODELS)
# =============================================================================
# For instruction-tuned models, the output already contains:
#   <think> REASONING </think> SUMMARY
# So we skip the summarization step and go directly to validation.
#
# For base models, this step would summarize the raw reasoning into a clean output.
# =============================================================================


# =============================================================================
# STEP 3: PREPARE VALIDATION PROMPTS
# =============================================================================
# Create columns with formatted validation prompts for each validation type
# Uses longer prompts for instruction-tuned models (thinking models)


@dataclass
class PrepValidationConfig:
    """Configuration for preparing validation prompts."""
    input_path: str
    output_path: str
    is_instruction_tuned: bool = True


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def prep_validation_prompts(config: PrepValidationConfig):
    """
    Create validation prompt columns combining question and answer.

    Uses different prompts based on model type:
    - Instruction-tuned: Longer prompts that allow thinking before verdict
    - Base models: Shorter prompts from experiments.self_instill.prompts
    """
    import ray.data

    if config.is_instruction_tuned:
        # Use longer prompts for instruction-tuned models (defined at module level)
        cycle_gen_prompt_template = CYCLE_QUESTION_GENERATION_PROMPT_INSTRUCT
        factual_prompt_template = FACTUAL_ERROR_PROMPT_INSTRUCT
        correctness_prompt_template = TOTAL_CORRECTNESS_PROMPT_INSTRUCT
    else:
        # Use shorter prompts for base models
        from experiments.self_instill.prompts import (
            CYCLE_QUESTION_GENERATION_PROMPT,
            FACTUAL_ERROR_PROMPT,
            TOTAL_CORRECTNESS_PROMPT,
        )
        cycle_gen_prompt_template = CYCLE_QUESTION_GENERATION_PROMPT
        factual_prompt_template = FACTUAL_ERROR_PROMPT
        correctness_prompt_template = TOTAL_CORRECTNESS_PROMPT

    def add_validation_prompts(row):
        """Add validation prompt columns."""
        # Drop conversations column if present (nested dicts not supported by PyArrow)
        if "conversations" in row:
            del row["conversations"]

        question = row.get("instruction_seed")
        answer = row.get("generated_text")

        # Fail explicitly if required fields are missing
        if not question or not isinstance(question, str):
            raise ValueError(f"Missing or invalid instruction_seed: {question}")
        if not answer or not isinstance(answer, str):
            raise ValueError(f"Missing or invalid generated_text: {answer}")

        question = question.strip()
        answer = answer.strip()

        # Cycle consistency - step 1: generate inferred question
        row["cycle_gen_prompt"] = cycle_gen_prompt_template.format(answer=answer)

        # Factual error prompt
        row["factual_prompt"] = factual_prompt_template.format(
            question=question,
            answer=answer,
        )

        # Total correctness prompt
        row["correctness_prompt"] = correctness_prompt_template.format(
            question=question,
            answer=answer,
        )

        return row

    ds = ray.data.read_parquet(config.input_path)
    ds = ds.map(add_validation_prompts)
    print(f"Rows prepared for validation: {ds.count()}")
    ds.write_parquet(config.output_path)


prep_validation = ExecutorStep(
    name=f"{BASE_PATH}/prep-validation",
    fn=prep_validation_prompts,
    config=PrepValidationConfig(
        input_path=filter_valid,  # Skip summarization for instruction-tuned models
        output_path=this_output_path(),
        is_instruction_tuned=IS_INSTRUCTION_TUNED,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 4: CYCLE CONSISTENCY - Generate inferred questions
# =============================================================================

cycle_gen_questions = ExecutorStep(
    name=f"{BASE_PATH}/cycle-gen",
    fn=run_generation_inference,
    config=TextGenerationInferenceConfig(
        # IO
        input_path=prep_validation,
        output_path=this_output_path(),

        # Model
        model_name=QWEN3_4B_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768,
        },
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
            "n": 3,  # Generate 3 inferred questions for voting
        },

        # Prompting
        template="{example}",
        prompt_column="cycle_gen_prompt",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 256),
        batch_size=BATCH_SIZE,
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),

        # Output column - save all 3 inferred questions as list
        generated_text_column_name="inferred_questions",

        # Checkpointing - use ms_id_sample since we have multiple rows per original ms_id
        checkpoint_id_column="ms_id_sample",

        # Save samples as list for later comparison
        save_samples_as_list=True,

        # Retry settings
        max_task_retries=10,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 5: CYCLE CONSISTENCY - Prepare comparison prompts
# =============================================================================


@dataclass
class PrepCycleCompareConfig:
    """Configuration for preparing cycle comparison prompts."""
    input_path: str
    output_path: str
    is_instruction_tuned: bool = True


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def prep_cycle_compare_prompts(config: PrepCycleCompareConfig):
    """Create cycle comparison prompts from inferred questions."""
    import ray.data

    if config.is_instruction_tuned:
        # Use longer prompt for instruction-tuned models
        comparison_template = CYCLE_COMPARISON_PROMPT_INSTRUCT
    else:
        from experiments.self_instill.prompts import CYCLE_COMPARISON_PROMPT
        comparison_template = CYCLE_COMPARISON_PROMPT

    def add_compare_prompts(row):
        """Add cycle comparison prompt columns for each inferred question."""
        original_question = row.get("instruction_seed")
        if not original_question or not isinstance(original_question, str):
            raise ValueError(f"Missing or invalid instruction_seed: {original_question}")
        original_question = original_question.strip()

        # inferred_questions is a list of 3 samples
        inferred_questions = row.get("inferred_questions", [])
        if not isinstance(inferred_questions, list):
            inferred_questions = [inferred_questions] if inferred_questions else []

        # Create comparison prompts for each inferred question
        compare_prompts = []
        cleaned_questions = []
        for inferred_q in inferred_questions:
            # Clean up - take first line only (or extract from thinking output)
            if inferred_q:
                # For thinking models, the question might be after </think>
                if "</think>" in inferred_q:
                    # Extract content after </think>
                    parts = inferred_q.split("</think>")
                    inferred_q_clean = parts[-1].strip().split('\n')[0].strip()
                else:
                    inferred_q_clean = inferred_q.strip().split('\n')[0].strip()
            else:
                inferred_q_clean = ""
            cleaned_questions.append(inferred_q_clean)

            # Create comparison prompt
            prompt = comparison_template.format(
                original_question=original_question,
                inferred_question=inferred_q_clean,
            )
            compare_prompts.append(prompt)

        row["cycle_compare_prompts"] = compare_prompts
        row["inferred_questions_cleaned"] = cleaned_questions

        # For the inference step, we'll use the first prompt
        row["cycle_compare_prompt"] = compare_prompts[0] if compare_prompts else ""

        return row

    ds = ray.data.read_parquet(config.input_path)
    ds = ds.map(add_compare_prompts)
    print(f"Rows prepared for cycle comparison: {ds.count()}")
    ds.write_parquet(config.output_path)


prep_cycle_compare = ExecutorStep(
    name=f"{BASE_PATH}/prep-cycle-compare",
    fn=prep_cycle_compare_prompts,
    config=PrepCycleCompareConfig(
        input_path=cycle_gen_questions,
        output_path=this_output_path(),
        is_instruction_tuned=IS_INSTRUCTION_TUNED,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 6: CYCLE CONSISTENCY - Compare prompts
# =============================================================================

cycle_compare = ExecutorStep(
    name=f"{BASE_PATH}/cycle-compare",
    fn=run_generation_inference,
    config=TextGenerationInferenceConfig(
        # IO
        input_path=prep_cycle_compare,
        output_path=this_output_path(),

        # Model
        model_name=QWEN3_4B_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768,
        },
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
            "n": 3,  # 3 samples for unanimous voting
        },

        # Prompting
        template="{example}",
        prompt_column="cycle_compare_prompt",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 256),
        batch_size=BATCH_SIZE,
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),

        # Output column - save as list for unanimous voting
        generated_text_column_name="cycle_results",

        # Checkpointing
        checkpoint_id_column="ms_id_sample",

        # Save samples as list for unanimous voting
        save_samples_as_list=True,

        # Retry settings
        max_task_retries=10,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 7: FACTUAL ERROR CHECK
# =============================================================================

factual_check = ExecutorStep(
    name=f"{BASE_PATH}/factual",
    fn=run_generation_inference,
    config=TextGenerationInferenceConfig(
        # IO
        input_path=cycle_compare,
        output_path=this_output_path(),

        # Model
        model_name=QWEN3_4B_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768,
        },
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
            "n": 3,  # 3 samples for unanimous voting
        },

        # Prompting
        template="{example}",
        prompt_column="factual_prompt",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 256),
        batch_size=BATCH_SIZE,
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),

        # Output column - save as list for unanimous voting
        generated_text_column_name="factual_results",

        # Checkpointing
        checkpoint_id_column="ms_id_sample",

        # Save samples as list for unanimous voting
        save_samples_as_list=True,

        # Retry settings
        max_task_retries=10,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 8: TOTAL CORRECTNESS CHECK
# =============================================================================

correctness_check = ExecutorStep(
    name=f"{BASE_PATH}/correctness",
    fn=run_generation_inference,
    config=TextGenerationInferenceConfig(
        # IO
        input_path=factual_check,
        output_path=this_output_path(),

        # Model
        model_name=QWEN3_4B_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768,
        },
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 32768,
            "n": 3,  # 3 samples for unanimous voting
        },

        # Prompting
        template="{example}",
        prompt_column="correctness_prompt",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 256),
        batch_size=BATCH_SIZE,
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),

        # Output column - save as list for unanimous voting
        generated_text_column_name="correctness_results",

        # Checkpointing
        checkpoint_id_column="ms_id_sample",

        # Save samples as list for unanimous voting
        save_samples_as_list=True,

        # Retry settings
        max_task_retries=10,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 9: FILTER VALIDATION RESULTS AND FORMAT OUTPUT
# =============================================================================


@dataclass
class FilterValidationConfig:
    """Configuration for filtering based on validation results."""
    input_path: str
    output_path: str
    is_instruction_tuned: bool = True


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def filter_and_format_output(config: FilterValidationConfig):
    """
    Filter based on unanimous validation and select first passing sample per original ms_id.

    Since we exploded rows (one per valid sample), we now need to:
    1. Check which samples passed all UQ validation
    2. For each original ms_id, pick the first sample (lowest sample_idx) that passed
    3. Format the final output
    """
    import re
    import ray.data
    from experiments.self_instill.prompts import REASONING_INSTRUCTION

    # ==========================================================================
    # Decision extraction for thinking models
    # ==========================================================================
    # Thinking models output reasoning before the verdict, so we need robust
    # extraction that finds [[Y]] or [[N]] anywhere in the output (prefer last).

    _DECISION_RE = re.compile(r"\[\[\s*([YN])\s*\]\]", re.IGNORECASE)

    def extract_decision_robust(text: str) -> bool:
        """
        Robustly extract Y/N decision from judge output.

        Priority:
          1) Last occurrence of [[Y]] / [[N]]
          2) Last occurrence of a standalone Y/N token
          3) Last occurrence of YES/NO

        Returns True for Y, False otherwise.
        """
        if not text:
            return False

        # 1) [[Y]] / [[N]] (prefer the last one)
        matches = list(_DECISION_RE.finditer(text))
        if matches:
            return matches[-1].group(1).upper() == "Y"

        # 2) Standalone Y/N token (common when small models ignore brackets)
        yn_tokens = re.findall(r"(?<![A-Za-z])([YN])(?![A-Za-z])", text.strip(), flags=re.IGNORECASE)
        if yn_tokens:
            return yn_tokens[-1].upper() == "Y"

        # 3) YES/NO
        yesno = re.findall(r"\b(YES|NO)\b", text.strip(), flags=re.IGNORECASE)
        if yesno:
            return yesno[-1].upper() == "YES"

        return False

    def check_unanimous(samples: list) -> bool:
        """Check if all samples have Y decision (unanimous voting)."""
        if not samples or not isinstance(samples, list):
            return False
        decisions = [extract_decision_robust(s) for s in samples]
        return all(decisions) if decisions else False

    def check_validation_and_format(row):
        """Check all validation results and format output if passed."""
        # Extract decisions from sample lists
        cycle_samples = row.get("cycle_results", [])
        factual_samples = row.get("factual_results", [])
        correctness_samples = row.get("correctness_results", [])

        # Check unanimous voting for each validation type
        cycle_pass = check_unanimous(cycle_samples)
        factual_pass = check_unanimous(factual_samples)
        correctness_pass = check_unanimous(correctness_samples)

        row["cycle_passed"] = cycle_pass
        row["factual_passed"] = factual_pass
        row["correctness_passed"] = correctness_pass
        row["all_validation_passed"] = cycle_pass and factual_pass and correctness_pass

        # Format final output if all validations passed
        if row["all_validation_passed"]:
            generated_text = row.get("generated_text")
            instruction_seed = row.get("instruction_seed")

            if not generated_text or not isinstance(generated_text, str):
                raise ValueError(f"Missing generated_text in validated row")
            if not instruction_seed or not isinstance(instruction_seed, str):
                raise ValueError(f"Missing instruction_seed in validated row")

            if config.is_instruction_tuned:
                # For instruction-tuned models, the output is already in the format:
                # <think> REASONING </think> SUMMARY
                # So we use it directly
                output_text = generated_text
            else:
                # For base models, we would combine reasoning and summary
                summary = row.get("summary", "")
                output_text = f"<think>\n{generated_text}\n</think>\n\n{summary}"

            # Store as separate columns instead of nested dict (PyArrow compatible)
            user_prompt = instruction_seed.strip() + "\n" + REASONING_INSTRUCTION + "\n"
            row["user_content"] = user_prompt
            row["assistant_content"] = output_text
        else:
            row["user_content"] = None
            row["assistant_content"] = None

        return row

    ds = ray.data.read_parquet(config.input_path)

    # First, check validation for all rows
    ds = ds.map(check_validation_and_format)

    # Log stats before selection
    total = ds.count()
    ds_passed = ds.filter(lambda x: x.get("all_validation_passed", False))
    passed_count = ds_passed.count()

    print(f"Validation results: {passed_count}/{total} sample-rows passed all validation")

    # Collect to pandas for global groupby (data size is manageable at ~100k rows max)
    import pandas as pd
    df_passed = ds_passed.to_pandas()

    if len(df_passed) == 0:
        print("No samples passed all validation!")
        # Write empty dataset
        ds_passed.write_parquet(config.output_path)
        return

    # Count unique ms_ids that have at least one passing sample
    unique_passed = df_passed["ms_id"].nunique()
    print(f"Unique original samples with at least one passing: {unique_passed}")

    # Sort by ms_id and sample_idx, then take first per ms_id
    # This selects the first sample (lowest sample_idx) that passed for each original input
    df_passed = df_passed.sort_values(["ms_id", "sample_idx"])
    df_selected = df_passed.groupby("ms_id").first().reset_index()

    final_count = len(df_selected)
    print(f"Final output: {final_count} rows (one per original sample that had a passing generation)")

    # Write back to parquet
    ray.data.from_pandas(df_selected).write_parquet(config.output_path)


filter_validation = ExecutorStep(
    name=f"{BASE_PATH}/validated",
    fn=filter_and_format_output,
    config=FilterValidationConfig(
        input_path=correctness_check,
        output_path=this_output_path(),
        is_instruction_tuned=IS_INSTRUCTION_TUNED,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# UPLOAD STEP
# =============================================================================

upload_self_instill = upload_dir_to_hf(
    input_path=filter_validation,
    repo_id=f"marin-community/self-instill-ot4-math-qwen3-4b-{ROUND}",
    repo_type="dataset",
)

upload_self_instill = replace(upload_self_instill, pip_dependency_groups=["vllm"])


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Stage 1: Download dataset (model downloaded via HF Hub on workers)
    print("Stage 1: Downloading dataset...")
    executor_main([download_ot4_math])

    # Stage 2: Preprocess - extract user message
    print("Stage 2: Preprocessing...")
    executor_main([preprocess_ot4])

    # Stage 3: Generate with multi-sample selection
    print("Stage 3: Generating with multi-sample selection...")
    executor_main([generate_with_selection])

    # Stage 4: Filter valid generations (checks <think>...</think> format for instruction-tuned)
    print("Stage 4: Filtering valid generations...")
    executor_main([filter_valid])

    # NOTE: Summarization step is SKIPPED for instruction-tuned models
    # The model already outputs in <think>...</think> SUMMARY format

    # Stage 5: Prepare validation prompts (uses longer prompts for instruction-tuned)
    print("Stage 5: Preparing validation prompts...")
    executor_main([prep_validation])

    # Stage 6: Cycle consistency - generate inferred questions
    print("Stage 6: Cycle consistency - generating inferred questions...")
    executor_main([cycle_gen_questions])

    # Stage 7: Cycle consistency - prepare comparison
    print("Stage 7: Cycle consistency - preparing comparison...")
    executor_main([prep_cycle_compare])

    # Stage 8: Cycle consistency - compare
    print("Stage 8: Cycle consistency - comparing...")
    executor_main([cycle_compare])

    # Stage 9: Factual error check
    print("Stage 9: Factual error check...")
    executor_main([factual_check])

    # Stage 10: Total correctness check
    print("Stage 10: Total correctness check...")
    executor_main([correctness_check])

    # Stage 11: Filter validation results and format output
    print("Stage 11: Filtering validation results and formatting output...")
    executor_main([filter_validation])

    # Stage 12: Upload to HuggingFace
    print("Stage 12: Uploading to HuggingFace...")
    executor_main([upload_self_instill])

    print("Pipeline complete!")
