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
2. Generate 4 samples per prompt, select first that passes static check
   - Static check: must have <think>...</think> format AND \\boxed{}
3. (SKIPPED for instruction-tuned) Summarization - model already outputs summary
4. Validate via LLM with EARLY EXIT optimization:
   - Cycle consistency: Infer question from answer, compare to original
     → If fails, skip factual and correctness (early exit)
   - Factual error: Check for math/logic errors
     → If fails, skip correctness (early exit)
   - Total correctness: Verify complete and correct solution
5. Format final output

Early Exit Optimization:
- Each validation check filters passed/failed rows BEFORE running the next check
- Rows that fail cycle skip factual and correctness entirely
- Rows that fail factual skip correctness entirely
- This saves ~30-50% compute compared to running all checks on all rows

Iterative UQ:
- For each row, try sample 0 first through all validation
- If fails, try sample 1, etc. up to sample 3
- First sample to pass all validation is used

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
RESOURCE_TYPE = "v5p-8"

# Tensor parallel size for vLLM inference
TENSOR_PARALLEL_SIZE = 4

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
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
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
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),

        # Output column
        generated_text_column_name="generated_text",

        # Checkpointing
        checkpoint_id_column="ms_id",

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
# STEP 1.5: FILTER - Collect valid samples as ordered list (don't explode)
# =============================================================================
# For instruction-tuned models: check for <think>...</think> format AND \boxed{}
# This ensures the model output is in the expected reasoning format.
#
# NEW DESIGN: Keep all valid samples together in one row as an ordered list.
# This enables iterative UQ checking - try sample 0 first, if fails try sample 1, etc.


@dataclass
class FilterCollectConfig:
    """Configuration for filtering and collecting valid samples."""
    input_path: str
    output_path: str
    is_instruction_tuned: bool = True


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def filter_collect_valid_samples(config: FilterCollectConfig):
    """
    Collect samples that pass static checks into an ordered list per row.

    For instruction-tuned models:
        - Must have <think>...</think> format
        - Must have \\boxed{} for final answer

    Output row contains:
        - valid_samples: list of samples that passed static check (in original order)
        - valid_indices: list of original indices (0-3) for each valid sample
        - num_valid_samples: count of valid samples
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
            has_think_format = bool(THINK_PATTERN.search(text))
            return has_think_format and has_boxed
        else:
            has_non_english = any(ch.isalpha() and ord(ch) > 127 for ch in text)
            return has_boxed and not has_non_english

    is_instruct = config.is_instruction_tuned

    def collect_valid_samples(row):
        """Collect all valid samples into ordered lists."""
        all_samples = row.get("all_samples", [])

        valid_samples = []
        valid_indices = []
        for idx, sample in enumerate(all_samples):
            if passes_static_check(sample, is_instruct):
                valid_samples.append(sample)
                valid_indices.append(idx)

        # Keep only essential columns plus the new ones
        return {
            "ms_id": row.get("ms_id", ""),
            "instruction_seed": row.get("instruction_seed", ""),
            "valid_samples": valid_samples,
            "valid_indices": valid_indices,
            "num_valid_samples": len(valid_samples),
        }

    ds = ray.data.read_parquet(config.input_path)
    ds = ds.map(collect_valid_samples)

    # Filter out rows with no valid samples
    ds = ds.filter(lambda x: x.get("num_valid_samples", 0) > 0)

    total_rows = ds.count()
    print(f"Rows with at least one valid sample: {total_rows} (is_instruction_tuned={is_instruct})")

    ds.write_parquet(config.output_path)


filter_collect = ExecutorStep(
    name=f"{BASE_PATH}/filtered-collected",
    fn=filter_collect_valid_samples,
    config=FilterCollectConfig(
        input_path=generate_with_selection,
        output_path=this_output_path(),
        is_instruction_tuned=IS_INSTRUCTION_TUNED,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# ITERATIVE UQ PROCESSING HELPERS
# =============================================================================
# These functions support the iterative approach:
# - Round 0: Try first valid sample for all rows
# - Round 1: For rows that failed Round 0, try second valid sample
# - Continue until a sample passes or no more samples


@dataclass
class ExtractSampleConfig:
    """Configuration for extracting a specific sample for UQ testing."""
    input_path: str
    output_path: str
    sample_round: int  # Which sample to extract (0 = first valid, 1 = second valid, etc.)


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def extract_sample_for_round(config: ExtractSampleConfig):
    """
    Extract the sample at the given round index for UQ testing.

    For round N, extracts valid_samples[N] if it exists.
    Rows without a sample at index N are filtered out.
    """
    import ray.data

    round_idx = config.sample_round

    def extract_sample(row):
        valid_samples = row.get("valid_samples", [])
        valid_indices = row.get("valid_indices", [])

        if round_idx >= len(valid_samples):
            return None  # No sample at this round index

        return {
            "ms_id": row.get("ms_id", ""),
            "instruction_seed": row.get("instruction_seed", ""),
            "current_sample": valid_samples[round_idx],
            "current_sample_idx": valid_indices[round_idx],
            "valid_samples": valid_samples,  # Keep for potential next round
            "valid_indices": valid_indices,
            "num_valid_samples": row.get("num_valid_samples", 0),
            "current_round": round_idx,
        }

    ds = ray.data.read_parquet(config.input_path)
    ds = ds.map(extract_sample)

    # Filter out None rows (no sample at this round index)
    ds = ds.filter(lambda x: x is not None)

    count = ds.count()
    print(f"Round {round_idx}: {count} rows have a sample at index {round_idx}")

    ds.write_parquet(config.output_path)


@dataclass
class CheckAndFilterConfig:
    """Configuration for checking one validation result and splitting passed/failed."""
    input_path: str
    passed_output_path: str
    failed_output_path: str
    result_column: str  # "cycle_results", "factual_results", or "correctness_results"
    check_name: str     # "cycle", "factual", "correctness" (for logging)


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def check_and_filter(config: CheckAndFilterConfig):
    """
    Check unanimous [[Y]] voting on a single result column and split into passed/failed.

    This enables early exit: if a row fails cycle check, we skip factual and correctness.
    Failed rows are collected for the next round (to try the next sample).
    Passed rows continue to the next validation step.
    """
    import re
    import ray.data

    _DECISION_RE = re.compile(r"\[\[\s*([YN])\s*\]\]", re.IGNORECASE)

    def extract_decision_robust(text: str) -> bool:
        if not text:
            return False
        matches = list(_DECISION_RE.finditer(text))
        if matches:
            return matches[-1].group(1).upper() == "Y"
        yn_tokens = re.findall(r"(?<![A-Za-z])([YN])(?![A-Za-z])", text.strip(), flags=re.IGNORECASE)
        if yn_tokens:
            return yn_tokens[-1].upper() == "Y"
        yesno = re.findall(r"\b(YES|NO)\b", text.strip(), flags=re.IGNORECASE)
        if yesno:
            return yesno[-1].upper() == "YES"
        return False

    def check_unanimous(samples: list) -> bool:
        if not samples or not isinstance(samples, list):
            return False
        decisions = [extract_decision_robust(s) for s in samples]
        return all(decisions) if decisions else False

    def add_check_result(row):
        samples = row.get(config.result_column, [])
        row[f"{config.check_name}_passed"] = check_unanimous(samples)
        return row

    ds = ray.data.read_parquet(config.input_path)
    ds = ds.map(add_check_result)

    total = ds.count()
    ds_passed = ds.filter(lambda x: x.get(f"{config.check_name}_passed", False))
    ds_failed = ds.filter(lambda x: not x.get(f"{config.check_name}_passed", False))

    passed_count = ds_passed.count()
    failed_count = ds_failed.count()

    print(f"{config.check_name.upper()}: {passed_count} passed, {failed_count} failed out of {total}")

    # Write outputs (handle empty case gracefully)
    if passed_count > 0:
        ds_passed.write_parquet(config.passed_output_path)
    else:
        # Write empty marker file so downstream steps can detect no data
        print(f"  No rows passed {config.check_name} check - skipping passed output")

    if failed_count > 0:
        ds_failed.write_parquet(config.failed_output_path)

    return passed_count


@dataclass
class MergeFailedConfig:
    """Configuration for merging failed rows from all validation stages."""
    cycle_failed_path: str
    factual_failed_path: str
    correctness_failed_path: str
    output_path: str


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def merge_all_failed(config: MergeFailedConfig):
    """
    Merge failed rows from cycle, factual, and correctness checks.

    These rows will be used as input for the next round, where we try
    the next valid sample for each row.
    """
    import ray.data

    datasets = []
    for path, name in [
        (config.cycle_failed_path, "cycle"),
        (config.factual_failed_path, "factual"),
        (config.correctness_failed_path, "correctness"),
    ]:
        try:
            ds = ray.data.read_parquet(path)
            count = ds.count()
            if count > 0:
                datasets.append(ds)
                print(f"  {name}_failed: {count} rows")
        except Exception as e:
            print(f"  {name}_failed: 0 rows (no file or empty)")

    if not datasets:
        print("No failed rows to merge - all rows passed!")
        return 0

    combined = datasets[0]
    for ds in datasets[1:]:
        combined = combined.union(ds)

    total = combined.count()
    print(f"Total failed rows for next round: {total}")
    combined.write_parquet(config.output_path)
    return total


def check_path_has_data(path: str) -> bool:
    """Check if a GCS/local path has parquet files with data."""
    import fsspec

    try:
        fs, path_without_protocol = fsspec.core.url_to_fs(path)
        # Check if path exists and has parquet files
        if not fs.exists(path_without_protocol):
            return False
        files = fs.glob(f"{path_without_protocol}/*.parquet")
        return len(files) > 0
    except Exception:
        return False


@dataclass
class FormatPassedConfig:
    """Configuration for formatting passed rows with final output."""
    input_path: str
    output_path: str


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def format_passed_output(config: FormatPassedConfig):
    """
    Format rows that passed all validation checks into final output format.

    Adds user_content and assistant_content columns for training.
    """
    import ray.data
    from experiments.self_instill.prompts import REASONING_INSTRUCTION

    def format_row(row):
        current_sample = row.get("current_sample", "")
        instruction_seed = row.get("instruction_seed", "")
        user_prompt = instruction_seed.strip() + "\n" + REASONING_INSTRUCTION + "\n"
        row["user_content"] = user_prompt
        row["assistant_content"] = current_sample
        row["final_sample_idx"] = row.get("current_sample_idx", 0)
        row["all_passed"] = True
        return row

    try:
        ds = ray.data.read_parquet(config.input_path)
        count = ds.count()
        if count > 0:
            ds = ds.map(format_row)
            ds.write_parquet(config.output_path)
            print(f"Formatted {count} passed rows")
            return count
        else:
            print("No rows to format (empty input)")
            return 0
    except Exception as e:
        print(f"No rows to format: {e}")
        return 0


@dataclass
class CombinePassedConfig:
    """Configuration for combining passed rows from all rounds."""
    input_paths: list  # List of paths to passed rows from each round
    output_path: str


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def combine_passed_rows(config: CombinePassedConfig):
    """Combine passed rows from all rounds into final output."""
    import ray.data

    datasets = []
    for path in config.input_paths:
        try:
            ds = ray.data.read_parquet(path)
            if ds.count() > 0:
                datasets.append(ds)
                print(f"Loaded {ds.count()} rows from {path}")
        except Exception as e:
            print(f"Could not read {path}: {e}")

    if not datasets:
        print("No passed rows from any round!")
        # Write empty dataset
        ray.data.from_items([]).write_parquet(config.output_path)
        return

    # Union all datasets
    combined = datasets[0]
    for ds in datasets[1:]:
        combined = combined.union(ds)

    final_count = combined.count()
    print(f"Combined {final_count} total passed rows from {len(datasets)} rounds")

    combined.write_parquet(config.output_path)


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
        # Use current_sample (from iterative extraction) or fall back to generated_text
        answer = row.get("current_sample") or row.get("generated_text")

        # Fail explicitly if required fields are missing
        if not question or not isinstance(question, str):
            raise ValueError(f"Missing or invalid instruction_seed: {question}")
        if not answer or not isinstance(answer, str):
            raise ValueError(f"Missing or invalid current_sample/generated_text: {answer}")

        question = question.strip()
        answer = answer.strip()

        # Extract final answer (after </think>) for cycle consistency
        # We only want to check if the conclusion addresses the question, not the reasoning
        if "</think>" in answer:
            final_answer = answer.split("</think>")[-1].strip()
        else:
            final_answer = answer

        # Cycle consistency - step 1: generate inferred question (uses final answer only)
        row["cycle_gen_prompt"] = cycle_gen_prompt_template.format(answer=final_answer)

        # Factual error prompt (uses final answer only, not reasoning trace)
        row["factual_prompt"] = factual_prompt_template.format(
            question=question,
            answer=final_answer,
        )

        # Total correctness prompt (uses final answer only, not reasoning trace)
        row["correctness_prompt"] = correctness_prompt_template.format(
            question=question,
            answer=final_answer,
        )

        return row

    ds = ray.data.read_parquet(config.input_path)
    ds = ds.map(add_validation_prompts)
    print(f"Rows prepared for validation: {ds.count()}")
    ds.write_parquet(config.output_path)


# NOTE: prep_validation and cycle_gen_questions ExecutorSteps are created dynamically
# in the main execution loop (see ITERATIVE UQ PROCESSING section below)


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


# NOTE: prep_cycle_compare and cycle_compare ExecutorSteps are created dynamically
# in the main execution loop (see ITERATIVE UQ PROCESSING section below)


# NOTE: factual_check and correctness_check ExecutorSteps are created dynamically
# in the main execution loop (see ITERATIVE UQ PROCESSING section below)


# =============================================================================
# MAIN EXECUTION - ITERATIVE UQ WITH EARLY EXIT
# =============================================================================
# Process samples iteratively with early exit optimization:
# - For each round, try sample N for all pending rows
# - Run validation checks sequentially: cycle → factual → correctness
# - EARLY EXIT: If a row fails cycle, skip factual & correctness (saves compute)
# - EARLY EXIT: If a row fails factual, skip correctness (saves compute)
# - Rows that pass all checks → final output
# - Rows that fail → merge all failures and try next sample in next round
#
# This is more efficient than running all checks on all samples upfront.

if __name__ == "__main__":
    # Stage 1: Download dataset (model downloaded via HF Hub on workers)
    print("Stage 1: Downloading dataset...")
    executor_main([download_ot4_math])

    # Stage 2: Preprocess - extract user message
    print("Stage 2: Preprocessing...")
    executor_main([preprocess_ot4])

    # Stage 3: Generate with multi-sample selection (CACHED)
    print("Stage 3: Generating with multi-sample selection...")
    executor_main([generate_with_selection])

    # Stage 4: Filter and collect valid samples (keep as ordered list, don't explode)
    print("Stage 4: Filtering and collecting valid samples...")
    executor_main([filter_collect])

    # ==========================================================================
    # ITERATIVE UQ PROCESSING
    # ==========================================================================
    # For each round (0-3), process the current sample index for all pending rows
    # Rows that pass move to final output, rows that fail continue to next round

    MAX_ROUNDS = 4  # Maximum number of samples to try (0, 1, 2, 3)
    passed_paths = []  # Collect paths to passed rows from each round

    # Track the input for each round
    # Round 0 input: filter_collect step (executor resolves the path)
    # Round N input: failed rows from round N-1
    current_input = filter_collect  # Start with the filter_collect step

    for round_idx in range(MAX_ROUNDS):
        print(f"\n{'='*60}")
        print(f"TRY {round_idx}: Processing sample index {round_idx}")
        print(f"{'='*60}")

        # Define paths for this round
        round_base = f"{BASE_PATH}/try{round_idx}"

        # Step 1: Extract sample for this round
        print(f"Try {round_idx} - Step 1: Extracting sample...")
        extract_step = ExecutorStep(
            name=f"{round_base}/extract",
            fn=extract_sample_for_round,
            config=ExtractSampleConfig(
                input_path=current_input,  # Step for round 0, string path for round 1+
                output_path=this_output_path(),
                sample_round=round_idx,
            ),
            pip_dependency_groups=["vllm"],
        )
        executor_main([extract_step])

        # Step 2: Prepare validation prompts
        print(f"Try {round_idx} - Step 2: Preparing validation prompts...")
        prep_val_step = ExecutorStep(
            name=f"{round_base}/prep-validation",
            fn=prep_validation_prompts,
            config=PrepValidationConfig(
                input_path=extract_step,  # Use step reference
                output_path=this_output_path(),
                is_instruction_tuned=IS_INSTRUCTION_TUNED,
            ),
            pip_dependency_groups=["vllm"],
        )
        executor_main([prep_val_step])

        # Step 3: Cycle consistency - generate inferred questions
        print(f"Try {round_idx} - Step 3: Generating inferred questions...")
        cycle_gen_step = ExecutorStep(
            name=f"{round_base}/cycle-gen",
            fn=run_generation_inference,
            config=TextGenerationInferenceConfig(
                input_path=prep_val_step,  # Use step reference
                output_path=this_output_path(),
                model_name=QWEN3_4B_HF_ID,
                engine_kwargs={"tensor_parallel_size": TENSOR_PARALLEL_SIZE, "max_model_len": 32768},
                generation_kwargs={"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768, "n": 3},
                template="{example}",
                prompt_column="cycle_gen_prompt",
                apply_chat_template=True,
                save_templated_prompt=False,
                max_doc_tokens=32768,
                num_instances=(1, 256),
                batch_size=BATCH_SIZE,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                preserve_order=False,
                filetype="parquet",
                output_filetype_override="parquet",
                resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),
                generated_text_column_name="inferred_questions",
                checkpoint_id_column="ms_id",
                save_samples_as_list=True,
                max_task_retries=10,
            ),
            pip_dependency_groups=["vllm"],
        )
        executor_main([cycle_gen_step])

        # Step 4: Prepare cycle comparison prompts
        print(f"Try {round_idx} - Step 4: Preparing cycle comparison...")
        prep_compare_step = ExecutorStep(
            name=f"{round_base}/prep-cycle-compare",
            fn=prep_cycle_compare_prompts,
            config=PrepCycleCompareConfig(
                input_path=cycle_gen_step,  # Use step reference
                output_path=this_output_path(),
                is_instruction_tuned=IS_INSTRUCTION_TUNED,
            ),
            pip_dependency_groups=["vllm"],
        )
        executor_main([prep_compare_step])

        # Step 5: Cycle consistency - compare
        print(f"Try {round_idx} - Step 5: Comparing questions...")
        cycle_compare_step = ExecutorStep(
            name=f"{round_base}/cycle-compare",
            fn=run_generation_inference,
            config=TextGenerationInferenceConfig(
                input_path=prep_compare_step,  # Use step reference
                output_path=this_output_path(),
                model_name=QWEN3_4B_HF_ID,
                engine_kwargs={"tensor_parallel_size": TENSOR_PARALLEL_SIZE, "max_model_len": 32768},
                generation_kwargs={"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768, "n": 3},
                template="{example}",
                prompt_column="cycle_compare_prompt",
                apply_chat_template=True,
                save_templated_prompt=False,
                max_doc_tokens=32768,
                num_instances=(1, 256),
                batch_size=BATCH_SIZE,
                tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                preserve_order=False,
                filetype="parquet",
                output_filetype_override="parquet",
                resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),
                generated_text_column_name="cycle_results",
                checkpoint_id_column="ms_id",
                save_samples_as_list=True,
                max_task_retries=10,
            ),
            pip_dependency_groups=["vllm"],
        )
        executor_main([cycle_compare_step])

        # ======================================================================
        # EARLY EXIT: Check cycle results before running factual
        # ======================================================================
        print(f"Try {round_idx} - Step 5b: Checking cycle results (early exit)...")
        cycle_passed_path = f"gs://marin-us-central1/{round_base}/cycle-passed"
        cycle_failed_path = f"gs://marin-us-central1/{round_base}/cycle-failed"

        check_cycle_step = ExecutorStep(
            name=f"{round_base}/check-cycle",
            fn=check_and_filter,
            config=CheckAndFilterConfig(
                input_path=cycle_compare_step,
                passed_output_path=cycle_passed_path,
                failed_output_path=cycle_failed_path,
                result_column="cycle_results",
                check_name="cycle",
            ),
            pip_dependency_groups=["vllm"],
        )
        executor_main([check_cycle_step])

        # Step 6: Factual error check - ONLY on cycle-passed rows
        # Skip if no rows passed cycle check (early exit optimization)
        if check_path_has_data(cycle_passed_path):
            print(f"Try {round_idx} - Step 6: Factual error check (only cycle-passed)...")
            factual_step = ExecutorStep(
                name=f"{round_base}/factual",
                fn=run_generation_inference,
                config=TextGenerationInferenceConfig(
                    input_path=cycle_passed_path,  # Only rows that passed cycle check!
                    output_path=this_output_path(),
                    model_name=QWEN3_4B_HF_ID,
                    engine_kwargs={"tensor_parallel_size": TENSOR_PARALLEL_SIZE, "max_model_len": 32768},
                    generation_kwargs={"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768, "n": 3},
                    template="{example}",
                    prompt_column="factual_prompt",
                    apply_chat_template=True,
                    save_templated_prompt=False,
                    max_doc_tokens=32768,
                    num_instances=(1, 256),
                    batch_size=BATCH_SIZE,
                    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                    preserve_order=False,
                    filetype="parquet",
                    output_filetype_override="parquet",
                    resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),
                    generated_text_column_name="factual_results",
                    checkpoint_id_column="ms_id",
                    save_samples_as_list=True,
                    max_task_retries=10,
                ),
                pip_dependency_groups=["vllm"],
            )
            executor_main([factual_step])
        else:
            print(f"Try {round_idx} - Step 6: SKIPPED (no rows passed cycle check)")

        # ======================================================================
        # EARLY EXIT: Check factual results before running correctness
        # ======================================================================
        factual_passed_path = f"gs://marin-us-central1/{round_base}/factual-passed"
        factual_failed_path = f"gs://marin-us-central1/{round_base}/factual-failed"

        # Only run check_factual if factual_step ran (i.e., cycle_passed had data)
        if check_path_has_data(cycle_passed_path):
            print(f"Try {round_idx} - Step 6b: Checking factual results (early exit)...")
            check_factual_step = ExecutorStep(
                name=f"{round_base}/check-factual",
                fn=check_and_filter,
                config=CheckAndFilterConfig(
                    input_path=factual_step,
                    passed_output_path=factual_passed_path,
                    failed_output_path=factual_failed_path,
                    result_column="factual_results",
                    check_name="factual",
                ),
                pip_dependency_groups=["vllm"],
            )
            executor_main([check_factual_step])
        else:
            print(f"Try {round_idx} - Step 6b: SKIPPED (no factual results to check)")

        # Step 7: Total correctness check - ONLY on factual-passed rows
        # Skip if no rows passed factual check
        if check_path_has_data(factual_passed_path):
            print(f"Try {round_idx} - Step 7: Correctness check (only factual-passed)...")
            correctness_step = ExecutorStep(
                name=f"{round_base}/correctness",
                fn=run_generation_inference,
                config=TextGenerationInferenceConfig(
                    input_path=factual_passed_path,  # Only rows that passed factual check!
                    output_path=this_output_path(),
                    model_name=QWEN3_4B_HF_ID,
                    engine_kwargs={"tensor_parallel_size": TENSOR_PARALLEL_SIZE, "max_model_len": 32768},
                    generation_kwargs={"temperature": 0.6, "top_p": 0.95, "max_tokens": 32768, "n": 3},
                    template="{example}",
                    prompt_column="correctness_prompt",
                    apply_chat_template=True,
                    save_templated_prompt=False,
                    max_doc_tokens=32768,
                    num_instances=(1, 256),
                    batch_size=BATCH_SIZE,
                    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
                    preserve_order=False,
                    filetype="parquet",
                    output_filetype_override="parquet",
                    resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),
                    generated_text_column_name="correctness_results",
                    checkpoint_id_column="ms_id",
                    save_samples_as_list=True,
                    max_task_retries=10,
                ),
                pip_dependency_groups=["vllm"],
            )
            executor_main([correctness_step])
        else:
            print(f"Try {round_idx} - Step 7: SKIPPED (no rows passed factual check)")

        # ======================================================================
        # EARLY EXIT: Check correctness results
        # ======================================================================
        correctness_passed_path = f"gs://marin-us-central1/{round_base}/correctness-passed"
        correctness_failed_path = f"gs://marin-us-central1/{round_base}/correctness-failed"
        passed_path = f"gs://marin-us-central1/{round_base}/passed"

        # Only run check_correctness if correctness_step ran
        if check_path_has_data(factual_passed_path):
            print(f"Try {round_idx} - Step 7b: Checking correctness results...")
            check_correctness_step = ExecutorStep(
                name=f"{round_base}/check-correctness",
                fn=check_and_filter,
                config=CheckAndFilterConfig(
                    input_path=correctness_step,
                    passed_output_path=correctness_passed_path,
                    failed_output_path=correctness_failed_path,
                    result_column="correctness_results",
                    check_name="correctness",
                ),
                pip_dependency_groups=["vllm"],
            )
            executor_main([check_correctness_step])

            # Step 8: Format passed rows with final output columns
            # format_passed_output handles empty input gracefully
            print(f"Try {round_idx} - Step 8: Formatting passed rows...")
            format_passed_step = ExecutorStep(
                name=f"{round_base}/format-passed",
                fn=format_passed_output,
                config=FormatPassedConfig(
                    input_path=correctness_passed_path,
                    output_path=passed_path,
                ),
                pip_dependency_groups=["vllm"],
            )
            executor_main([format_passed_step])
        else:
            print(f"Try {round_idx} - Step 7b & 8: SKIPPED (no correctness results to check)")

        # Track passed path for final combination
        passed_paths.append(passed_path)

        # Step 9: Merge all failed rows for next round
        print(f"Try {round_idx} - Step 9: Merging failed rows for next round...")
        failed_merged_path = f"gs://marin-us-central1/{round_base}/failed-merged"

        merge_failed_step = ExecutorStep(
            name=f"{round_base}/merge-failed",
            fn=merge_all_failed,
            config=MergeFailedConfig(
                cycle_failed_path=cycle_failed_path,
                factual_failed_path=factual_failed_path,
                correctness_failed_path=correctness_failed_path,
                output_path=failed_merged_path,
            ),
            pip_dependency_groups=["vllm"],
        )
        executor_main([merge_failed_step])

        # Update input for next round to be the merged failed rows
        current_input = failed_merged_path

        print(f"Try {round_idx} complete. Passed rows saved to {passed_path}")

    # ==========================================================================
    # COMBINE ALL PASSED ROWS
    # ==========================================================================
    print(f"\n{'='*60}")
    print("COMBINING PASSED ROWS FROM ALL ROUNDS")
    print(f"{'='*60}")

    combine_step = ExecutorStep(
        name=f"{BASE_PATH}/combined-passed",
        fn=combine_passed_rows,
        config=CombinePassedConfig(
            input_paths=passed_paths,
            output_path=this_output_path(),
        ),
        pip_dependency_groups=["vllm"],
    )
    executor_main([combine_step])

    # ==========================================================================
    # UPLOAD TO HUGGINGFACE
    # ==========================================================================
    print("\nUploading to HuggingFace...")
    combined_output = f"gs://marin-us-central1/{BASE_PATH}/combined-passed"

    upload_step = upload_dir_to_hf(
        input_path=combined_output,
        repo_id=f"marin-community/self-instill-ot4-math-qwen3-4b-{ROUND}",
        repo_type="dataset",
    )
    upload_step = replace(upload_step, pip_dependency_groups=["vllm"])
    executor_main([upload_step])

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
