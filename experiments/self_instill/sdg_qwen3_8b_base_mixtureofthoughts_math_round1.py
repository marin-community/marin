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
Self-Instill: Mixture-of-Thoughts Math Experiment

This experiment generates high-quality synthetic reasoning data from the
open-r1/Mixture-of-Thoughts dataset using the self-instill pipeline.

Pipeline:
1. Preprocess: Extract user message from messages[0]["content"]
2. Generate 4 samples per prompt, select longest that passes static check
3. Summarize selected samples
4. Validate via LLM:
   - Cycle consistency: Infer question from answer, compare to original
   - Factual error: Check for math/logic errors
   - Total correctness: Verify complete and correct solution
5. Format final output

Usage:
    python experiments/self_instill/sdg_qwen3_8b_base_mixtureofthoughts_math_round1.py
"""

from dataclasses import dataclass, replace

import ray

from fray.cluster import ResourceConfig

from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.export.hf_upload import upload_dir_to_hf
from marin.generation.inference import TextGenerationInferenceConfig
from marin.generation.inference import run_inference as run_generation_inference

# Model HuggingFace ID (using HF Hub directly instead of local gcsfuse path)
QWEN3_8B_BASE_HF_ID = "Qwen/Qwen3-8B-Base"

# Import prompts (only import what's used at module level)
from experiments.self_instill.prompts import REASONING_LONG_INSTRUCTION

# =============================================================================
# ROUND CONFIGURATION
# =============================================================================
# Change this to run different rounds (e.g., "round1", "round2", etc.)
ROUND = "round1"
BASE_PATH = f"documents/self-instill/qwen3-8b/{ROUND}"

# Summarization template with {example} placeholder for marin compatibility
SUMMARIZATION_TEMPLATE = """Summarize the solution as a clear explanation (like the final write-up), not a one-liner.
Include the key reasoning steps that justify the result (setup -> method -> crucial computations -> conclusion).
Paraphrase, don't copy sentences from the original.
End with the final answer within \\boxed{{...}}

Original solution:
{example}

Explanation:
"""


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Define dataset config directly to avoid import issues
MOT_MATH_HF_ID = "open-r1/Mixture-of-Thoughts"
MOT_MATH_REVISION = "e55fa28"

download_mot_math = ExecutorStep(
    name="raw/open-r1/Mixture-of-Thoughts-math",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=MOT_MATH_HF_ID,
        revision=versioned(MOT_MATH_REVISION),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path=f"raw/open-r1/Mixture-of-Thoughts-math-{MOT_MATH_REVISION}",
    pip_dependency_groups=["vllm"],
).cd("math")


# =============================================================================
# STEP 0: PREPROCESS - Extract user message from messages format
# =============================================================================


@dataclass
class PreprocessConfig:
    """Configuration for preprocessing MoT data."""
    input_path: str
    output_path: str


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def preprocess_mot_data(config: PreprocessConfig):
    """Extract user message from messages[0]['content'] and add instruction_seed column."""
    import hashlib
    import ray.data

    def extract_user_message(row):
        """Extract user message and create instruction_seed column."""
        messages = row.get("messages", [])
        if messages and len(messages) > 0:
            user_msg = messages[0].get("content", "")
        else:
            user_msg = ""

        # Create a unique ID based on the user message
        ms_id = hashlib.md5(user_msg.encode()).hexdigest()[:16]

        row["instruction_seed"] = user_msg
        row["ms_id"] = ms_id
        return row

    # Read and transform
    ds = ray.data.read_parquet(config.input_path)
    ds = ds.map(extract_user_message)

    # Filter out empty messages
    ds = ds.filter(lambda x: x["instruction_seed"] and len(x["instruction_seed"]) > 0)

    print(f"Preprocessed rows: {ds.count()}")

    # Write output
    ds.write_parquet(config.output_path)


preprocess_mot = ExecutorStep(
    name=f"{BASE_PATH}/preprocessed",
    fn=preprocess_mot_data,
    config=PreprocessConfig(
        input_path=download_mot_math,
        output_path=this_output_path(),
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 1: GENERATION WITH MULTI-SAMPLE SELECTION
# =============================================================================
# Generate 4 samples per prompt, select longest that passes static validation
# (requires \boxed{}, no non-English characters)

generate_with_selection = ExecutorStep(
    name=f"{BASE_PATH}/generated",
    fn=run_generation_inference,
    config=TextGenerationInferenceConfig(
        # IO
        input_path=preprocess_mot,
        output_path=this_output_path(),

        # Model
        model_name=QWEN3_8B_BASE_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768 + 2048,
        },
        generation_kwargs={
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 30000,
            "n": 4,  # Generate 4 samples per prompt
        },

        # Prompting - use instruction_seed column with reasoning instruction
        template="{example}\n" + REASONING_LONG_INSTRUCTION + "\n",
        prompt_column="instruction_seed",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 32),
        batch_size=16,  # Smaller batch since generating 4 samples each
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu("v5p-8"),

        # Output column
        generated_text_column_name="generated_text",

        # Checkpointing
        checkpoint_id_column="ms_id",

        # Multi-sample selection
        enable_multi_sample_selection=True,
        save_all_samples=True,  # Save for debugging
        all_samples_column_name="all_samples",
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 1.5: FILTER - Remove rows where no sample passed static check
# =============================================================================


@dataclass
class FilterConfig:
    """Configuration for filtering rows."""
    input_path: str
    output_path: str


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def filter_valid_generations(config: FilterConfig):
    """Filter out rows where generated_text is None (no sample passed static check)."""
    import ray.data

    ds = ray.data.read_parquet(config.input_path)

    # Filter to only rows with valid generated text
    ds = ds.filter(lambda x: x.get("generated_text") is not None and len(x.get("generated_text", "")) > 0)

    print(f"Rows after filtering: {ds.count()}")

    ds.write_parquet(config.output_path)


filter_valid = ExecutorStep(
    name=f"{BASE_PATH}/filtered",
    fn=filter_valid_generations,
    config=FilterConfig(
        input_path=generate_with_selection,
        output_path=this_output_path(),
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 2: SUMMARIZATION
# =============================================================================
# Summarize the selected samples

summarize_selected = ExecutorStep(
    name=f"{BASE_PATH}/summarized",
    fn=run_generation_inference,
    config=TextGenerationInferenceConfig(
        # IO
        input_path=filter_valid,
        output_path=this_output_path(),

        # Model
        model_name=QWEN3_8B_BASE_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768 + 2048,
        },
        generation_kwargs={
            "temperature": 0.4,
            "top_p": 0.95,
            "max_tokens": 10000,
        },

        # Prompting - use summarization template on generated text
        template=SUMMARIZATION_TEMPLATE,
        prompt_column="generated_text",  # Summarize the generated reasoning
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 32),
        batch_size=32,
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu("v5p-8"),

        # Output column
        generated_text_column_name="summary",

        # Checkpointing
        checkpoint_id_column="ms_id",
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 3: PREPARE VALIDATION PROMPTS
# =============================================================================
# Create columns with formatted validation prompts for each validation type


@dataclass
class PrepValidationConfig:
    """Configuration for preparing validation prompts."""
    input_path: str
    output_path: str


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def prep_validation_prompts(config: PrepValidationConfig):
    """Create validation prompt columns combining question and answer."""
    import ray.data
    from experiments.self_instill.prompts import (
        CYCLE_QUESTION_GENERATION_PROMPT,
        FACTUAL_ERROR_PROMPT,
        TOTAL_CORRECTNESS_PROMPT,
    )

    def add_validation_prompts(row):
        """Add validation prompt columns."""
        question = row.get("instruction_seed", "").strip()
        answer = row.get("generated_text", "").strip()

        # Cycle consistency - step 1: generate inferred question
        row["cycle_gen_prompt"] = CYCLE_QUESTION_GENERATION_PROMPT.format(answer=answer)

        # Factual error prompt
        row["factual_prompt"] = FACTUAL_ERROR_PROMPT.format(
            question=question,
            answer=answer,
        )

        # Total correctness prompt
        row["correctness_prompt"] = TOTAL_CORRECTNESS_PROMPT.format(
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
        input_path=summarize_selected,
        output_path=this_output_path(),
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
        model_name=QWEN3_8B_BASE_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768 + 2048,
        },
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 200,
            "n": 3,  # Generate 3 inferred questions for voting
        },

        # Prompting
        template="{example}",
        prompt_column="cycle_gen_prompt",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 32),
        batch_size=64,
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu("v5p-8"),

        # Output column - save all 3 inferred questions as list
        generated_text_column_name="inferred_questions",

        # Checkpointing
        checkpoint_id_column="ms_id",

        # Save samples as list for later comparison
        save_samples_as_list=True,
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


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def prep_cycle_compare_prompts(config: PrepCycleCompareConfig):
    """Create cycle comparison prompts from inferred questions."""
    import ray.data
    from experiments.self_instill.prompts import CYCLE_COMPARISON_PROMPT

    def add_compare_prompts(row):
        """Add cycle comparison prompt columns for each inferred question."""
        original_question = row.get("instruction_seed", "").strip()

        # inferred_questions is a list of 3 samples
        inferred_questions = row.get("inferred_questions", [])
        if not isinstance(inferred_questions, list):
            inferred_questions = [inferred_questions] if inferred_questions else []

        # Create comparison prompts for each inferred question
        compare_prompts = []
        cleaned_questions = []
        for inferred_q in inferred_questions:
            # Clean up - take first line only
            inferred_q_clean = inferred_q.strip().split('\n')[0].strip() if inferred_q else ""
            cleaned_questions.append(inferred_q_clean)

            # Create comparison prompt
            prompt = CYCLE_COMPARISON_PROMPT.format(
                original_question=original_question,
                inferred_question=inferred_q_clean,
            )
            compare_prompts.append(prompt)

        row["cycle_compare_prompts"] = compare_prompts
        row["inferred_questions_cleaned"] = cleaned_questions

        # For the inference step, we'll use the first prompt
        # (we'll run with n=1 for each of the 3 prompts via a different approach)
        # For now, just use the first one and rely on n=3 for voting
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
        model_name=QWEN3_8B_BASE_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768 + 2048,
        },
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 50,
            "n": 3,  # 3 samples for unanimous voting
        },

        # Prompting
        template="{example}",
        prompt_column="cycle_compare_prompt",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 32),
        batch_size=128,
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu("v5p-8"),

        # Output column - save as list for unanimous voting
        generated_text_column_name="cycle_results",

        # Checkpointing
        checkpoint_id_column="ms_id",

        # Save samples as list for unanimous voting
        save_samples_as_list=True,
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
        model_name=QWEN3_8B_BASE_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768 + 2048,
        },
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 50,
            "n": 3,  # 3 samples for unanimous voting
        },

        # Prompting
        template="{example}",
        prompt_column="factual_prompt",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 32),
        batch_size=128,
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu("v5p-8"),

        # Output column - save as list for unanimous voting
        generated_text_column_name="factual_results",

        # Checkpointing
        checkpoint_id_column="ms_id",

        # Save samples as list for unanimous voting
        save_samples_as_list=True,
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
        model_name=QWEN3_8B_BASE_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 32768 + 2048,
        },
        generation_kwargs={
            "temperature": 0.6,
            "top_p": 0.95,
            "max_tokens": 50,
            "n": 3,  # 3 samples for unanimous voting
        },

        # Prompting
        template="{example}",
        prompt_column="correctness_prompt",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,

        # Ray Data
        num_instances=(1, 32),
        batch_size=128,
        tensor_parallel_size=1,
        preserve_order=False,

        # File
        filetype="parquet",
        output_filetype_override="parquet",

        # Hardware
        resource_config=ResourceConfig.with_tpu("v5p-8"),

        # Output column - save as list for unanimous voting
        generated_text_column_name="correctness_results",

        # Checkpointing
        checkpoint_id_column="ms_id",

        # Save samples as list for unanimous voting
        save_samples_as_list=True,
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


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def filter_and_format_output(config: FilterValidationConfig):
    """Filter based on unanimous validation and format final output."""
    import re
    import ray.data
    from experiments.self_instill.prompts import REASONING_INSTRUCTION

    # Decision extraction pattern
    decision_re = re.compile(r"\[\[\s*([YN])\s*\]\]", re.IGNORECASE)

    def extract_decision_local(text: str) -> bool:
        """Extract Y/N decision from LLM output."""
        if not text:
            return False
        matches = list(decision_re.finditer(text))
        if matches:
            return matches[-1].group(1).upper() == "Y"
        # Fallback to standalone Y/N
        yn_tokens = re.findall(r"(?<![A-Za-z])([YN])(?![A-Za-z])", text.strip(), flags=re.IGNORECASE)
        if yn_tokens:
            return yn_tokens[-1].upper() == "Y"
        return False

    def check_unanimous(samples: list) -> bool:
        """Check if all samples have Y decision (unanimous voting)."""
        if not samples or not isinstance(samples, list):
            return False
        decisions = [extract_decision_local(s) for s in samples]
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
            generated_text = row.get("generated_text", "")
            summary = row.get("summary", "")
            instruction_seed = row.get("instruction_seed", "")

            # Format as <think>...</think> + summary
            output_text = f"<think>\n{generated_text}\n</think>\n\n{summary}"

            # Create conversation format
            user_prompt = instruction_seed.strip() + "\n" + REASONING_INSTRUCTION + "\n"
            row["messages"] = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": output_text},
            ]
        else:
            row["messages"] = None

        return row

    ds = ray.data.read_parquet(config.input_path)
    ds = ds.map(check_validation_and_format)

    # Log stats before filtering
    total = ds.count()

    # Filter to only validated rows
    ds_valid = ds.filter(lambda x: x.get("all_validation_passed", False))
    valid_count = ds_valid.count()

    print(f"Validation results: {valid_count}/{total} rows passed all validation ({100*valid_count/total:.1f}%)")

    ds_valid.write_parquet(config.output_path)


filter_validation = ExecutorStep(
    name=f"{BASE_PATH}/validated",
    fn=filter_and_format_output,
    config=FilterValidationConfig(
        input_path=correctness_check,
        output_path=this_output_path(),
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# UPLOAD STEP
# =============================================================================

upload_self_instill = upload_dir_to_hf(
    input_path=filter_validation,
    repo_id=f"marin-community/self-instill-mot-math-qwen3-8b-base-{ROUND}",
    repo_type="dataset",
)

upload_self_instill = replace(upload_self_instill, pip_dependency_groups=["vllm"])


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Stage 1: Download dataset (model downloaded via HF Hub on workers)
    print("Stage 1: Downloading dataset...")
    executor_main([download_mot_math])

    # Stage 2: Preprocess - extract user message
    print("Stage 2: Preprocessing...")
    executor_main([preprocess_mot])

    # Stage 3: Generate with multi-sample selection
    print("Stage 3: Generating with multi-sample selection...")
    executor_main([generate_with_selection])

    # Stage 4: Filter valid generations
    print("Stage 4: Filtering valid generations...")
    executor_main([filter_valid])

    # Stage 5: Summarize
    print("Stage 5: Summarizing...")
    executor_main([summarize_selected])

    # Stage 6: Prepare validation prompts
    print("Stage 6: Preparing validation prompts...")
    executor_main([prep_validation])

    # Stage 7: Cycle consistency - generate inferred questions
    print("Stage 7: Cycle consistency - generating inferred questions...")
    executor_main([cycle_gen_questions])

    # Stage 8: Cycle consistency - prepare comparison
    print("Stage 8: Cycle consistency - preparing comparison...")
    executor_main([prep_cycle_compare])

    # Stage 9: Cycle consistency - compare
    print("Stage 9: Cycle consistency - comparing...")
    executor_main([cycle_compare])

    # Stage 10: Factual error check
    print("Stage 10: Factual error check...")
    executor_main([factual_check])

    # Stage 11: Total correctness check
    print("Stage 11: Total correctness check...")
    executor_main([correctness_check])

    # Stage 12: Filter validation results and format output
    print("Stage 12: Filtering validation results and formatting output...")
    executor_main([filter_validation])

    # Stage 13: Upload to HuggingFace
    print("Stage 13: Uploading to HuggingFace...")
    executor_main([upload_self_instill])

    print("Pipeline complete!")
