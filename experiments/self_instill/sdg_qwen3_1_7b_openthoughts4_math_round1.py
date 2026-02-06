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
Self-Instill: OpenThoughts4 Math Experiment (Qwen3-1.7B Instruction-Tuned)

This experiment generates high-quality synthetic reasoning data from the
marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-32768-tokens dataset
using the self-instill pipeline.

NOTE: This is for INSTRUCTION-TUNED models (Qwen3-1.7B), which output in the format:
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
    python experiments/self_instill/sdg_qwen3_1_7b_openthoughts4_math_round1.py
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
from experiments.self_instill.uq_verification import (
    FilterCollectConfig,
    filter_collect_valid_samples,
    ExtractSampleConfig,
    extract_sample_for_round,
    PrepValidationConfig,
    prep_validation_prompts,
    PrepCycleCompareConfig,
    prep_cycle_compare_prompts,
    CheckAndFilterConfig,
    check_and_filter,
    MergeFailedConfig,
    merge_all_failed,
    FormatPassedConfig,
    format_passed_output,
    CombinePassedConfig,
    combine_passed_rows,
    check_path_has_data,
)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
QWEN3_1_7B_HF_ID = "Qwen/Qwen3-1.7B"
RESOURCE_TYPE = "v5p-8"
TENSOR_PARALLEL_SIZE = 4
BATCH_SIZE = 16
IS_INSTRUCTION_TUNED = True

# =============================================================================
# ROUND CONFIGURATION
# =============================================================================
ROUND = "round1"
BASE_PATH = f"documents/self-instill/qwen3-1_7b-openthoughts4/{ROUND}"

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
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

generate_with_selection = ExecutorStep(
    name=f"{BASE_PATH}/generated",
    fn=run_generation_inference,
    config=TextGenerationInferenceConfig(
        input_path=preprocess_ot4,
        output_path=this_output_path(),
        model_name=QWEN3_1_7B_HF_ID,
        engine_kwargs={
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "max_model_len": 32768,
        },
        generation_kwargs={
            "temperature": 0.8,
            "top_p": 0.95,
            "max_tokens": 32768,
            "n": 4,
        },
        template="{example}\n" + REASONING_INSTRUCTION + "\n",
        prompt_column="instruction_seed",
        apply_chat_template=True,
        save_templated_prompt=False,
        max_doc_tokens=32768,
        num_instances=(1, 128),
        batch_size=BATCH_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        preserve_order=False,
        filetype="parquet",
        output_filetype_override="parquet",
        resource_config=ResourceConfig.with_tpu(RESOURCE_TYPE),
        generated_text_column_name="generated_text",
        checkpoint_id_column="ms_id",
        enable_multi_sample_selection=True,
        save_all_samples=True,
        all_samples_column_name="all_samples",
        static_check_type="boxed_only",
        selection_strategy="first",
        max_task_retries=10,
    ),
    pip_dependency_groups=["vllm"],
)


# =============================================================================
# STEP 1.5: FILTER - Collect valid samples as ordered list
# =============================================================================

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
# MAIN EXECUTION - ITERATIVE UQ WITH EARLY EXIT
# =============================================================================

if __name__ == "__main__":
    # Stage 1: Download dataset
    print("Stage 1: Downloading dataset...")
    executor_main([download_ot4_math])

    # Stage 2: Preprocess - extract user message
    print("Stage 2: Preprocessing...")
    executor_main([preprocess_ot4])

    # Stage 3: Generate with multi-sample selection
    print("Stage 3: Generating with multi-sample selection...")
    executor_main([generate_with_selection])

    # Stage 4: Filter and collect valid samples
    print("Stage 4: Filtering and collecting valid samples...")
    executor_main([filter_collect])

    # ==========================================================================
    # ITERATIVE UQ PROCESSING
    # ==========================================================================
    MAX_ROUNDS = 4
    passed_paths = []
    current_input = filter_collect

    for round_idx in range(MAX_ROUNDS):
        print(f"\n{'='*60}")
        print(f"TRY {round_idx}: Processing sample index {round_idx}")
        print(f"{'='*60}")

        round_base = f"{BASE_PATH}/try{round_idx}"

        # Step 1: Extract sample for this round
        print(f"Try {round_idx} - Step 1: Extracting sample...")
        extract_step = ExecutorStep(
            name=f"{round_base}/extract",
            fn=extract_sample_for_round,
            config=ExtractSampleConfig(
                input_path=current_input,
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
                input_path=extract_step,
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
                input_path=prep_val_step,
                output_path=this_output_path(),
                model_name=QWEN3_1_7B_HF_ID,
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
                input_path=cycle_gen_step,
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
                input_path=prep_compare_step,
                output_path=this_output_path(),
                model_name=QWEN3_1_7B_HF_ID,
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

        # EARLY EXIT: Check cycle results before running factual
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
        factual_passed_path = f"gs://marin-us-central1/{round_base}/factual-passed"
        factual_failed_path = f"gs://marin-us-central1/{round_base}/factual-failed"

        if check_path_has_data(cycle_passed_path):
            print(f"Try {round_idx} - Step 6: Factual error check (only cycle-passed)...")
            factual_step = ExecutorStep(
                name=f"{round_base}/factual",
                fn=run_generation_inference,
                config=TextGenerationInferenceConfig(
                    input_path=cycle_passed_path,
                    output_path=this_output_path(),
                    model_name=QWEN3_1_7B_HF_ID,
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

            # EARLY EXIT: Check factual results before running correctness
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
            print(f"Try {round_idx} - Step 6: SKIPPED (no rows passed cycle check)")

        # Step 7: Total correctness check - ONLY on factual-passed rows
        correctness_passed_path = f"gs://marin-us-central1/{round_base}/correctness-passed"
        correctness_failed_path = f"gs://marin-us-central1/{round_base}/correctness-failed"
        passed_path = f"gs://marin-us-central1/{round_base}/passed"

        if check_path_has_data(factual_passed_path):
            print(f"Try {round_idx} - Step 7: Correctness check (only factual-passed)...")
            correctness_step = ExecutorStep(
                name=f"{round_base}/correctness",
                fn=run_generation_inference,
                config=TextGenerationInferenceConfig(
                    input_path=factual_passed_path,
                    output_path=this_output_path(),
                    model_name=QWEN3_1_7B_HF_ID,
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

            # Check correctness results
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

            # Step 8: Format passed rows
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
            print(f"Try {round_idx} - Step 7 & 8: SKIPPED (no rows passed factual check)")

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

        # Update input for next round
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
        repo_id=f"marin-community/self-instill-ot4-math-qwen3-1_7b-{ROUND}",
        repo_type="dataset",
    )
    upload_step = replace(upload_step, pip_dependency_groups=["vllm"])
    executor_main([upload_step])

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
