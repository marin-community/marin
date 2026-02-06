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
Shared UQ (Uncertainty Quantification) verification functions for Self-Instill pipeline.

This module contains all the common data processing and validation logic used
across different Self-Instill experiments, including:
- Sample filtering and collection
- Validation prompt preparation
- Decision extraction and checking
- Pass/fail splitting with early exit optimization
- Output formatting and combination

The functions are designed to work with both instruction-tuned models (thinking models)
and base models, with appropriate handling for each.
"""

from dataclasses import dataclass
import re

import ray


# =============================================================================
# REGEX PATTERNS
# =============================================================================

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{[^}]+\}")
DECISION_RE = re.compile(r"\[\[\s*([YN])\s*\]\]", re.IGNORECASE)


# =============================================================================
# STATIC CHECK FUNCTIONS
# =============================================================================


def passes_static_check(text: str, is_instruction_tuned: bool) -> bool:
    """
    Check if text passes static validation.

    For instruction-tuned models:
        - Must have <think>...</think> format
        - Must have \\boxed{} for final answer

    For base models:
        - Must have \\boxed{} for final answer
        - Must not have non-English characters

    Args:
        text: The generated text to check
        is_instruction_tuned: Whether the model is instruction-tuned

    Returns:
        True if the text passes static validation
    """
    if not text or len(text) == 0:
        return False

    has_boxed = bool(BOXED_PATTERN.search(text))

    if is_instruction_tuned:
        has_think_format = bool(THINK_PATTERN.search(text))
        return has_think_format and has_boxed
    else:
        has_non_english = any(ch.isalpha() and ord(ch) > 127 for ch in text)
        return has_boxed and not has_non_english


# =============================================================================
# DECISION EXTRACTION FUNCTIONS
# =============================================================================


def extract_decision_robust(text: str) -> bool:
    """
    Robustly extract Y/N decision from judge output.

    Priority:
      1) Last occurrence of [[Y]] / [[N]]
      2) Last occurrence of a standalone Y/N token
      3) Last occurrence of YES/NO

    Args:
        text: The judge output text

    Returns:
        True for Y/YES, False otherwise.
    """
    if not text:
        return False

    # 1) [[Y]] / [[N]] (prefer the last one)
    matches = list(DECISION_RE.finditer(text))
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
    """
    Check if all samples have Y decision (unanimous voting).

    Args:
        samples: List of judge output strings

    Returns:
        True if all samples have Y decision
    """
    if not samples or not isinstance(samples, list):
        return False
    decisions = [extract_decision_robust(s) for s in samples]
    return all(decisions) if decisions else False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def check_path_has_data(path: str) -> bool:
    """
    Check if a GCS/local path has parquet files with data.

    Args:
        path: The path to check (GCS or local)

    Returns:
        True if the path exists and has parquet files
    """
    import fsspec

    try:
        fs, path_without_protocol = fsspec.core.url_to_fs(path)
        if not fs.exists(path_without_protocol):
            return False
        files = fs.glob(f"{path_without_protocol}/*.parquet")
        return len(files) > 0
    except Exception:
        return False


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================


@dataclass
class FilterCollectConfig:
    """Configuration for filtering and collecting valid samples."""
    input_path: str
    output_path: str
    is_instruction_tuned: bool = True


@dataclass
class ExtractSampleConfig:
    """Configuration for extracting a specific sample for UQ testing."""
    input_path: str
    output_path: str
    sample_round: int  # Which sample to extract (0 = first valid, 1 = second valid, etc.)


@dataclass
class PrepValidationConfig:
    """Configuration for preparing validation prompts."""
    input_path: str
    output_path: str
    is_instruction_tuned: bool = True


@dataclass
class PrepCycleCompareConfig:
    """Configuration for preparing cycle comparison prompts."""
    input_path: str
    output_path: str
    is_instruction_tuned: bool = True


@dataclass
class CheckAndFilterConfig:
    """Configuration for checking one validation result and splitting passed/failed."""
    input_path: str
    passed_output_path: str
    failed_output_path: str
    result_column: str  # "cycle_results", "factual_results", or "correctness_results"
    check_name: str     # "cycle", "factual", "correctness" (for logging)


@dataclass
class CheckUQResultsConfig:
    """Configuration for checking UQ results and splitting passed/failed (non-early-exit)."""
    input_path: str
    passed_output_path: str
    failed_output_path: str


@dataclass
class MergeFailedConfig:
    """Configuration for merging failed rows from all validation stages."""
    cycle_failed_path: str
    factual_failed_path: str
    correctness_failed_path: str
    output_path: str


@dataclass
class FormatPassedConfig:
    """Configuration for formatting passed rows with final output."""
    input_path: str
    output_path: str


@dataclass
class CombinePassedConfig:
    """Configuration for combining passed rows from all rounds."""
    input_paths: list  # List of paths to passed rows from each round
    output_path: str


# =============================================================================
# RAY REMOTE FUNCTIONS
# =============================================================================


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
    import ray.data

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
        from experiments.self_instill.prompts import (
            CYCLE_QUESTION_GENERATION_PROMPT_INSTRUCT,
            FACTUAL_ERROR_PROMPT_INSTRUCT,
            TOTAL_CORRECTNESS_PROMPT_INSTRUCT,
        )
        cycle_gen_prompt_template = CYCLE_QUESTION_GENERATION_PROMPT_INSTRUCT
        factual_prompt_template = FACTUAL_ERROR_PROMPT_INSTRUCT
        correctness_prompt_template = TOTAL_CORRECTNESS_PROMPT_INSTRUCT
    else:
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
        # Drop nested dict columns if present (not supported by PyArrow)
        if "conversations" in row:
            del row["conversations"]
        if "messages" in row:
            del row["messages"]

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


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def prep_cycle_compare_prompts(config: PrepCycleCompareConfig):
    """Create cycle comparison prompts from inferred questions."""
    import ray.data

    if config.is_instruction_tuned:
        from experiments.self_instill.prompts import CYCLE_COMPARISON_PROMPT_INSTRUCT
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


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def check_and_filter(config: CheckAndFilterConfig):
    """
    Check unanimous [[Y]] voting on a single result column and split into passed/failed.

    This enables early exit: if a row fails cycle check, we skip factual and correctness.
    Failed rows are collected for the next round (to try the next sample).
    Passed rows continue to the next validation step.
    """
    import ray.data

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
        except Exception:
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


@ray.remote(num_cpus=0, resources={"head_node": 0.001})
def check_uq_and_split(config: CheckUQResultsConfig):
    """
    Check UQ results and split into passed/failed rows (non-early-exit version).

    A row passes if ALL of:
    - cycle_results: unanimous [[Y]]
    - factual_results: unanimous [[Y]]
    - correctness_results: unanimous [[Y]]

    Passed rows get their final output formatted.
    Failed rows continue to the next round (if more samples available).
    """
    import ray.data
    from experiments.self_instill.prompts import REASONING_INSTRUCTION

    def check_and_format(row):
        cycle_samples = row.get("cycle_results", [])
        factual_samples = row.get("factual_results", [])
        correctness_samples = row.get("correctness_results", [])

        cycle_pass = check_unanimous(cycle_samples)
        factual_pass = check_unanimous(factual_samples)
        correctness_pass = check_unanimous(correctness_samples)

        row["cycle_passed"] = cycle_pass
        row["factual_passed"] = factual_pass
        row["correctness_passed"] = correctness_pass
        row["all_passed"] = cycle_pass and factual_pass and correctness_pass

        if row["all_passed"]:
            # Format final output
            current_sample = row.get("current_sample", "")
            instruction_seed = row.get("instruction_seed", "")
            user_prompt = instruction_seed.strip() + "\n" + REASONING_INSTRUCTION + "\n"
            row["user_content"] = user_prompt
            row["assistant_content"] = current_sample
            row["final_sample_idx"] = row.get("current_sample_idx", 0)

        return row

    ds = ray.data.read_parquet(config.input_path)
    ds = ds.map(check_and_format)

    total = ds.count()
    ds_passed = ds.filter(lambda x: x.get("all_passed", False))
    ds_failed = ds.filter(lambda x: not x.get("all_passed", False))

    passed_count = ds_passed.count()
    failed_count = ds_failed.count()

    print(f"UQ Results: {passed_count} passed, {failed_count} failed out of {total}")

    ds_passed.write_parquet(config.passed_output_path)
    ds_failed.write_parquet(config.failed_output_path)
