#!/usr/bin/env python3
"""
Experiment to tokenize FineWeb-Edu with GPT-OSS tokenizer and create a 10B token subcache.
This creates a GPT-OSS tokenized version equivalent to the Marin tokenizer subcache.
"""

from experiments.defaults import default_tokenize
from experiments.pretraining_datasets import fineweb_edu
from marin.execution.executor import executor_main
from marin.tokenize.slice_cache import slice_cache
from marin.processing.tokenize import step_to_lm_mixture_component

# Use the GPT-OSS tokenizer
GPT_OSS_TOKENIZER = "openai/gpt-oss-20b"

# Step 1: Tokenize the full FineWeb-Edu with GPT-OSS tokenizer
# This uses the original raw data (has 'text' field) instead of pretokenized cache
fineweb_edu_gpt_oss_full_tokenized = default_tokenize(
    name="fineweb_edu_gpt_oss_full",
    dataset=fineweb_edu,  # ← Uses original raw data with 'text' field
    tokenizer=GPT_OSS_TOKENIZER,
    # Note: FineWeb-Edu is text data, so we use default TextLmDatasetFormat (no chat format needed)
)

# Step 2: Create a 10B token subcache from the GPT-OSS tokenized data
# This mimics what build_prebuilt_caches.py does for the Marin tokenizer
fineweb_edu_gpt_oss_subcache_10B = slice_cache(
    output_path="tokenized/subcache/fineweb-edu-gpt-oss-10B",
    input_config=step_to_lm_mixture_component(fineweb_edu_gpt_oss_full_tokenized, include_raw_paths=True),
    num_tokens=10_000_000_000,  # 10B tokens
)

if __name__ == "__main__":
    # Run both steps: full tokenization then create 10B subcache
    executor_main(steps=[fineweb_edu_gpt_oss_subcache_10B])