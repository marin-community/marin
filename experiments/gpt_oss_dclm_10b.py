#!/usr/bin/env python3
"""
Experiment to tokenize kothasuhas/dclm_10B_tokens with GPT-OSS tokenizer.
This creates a GPT-OSS tokenized version of the 10B token DCLM dataset with proper train/validation splits.
"""

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize import tokenize
from marin.processing.tokenize.tokenize import HfTokenizeConfig

# Use the GPT-OSS tokenizer
GPT_OSS_TOKENIZER = "openai/gpt-oss-20b"

# Tokenize directly from HuggingFace dataset with proper train/validation split handling
dclm_10b_gpt_oss_tokenized = ExecutorStep(
    name="tokenized/dclm_10B_tokens_gpt_oss",
    fn=tokenize,
    config=HfTokenizeConfig(
        id="kothasuhas/dclm_10B_tokens",
        cache_path=this_output_path(),
        tokenizer=GPT_OSS_TOKENIZER,
        # HfTokenizeConfig automatically handles train and validation splits from the HF dataset
    ),
    pip_dependency_groups=["tokenize_train"],
)

if __name__ == "__main__":
    # Run tokenization step (which depends on download step)
    executor_main(steps=[dclm_10b_gpt_oss_tokenized])