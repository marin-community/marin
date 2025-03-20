"""
This experiment evaluates the quality of arXiv data for model cooldown using `default_quality_ablation`
which fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% arXiv dataset (markdownified using Resiliparse)
- 15% Dolma/FLAN dataset

Reference Issue: https://github.com/stanford-crfm/marin/issues/846
"""

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.exp579_ar5iv_markdownify import ar5iv_no_problem_resiliparse_custom_fork
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main

# Tokenize the arXiv dataset
markdownified_arxiv_tokenized = default_tokenize(
    "ar5iv-no-problem-markdownified",
    ar5iv_no_problem_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
)

# Conduct the cooldown experiment over v4-128 TPU else the v5litepod-128
# TPU is used which is not available in us-central2
cooldown_config = QualityAblationConfig(
    tpu_type="v4-128",
)

# Conduct the cooldown experiment
ar5iv_cooldown_ablation = default_quality_ablation(
    markdownified_arxiv_tokenized,
    cooldown_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            markdownified_arxiv_tokenized,
            ar5iv_cooldown_ablation,
        ]
    )
