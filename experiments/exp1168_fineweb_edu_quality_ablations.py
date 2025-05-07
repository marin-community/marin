"""
This experiment evaluates the quality of fineweb-edu crawl data for model cooldown using `default_quality_ablation`
which fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% fineweb-edu crawl data (markdownified using Resiliparse)
- 15% Dolma/FLAN dataset

Reference Issue: https://github.com/stanford-crfm/marin/issues/1168
"""

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorMainConfig, executor_main

executor_main_config = ExecutorMainConfig(force_run_failed=True)
# Tokenize the fineweb-edu dataset
fineweb_edu_tokenized = default_tokenize(
    "fineweb-edu-crawled",
    "gs://marin-us-central2/scratch/nfliu/text/fineweb_edu_unique_100M_passing_minhash_against_fineweb_edu",
    tokenizer=llama3_tokenizer,
)

# Conduct the cooldown experiment over v4-128 TPU else the v5litepod-128
# TPU is used which is not available in us-central2
cooldown_config = QualityAblationConfig(
    tpu_type="v4-128",
)

# Conduct the cooldown experiment
fineweb_edu_ablation = default_quality_ablation(
    fineweb_edu_tokenized,
    cooldown_config,
)

if __name__ == "__main__":
    executor_main(
        executor_main_config,
        steps=[
            fineweb_edu_tokenized,
            fineweb_edu_ablation,
        ],
    )
