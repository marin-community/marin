"""
This experiment evaluates the quality of Wikipedia data for model cooldown.

It fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% Wikipedia dataset (markdownified using Resiliparse)
- 15% Dolma/FLAN dataset

Reference Issue: https://github.com/stanford-crfm/marin/issues/845
"""

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.exp575_wikipedia_markdownify import wikipedia_resiliparse_custom_fork
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main

# Tokenize the Wikipedia dataset
markdownified_wiki_tokenized = default_tokenize(
    "wikipedia-markdownified",
    wikipedia_resiliparse_custom_fork,
    tokenizer=llama3_tokenizer,
)

# Conduct the cooldown experiment over v4-128 TPU else the v5litepod-128
# TPU is used which is not available in us-central2
cooldown_config = QualityAblationConfig(
    tpu_type="v4-128",
)

# Conduct the cooldown experiment
wikipedia_cooldown_ablation = default_quality_ablation(
    markdownified_wiki_tokenized,
    cooldown_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            markdownified_wiki_tokenized,
            wikipedia_cooldown_ablation,
        ]
    )
