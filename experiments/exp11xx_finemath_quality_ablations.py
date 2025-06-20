"""
This experiment evaluates the quality of finemath crawl data for model cooldown using `default_quality_ablation`
which fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% finemath crawl data
- 15% Dolma/FLAN dataset

Reference Issue: https://github.com/stanford-crfm/marin/issues/1167
"""

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import InputName, executor_main
from marin.resources import TpuPodConfig

# Tokenize the finemath crawl dataset
finemath_crawl_tokenized = default_tokenize(
    "finemath-crawled",
    InputName.hardcoded(
        "gs://marin-us-central2/crawl/finemath-3plus/minhash-31a182/finemath-3plus_passing_minhash_against_finemath-3plus/deduplicated_output"
    ),
    tokenizer=llama3_tokenizer,
)

finemath_raw_tokenized = default_tokenize(
    "finemath-control",
    InputName.hardcoded(
        "gs://marin-us-central2/raw/finemath-7090a5/finemath-3plus"
    ),
    tokenizer=llama3_tokenizer,
)

# Conduct the cooldown experiment over v4-128 TPU else the v5litepod-128
# TPU is used which is not available in us-central2
cooldown_config = QualityAblationConfig(
    resources=TpuPodConfig(tpu_type="v4-128"),
)

# Conduct the cooldown experiment
finemath_crawl_ablation = default_quality_ablation(
    finemath_crawl_tokenized,
    cooldown_config,
)

finemath_raw_ablation = default_quality_ablation(
    finemath_raw_tokenized,
    cooldown_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            finemath_crawl_tokenized,
            finemath_crawl_ablation,
            finemath_raw_tokenized,
            finemath_raw_ablation,
        ],
    )
