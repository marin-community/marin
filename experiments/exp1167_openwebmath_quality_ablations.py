"""
This experiment evaluates the quality of openwebmath crawl data for model cooldown using `default_quality_ablation`
which fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% openwebmath crawl data
- 15% Dolma/FLAN dataset

Reference Issue: https://github.com/stanford-crfm/marin/issues/1167
"""

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.resources import TpuPodConfig

executor_main_config = ExecutorMainConfig(force_run_failed=True)
# Tokenize the openwebmath dataset
openwebmath_crawl_tokenized = default_tokenize(
    "openwebmath-crawled",
    "gs://marin-us-central2/scratch/nfliu/text/open_web_math_100M_passing_minhash_against_open_web_math",
    tokenizer=llama3_tokenizer,
)

# Conduct the cooldown experiment over v4-128 TPU else the v5litepod-128
# TPU is used which is not available in us-central2
cooldown_config = QualityAblationConfig(
    resources=TpuPodConfig(tpu_type="v4-128"),
)

# Conduct the cooldown experiment
openwebmath_crawl_ablation = default_quality_ablation(
    openwebmath_crawl_tokenized,
    cooldown_config,
)

if __name__ == "__main__":
    executor_main(
        executor_main_config,
        steps=[
            openwebmath_crawl_tokenized,
            openwebmath_crawl_ablation,
        ],
    )
