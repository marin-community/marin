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
This experiment evaluates the quality of finemath crawl data for model cooldown using `default_quality_ablation`
which fine-tunes an 8B model on a mixture of:
- 70% DCLM baseline data
- 15% finemath crawl data
- 15% Dolma/FLAN dataset

Reference Issue: https://github.com/stanford-crfm/marin/issues/1167
"""

from experiments.anneal_config import AnnealConfig
from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.defaults import default_anneal, default_tokenize
from experiments.dolma.tokenize_dolma import tokenize_dolma_steps
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import MMLU_TASKS
from experiments.llama import llama3_tokenizer
from marin.execution.executor import InputName, executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
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
    InputName.hardcoded("gs://marin-us-central2/raw/finemath-7090a5/finemath-3plus"),
    tokenizer=llama3_tokenizer,
)

# Conduct the cooldown experiment over v4-128 TPU else the v5litepod-128
# TPU is used which is not available in us-central2
cooldown_config = QualityAblationConfig(
    resources=TpuPodConfig(tpu_type="v4-128"),
    permutation_type="linear",
)

finemath_crawl_ablation = default_quality_ablation(
    finemath_crawl_tokenized,
    cooldown_config,
)

finemath_raw_ablation = default_quality_ablation(
    finemath_raw_tokenized,
    cooldown_config,
)


baseline_component = dclm_components_llama3["dclm_baseline"]
mcq_component = tokenize_dolma_steps()["dolma/flan"]

baseline_weight = 0.7
mcq_weight = 0.15
candidate_weight_1 = 0.015
candidate_weight_2 = 0.135


crawl_control_lm_mixture_data_config = lm_mixture_data_config(
    components={
        "baseline": baseline_component,
        "mcq": mcq_component,
        "candidate_1": finemath_crawl_tokenized,
        "candidate_2": finemath_raw_tokenized,
    },
    weights={
        "baseline": baseline_weight,
        "mcq": mcq_weight,
        "candidate_1": candidate_weight_1,
        "candidate_2": candidate_weight_2,
    },
    permutation_type="linear",
)

crawl_control_lm_mixture_anneal_config = AnnealConfig(
    dataset_config=crawl_control_lm_mixture_data_config,
    resources=TpuPodConfig(tpu_type="v4-256"),
)

crawl_control_ablation = default_anneal(
    name="8b-quality-eval-finemath-crawl-control-token-proportional",
    anneal_config=crawl_control_lm_mixture_anneal_config,
)

crawl_control_ablation_eval = default_eval(
    step=crawl_control_ablation,
    evals=MMLU_TASKS,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            finemath_crawl_tokenized,
            finemath_crawl_ablation,
            finemath_raw_tokenized,
            finemath_raw_ablation,
            crawl_control_ablation,
            crawl_control_ablation_eval,
        ],
    )
