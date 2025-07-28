"""Filter C4 training data based on LZ4 compression ratios.

Filter C4 documents based on their LZ4 compression ratios (compressed_size/original_size),
keeping only those with ratios between 0.65 and 0.8.

Dataset: C4 (allenai/c4, en subset) from Dolma v1.7
Model: llama_1_4b

Pipeline: compression ratio calculation -> filtering -> tokenization -> training

Usage:
    python marin/run/ray_run.py --env_vars WANDB_API_KEY <your_key> -- python experiments/exp_c4_compression_filtering.py

    For forced re-run of specific steps:
    python marin/run/ray_run.py --env_vars WANDB_API_KEY <your_key> -- \
     python experiments/exp_c4_compression_filtering.py --force_run_failed

Note:
    - C4 data is sourced from the pre-downloaded Dolma v1.7 dataset
    - Compression filtering uses LZ4 with thresholds: 0.65 ≤ ratio ≤ 0.8
    - Final model will be saved to: gs://marin-us-central2/checkpoints/compression_filtering/c4-compression-ratio-filter-065-08/
"""

import logging

from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import BASE_DIR_DOLMA, tokenize_dolma_steps
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.core.runtime import TaskConfig
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.classification.config.inference_config import RuntimeConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.inference import InferenceConfig, run_inference
from marin.processing.tokenize import lm_mixture_data_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")

EXPERIMENT_NAME = "c4-compression-ratio-filter-065-08"

input_data_source = "dolma-c4"
input_data_path = BASE_DIR_DOLMA / "c4*.json.gz"

# Calculate compression ratios
compression_step = ExecutorStep(
    name=f"attributes/compression_filtering/{EXPERIMENT_NAME}/{input_data_source}",
    fn=run_inference,
    config=InferenceConfig(
        input_path=input_data_path,
        output_path=this_output_path(),
        model_type="compression",  # Use our compression classifier
        model_name=None,  # This doesn't matter for compression
        attribute_name=versioned("compression_ratio"),
        runtime=RuntimeConfig(
            memory_limit_gb=200,  # this is insane, but I don't want to figure out why it's happening
        ),
        task=TaskConfig(max_in_flight=500),
        filetype=None,
    ),
    pip_dependency_groups=["lz4", "datasets", "filelock"],
)

# Filter based on compression ratios (0.65-0.8 range)
consolidate_step = ExecutorStep(
    name=f"documents/compression_filtering/{EXPERIMENT_NAME}/{input_data_source}",
    fn=consolidate,
    config=ConsolidateConfig(
        input_path=input_data_path,
        output_path=this_output_path(),
        filters=[
            FilterConfig(
                type=versioned("classify"),
                attribute_path=output_path_of(compression_step),
                name=versioned("compression_ratio"),
                threshold=versioned(0.65),  # Lower bound (increased from 0.6)
                upper_threshold=versioned(0.8),  # Upper bound (decreased from 0.9)
            ),
        ],
        ray_memory_limit_gb=200,
        filetype=None,
    ),
    pip_dependency_groups=["ddsketch", "lz4"],
)

filtered_dolma = tokenize_dolma_steps(
    base_path=f"tokenized/compression_filtering/{EXPERIMENT_NAME}/{input_data_source}",
    input_base_path=output_path_of(consolidate_step),
    tokenizer=llama3_tokenizer,
)

dolma_to_use = {
    "filtered_dolma/c4": filtered_dolma["dolma/c4"],
}


weights = {
    "filtered_dolma/c4": 1.0,  # Assuming we only have C4 English data
}


data_config = lm_mixture_data_config(components=dolma_to_use, weights=weights)

train_step = default_train(
    # initial version was using the original c4 by mistake
    name=f"compression_filtering/{EXPERIMENT_NAME}-v2",
    tokenized=data_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    eval_harness_tasks=[],
).with_output_path(f"compression_filtering/{EXPERIMENT_NAME}")

# also need a baseline

baseline_dolma = tokenize_dolma_steps(
    tokenizer=llama3_tokenizer,
)
baseline_dolma_to_use = {
    "dolma/c4": baseline_dolma["dolma/c4"],
}

baseline_weights = {
    "dolma/c4": 1.0,  # Assuming we only have C4 English data
}

baseline_data_config = lm_mixture_data_config(components=baseline_dolma_to_use, weights=baseline_weights)

baseline_train_step = default_train(
    name=f"baseline_dolma/{EXPERIMENT_NAME}",
    tokenized=baseline_data_config,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    eval_harness_tasks=[],
).with_output_path(f"compression_filtering/{EXPERIMENT_NAME}-baseline")


if __name__ == "__main__":
    executor_main(steps=[train_step, baseline_train_step])
