"""
Sharded n-gram overlap pipeline for all SFT instruction datasets from the mixture config.

This script uses get_instruction_dataset to fetch and transform each dataset,
then tokenizes via Marin tokenizer, computes n-gram overlaps in a sharded
manner, and consolidates results.
"""

import logging

from experiments.exp808_sft_mixture import DATASETS
from experiments.instruction_datasets import get_instruction_dataset
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path

# Add imports for Dolma conversion
from operations.transform.conversation.conversation_to_dolma import (
    ConversationToDolmaConfig,
    convert_conversation_to_dolma,
)
from train_test_overlap.run_overlap_shards import ShardedOverlapConfig, run_all_shards

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1) Fetch & convert SFT datasets into Dolma format
# -------------------------------------------------------------
dolma_datasets = {}
for short_name, hf_name in DATASETS.items():
    # Download & transform instruction dataset into document shards
    dataset_step = get_instruction_dataset(hf_name)
    # Convert to Dolma format
    dolma_step = ExecutorStep(
        name=f"dolma/{short_name}",
        fn=convert_conversation_to_dolma,
        config=ConversationToDolmaConfig(output_path_of(dataset_step)),
    )
    dolma_datasets[short_name] = dolma_step

# 2) Define overlap settings
# --------------------------
# TODO: set this to your evaluation scenarios JSONL path
scenario_data = "gs://marin-us-central2/scenarios/consolidated_eval_scenarios_final-50b720/consolidated_scenarios.jsonl"
n_values = [10, 15]

# 3) Create sharded overlap & consolidation steps
# -----------------------------------------------
overlap_steps = []
for _idx, (short_name, dolma_step) in enumerate(dolma_datasets.items()):
    # Sharded overlap computation
    overlap_config = ShardedOverlapConfig(
        base_input_dir=output_path_of(dolma_step),
        scenario_data=scenario_data,
        output_base=this_output_path(),
        N=n_values,
        max_in_flight=512,
    )
    overlap_step = ExecutorStep(
        name=f"train_test_overlap/ngrams/{short_name}",
        fn=run_all_shards,
        config=overlap_config,
    )
    overlap_steps.append(overlap_step)

if __name__ == "__main__":
    executor_main(
        steps=overlap_steps,
        description="Sharded n-gram overlap pipeline for SFT instruction datasets",
    )
